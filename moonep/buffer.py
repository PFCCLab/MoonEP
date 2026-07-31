from __future__ import annotations

import os
import shutil
import socket
import struct
import tempfile

import torch
import torch.distributed as dist

from moonep._C import (
    nvl_dist_alloc,
    nvl_dist_map,
    nvl_dist_alloc_fabric,
    nvl_dist_map_fabric,
    nvl_release_mem_handle,
    get_vmm_granularity as _get_vmm_granularity,
    get_vmm_granularity_fabric,
    get_multicast_granularity as _get_multicast_granularity,
    get_multicast_granularity_fabric,
    nvl_fabric_supported,
    nvl_multicast_supported,
    nvl_multicast_create,
    nvl_multicast_import,
    nvl_multicast_create_fabric,
    nvl_multicast_import_fabric,
    nvl_multicast_add_device,
    nvl_multicast_bind_map,
    nvl_multicast_bind_map_fabric,
)

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
}


def _get_global_rank(group, group_rank: int) -> int:
    if group is None:
        return group_rank
    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is not None:
        return get_global_rank(group, group_rank)
    return group.get_global_rank(group_rank)


def _mnnvl_enabled() -> bool:
    return os.environ.get("MOONEP_USE_MNNVL", "0") == "1"


def mnnvl_available() -> bool:
    return nvl_fabric_supported() and nvl_multicast_supported()


def get_vmm_granularity() -> int:
    if _mnnvl_enabled():
        return get_vmm_granularity_fabric()
    return _get_vmm_granularity()


def get_multicast_granularity(num_devices: int) -> int:
    if _mnnvl_enabled():
        return get_multicast_granularity_fabric(num_devices)
    return _get_multicast_granularity(num_devices)


def pad_to_granularity(nbytes: int) -> int:
    """Round up nbytes to VMM granularity."""
    gran = get_vmm_granularity()
    return ((nbytes + gran - 1) // gran) * gran


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype) -> int:
    """Compute the padded dim0 so that chunk bytes are aligned to VMM granularity.

    Returns the padded dim0 value (>= chunk_shape[0]).
    """
    elem_size = _ELEM_SIZE[dtype]
    inner_size = elem_size
    for d in chunk_shape[1:]:
        inner_size *= d  # bytes per row

    nbytes = chunk_shape[0] * inner_size
    padded_bytes = pad_to_granularity(nbytes)
    padded_dim0 = padded_bytes // inner_size
    # Ensure exact alignment
    while padded_dim0 * inner_size % get_vmm_granularity() != 0:
        padded_dim0 += 1
    return padded_dim0


def _exchange_ipc_fds(
    local_fd: int | None,
    sender_ranks: list[int],
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> dict[int, int]:
    """Pass POSIX fds between ranks via per-rank unix datagram sockets.

    Ranks listed in ``sender_ranks`` must pass their exported fd as
    ``local_fd`` (other ranks pass None). Every rank receives one fd from
    each sender; the kernel dups the fd into the receiving process
    (SCM_RIGHTS), so the returned fds are owned by this process and must be
    closed by the caller after import. The sender's own fd may be closed as
    soon as this function returns — a trailing barrier guarantees all peers
    have already received their copy.
    """
    # Group-rank 0 creates a shared dir; broadcast the path to the group.
    if local_rank == 0:
        dir_path = tempfile.mkdtemp(prefix="moonep_ipc_")
    else:
        dir_path = None
    obj = [dir_path]
    src = _get_global_rank(group, 0)
    dist.broadcast_object_list(obj, src=src, group=group)
    dir_path = obj[0]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(os.path.join(dir_path, f"rank_{local_rank}"))
    sock.settimeout(120)
    # All sockets must be bound before any send.
    dist.barrier(group=group)

    try:
        if local_fd is not None:
            payload = struct.pack("<i", local_rank)
            anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                    struct.pack("<i", local_fd))]
            for dst in range(world_size):
                sock.sendmsg([payload], anc, 0,
                             os.path.join(dir_path, f"rank_{dst}"))

        fds = {}
        while len(fds) < len(sender_ranks):
            msg, ancdata, _flags, _addr = sock.recvmsg(16, socket.CMSG_SPACE(4))
            src_rank = struct.unpack("<i", msg[:4])[0]
            for level, ctype, cdata in ancdata:
                if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                    fds[src_rank] = struct.unpack("<i", cdata[:4])[0]
                    break
            else:
                raise RuntimeError("received IPC message without an fd")
    finally:
        sock.close()
        # Everyone has received their fds; safe to tear down the sockets and
        # for senders to close their original fd.
        dist.barrier(group=group)
        if local_rank == 0:
            shutil.rmtree(dir_path, ignore_errors=True)
    return fds


def _exchange_fabric_handles(
    local_handle: list[int],
    world_size: int,
    group: dist.ProcessGroup | None,
) -> list[list[int]]:
    handles = [None] * world_size
    dist.all_gather_object(handles, local_handle, group=group)
    return handles


def _alloc_nvl_chunk(
    chunk_shape: list[int],
    dtype: torch.dtype,
):
    alloc = nvl_dist_alloc_fabric if _mnnvl_enabled() else nvl_dist_alloc
    return alloc(shape=chunk_shape, dtype=dtype)


def _map_nvl_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_handle,
    keepalive: torch.Tensor,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if _mnnvl_enabled():
        handles = _exchange_fabric_handles(local_handle, world_size, group)
        full_tensor = nvl_dist_map_fabric(
            chunk_shape=chunk_shape,
            dtype=dtype,
            fabric_handles=handles,
            local_rank=local_rank,
            world_size=world_size,
        )
    else:
        fds = _exchange_ipc_fds(
            local_handle, list(range(world_size)),
            local_rank, world_size, group)
        os.close(local_handle)
        all_fds = [fds[r] for r in range(world_size)]
        try:
            full_tensor = nvl_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                fds=all_fds,
                local_rank=local_rank,
                world_size=world_size,
            )
        finally:
            for fd in all_fds:
                os.close(fd)
    full_tensor._keepalive = keepalive
    return full_tensor


def create_nvl_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Allocate an NVLink distributed tensor with all-RW access.

    chunk_shape MUST already be padded to VMM granularity alignment.
    Use pad_dim0_for_alignment() to compute the padded dim0.

    `local_rank` and `world_size` must match the given `group` (or the default
    group when `group is None`).
    """
    keepalive, local_handle, owned_handle = _alloc_nvl_chunk(
        chunk_shape, dtype)
    try:
        return _map_nvl_dist_tensor(
            chunk_shape, dtype, local_handle, keepalive,
            local_rank, world_size, group,
        )
    finally:
        nvl_release_mem_handle(owned_handle)


def create_nvl_dist_multicast_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an NVLink distributed tensor and its multicast view together.

    Returns `(full_tensor, mc_view)`. `full_tensor` is the regular all-rank RW
    VMM mapping; `mc_view` is a single-chunk multicast VA whose `.data_ptr()` can
    be used by `multimem.st` to fan out writes to every rank's local chunk.

    The owned allocation handle needed by `cuMulticastBindMem` stays internal to
    this helper and is released after both mappings have been created.
    """
    keepalive, local_handle, owned_handle = _alloc_nvl_chunk(
        chunk_shape, dtype)
    try:
        full_tensor = _map_nvl_dist_tensor(
            chunk_shape, dtype, local_handle, keepalive,
            local_rank, world_size, group,
        )
        mc_view = _create_nvl_multicast_view(
            full_tensor, owned_handle, local_rank, world_size, group,
        )
        return full_tensor, mc_view
    finally:
        nvl_release_mem_handle(owned_handle)


def _create_nvl_multicast_view(
    meta_buf: torch.Tensor,
    owned_handle: int,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Overlay a multicast (NVSwitch SHARP) mapping on an existing NVL chunk.

    Binds each rank's own chunk physical memory (the chunk it `cuMemCreate`d
    inside `meta_buf`) to a single multicast object, then maps a multicast VA.
    A `multimem.st` to the returned VA fan-outs the hardware-replicated write to
    every rank's local chunk at the same offset.

    Registration is one-shot and persistent — the caller keeps the returned
    tensor alive (same lifetime as `meta_buf`). No extra device memory is used
    beyond an additional virtual address mapping over the existing chunk.

    The returned tensor's `.data_ptr()` is the multimem address, laid out as a
    single chunk (write `mc[off]` hits every rank's chunk `[off]`).
    """
    assert nvl_multicast_supported(), "Multicast not supported on this device"
    chunk_elems = meta_buf.shape.numel() // world_size
    size_bytes = chunk_elems * meta_buf.element_size()
    is_root = local_rank == 0

    if _mnnvl_enabled():
        if is_root:
            mc_handle, shared_handle = nvl_multicast_create_fabric(
                size_bytes, world_size)
        else:
            mc_handle, shared_handle = 0, None
        obj = [shared_handle]
        src = _get_global_rank(group, 0)
        dist.broadcast_object_list(obj, src=src, group=group)
        if not is_root:
            mc_handle = nvl_multicast_import_fabric(obj[0])
        bind_map = nvl_multicast_bind_map_fabric
    else:
        if is_root:
            mc_handle, shared_handle = nvl_multicast_create(
                size_bytes, world_size)
        else:
            mc_handle, shared_handle = 0, None
        fds = _exchange_ipc_fds(
            shared_handle, [0], local_rank, world_size, group)
        if is_root:
            os.close(shared_handle)
        root_fd = fds[0]
        try:
            if not is_root:
                mc_handle = nvl_multicast_import(root_fd)
        finally:
            os.close(root_fd)
        bind_map = nvl_multicast_bind_map

    nvl_multicast_add_device(mc_handle)
    dist.barrier(group=group)
    mc_view = bind_map(
        mc_handle, owned_handle, size_bytes, world_size)
    dist.barrier(group=group)
    return mc_view


def create_nvl_single_owner_tensor(
    shape: list[int],
    dtype: torch.dtype,
    owner_rank: int,
    local_rank: int,
) -> torch.Tensor:
    """Allocate a VMM tensor on one GPU, visible to all ranks via NVLink.

    The physical memory resides on owner_rank's GPU.  All ranks get an RW
    mapping so the resulting tensor can be read/written from any rank (remote
    accesses go over NVLink).  shape must already be padded to VMM granularity
    (use pad_dim0_for_alignment).
    """
    world_size = dist.get_world_size()
    if local_rank == owner_rank:
        keepalive, local_handle, owned_handle = _alloc_nvl_chunk(shape, dtype)
        nvl_release_mem_handle(owned_handle)
    else:
        local_handle = None

    if _mnnvl_enabled():
        obj = [local_handle]
        dist.broadcast_object_list(obj, src=owner_rank)
        tensor = nvl_dist_map_fabric(
            chunk_shape=shape, dtype=dtype,
            fabric_handles=[obj[0]], local_rank=0, world_size=1,
        )
        if local_rank == owner_rank:
            tensor._keepalive = keepalive
        return tensor

    fds = _exchange_ipc_fds(local_handle, [owner_rank], local_rank, world_size,
                            group=None)
    if local_handle is not None:
        os.close(local_handle)
    owner_fd = fds[owner_rank]
    try:
        tensor = nvl_dist_map(
            chunk_shape=shape, dtype=dtype,
            fds=[owner_fd], local_rank=0, world_size=1,
        )
    finally:
        os.close(owner_fd)

    if local_rank == owner_rank:
        tensor._keepalive = keepalive
    return tensor
