"""Focused GB200 MNNVL capability and VMM handle tests.

Run with:
    MOONEP_USE_MNNVL=1 torchrun --nproc_per_node=4 -m pytest tests/test_mnnvl.py
    MOONEP_USE_MNNVL=1 torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=29500 -m pytest tests/test_mnnvl.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from moonep._C import nvl_fabric_supported, nvl_multicast_supported
from moonep.buffer import (
    create_nvl_dist_multicast_tensor,
    create_nvl_dist_tensor,
    get_multicast_granularity,
    get_vmm_granularity,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("MOONEP_USE_MNNVL") != "1",
    reason="set MOONEP_USE_MNNVL=1 to run MNNVL tests",
)


def test_mnnvl_capability(dist_env):
    assert nvl_fabric_supported(), (
        "CUDA fabric allocation is unavailable; check GB200 fabric support "
        "and the container's IMEX channel"
    )
    assert nvl_multicast_supported(), "CUDA multicast is unavailable"


def test_mnnvl_shared_buffer_mapping(dist_env):
    rank, world_size = dist_env
    alignment = max(
        get_vmm_granularity(),
        get_multicast_granularity(world_size),
    )
    chunk_elems = alignment // torch.tensor([], dtype=torch.int32).element_size()

    shared = create_nvl_dist_tensor(
        [chunk_elems], torch.int32, rank, world_size)
    chunks = shared.view(world_size, chunk_elems)
    chunks[rank, :8].fill_(rank + 1)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    for src_rank in range(world_size):
        expected = torch.full_like(chunks[src_rank, :8], src_rank + 1)
        assert torch.equal(chunks[src_rank, :8], expected)

    remote_rank = (rank + 1) % world_size
    chunks[remote_rank, 8:16].fill_(100 + rank)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    for dst_rank in range(world_size):
        src_rank = (dst_rank - 1) % world_size
        expected = torch.full_like(chunks[dst_rank, 8:16], 100 + src_rank)
        assert torch.equal(chunks[dst_rank, 8:16], expected)

    meta, multicast = create_nvl_dist_multicast_tensor(
        [chunk_elems], torch.int32, rank, world_size)
    assert meta.numel() == world_size * chunk_elems
    assert multicast.numel() == chunk_elems
