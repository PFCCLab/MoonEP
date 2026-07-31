#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "paddle/common/ddim.h"
#include "paddle/phi/api/include/compat/utils/scalar_type_conversion.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

#ifndef CUCHECK
#define CUCHECK(cmd) do {                                      \
    CUresult err = cmd;                                            \
    if (err != CUDA_SUCCESS) {                                     \
        const char *errStr;                                        \
        cuGetErrorString(err, &errStr);                            \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, errStr);                           \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
#endif

#ifndef CUDACHECK
#define CUDACHECK(cmd) do {                                    \
    cudaError_t err = cmd;                                         \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
#endif

static inline std::tuple<size_t, size_t, int> nvl_prepare_for_handle_type(
    const std::vector<int64_t> &shape,
    at::ScalarType dtype,
    CUmemAllocationHandleType handle_type
) {
    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");

    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));

    size_t size = c10::elementSize(dtype);
    for (auto dim : shape) {
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        size *= static_cast<size_t>(dim);
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = handle_type;

    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t allocated_size = (size + granularity - 1) / granularity * granularity;

    return {size, allocated_size, device_id};
}

static inline std::tuple<size_t, size_t, int> nvl_prepare(
    const std::vector<int64_t> &shape,
    at::ScalarType dtype
) {
    return nvl_prepare_for_handle_type(
        shape, dtype, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

static inline at::Tensor make_external_cuda_tensor(
    void *ptr,
    size_t nbytes,
    int device_id,
    const std::vector<int64_t> &shape,
    at::ScalarType dtype,
    std::function<void(void *)> deleter
) {
    // TODO(Paddle compat): use at::from_blob once it accepts peer/multicast VMM
    // addresses whose physical allocation device differs from the logical
    // current device. The holder below preserves that logical place without
    // changing MoonEP's VMM layout or lifetime.
    auto holder = std::shared_ptr<phi::Allocation>(
        new phi::Allocation(
            ptr, nbytes, phi::GPUPlace(device_id)),
        [deleter = std::move(deleter)](phi::Allocation *allocation) mutable {
            if (deleter) {
                deleter(allocation->ptr());
            }
            delete allocation;
        });
    auto meta = phi::DenseTensorMeta(
        compat::_PD_AtenScalarTypeToPhiDataType(dtype),
        common::make_ddim(shape),
        phi::DataLayout::NCHW);
    paddle::Tensor tensor(
        std::make_shared<phi::DenseTensor>(holder, meta));
    return at::Tensor(tensor);
}

static inline at::Tensor make_vmm_tensor(
    void *ptr,
    size_t nbytes,
    size_t allocated_size,
    int device_id,
    const std::vector<int64_t> &shape,
    at::ScalarType dtype
) {
    auto deleter = [device_id, ptr, allocated_size](void *p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
    };

    return make_external_cuda_tensor(
        ptr, nbytes, device_id, shape, dtype, std::move(deleter));
}

static inline std::tuple<at::Tensor, CUmemGenericAllocationHandle>
nvl_dist_alloc_for_handle_type(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    CUmemAllocationHandleType handle_type
) {
    auto [nbytes, allocated_size, device_id] =
        nvl_prepare_for_handle_type(chunk_shape, dtype, handle_type);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = handle_type;

    CUmemGenericAllocationHandle mem_handle;
    CUCHECK(cuMemCreate(&mem_handle, allocated_size, &prop, 0));

    CUdeviceptr dptr;
    CUCHECK(cuMemAddressReserve(&dptr, allocated_size, 0, 0, 0));
    CUCHECK(cuMemMap(dptr, allocated_size, 0, mem_handle, 0));

    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, allocated_size, &access_desc, 1));

    auto keepalive = make_vmm_tensor(
        reinterpret_cast<void *>(dptr), nbytes, allocated_size,
        device_id, chunk_shape, dtype);
    return {std::move(keepalive), mem_handle};
}

// Returns (keepalive VA tensor, exported POSIX fd, owned mem handle as int64).
// The caller releases the owned handle after optional multicast binding, and
// closes the fd after every peer has imported it.
inline std::tuple<at::Tensor, int64_t, int64_t> nvl_dist_alloc(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype
) {
    auto [keepalive, mem_handle] = nvl_dist_alloc_for_handle_type(
        chunk_shape, dtype, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    int fd = -1;
    CUCHECK(cuMemExportToShareableHandle(
        &fd, mem_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");
    return {std::move(keepalive), static_cast<int64_t>(fd),
            static_cast<int64_t>(mem_handle)};
}

inline std::tuple<at::Tensor, std::vector<uint8_t>, int64_t>
nvl_dist_alloc_fabric(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype
) {
    auto [keepalive, mem_handle] = nvl_dist_alloc_for_handle_type(
        chunk_shape, dtype, CU_MEM_HANDLE_TYPE_FABRIC);
    std::vector<uint8_t> fabric_handle(sizeof(CUmemFabricHandle));
    CUCHECK(cuMemExportToShareableHandle(
        fabric_handle.data(), mem_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    return {std::move(keepalive), std::move(fabric_handle),
            static_cast<int64_t>(mem_handle)};
}

// Release the owned mem handle returned by nvl_dist_alloc (the physical memory
// stays alive via the existing maps).
static inline void nvl_release_mem_handle(int64_t mem_handle_u64) {
    CUCHECK(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(mem_handle_u64)));
}

template <typename ImportHandle>
static inline at::Tensor nvl_dist_map_impl(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    int64_t world_size,
    CUmemAllocationHandleType handle_type,
    ImportHandle import_handle
) {
    auto [chunk_nbytes, chunk_allocated_size, device_id] =
        nvl_prepare_for_handle_type(chunk_shape, dtype, handle_type);

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        "are a multiple of the allocation granularity.");

    size_t total_allocated_size = chunk_allocated_size * world_size;

    CUdeviceptr dptr;
    CUCHECK(cuMemAddressReserve(&dptr, total_allocated_size, 0, 0, 0));

    for (int64_t i = 0; i < world_size; i++) {
        CUmemGenericAllocationHandle mem_handle = import_handle(i);
        CUdeviceptr chunk_va = dptr + i * chunk_allocated_size;
        CUCHECK(cuMemMap(chunk_va, chunk_allocated_size, 0, mem_handle, 0));

        // All chunks get RW access for dispatch writes
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(chunk_va, chunk_allocated_size, &access, 1));

        CUCHECK(cuMemRelease(mem_handle));
    }

    std::vector<int64_t> full_shape = chunk_shape;
    full_shape[0] *= world_size;
    size_t total_nbytes = chunk_nbytes * world_size;

    int ws = world_size;
    size_t cas = chunk_allocated_size;
    size_t tas = total_allocated_size;
    auto deleter = [device_id, dptr, cas, ws, tas](void *p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        for (int i = 0; i < ws; i++) {
            cuMemUnmap(dptr + i * cas, cas);
        }
        cuMemAddressFree(dptr, tas);
    };

    return make_external_cuda_tensor(
        reinterpret_cast<void *>(dptr), total_nbytes, device_id,
        full_shape, dtype, std::move(deleter));
}

/**
 * Map all POSIX-fd chunks into one contiguous VA region with RW access.
 * The caller may close each fd after this function imports it.
 */
inline at::Tensor nvl_dist_map(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    const std::vector<int64_t> &fds,
    int64_t local_rank,
    int64_t world_size
) {
    TORCH_CHECK((int64_t)fds.size() == world_size,
        "fds.size()=", fds.size(), " != world_size=", world_size);
    return nvl_dist_map_impl(
        chunk_shape, dtype, world_size,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        [&fds](int64_t i) {
            CUmemGenericAllocationHandle mem_handle;
            int fd = static_cast<int>(fds[i]);
            CUCHECK(cuMemImportFromShareableHandle(
                &mem_handle,
                reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            return mem_handle;
        });
}

// Fabric handles are ordinary bytes and can cross separate OS instances.
inline at::Tensor nvl_dist_map_fabric(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    const std::vector<std::vector<uint8_t>> &fabric_handles,
    int64_t local_rank,
    int64_t world_size
) {
    TORCH_CHECK((int64_t)fabric_handles.size() == world_size,
        "fabric_handles.size()=", fabric_handles.size(),
        " != world_size=", world_size);
    return nvl_dist_map_impl(
        chunk_shape, dtype, world_size, CU_MEM_HANDLE_TYPE_FABRIC,
        [&fabric_handles](int64_t i) {
            TORCH_CHECK(
                fabric_handles[i].size() == sizeof(CUmemFabricHandle),
                "invalid fabric handle size ", fabric_handles[i].size(),
                ", expected ", sizeof(CUmemFabricHandle));
            CUmemGenericAllocationHandle mem_handle;
            CUCHECK(cuMemImportFromShareableHandle(
                &mem_handle,
                const_cast<uint8_t *>(fabric_handles[i].data()),
                CU_MEM_HANDLE_TYPE_FABRIC));
            return mem_handle;
        });
}

// ============================================================
// Multicast (NVSwitch SHARP) — register once, hold persistently
// ============================================================
//
// Adds a multicast mapping on top of an already-allocated NVLink chunk (each
// rank's own cuMemCreate physical memory): bind every rank's chunk physical
// memory to the same multicast object, then map a multicast VA. A single
// multimem.st write to the multicast VA is fanned out by NVSwitch hardware to
// the corresponding offset of every rank's local physical memory.
//
// Uses no extra device memory — it only adds one more VA mapping over the
// existing chunk physical memory. Lifetime: the caller (MoonEP Buffer) holds
// the returned mc VA tensor, same lifetime as meta_buf; the mc VA tensor's
// deleter releases the multicast handle after unmap/free of the VA.
//
// Registration must follow the Python-side coordination order (separated by
// dist.barrier, otherwise bind fails with CUDA_ERROR_ILLEGAL_STATE or hangs):
//   root: nvl_multicast_create -> send the exported POSIX fd to other ranks
//   non-root: nvl_multicast_import
//   all ranks: nvl_multicast_add_device
//   --- dist.barrier ---
//   all ranks: nvl_multicast_bind_map
//   --- dist.barrier ---

// Check both the device attribute and whether this process can actually create
// a fabric allocation (the latter requires a usable IMEX channel).
static inline bool nvl_fabric_supported() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));

    int supported = 0;
    CUCHECK(cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    if (!supported) return false;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t size = 0;
    if (cuMemGetAllocationGranularity(
            &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
        return false;
    }

    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) return false;
    CUCHECK(cuMemRelease(handle));
    return true;
}

// Whether the device supports multicast.
static inline bool nvl_multicast_supported() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    int supported = 0;
    CUCHECK(cuDeviceGetAttribute(&supported,
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_device));
    return supported != 0;
}

// Recommended multicast alignment granularity (bytes). Both the addr and size
// of a bind must be aligned to it.
static inline size_t nvl_multicast_granularity_for_handle_type(
    int num_devices,
    CUmemAllocationHandleType handle_type
) {
    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = 0;
    prop.handleTypes = handle_type;
    prop.flags = 0;
    size_t gran = 0;
    CUCHECK(cuMulticastGetGranularity(
        &gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    return gran;
}

static inline size_t nvl_multicast_granularity(int num_devices) {
    return nvl_multicast_granularity_for_handle_type(
        num_devices, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

static inline int64_t get_multicast_granularity(int64_t num_devices) {
    return static_cast<int64_t>(
        nvl_multicast_granularity(static_cast<int>(num_devices)));
}

static inline int64_t get_multicast_granularity_fabric(int64_t num_devices) {
    return static_cast<int64_t>(nvl_multicast_granularity_for_handle_type(
        static_cast<int>(num_devices), CU_MEM_HANDLE_TYPE_FABRIC));
}

static inline CUmemGenericAllocationHandle
nvl_multicast_create_for_handle_type(
    int64_t size_bytes,
    int64_t num_devices,
    CUmemAllocationHandleType handle_type
) {
    TORCH_CHECK(nvl_multicast_supported(),
        "Multicast not supported on this device");
    const size_t gran = nvl_multicast_granularity_for_handle_type(
        static_cast<int>(num_devices), handle_type);
    const size_t aligned =
        (static_cast<size_t>(size_bytes) + gran - 1) / gran * gran;

    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = aligned;
    prop.handleTypes = handle_type;
    prop.flags = 0;

    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMulticastCreate(&mc_handle, &prop));
    return mc_handle;
}

// Root creates the multicast object and exports a POSIX fd.
inline std::tuple<int64_t, int64_t> nvl_multicast_create(
    int64_t size_bytes, int64_t num_devices
) {
    CUmemGenericAllocationHandle mc_handle =
        nvl_multicast_create_for_handle_type(
            size_bytes, num_devices, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    int fd = -1;
    CUCHECK(cuMemExportToShareableHandle(
        &fd, mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");

    return {static_cast<int64_t>(mc_handle), static_cast<int64_t>(fd)};
}

// non-root: import the multicast object from the POSIX fd sent by root. The
// caller may close the fd once it is imported.
inline int64_t nvl_multicast_import(int64_t fd) {
    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMemImportFromShareableHandle(
        &mc_handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    return static_cast<int64_t>(mc_handle);
}

inline std::tuple<int64_t, std::vector<uint8_t>> nvl_multicast_create_fabric(
    int64_t size_bytes, int64_t num_devices
) {
    CUmemGenericAllocationHandle mc_handle =
        nvl_multicast_create_for_handle_type(
            size_bytes, num_devices, CU_MEM_HANDLE_TYPE_FABRIC);
    std::vector<uint8_t> fabric_handle(sizeof(CUmemFabricHandle));
    CUCHECK(cuMemExportToShareableHandle(
        fabric_handle.data(), mc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    return {static_cast<int64_t>(mc_handle), std::move(fabric_handle)};
}

inline int64_t nvl_multicast_import_fabric(
    const std::vector<uint8_t> &fabric_handle
) {
    TORCH_CHECK(fabric_handle.size() == sizeof(CUmemFabricHandle),
        "invalid fabric handle size ", fabric_handle.size(),
        ", expected ", sizeof(CUmemFabricHandle));
    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMemImportFromShareableHandle(
        &mc_handle, const_cast<uint8_t *>(fabric_handle.data()),
        CU_MEM_HANDLE_TYPE_FABRIC));
    return static_cast<int64_t>(mc_handle);
}

// all ranks: add this rank's device to the multicast group. Must happen before
// bind, and bind may only happen after all ranks have added (dist.barrier).
inline void nvl_multicast_add_device(int64_t mc_handle_u64) {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    CUCHECK(cuMulticastAddDevice(
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64), cu_device));
}

// all ranks: bind this rank's local chunk owned physical memory (the mem
// handle kept by nvl_dist_alloc) to the multicast object, then reserve a VA
// for the multicast handle + map + setAccess. Returns an int32 keepalive
// tensor holding the multicast VA (.data_ptr() is the multimem address). Uses
// BindMem (owned handle) instead of BindAddr: BindAddr returns invalid
// argument for IPC-imported VAs on this platform.
static inline at::Tensor nvl_multicast_bind_map_for_handle_type(
    int64_t mc_handle_u64, int64_t owned_mem_handle_u64,
    int64_t size_bytes, int64_t num_devices,
    CUmemAllocationHandleType handle_type
) {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUmemGenericAllocationHandle mc_handle =
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64);
    CUmemGenericAllocationHandle mem_handle =
        static_cast<CUmemGenericAllocationHandle>(owned_mem_handle_u64);

    const size_t gran = nvl_multicast_granularity_for_handle_type(
        static_cast<int>(num_devices), handle_type);
    TORCH_CHECK(static_cast<size_t>(size_bytes) % gran == 0,
        "multicast bind size (", size_bytes, ") must be a multiple of multicast "
        "granularity (", gran, "); pad meta_buf chunk accordingly");
    const size_t size = static_cast<size_t>(size_bytes);

    // Bind this rank's owned physical allocation to offset 0 of the multicast object.
    CUCHECK(cuMulticastBindMem(
        mc_handle, /*mcOffset=*/0, mem_handle, /*memOffset=*/0, size, 0));

    // Reserve a separate VA for the multicast handle and map it. Pass 0 for
    // alignment (the driver aligns to granularity by default, same as
    // nvl_dist_alloc/nvl_dist_map; a non-zero alignment is rejected by
    // cuMemAddressReserve as invalid argument).
    CUdeviceptr mc_va;
    CUCHECK(cuMemAddressReserve(&mc_va, size, 0, 0, 0));
    CUCHECK(cuMemMap(mc_va, size, 0, mc_handle, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(mc_va, size, &access, 1));

    // keepalive: the deleter unmaps + frees the VA, then releases mc_handle.
    // data_ptr() = multimem address.
    auto deleter = [device_id, mc_va, size, mc_handle](void* p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        cuMemUnmap(mc_va, size);
        cuMemAddressFree(mc_va, size);
        cuMemRelease(mc_handle);
    };

    std::vector<int64_t> shape = {static_cast<int64_t>(size / sizeof(int32_t))};
    return make_external_cuda_tensor(
        reinterpret_cast<void*>(mc_va), size, device_id,
        shape, at::kInt, std::move(deleter));
}

inline at::Tensor nvl_multicast_bind_map(
    int64_t mc_handle_u64, int64_t owned_mem_handle_u64,
    int64_t size_bytes, int64_t num_devices
) {
    return nvl_multicast_bind_map_for_handle_type(
        mc_handle_u64, owned_mem_handle_u64, size_bytes, num_devices,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

inline at::Tensor nvl_multicast_bind_map_fabric(
    int64_t mc_handle_u64, int64_t owned_mem_handle_u64,
    int64_t size_bytes, int64_t num_devices
) {
    return nvl_multicast_bind_map_for_handle_type(
        mc_handle_u64, owned_mem_handle_u64, size_bytes, num_devices,
        CU_MEM_HANDLE_TYPE_FABRIC);
}
