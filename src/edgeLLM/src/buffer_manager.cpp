/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"

#include "llm_on_edge/memory/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace llm_on_edge::memory
{

BufferManager::BufferManager(std::shared_ptr<MemoryStats> stats, int device_id)
    : m_stats(stats ? std::move(stats) : MemoryStats::global())
    , m_device_id(device_id)
{
}

Buffer::SharedPtr BufferManager::gpu(std::size_t size_bytes)
{
    if (size_bytes == 0)
    {
        return Buffer::SharedPtr(new Buffer(nullptr, 0, 0, MemoryType::kGPU, m_device_id, m_stats));
    }
    LLMONEDGE_CUDA_CHECK(cudaSetDevice(m_device_id));
    void* p = nullptr;
    LLMONEDGE_CUDA_CHECK(cudaMalloc(&p, size_bytes));
    m_stats->on_allocate(MemoryType::kGPU, size_bytes);
    return Buffer::SharedPtr(new Buffer(p, size_bytes, size_bytes, MemoryType::kGPU, m_device_id, m_stats));
}

Buffer::SharedPtr BufferManager::cpu(std::size_t size_bytes)
{
    if (size_bytes == 0)
    {
        return Buffer::SharedPtr(new Buffer(nullptr, 0, 0, MemoryType::kCPU, 0, m_stats));
    }
    void* p = std::malloc(size_bytes);
    if (!p)
    {
        throw std::bad_alloc();
    }
    m_stats->on_allocate(MemoryType::kCPU, size_bytes);
    return Buffer::SharedPtr(new Buffer(p, size_bytes, size_bytes, MemoryType::kCPU, 0, m_stats));
}

Buffer::SharedPtr BufferManager::pinned(std::size_t size_bytes)
{
    if (size_bytes == 0)
    {
        return Buffer::SharedPtr(new Buffer(nullptr, 0, 0, MemoryType::kPINNED, 0, m_stats));
    }
    void* p = nullptr;
    LLMONEDGE_CUDA_CHECK(cudaHostAlloc(&p, size_bytes, cudaHostAllocDefault));
    m_stats->on_allocate(MemoryType::kPINNED, size_bytes);
    return Buffer::SharedPtr(new Buffer(p, size_bytes, size_bytes, MemoryType::kPINNED, 0, m_stats));
}

void BufferManager::copy(Buffer const& src, Buffer& dst, cudaStream_t stream) const
{
    if (src.size_bytes() != dst.size_bytes())
    {
        throw std::invalid_argument("BufferManager::copy: size mismatch");
    }
    std::size_t const n = src.size_bytes();
    if (n == 0)
    {
        return;
    }

    cudaStream_t const s = stream ? stream : static_cast<cudaStream_t>(0);

    MemoryType const st = src.memory_type();
    MemoryType const dt = dst.memory_type();

    if (st == MemoryType::kGPU && dt == MemoryType::kGPU)
    {
        if (src.device_id() != dst.device_id())
        {
            throw std::invalid_argument("BufferManager::copy: D2D cross-device not supported in minimal build");
        }
        LLMONEDGE_CUDA_CHECK(cudaSetDevice(dst.device_id()));
        LLMONEDGE_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), n, cudaMemcpyDeviceToDevice, s));
        return;
    }

    if ((st == MemoryType::kCPU || st == MemoryType::kPINNED) && (dt == MemoryType::kCPU || dt == MemoryType::kPINNED))
    {
        std::memcpy(dst.data(), src.data(), n);
        return;
    }

    if ((st == MemoryType::kCPU || st == MemoryType::kPINNED) && dt == MemoryType::kGPU)
    {
        LLMONEDGE_CUDA_CHECK(cudaSetDevice(dst.device_id()));
        LLMONEDGE_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), n, cudaMemcpyHostToDevice, s));
        return;
    }

    if (st == MemoryType::kGPU && (dt == MemoryType::kCPU || dt == MemoryType::kPINNED))
    {
        LLMONEDGE_CUDA_CHECK(cudaSetDevice(src.device_id()));
        LLMONEDGE_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), n, cudaMemcpyDeviceToHost, s));
        return;
    }
}

} // namespace llm_on_edge::memory
