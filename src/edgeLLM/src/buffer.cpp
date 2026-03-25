/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer.h"

#include "llm_on_edge/memory/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace llm_on_edge::memory
{

Buffer::Buffer(void* ptr, std::size_t size_bytes, std::size_t capacity_bytes, MemoryType type, int device_id,
    std::shared_ptr<MemoryStats> stats) noexcept
    : m_data(ptr)
    , m_size_bytes(size_bytes)
    , m_capacity_bytes(capacity_bytes)
    , m_type(type)
    , m_device_id(device_id)
    , m_stats(std::move(stats))
{
}

Buffer::~Buffer()
{
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : m_data(other.m_data)
    , m_size_bytes(other.m_size_bytes)
    , m_capacity_bytes(other.m_capacity_bytes)
    , m_type(other.m_type)
    , m_device_id(other.m_device_id)
    , m_stats(std::move(other.m_stats))
{
    other.m_data = nullptr;
    other.m_size_bytes = 0;
    other.m_capacity_bytes = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    destroy();
    m_data = other.m_data;
    m_size_bytes = other.m_size_bytes;
    m_capacity_bytes = other.m_capacity_bytes;
    m_type = other.m_type;
    m_device_id = other.m_device_id;
    m_stats = std::move(other.m_stats);
    other.m_data = nullptr;
    other.m_size_bytes = 0;
    other.m_capacity_bytes = 0;
    return *this;
}

void Buffer::destroy() noexcept
{
    if (!m_data || !m_stats)
    {
        m_data = nullptr;
        m_size_bytes = 0;
        m_capacity_bytes = 0;
        return;
    }

    void* const ptr = m_data;
    std::size_t const cap = m_capacity_bytes;
    MemoryType const t = m_type;
    int const dev = m_device_id;
    std::shared_ptr<MemoryStats> stats = m_stats;

    m_data = nullptr;
    m_size_bytes = 0;
    m_capacity_bytes = 0;

    switch (t)
    {
    case MemoryType::kGPU:
        if (cap > 0)
        {
            cudaError_t e = cudaSetDevice(dev);
            if (e == cudaSuccess)
            {
                (void)cudaFree(ptr);
            }
            stats->on_deallocate(MemoryType::kGPU, cap);
        }
        break;
    case MemoryType::kCPU:
        if (cap > 0)
        {
            std::free(ptr);
            stats->on_deallocate(MemoryType::kCPU, cap);
        }
        break;
    case MemoryType::kPINNED:
        if (cap > 0)
        {
            (void)cudaFreeHost(ptr);
            stats->on_deallocate(MemoryType::kPINNED, cap);
        }
        break;
    }
}

void Buffer::release() noexcept
{
    destroy();
}

void Buffer::resize(std::size_t new_size_bytes)
{
    if (new_size_bytes <= m_capacity_bytes)
    {
        m_size_bytes = new_size_bytes;
        return;
    }

    if (!m_stats)
    {
        throw std::logic_error("Buffer::resize: stats not set");
    }

    void* old_ptr = m_data;
    std::size_t const old_size = m_size_bytes;
    std::size_t const old_cap = m_capacity_bytes;
    MemoryType const t = m_type;
    int const dev = m_device_id;
    std::shared_ptr<MemoryStats> stats = m_stats;

    void* new_ptr = nullptr;
    switch (t)
    {
    case MemoryType::kGPU:
        LLMONEDGE_CUDA_CHECK(cudaSetDevice(dev));
        LLMONEDGE_CUDA_CHECK(cudaMalloc(&new_ptr, new_size_bytes));
        stats->on_allocate(MemoryType::kGPU, new_size_bytes);
        break;
    case MemoryType::kCPU:
        new_ptr = std::malloc(new_size_bytes);
        if (!new_ptr)
        {
            throw std::bad_alloc();
        }
        stats->on_allocate(MemoryType::kCPU, new_size_bytes);
        break;
    case MemoryType::kPINNED:
        LLMONEDGE_CUDA_CHECK(cudaHostAlloc(&new_ptr, new_size_bytes, cudaHostAllocDefault));
        stats->on_allocate(MemoryType::kPINNED, new_size_bytes);
        break;
    }

    if (old_ptr && old_size > 0)
    {
        if (t == MemoryType::kGPU)
        {
            LLMONEDGE_CUDA_CHECK(cudaSetDevice(dev));
            LLMONEDGE_CUDA_CHECK(cudaMemcpy(new_ptr, old_ptr, old_size, cudaMemcpyDeviceToDevice));
        }
        else
        {
            std::memcpy(new_ptr, old_ptr, old_size);
        }
    }

    if (old_ptr && old_cap > 0)
    {
        switch (t)
        {
        case MemoryType::kGPU:
            LLMONEDGE_CUDA_CHECK(cudaSetDevice(dev));
            LLMONEDGE_CUDA_CHECK(cudaFree(old_ptr));
            stats->on_deallocate(MemoryType::kGPU, old_cap);
            break;
        case MemoryType::kCPU:
            std::free(old_ptr);
            stats->on_deallocate(MemoryType::kCPU, old_cap);
            break;
        case MemoryType::kPINNED:
            LLMONEDGE_CUDA_CHECK(cudaFreeHost(old_ptr));
            stats->on_deallocate(MemoryType::kPINNED, old_cap);
            break;
        }
    }

    m_data = new_ptr;
    m_size_bytes = new_size_bytes;
    m_capacity_bytes = new_size_bytes;
}

} // namespace llm_on_edge::memory
