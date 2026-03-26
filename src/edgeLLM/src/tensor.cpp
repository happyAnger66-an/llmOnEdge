/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/tensor.h"

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/cuda_check.h"

#include <cuda_runtime.h>

#include <cstring>
#include <numeric>

namespace llm_on_edge::memory
{

namespace
{
BufferManager& global_buffer_manager()
{
    static BufferManager manager;
    return manager;
}

std::int64_t volume(std::vector<int64_t> const& shape)
{
    if (shape.empty())
    {
        return 0;
    }
    std::int64_t v = 1;
    for (std::int64_t d : shape)
    {
        if (d < 0)
        {
            throw std::invalid_argument("Tensor: negative dimension");
        }
        v *= d;
    }
    return v;
}
} // namespace

Tensor::Tensor(std::vector<int64_t> shape, ElementType element_type, MemoryType memory_type)
    : Tensor(global_buffer_manager(), std::move(shape), element_type, memory_type)
{
}

Tensor::Tensor(BufferManager& manager, std::vector<int64_t> shape, ElementType element_type, MemoryType memory_type)
    : m_shape(std::move(shape))
    , m_element_type(element_type)
{
    std::int64_t const v = volume(m_shape);
    std::size_t const bytes = static_cast<std::size_t>(v) * element_size(m_element_type);

    switch (memory_type)
    {
    case MemoryType::kGPU: m_buffer = manager.gpu(bytes); break;
    case MemoryType::kCPU: m_buffer = manager.cpu(bytes); break;
    case MemoryType::kPINNED: m_buffer = manager.pinned(bytes); break;
    }
    register_tensor();
}

Tensor::Tensor(Buffer::SharedPtr buffer, std::vector<int64_t> shape, ElementType element_type)
    : m_buffer(std::move(buffer))
    , m_shape(std::move(shape))
    , m_element_type(element_type)
{
    if (!m_buffer)
    {
        throw std::invalid_argument("Tensor: null buffer");
    }
    std::size_t const need = static_cast<std::size_t>(volume(m_shape)) * element_size(m_element_type);
    if (need != m_buffer->size_bytes())
    {
        throw std::invalid_argument("Tensor: shape * element_size != buffer size");
    }
    register_tensor();
}

Tensor::~Tensor()
{
    unregister_tensor();
}

Tensor::Tensor(Tensor&& other) noexcept
    : m_buffer(std::move(other.m_buffer))
    , m_shape(std::move(other.m_shape))
    , m_element_type(other.m_element_type)
{
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    unregister_tensor();
    m_buffer = std::move(other.m_buffer);
    m_shape = std::move(other.m_shape);
    m_element_type = other.m_element_type;
    return *this;
}

void Tensor::register_tensor()
{
    if (!m_buffer)
    {
        return;
    }
    if (auto s = m_buffer->memory_stats())
    {
        s->tensor_live_add(m_buffer->size_bytes());
    }
}

void Tensor::unregister_tensor()
{
    if (!m_buffer)
    {
        return;
    }
    if (auto s = m_buffer->memory_stats())
    {
        s->tensor_live_remove(m_buffer->size_bytes());
    }
}

std::size_t Tensor::num_elements() const
{
    return static_cast<std::size_t>(volume(m_shape));
}

std::size_t Tensor::size_bytes() const
{
    return m_buffer->size_bytes();
}

void Tensor::reshape(std::vector<int64_t> new_shape)
{
    std::int64_t const v = volume(new_shape);
    if (static_cast<std::size_t>(v) * element_size(m_element_type) != m_buffer->size_bytes())
    {
        throw std::invalid_argument("Tensor::reshape: volume mismatch");
    }
    m_shape = std::move(new_shape);
}

Tensor Tensor::copy_to(BufferManager& manager, MemoryType memory_type, cudaStream_t stream) const
{
    Tensor dst(manager, m_shape, m_element_type, memory_type);
    copy_to(dst, manager, stream);
    return dst;
}

Tensor Tensor::copy_to(MemoryType memory_type, cudaStream_t stream) const
{
    return copy_to(global_buffer_manager(), memory_type, stream);
}

void Tensor::copy_to(Tensor& dst, BufferManager const& manager, cudaStream_t stream) const
{
    if (!m_buffer || !dst.m_buffer)
    {
        throw std::invalid_argument("Tensor::copy_to: null buffer");
    }
    if (size_bytes() != dst.size_bytes())
    {
        throw std::invalid_argument("Tensor::copy_to: size mismatch");
    }
    manager.copy(*m_buffer, *dst.m_buffer, stream);
}

void Tensor::copy_to(Tensor& dst, cudaStream_t stream) const
{
    copy_to(dst, global_buffer_manager(), stream);
}

void Tensor::set_zero(cudaStream_t stream)
{
    if (!m_buffer || m_buffer->size_bytes() == 0)
    {
        return;
    }
    if (m_buffer->memory_type() == MemoryType::kGPU)
    {
        LLMONEDGE_CUDA_CHECK(cudaSetDevice(m_buffer->device_id()));
        LLMONEDGE_CUDA_CHECK(cudaMemsetAsync(m_buffer->data(), 0, m_buffer->size_bytes(), stream));
        return;
    }
    std::memset(m_buffer->data(), 0, m_buffer->size_bytes());
}

void Tensor::set_from(Tensor const& src, BufferManager const& manager, cudaStream_t stream)
{
    src.copy_to(*this, manager, stream);
}

void Tensor::set_from(Tensor const& src, cudaStream_t stream)
{
    set_from(src, global_buffer_manager(), stream);
}

} // namespace llm_on_edge::memory
