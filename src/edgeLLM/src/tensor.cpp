/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/tensor.h"

#include <numeric>

namespace llm_on_edge::memory
{

namespace
{
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

} // namespace llm_on_edge::memory
