/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/buffer.h"
#include "llm_on_edge/memory/element_type.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llm_on_edge::memory
{

/// Contiguous tensor view over a Buffer (TRT-LLM ITensor style, minimal).
class Tensor
{
public:
    Tensor(Buffer::SharedPtr buffer, std::vector<int64_t> shape, ElementType element_type);

    [[nodiscard]] Buffer::SharedPtr buffer() const noexcept { return m_buffer; }
    [[nodiscard]] std::vector<int64_t> const& shape() const noexcept { return m_shape; }
    [[nodiscard]] ElementType element_type() const noexcept { return m_element_type; }

    [[nodiscard]] std::size_t num_elements() const;
    [[nodiscard]] std::size_t size_bytes() const;

    /// Same element count; may update shape only.
    void reshape(std::vector<int64_t> new_shape);

private:
    Buffer::SharedPtr m_buffer;
    std::vector<int64_t> m_shape;
    ElementType m_element_type;
};

} // namespace llm_on_edge::memory
