/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace llm_on_edge::memory
{

enum class ElementType : std::uint8_t
{
    kUInt8 = 0,
    kInt32 = 1,
    kFloat32 = 2,
    kFloat16 = 3,
};

[[nodiscard]] std::size_t element_size(ElementType t) noexcept;

} // namespace llm_on_edge::memory
