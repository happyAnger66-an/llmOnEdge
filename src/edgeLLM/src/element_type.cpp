/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/element_type.h"

#include <cstdint>

namespace llm_on_edge::memory
{

std::size_t element_size(ElementType t) noexcept
{
    switch (t)
    {
    case ElementType::kUInt8: return 1;
    case ElementType::kInt32: return sizeof(std::int32_t);
    case ElementType::kFloat32: return sizeof(float);
    case ElementType::kFloat16: return 2;
    default: return 1;
    }
}

} // namespace llm_on_edge::memory
