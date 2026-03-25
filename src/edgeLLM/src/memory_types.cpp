/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/memory_types.h"

namespace llm_on_edge::memory
{

char const* memory_type_name(MemoryType t) noexcept
{
    switch (t)
    {
    case MemoryType::kGPU: return "GPU";
    case MemoryType::kCPU: return "CPU";
    case MemoryType::kPINNED: return "PINNED";
    default: return "UNKNOWN";
    }
}

} // namespace llm_on_edge::memory
