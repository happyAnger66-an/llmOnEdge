/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace llm_on_edge::memory
{

/// Logical memory kind (aligned with common TRT-LLM naming; extend as needed).
enum class MemoryType : std::int32_t
{
    kGPU = 0,
    kCPU = 1,
    kPINNED = 2,
};

char const* memory_type_name(MemoryType t) noexcept;

} // namespace llm_on_edge::memory
