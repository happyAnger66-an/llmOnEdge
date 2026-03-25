/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/memory_types.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace llm_on_edge::memory
{

/// Thread-safe byte counters per memory class, plus last diff for logging (TRT-LLM MemoryCounters style).
class MemoryStats
{
public:
    using SizeType = std::uint64_t;
    using DiffType = std::int64_t;

    MemoryStats() = default;

    void on_allocate(MemoryType t, SizeType bytes) noexcept;
    void on_deallocate(MemoryType t, SizeType bytes) noexcept;

    [[nodiscard]] SizeType current_gpu() const noexcept { return m_gpu.load(std::memory_order_relaxed); }
    [[nodiscard]] SizeType current_cpu() const noexcept { return m_cpu.load(std::memory_order_relaxed); }
    [[nodiscard]] SizeType current_pinned() const noexcept { return m_pinned.load(std::memory_order_relaxed); }

    [[nodiscard]] DiffType last_diff_gpu() const noexcept { return m_last_diff_gpu.load(std::memory_order_relaxed); }
    [[nodiscard]] DiffType last_diff_cpu() const noexcept { return m_last_diff_cpu.load(std::memory_order_relaxed); }
    [[nodiscard]] DiffType last_diff_pinned() const noexcept
    {
        return m_last_diff_pinned.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static std::string bytes_to_string(SizeType bytes, int precision = 2);
    [[nodiscard]] std::string to_string() const;

private:
    void bump(MemoryType t, DiffType delta) noexcept;

    std::atomic<SizeType> m_gpu{};
    std::atomic<SizeType> m_cpu{};
    std::atomic<SizeType> m_pinned{};

    std::atomic<DiffType> m_last_diff_gpu{};
    std::atomic<DiffType> m_last_diff_cpu{};
    std::atomic<DiffType> m_last_diff_pinned{};
};

} // namespace llm_on_edge::memory
