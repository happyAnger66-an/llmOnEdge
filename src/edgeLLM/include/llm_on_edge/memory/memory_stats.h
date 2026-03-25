/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/memory_types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace llm_on_edge::memory
{

/// Thread-safe byte counters per memory class, plus live Tensor footprint (TRT-LLM MemoryCounters style).
///
/// **进程内全局统计**：`global()` 返回的 `MemoryStats` 全进程唯一；`BufferManager` 默认悬挂同一实例，
/// 这样 GPU/CPU/Pinned 缓冲与 Tensor 存活量在同一处汇总。单元测试如需隔离可自建 `std::make_shared<MemoryStats>()`
/// 并传入 `BufferManager(stats, device_id)`。
class MemoryStats
{
public:
    using SizeType = std::uint64_t;
    using DiffType = std::int64_t;

    /// 允许自建实例（单测隔离等）；生产环境请用 `global()` + 默认 `BufferManager`。
    MemoryStats() = default;

    /// 进程内唯一计数器（所有默认 `BufferManager` 共享此对象的同一 `shared_ptr` 控制块）。
    [[nodiscard]] static std::shared_ptr<MemoryStats> global() noexcept;

    /// @deprecated 与 `global()` 相同，仅为旧代码保留。
    [[nodiscard]] static std::shared_ptr<MemoryStats> shared_process() noexcept { return global(); }

    void on_allocate(MemoryType t, SizeType bytes) noexcept;
    void on_deallocate(MemoryType t, SizeType bytes) noexcept;

    /// Live Tensor objects (see Tensor ctor/dtor); bytes match sum of their backing buffer sizes.
    void tensor_live_add(SizeType bytes) noexcept;
    void tensor_live_remove(SizeType bytes) noexcept;
    [[nodiscard]] SizeType live_tensor_bytes() const noexcept { return m_tensors_bytes.load(std::memory_order_relaxed); }
    [[nodiscard]] SizeType live_tensor_count() const noexcept { return m_tensors_count.load(std::memory_order_relaxed); }
    [[nodiscard]] DiffType last_diff_tensor_bytes() const noexcept
    {
        return m_last_diff_tensors_bytes.load(std::memory_order_relaxed);
    }

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

    std::atomic<SizeType> m_tensors_bytes{};
    std::atomic<SizeType> m_tensors_count{};
    std::atomic<DiffType> m_last_diff_tensors_bytes{};
};

} // namespace llm_on_edge::memory
