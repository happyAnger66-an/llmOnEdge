/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/buffer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace llm_on_edge::memory
{

/// Central factory for buffers + copies (TRT-LLM BufferManager style).
/// `stats == nullptr` 时使用 `MemoryStats::global()`，全进程共一份统计。
class BufferManager
{
public:
    explicit BufferManager(std::shared_ptr<MemoryStats> stats = nullptr, int device_id = 0);

    [[nodiscard]] std::shared_ptr<MemoryStats> stats() const noexcept { return m_stats; }
    [[nodiscard]] int device_id() const noexcept { return m_device_id; }

    [[nodiscard]] Buffer::SharedPtr gpu(std::size_t size_bytes);
    [[nodiscard]] Buffer::SharedPtr cpu(std::size_t size_bytes);
    [[nodiscard]] Buffer::SharedPtr pinned(std::size_t size_bytes);

    /// Full copy when sizes match; uses stream for async GPU copies.
    void copy(Buffer const& src, Buffer& dst, cudaStream_t stream = nullptr) const;

private:
    std::shared_ptr<MemoryStats> m_stats;
    int m_device_id{0};
};

} // namespace llm_on_edge::memory
