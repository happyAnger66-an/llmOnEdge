/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/memory_stats.h"
#include "llm_on_edge/memory/memory_types.h"

#include <cstddef>
#include <memory>

namespace llm_on_edge::memory
{

class BufferManager;

/// Owns a contiguous byte range; updates MemoryStats on destroy. Non-copyable, movable.
class Buffer
{
public:
    using SharedPtr = std::shared_ptr<Buffer>;
    using ConstSharedPtr = std::shared_ptr<Buffer const>;

    ~Buffer();

    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return m_data; }
    [[nodiscard]] void const* data() const noexcept { return m_data; }

    [[nodiscard]] std::size_t size_bytes() const noexcept { return m_size_bytes; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return m_capacity_bytes; }
    [[nodiscard]] MemoryType memory_type() const noexcept { return m_type; }
    [[nodiscard]] int device_id() const noexcept { return m_device_id; }

    /// Grow or shrink logical size; may reallocate if larger than capacity.
    void resize(std::size_t new_size_bytes);

    /// Free storage and zero counters (safe to call multiple times).
    void release() noexcept;

private:
    friend class BufferManager;

    Buffer(void* ptr, std::size_t size_bytes, std::size_t capacity_bytes, MemoryType type, int device_id,
        std::shared_ptr<MemoryStats> stats) noexcept;

    void destroy() noexcept;

    void* m_data{nullptr};
    std::size_t m_size_bytes{0};
    std::size_t m_capacity_bytes{0};
    MemoryType m_type{MemoryType::kCPU};
    int m_device_id{0};
    std::shared_ptr<MemoryStats> m_stats{};
};

} // namespace llm_on_edge::memory
