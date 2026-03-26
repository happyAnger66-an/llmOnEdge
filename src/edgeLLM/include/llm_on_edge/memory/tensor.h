/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/buffer.h"
#include "llm_on_edge/memory/element_type.h"
#include "llm_on_edge/memory/memory_types.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llm_on_edge::memory
{

class BufferManager;

/// Contiguous tensor over a Buffer. Prefer `Tensor(manager, shape, type[, memory_type])` for simple construction.
/// Live instances are counted in `MemoryStats::live_tensor_*` (via buffer’s stats; default buffers use `MemoryStats::global()`).
class Tensor
{
public:
    /// TRT-LLM style quick factories using process-global BufferManager.
    [[nodiscard]] static Tensor cpu(std::vector<int64_t> shape, ElementType element_type);
    [[nodiscard]] static Tensor pinned(std::vector<int64_t> shape, ElementType element_type);
    [[nodiscard]] static Tensor gpu(std::vector<int64_t> shape, ElementType element_type);

    /// Allocates storage via process-global BufferManager (MemoryStats::global()).
    Tensor(std::vector<int64_t> shape, ElementType element_type, MemoryType memory_type = MemoryType::kCPU);

    /// Allocates storage via `manager` from shape × element size. Memory class: GPU / CPU / PINNED.
    Tensor(BufferManager& manager, std::vector<int64_t> shape, ElementType element_type,
        MemoryType memory_type = MemoryType::kCPU);

    /// Wraps an existing buffer (must match shape × element size). Do not wrap the same buffer in multiple Tensor
    /// objects (accounting would double-count).
    Tensor(Buffer::SharedPtr buffer, std::vector<int64_t> shape, ElementType element_type);

    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(Tensor const&) = delete;
    Tensor& operator=(Tensor const&) = delete;

    [[nodiscard]] Buffer::SharedPtr buffer() const noexcept { return m_buffer; }
    [[nodiscard]] std::vector<int64_t> const& shape() const noexcept { return m_shape; }
    [[nodiscard]] ElementType element_type() const noexcept { return m_element_type; }
    [[nodiscard]] MemoryType memory_type() const noexcept
    {
        return m_buffer ? m_buffer->memory_type() : MemoryType::kCPU;
    }

    [[nodiscard]] std::size_t num_elements() const;
    [[nodiscard]] std::size_t size_bytes() const;

    /// Same element count; may update shape only.
    void reshape(std::vector<int64_t> new_shape);

    /// Allocate a new tensor and copy content into it.
    [[nodiscard]] Tensor copy_to(MemoryType memory_type, cudaStream_t stream = nullptr) const;
    [[nodiscard]] Tensor copy_to_cpu(cudaStream_t stream = nullptr) const;
    [[nodiscard]] Tensor copy_to_pinned(cudaStream_t stream = nullptr) const;
    [[nodiscard]] Tensor copy_to_gpu(cudaStream_t stream = nullptr) const;

    /// Allocate a new tensor and copy content into it.
    [[nodiscard]] Tensor copy_to(BufferManager& manager, MemoryType memory_type, cudaStream_t stream = nullptr) const;

    /// Copy content into an existing destination tensor with same bytes.
    void copy_to(Tensor& dst, cudaStream_t stream = nullptr) const;

    /// Copy content into an existing destination tensor with same bytes.
    void copy_to(Tensor& dst, BufferManager const& manager, cudaStream_t stream = nullptr) const;

    /// Fill with zero, async for GPU, sync for CPU/PINNED.
    void set_zero(cudaStream_t stream = nullptr);

    /// Copy from source tensor into this tensor; bytes must match.
    void set_from(Tensor const& src, cudaStream_t stream = nullptr);

    /// Copy from source tensor into this tensor; bytes must match.
    void set_from(Tensor const& src, BufferManager const& manager, cudaStream_t stream = nullptr);

private:
    void register_tensor();
    void unregister_tensor();

    Buffer::SharedPtr m_buffer;
    std::vector<int64_t> m_shape;
    ElementType m_element_type;
};

} // namespace llm_on_edge::memory
