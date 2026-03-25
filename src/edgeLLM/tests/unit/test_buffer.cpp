/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"

#include <gtest/gtest.h>

#include <cstring>

using namespace llm_on_edge::memory;

TEST(Buffer, CpuAllocateAndStats)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    {
        auto b = mgr.cpu(256);
        ASSERT_NE(b->data(), nullptr);
        EXPECT_EQ(b->size_bytes(), 256u);
        EXPECT_EQ(b->memory_type(), MemoryType::kCPU);
        std::memset(b->data(), 0xAB, 256);
    }
    EXPECT_EQ(stats->current_cpu(), 0u);
}

TEST(Buffer, PinnedAllocate)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    {
        auto b = mgr.pinned(64);
        ASSERT_NE(b->data(), nullptr);
        EXPECT_EQ(b->memory_type(), MemoryType::kPINNED);
    }
    EXPECT_EQ(stats->current_pinned(), 0u);
}

TEST(Buffer, ResizeCpuGrows)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    auto b = mgr.cpu(10);
    std::uint64_t const before = stats->current_cpu();
    b->resize(100);
    EXPECT_GE(b->capacity_bytes(), 100u);
    EXPECT_EQ(b->size_bytes(), 100u);
    EXPECT_GE(stats->current_cpu(), before);
}
