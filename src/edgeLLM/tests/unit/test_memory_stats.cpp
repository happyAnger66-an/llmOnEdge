/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/memory_stats.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

using namespace llm_on_edge::memory;

TEST(MemoryStats, AllocateDeallocateBalances)
{
    MemoryStats s;
    s.on_allocate(MemoryType::kCPU, 100);
    EXPECT_EQ(s.current_cpu(), 100u);
    s.on_deallocate(MemoryType::kCPU, 100);
    EXPECT_EQ(s.current_cpu(), 0u);

    s.on_allocate(MemoryType::kGPU, 4096);
    s.on_allocate(MemoryType::kPINNED, 512);
    EXPECT_EQ(s.current_gpu(), 4096u);
    EXPECT_EQ(s.current_pinned(), 512u);
    s.on_deallocate(MemoryType::kGPU, 4096);
    s.on_deallocate(MemoryType::kPINNED, 512);
    EXPECT_EQ(s.current_gpu(), 0u);
    EXPECT_EQ(s.current_pinned(), 0u);
}

TEST(MemoryStats, LastDiff)
{
    MemoryStats s;
    s.on_allocate(MemoryType::kGPU, 50);
    EXPECT_EQ(s.last_diff_gpu(), 50);
    s.on_deallocate(MemoryType::kGPU, 50);
    EXPECT_EQ(s.last_diff_gpu(), -50);
}

TEST(MemoryStats, BytesToString)
{
    EXPECT_NE(MemoryStats::bytes_to_string(0).find('B'), std::string::npos);
    EXPECT_NE(MemoryStats::bytes_to_string(1024).find('K'), std::string::npos);
}

TEST(MemoryStats, ConcurrentIncrements)
{
    MemoryStats s;
    std::vector<std::thread> threads;
    constexpr int kThreads = 8;
    constexpr std::uint64_t kPer = 1000;
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back([&]() {
            for (std::uint64_t j = 0; j < kPer; ++j)
            {
                s.on_allocate(MemoryType::kCPU, 1);
            }
        });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(s.current_cpu(), kThreads * kPer);
}
