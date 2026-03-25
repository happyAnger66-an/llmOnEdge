/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/cuda_check.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstring>

using namespace llm_on_edge::memory;

static bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

TEST(BufferManager, GpuAllocate)
{
    if (!cuda_available())
    {
        GTEST_SKIP() << "No CUDA device";
    }
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats, 0);
    {
        auto g = mgr.gpu(1024);
        ASSERT_NE(g->data(), nullptr);
        EXPECT_EQ(g->memory_type(), MemoryType::kGPU);
        EXPECT_EQ(g->device_id(), 0);
    }
    EXPECT_EQ(stats->current_gpu(), 0u);
}

TEST(BufferManager, CopyHostToDeviceRoundTrip)
{
    if (!cuda_available())
    {
        GTEST_SKIP() << "No CUDA device";
    }
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats, 0);
    constexpr std::size_t n = 32;
    auto h = mgr.pinned(n);
    auto d = mgr.gpu(n);
    std::uint8_t* hp = static_cast<std::uint8_t*>(h->data());
    for (std::size_t i = 0; i < n; ++i)
    {
        hp[i] = static_cast<std::uint8_t>(i + 1);
    }
    auto h2 = mgr.pinned(n);
    mgr.copy(*h, *d, nullptr);
    mgr.copy(*d, *h2, nullptr);
    LLMONEDGE_CUDA_CHECK(cudaDeviceSynchronize());
    std::uint8_t* hp2 = static_cast<std::uint8_t*>(h2->data());
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(hp2[i], hp[i]);
    }
}

TEST(BufferManager, CopySizeMismatchThrows)
{
    BufferManager mgr;
    auto a = mgr.cpu(4);
    auto b = mgr.cpu(8);
    EXPECT_THROW(mgr.copy(*a, *b, nullptr), std::invalid_argument);
}
