/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/cuda_check.h"
#include "llm_on_edge/memory/tensor.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

using namespace llm_on_edge::memory;

TEST(Tensor, CreateFromShapeAndType)
{
    Tensor t({2, 3}, ElementType::kUInt8);
    EXPECT_EQ(t.num_elements(), 6u);
    EXPECT_EQ(t.size_bytes(), 6u);
    EXPECT_EQ(t.shape().size(), 2u);
    EXPECT_GE(MemoryStats::global()->live_tensor_count(), 1u);
}

TEST(Tensor, LiveStatsAfterScope)
{
    auto stats = std::make_shared<MemoryStats>();
    EXPECT_EQ(stats->live_tensor_count(), 0u);
    {
        BufferManager mgr(stats);
        Tensor t(mgr, {4}, ElementType::kInt32);
        EXPECT_EQ(stats->live_tensor_count(), 1u);
        EXPECT_EQ(stats->live_tensor_bytes(), 16u);
    }
    EXPECT_EQ(stats->live_tensor_count(), 0u);
    EXPECT_EQ(stats->live_tensor_bytes(), 0u);
}

TEST(Tensor, MoveTransfersOwnershipNotDoubleCount)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    EXPECT_EQ(stats->live_tensor_count(), 0u);
    {
        Tensor a(mgr, {2}, ElementType::kFloat32);
        EXPECT_EQ(stats->live_tensor_count(), 1u);
        Tensor b(std::move(a));
        EXPECT_EQ(stats->live_tensor_count(), 1u);
        EXPECT_EQ(stats->live_tensor_bytes(), sizeof(float) * 2);
    }
    EXPECT_EQ(stats->live_tensor_count(), 0u);
}

TEST(Tensor, ShapeAndBytesFromBuffer)
{
    BufferManager mgr;
    auto buf = mgr.cpu(6);
    Tensor t(buf, {2, 3}, ElementType::kUInt8);
    EXPECT_EQ(t.num_elements(), 6u);
    EXPECT_EQ(t.size_bytes(), 6u);
    EXPECT_EQ(mgr.stats()->live_tensor_count(), 1u);
}

TEST(Tensor, ReshapeSameVolume)
{
    BufferManager mgr;
    Tensor t(mgr, {3, 4}, ElementType::kUInt8);
    t.reshape({2, 2, 3});
    EXPECT_EQ(t.num_elements(), 12u);
}

TEST(Tensor, InvalidVolumeThrows)
{
    BufferManager mgr;
    auto buf = mgr.cpu(10);
    EXPECT_THROW((Tensor(buf, {2, 3}, ElementType::kUInt8)), std::invalid_argument);
}

TEST(Tensor, CopyToAllocatesAndTracksTensorStats)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    Tensor src(mgr, {8}, ElementType::kUInt8, MemoryType::kCPU);
    auto* p = static_cast<std::uint8_t*>(src.buffer()->data());
    for (std::size_t i = 0; i < 8; ++i)
    {
        p[i] = static_cast<std::uint8_t>(i + 3);
    }

    EXPECT_EQ(stats->live_tensor_count(), 1u);
    Tensor dst = src.copy_to(mgr, MemoryType::kPINNED);
    EXPECT_EQ(stats->live_tensor_count(), 2u);
    EXPECT_EQ(stats->live_tensor_bytes(), 16u);
    auto const* q = static_cast<std::uint8_t const*>(dst.buffer()->data());
    for (std::size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(q[i], p[i]);
    }
}

TEST(Tensor, CopyToWithoutManagerAndSetFromWithoutManager)
{
    auto stats = std::make_shared<MemoryStats>();
    BufferManager mgr(stats);
    Tensor src(mgr, {6}, ElementType::kUInt8, MemoryType::kCPU);
    auto* p = static_cast<std::uint8_t*>(src.buffer()->data());
    for (std::size_t i = 0; i < 6; ++i)
    {
        p[i] = static_cast<std::uint8_t>(50 + i);
    }

    Tensor dst = src.copy_to(MemoryType::kCPU);
    auto const* q = static_cast<std::uint8_t const*>(dst.buffer()->data());
    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(q[i], p[i]);
    }

    Tensor same_shape({6}, ElementType::kUInt8, MemoryType::kCPU);
    same_shape.set_from(dst);
    auto const* z = static_cast<std::uint8_t const*>(same_shape.buffer()->data());
    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(z[i], q[i]);
    }
}

TEST(Tensor, QuickFactoriesAndCopyHelpers)
{
    Tensor a = Tensor::cpu({4}, ElementType::kUInt8);
    auto* pa = static_cast<std::uint8_t*>(a.buffer()->data());
    for (std::size_t i = 0; i < 4; ++i)
    {
        pa[i] = static_cast<std::uint8_t>(20 + i);
    }

    Tensor b = a.copy_to_pinned();
    auto const* pb = static_cast<std::uint8_t const*>(b.buffer()->data());
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(pb[i], pa[i]);
    }

    Tensor c = b.copy_to_cpu();
    auto const* pc = static_cast<std::uint8_t const*>(c.buffer()->data());
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(pc[i], pa[i]);
    }
}

TEST(Tensor, SetZeroCpu)
{
    BufferManager mgr;
    Tensor t(mgr, {4}, ElementType::kInt32, MemoryType::kCPU);
    auto* p = static_cast<std::int32_t*>(t.buffer()->data());
    p[0] = 1;
    p[1] = 2;
    p[2] = 3;
    p[3] = 4;
    t.set_zero();
    EXPECT_EQ(p[0], 0);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 0);
    EXPECT_EQ(p[3], 0);
}

TEST(Tensor, SetFromCopiesAndSizeMismatchThrows)
{
    BufferManager mgr;
    Tensor a(mgr, {4}, ElementType::kUInt8, MemoryType::kCPU);
    Tensor b(mgr, {4}, ElementType::kUInt8, MemoryType::kCPU);
    auto* pa = static_cast<std::uint8_t*>(a.buffer()->data());
    auto const* pb = static_cast<std::uint8_t const*>(b.buffer()->data());
    for (std::size_t i = 0; i < 4; ++i)
    {
        pa[i] = static_cast<std::uint8_t>(10 + i);
    }
    b.set_from(a, mgr);
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(pb[i], pa[i]);
    }

    Tensor c(mgr, {5}, ElementType::kUInt8, MemoryType::kCPU);
    EXPECT_THROW(c.set_from(a, mgr), std::invalid_argument);
}

static bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

TEST(Tensor, SetZeroGpu)
{
    if (!cuda_available())
    {
        GTEST_SKIP() << "No CUDA device";
    }
    BufferManager mgr;
    Tensor host(mgr, {8}, ElementType::kUInt8, MemoryType::kPINNED);
    Tensor dev(mgr, {8}, ElementType::kUInt8, MemoryType::kGPU);
    Tensor out(mgr, {8}, ElementType::kUInt8, MemoryType::kPINNED);

    auto* h = static_cast<std::uint8_t*>(host.buffer()->data());
    for (std::size_t i = 0; i < 8; ++i)
    {
        h[i] = 0x7f;
    }
    mgr.copy(*host.buffer(), *dev.buffer(), nullptr);
    dev.set_zero(nullptr);
    mgr.copy(*dev.buffer(), *out.buffer(), nullptr);
    LLMONEDGE_CUDA_CHECK(cudaDeviceSynchronize());

    auto const* q = static_cast<std::uint8_t const*>(out.buffer()->data());
    for (std::size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(q[i], 0);
    }
}
