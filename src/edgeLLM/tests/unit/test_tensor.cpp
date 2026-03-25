/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace llm_on_edge::memory;

TEST(Tensor, CreateFromShapeAndType)
{
    BufferManager mgr;
    Tensor t(mgr, {2, 3}, ElementType::kUInt8);
    EXPECT_EQ(t.num_elements(), 6u);
    EXPECT_EQ(t.size_bytes(), 6u);
    EXPECT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(mgr.stats()->live_tensor_count(), 1u);
    EXPECT_EQ(mgr.stats()->live_tensor_bytes(), 6u);
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
