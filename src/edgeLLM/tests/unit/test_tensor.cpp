/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace llm_on_edge::memory;

TEST(Tensor, ShapeAndBytes)
{
    BufferManager mgr;
    auto buf = mgr.cpu(6);
    Tensor t(buf, {2, 3}, ElementType::kUInt8);
    EXPECT_EQ(t.num_elements(), 6u);
    EXPECT_EQ(t.size_bytes(), 6u);
    EXPECT_EQ(t.shape().size(), 2u);
}

TEST(Tensor, ReshapeSameVolume)
{
    BufferManager mgr;
    auto buf = mgr.cpu(12);
    Tensor t(buf, {3, 4}, ElementType::kUInt8);
    t.reshape({2, 2, 3});
    EXPECT_EQ(t.num_elements(), 12u);
}

TEST(Tensor, InvalidVolumeThrows)
{
    BufferManager mgr;
    auto buf = mgr.cpu(10);
    EXPECT_THROW((Tensor(buf, {2, 3}, ElementType::kUInt8)), std::invalid_argument);
}
