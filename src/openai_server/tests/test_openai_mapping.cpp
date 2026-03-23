/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openai_mapping.h"

#include "llm_input_parse.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

TEST(OpenAiMapping, ValidateRejectsStreamUnlessPseudoEnabled)
{
    json body;
    body["messages"] = json::array({{{"role", "user"}, {"content", "hi"}}});
    body["stream"] = true;
    EXPECT_TRUE(llm_on_edge::openai::validateChatCompletionBody(body, false).has_value());
    EXPECT_FALSE(llm_on_edge::openai::validateChatCompletionBody(body, true).has_value());
}

TEST(OpenAiMapping, ValidateAcceptsMinimal)
{
    json body;
    body["messages"] = json::array({{{"role", "user"}, {"content", "hi"}}});
    EXPECT_FALSE(llm_on_edge::openai::validateChatCompletionBody(body).has_value());
}

TEST(OpenAiMapping, PseudoSseContainsChunksAndDone)
{
    std::string const sse = llm_on_edge::openai::buildPseudoChatCompletionSse(
        "m", "chatcmpl-test", u8"\u4e2dab", 1); // 中 + a + b as separate deltas when cp=1
    EXPECT_NE(sse.find("chat.completion.chunk"), std::string::npos);
    EXPECT_NE(sse.find("[DONE]"), std::string::npos);
    EXPECT_NE(sse.find("\"role\":\"assistant\""), std::string::npos);
    EXPECT_NE(sse.find("data: "), std::string::npos);
}

TEST(OpenAiMapping, EdgeJsonParsesThroughLlmInputParse)
{
    json openaiBody;
    openaiBody["messages"] = json::array({{{"role", "user"}, {"content", "hello"}}});
    openaiBody["max_tokens"] = 32;
    openaiBody["temperature"] = 0.5f;
    openaiBody["top_p"] = 0.9f;

    json edge = llm_on_edge::openai::chatCompletionBodyToEdgeLlmJson(openaiBody);
    EXPECT_EQ(edge["batch_size"], 1);
    EXPECT_EQ(edge["max_generate_length"], 32);
    EXPECT_DOUBLE_EQ(edge["temperature"].get<double>(), 0.5);
    EXPECT_DOUBLE_EQ(edge["top_p"].get<double>(), 0.9);

    auto parsed = llm_on_edge::parseInputFromJson(edge, -1, -1);
    auto& batches = parsed.second;
    ASSERT_EQ(batches.size(), 1U);
    ASSERT_EQ(batches[0].requests.size(), 1U);
    EXPECT_EQ(batches[0].maxGenerateLength, 32);
}
