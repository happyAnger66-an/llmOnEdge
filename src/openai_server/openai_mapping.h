/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace llm_on_edge::openai
{

/// Human-readable validation error, or std::nullopt when the body is acceptable.
/// When \p allowPseudoStream is false, `stream: true` is rejected (default server behavior).
std::optional<std::string> validateChatCompletionBody(
    nlohmann::json const& body, bool allowPseudoStream = false);

/*!
 * OpenAI-compatible SSE body for pseudo-streaming: full assistant text is split into UTF-8 codepoint chunks
 * after inference; wire format matches chat.completion.chunk events + trailing `data: [DONE]`.
 * \p assistantUtf8Text should already be safe for JSON (e.g. passed through sanitizeUtf8ForJson).
 */
std::string buildPseudoChatCompletionSse(std::string const& modelId, std::string const& completionId,
    std::string const& assistantUtf8Text, std::size_t codepointsPerChunk = 8);

/*!
 * Map an OpenAI-style POST /v1/chat/completions JSON body to the Edge LLM file schema
 * consumed by parseInputFromJson (batch_size 1, single request).
 */
nlohmann::json chatCompletionBodyToEdgeLlmJson(nlohmann::json const& body);

} // namespace llm_on_edge::openai
