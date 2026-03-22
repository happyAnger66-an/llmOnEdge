/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace llm_on_edge::openai
{

/// Human-readable validation error, or std::nullopt when the body is acceptable.
std::optional<std::string> validateChatCompletionBody(nlohmann::json const& body);

/*!
 * Map an OpenAI-style POST /v1/chat/completions JSON body to the Edge LLM file schema
 * consumed by parseInputFromJson (batch_size 1, single request).
 */
nlohmann::json chatCompletionBodyToEdgeLlmJson(nlohmann::json const& body);

} // namespace llm_on_edge::openai
