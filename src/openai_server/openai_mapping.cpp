/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openai_mapping.h"

namespace llm_on_edge::openai
{

std::optional<std::string> validateChatCompletionBody(nlohmann::json const& body)
{
    if (!body.contains("messages") || !body.at("messages").is_array())
    {
        return "messages must be an array";
    }
    if (body.at("messages").empty())
    {
        return "messages must not be empty";
    }
    if (body.value("stream", false))
    {
        return "stream=true is not supported (non-streaming only)";
    }
    return std::nullopt;
}

nlohmann::json chatCompletionBodyToEdgeLlmJson(nlohmann::json const& body)
{
    nlohmann::json edge;
    edge["batch_size"] = 1;
    edge["temperature"] = body.value("temperature", 1.0f);
    edge["top_p"] = body.value("top_p", 1.0f);
    edge["top_k"] = body.value("top_k", 50);
    edge["max_generate_length"] = body.value("max_tokens", 256);
    edge["apply_chat_template"] = body.value("apply_chat_template", true);
    edge["add_generation_prompt"] = body.value("add_generation_prompt", true);
    edge["enable_thinking"] = body.value("enable_thinking", false);

    if (body.contains("available_lora_weights") && body["available_lora_weights"].is_object())
    {
        edge["available_lora_weights"] = body["available_lora_weights"];
    }

    nlohmann::json req = nlohmann::json::object();
    req["messages"] = body.at("messages");
    if (body.contains("lora_name") && !body["lora_name"].is_null())
    {
        req["lora_name"] = body["lora_name"];
    }
    if (body.contains("save_system_prompt_kv_cache"))
    {
        req["save_system_prompt_kv_cache"] = body["save_system_prompt_kv_cache"];
    }
    if (body.contains("disable_spec_decode"))
    {
        req["disable_spec_decode"] = body["disable_spec_decode"];
    }

    edge["requests"] = nlohmann::json::array({req});
    return edge;
}

} // namespace llm_on_edge::openai
