/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openai_mapping.h"

#include <cstdint>
#include <ctime>

namespace llm_on_edge::openai
{
namespace
{
/// Returns byte length of one UTF-8 code point starting at \p s[\p i], or 1 if invalid / truncated.
std::size_t utf8CodepointByteLength(std::string const& s, std::size_t i)
{
    if (i >= s.size())
    {
        return 0;
    }
    auto const c = static_cast<std::uint8_t>(s[i]);
    if (c <= 0x7Fu)
    {
        return 1;
    }
    if ((c >> 5u) == 0x6u && i + 2 <= s.size())
    {
        return 2;
    }
    if ((c >> 4u) == 0xeu && i + 3 <= s.size())
    {
        return 3;
    }
    if ((c >> 3u) == 0x1eu && i + 4 <= s.size())
    {
        return 4;
    }
    return 1;
}

void appendSseDataLine(std::string& out, nlohmann::json const& payload)
{
    out += "data: ";
    out += payload.dump();
    out += "\n\n";
}
} // namespace

std::optional<std::string> validateChatCompletionBody(nlohmann::json const& body, bool allowPseudoStream)
{
    if (!body.contains("messages") || !body.at("messages").is_array())
    {
        return "messages must be an array";
    }
    if (body.at("messages").empty())
    {
        return "messages must not be empty";
    }
    if (body.contains("stream"))
    {
        if (!body.at("stream").is_boolean())
        {
            return "stream must be a boolean";
        }
        if (body.at("stream").get<bool>() && !allowPseudoStream)
        {
            return "stream=true is not supported (use server flag --pseudo-stream for protocol-compatible "
                   "pseudo streaming)";
        }
    }
    return std::nullopt;
}

std::string buildPseudoChatCompletionSse(std::string const& modelId, std::string const& completionId,
    std::string const& assistantUtf8Text, std::size_t codepointsPerChunk)
{
    if (codepointsPerChunk == 0)
    {
        codepointsPerChunk = 1;
    }

    int64_t const created = static_cast<int64_t>(std::time(nullptr));
    std::string out;
    out.reserve(assistantUtf8Text.size() + 512);

    auto const chunkBase = [&]() {
        nlohmann::json j;
        j["id"] = completionId;
        j["object"] = "chat.completion.chunk";
        j["created"] = created;
        j["model"] = modelId;
        return j;
    };

    {
        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"] = nlohmann::json{{"role", "assistant"}, {"content", ""}};
        choice["finish_reason"] = nullptr;
        nlohmann::json j = chunkBase();
        j["choices"] = nlohmann::json::array({choice});
        appendSseDataLine(out, j);
    }

    for (std::size_t i = 0; i < assistantUtf8Text.size();)
    {
        std::size_t end = i;
        for (std::size_t n = 0; n < codepointsPerChunk && end < assistantUtf8Text.size(); ++n)
        {
            std::size_t const len = utf8CodepointByteLength(assistantUtf8Text, end);
            if (len == 0)
            {
                break;
            }
            end += len;
        }
        if (end == i)
        {
            break;
        }
        std::string const piece = assistantUtf8Text.substr(i, end - i);
        i = end;

        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"] = nlohmann::json{{"content", piece}};
        choice["finish_reason"] = nullptr;
        nlohmann::json j = chunkBase();
        j["choices"] = nlohmann::json::array({choice});
        appendSseDataLine(out, j);
    }

    {
        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"] = nlohmann::json::object();
        choice["finish_reason"] = "stop";
        nlohmann::json j = chunkBase();
        j["choices"] = nlohmann::json::array({choice});
        appendSseDataLine(out, j);
    }

    out += "data: [DONE]\n\n";
    return out;
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
