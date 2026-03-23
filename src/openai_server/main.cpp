/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_input_parse.h"
#include "llm_session.h"
#include "openai_mapping.h"
#include "profileFormatter.h"

#include <httplib.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <tuple>

using json = nlohmann::json;

namespace
{

struct ServerArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string host{"0.0.0.0"};
    int port{8000};
    std::string model{"tensorrt-edgellm"};
    /// When set, `stream: true` is accepted and returns SSE pseudo-streaming (full decode first, then chunked wire).
    bool pseudoStream{false};
};

void printUsage(char const* argv0)
{
    std::cerr << "Usage: " << argv0 << " --engine-dir <path> [options]\n"
              << "Options:\n"
              << "  --engine-dir PATH          LLM engine directory (required)\n"
              << "  --multimodal-engine-dir    Optional multimodal engine directory\n"
              << "  --host HOST                Bind address (default 0.0.0.0)\n"
              << "  --port N                   TCP port (default 8000)\n"
              << "  --model NAME               model id for /v1/models and responses (default tensorrt-edgellm)\n"
              << "  --pseudo-stream            allow stream=true (protocol-compatible pseudo SSE; off by default)\n"
              << "  --help                     This message\n";
}

bool parseServerArgs(int argc, char** argv, ServerArgs& out)
{
    enum OptId : int
    {
        HELP = 1000,
        ENGINE_DIR = 1001,
        MM_ENGINE_DIR = 1002,
        HOST = 1003,
        PORT = 1004,
        MODEL = 1005,
        PSEUDO_STREAM = 1006
    };
    static struct option longOpts[] = {{"help", no_argument, nullptr, HELP},
        {"engine-dir", required_argument, nullptr, ENGINE_DIR},
        {"multimodal-engine-dir", required_argument, nullptr, MM_ENGINE_DIR},
        {"host", required_argument, nullptr, HOST},
        {"port", required_argument, nullptr, PORT},
        {"model", required_argument, nullptr, MODEL},
        {"pseudo-stream", no_argument, nullptr, PSEUDO_STREAM},
        {nullptr, 0, nullptr, 0}};
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "", longOpts, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: out.help = true; break;
        case ENGINE_DIR: out.engineDir = optarg; break;
        case MM_ENGINE_DIR: out.multimodalEngineDir = optarg; break;
        case HOST: out.host = optarg; break;
        case PORT:
            try
            {
                out.port = std::stoi(optarg);
            }
            catch (...)
            {
                return false;
            }
            break;
        case MODEL: out.model = optarg; break;
        case PSEUDO_STREAM: out.pseudoStream = true; break;
        default: return false;
        }
    }
    return true;
}

json makeOpenAiError(std::string const& msg, std::string const& type = "invalid_request_error")
{
    return json{{"error",
        {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
}

std::string newCompletionId()
{
    using clock = std::chrono::system_clock;
    auto const sec = std::chrono::duration_cast<std::chrono::seconds>(clock::now().time_since_epoch()).count();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;
    return "chatcmpl-" + std::to_string(sec) + "-" + std::to_string(dist(gen));
}

json buildChatCompletionResponse(std::string const& modelId, std::string const& content, std::string const& id,
    int64_t completionTokens, double inferenceSeconds)
{
    int64_t created = static_cast<int64_t>(std::time(nullptr));
    json choice;
    choice["index"] = 0;
    choice["message"] = json{{"role", "assistant"}, {"content", sanitizeUtf8ForJson(content)}};
    choice["finish_reason"] = "stop";

    json out{{"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", modelId},
        {"choices", json::array({choice})},
        {"usage",
            {{"prompt_tokens", 0},
                {"completion_tokens", completionTokens},
                {"total_tokens", completionTokens}}}};

    // Wall-clock tokens/s over full handleRequest (prefill + decode). prompt_tokens 未统计，保持为 0。
    if (inferenceSeconds > 0.0 && completionTokens > 0)
    {
        double const tps = static_cast<double>(completionTokens) / inferenceSeconds;
        out["llm_on_edge"] = json{{"inference_time_sec", inferenceSeconds}, {"output_tokens_per_sec", tps}};
    }

    return out;
}

json buildModelsList(std::string const& modelId)
{
    int64_t created = static_cast<int64_t>(std::time(nullptr));
    json item;
    item["id"] = modelId;
    item["object"] = "model";
    item["created"] = created;
    item["owned_by"] = "tensorrt-edgellm";
    return json{{"object", "list"}, {"data", json::array({item})}};
}

} // namespace

int main(int argc, char** argv)
{
    ServerArgs args;
    if (!parseServerArgs(argc, argv, args))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (args.engineDir.empty())
    {
        std::cerr << "error: --engine-dir is required\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::unique_ptr<llm_on_edge::openai::LlmEngineSession> session;
    try
    {
        std::unordered_map<std::string, std::string> lora;
        session = std::make_unique<llm_on_edge::openai::LlmEngineSession>(
            args.engineDir, args.multimodalEngineDir, std::move(lora));
    }
    catch (std::exception const& e)
    {
        std::cerr << "Failed to create LLM runtime: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    httplib::Server svr;

    svr.Get("/health", [](httplib::Request const&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Get("/v1/models", [&args](httplib::Request const&, httplib::Response& res) {
        res.set_content(buildModelsList(args.model).dump(2), "application/json");
    });

    svr.Post("/v1/chat/completions", [&args, &session](httplib::Request const& req, httplib::Response& res) {
        json body;
        try
        {
            body = json::parse(req.body);
        }
        catch (json::parse_error const& e)
        {
            res.status = 400;
            res.set_content(makeOpenAiError(std::string("Invalid JSON: ") + e.what()).dump(), "application/json");
            return;
        }

        if (auto err = llm_on_edge::openai::validateChatCompletionBody(body, args.pseudoStream))
        {
            res.status = 400;
            res.set_content(makeOpenAiError(*err).dump(), "application/json");
            return;
        }

        bool const wantStream = body.value("stream", false);

        json edgeBody = llm_on_edge::openai::chatCompletionBodyToEdgeLlmJson(body);

        std::vector<trt_edgellm::rt::LLMGenerationRequest> batches;
        try
        {
            std::tie(std::ignore, batches) = llm_on_edge::parseInputFromJson(edgeBody, -1, -1);
        }
        catch (std::exception const& e)
        {
            res.status = 400;
            res.set_content(makeOpenAiError(e.what()).dump(), "application/json");
            return;
        }

        if (batches.size() != 1 || batches[0].requests.size() != 1)
        {
            res.status = 500;
            res.set_content(makeOpenAiError("Internal batching error", "server_error").dump(), "application/json");
            return;
        }

        trt_edgellm::rt::LLMGenerationResponse genResponse;
        auto const inferT0 = std::chrono::steady_clock::now();
        bool ok = session->handleRequest(batches[0], genResponse);
        auto const inferT1 = std::chrono::steady_clock::now();
        double const inferenceSeconds
            = std::chrono::duration<double>(inferT1 - inferT0).count();

        if (!ok || genResponse.outputTexts.empty())
        {
            res.status = 500;
            res.set_content(
                makeOpenAiError("Inference failed", "server_error").dump(), "application/json");
            return;
        }

        int64_t completionTokens = 0;
        if (!genResponse.outputIds.empty())
        {
            completionTokens = static_cast<int64_t>(genResponse.outputIds[0].size());
        }

        if (inferenceSeconds > 0.0 && completionTokens > 0)
        {
            double const tps = static_cast<double>(completionTokens) / inferenceSeconds;
            std::cerr << "[llm_on_edge] completion_tokens=" << completionTokens << " inference_time_sec=" << std::fixed
                      << std::setprecision(3) << inferenceSeconds << " output_tokens_per_sec=" << std::setprecision(2)
                      << tps << "\n";
        }

        std::string const& text = genResponse.outputTexts[0];
        std::string const cmplId = newCompletionId();

        if (wantStream)
        {
            std::string const sse = llm_on_edge::openai::buildPseudoChatCompletionSse(
                args.model, cmplId, sanitizeUtf8ForJson(text), 8);
            res.set_header("Cache-Control", "no-cache");
            res.set_content(sse, "text/event-stream");
            return;
        }

        res.set_content(
            buildChatCompletionResponse(args.model, text, cmplId, completionTokens, inferenceSeconds).dump(2),
            "application/json");
    });

    std::cout << "Listening on http://" << args.host << ":" << args.port << "\n";
    if (!svr.listen(args.host.c_str(), args.port))
    {
        std::cerr << "Failed to bind " << args.host << ":" << args.port << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
