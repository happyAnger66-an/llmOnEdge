/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llm_on_edge::openai
{

/// Owns TensorRT Edge LLM runtime + stream; thread-safe handleRequest (serialized).
class LlmEngineSession
{
public:
    LlmEngineSession(std::string engineDir, std::string multimodalEngineDir,
        std::unordered_map<std::string, std::string> loraWeightsMap);
    ~LlmEngineSession();

    LlmEngineSession(LlmEngineSession const&) = delete;
    LlmEngineSession& operator=(LlmEngineSession const&) = delete;

    bool handleRequest(trt_edgellm::rt::LLMGenerationRequest const& request,
        trt_edgellm::rt::LLMGenerationResponse& response);

private:
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> mRuntime;
    cudaStream_t mStream{};
    std::mutex mMutex;
};

} // namespace llm_on_edge::openai
