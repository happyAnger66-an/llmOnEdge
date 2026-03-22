/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_session.h"

#include "common/checkMacros.h"
#include "common/trtUtils.h"

#include <utility>

namespace llm_on_edge::openai
{
// LOG_* / loadEdgellmPluginLib 依赖 trt_edgellm 中的 gLogger、format::fmtstr 与符号解析。
using namespace trt_edgellm;

namespace
{
std::once_flag gPluginLoadFlag;

void ensurePluginsLoaded()
{
    std::call_once(gPluginLoadFlag, []() { static auto handles [[maybe_unused]] = loadEdgellmPluginLib(); });
}
} // namespace

LlmEngineSession::LlmEngineSession(
    std::string engineDir, std::string multimodalEngineDir, std::unordered_map<std::string, std::string> loraWeightsMap)
{
    ensurePluginsLoaded();
    CUDA_CHECK(cudaStreamCreate(&mStream));
    mRuntime = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
        std::move(engineDir), std::move(multimodalEngineDir), std::move(loraWeightsMap), mStream);
    if (!mRuntime->captureDecodingCUDAGraph(mStream))
    {
        LOG_WARNING("CUDA graph capture for decoding failed; continuing without graph.");
    }
}

LlmEngineSession::~LlmEngineSession()
{
    mRuntime.reset();
    if (mStream)
    {
        cudaStreamDestroy(mStream);
    }
}

bool LlmEngineSession::handleRequest(trt_edgellm::rt::LLMGenerationRequest const& request,
    trt_edgellm::rt::LLMGenerationResponse& response)
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mRuntime->handleRequest(request, response, mStream);
}

} // namespace llm_on_edge::openai
