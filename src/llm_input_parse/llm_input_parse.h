/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/llmRuntimeUtils.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm_on_edge
{

/*!
 * Parse Edge LLM batch JSON (same schema as llm_inference input files).
 */
std::pair<std::unordered_map<std::string, std::string>, std::vector<trt_edgellm::rt::LLMGenerationRequest>>
parseInputFromJson(nlohmann::json const& inputData, int32_t batchSizeOverride = -1,
    int64_t maxGenerateLengthOverride = -1);

/*!
 * Read JSON from disk and parse via parseInputFromJson.
 */
std::pair<std::unordered_map<std::string, std::string>, std::vector<trt_edgellm::rt::LLMGenerationRequest>>
parseInputFile(std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1,
    int64_t maxGenerateLengthOverride = -1);

} // namespace llm_on_edge
