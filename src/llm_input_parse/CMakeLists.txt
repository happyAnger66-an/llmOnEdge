# SPDX-License-Identifier: Apache-2.0
#
# JSON → LLMGenerationRequest parsing (same schema as upstream llm_inference input files).

add_library(llmOnEdgeInputParse STATIC llm_input_parse.cpp)
# 上游 cpp 目标在子工程内通过 include_directories 拿到 CUDA/TRT；本目标在父工程 add_subdirectory，
# 不继承该目录属性，需显式加入头路径（tensor.h → cuda.h、NvInferRuntime.h）。
target_include_directories(
  llmOnEdgeInputParse SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EDGELLM_COMMON_INCLUDE_DIRS}
                                    ${CUDAToolkit_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
target_link_libraries(llmOnEdgeInputParse PUBLIC edgellmCore CUDA::cudart)
