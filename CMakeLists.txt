# SPDX-License-Identifier: Apache-2.0
#
# llmOnEdge: 业务代码在 src/；TensorRT-Edge-LLM 为 third_party 子模块。

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(llmOnEdge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# TensorRT-Edge-LLM 默认 CUDA_DIR=/usr/local/cuda-${CUDA_CTK_VERSION}，本机常为 /usr/local/cuda 或其它版本，
# 会导致 CUDART_LIB / CUDA_RUNTIME_API_INCLUDE_DIR 为 NOTFOUND。先用 CUDAToolkit 解析并写入 CACHE，
# 子工程里的 find_* 会复用这些变量。
find_package(CUDAToolkit REQUIRED)
find_path(
  CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  REQUIRED)
find_library(
  CUDART_LIB cudart
  NAMES cudart
  PATHS ${CUDAToolkit_LIBRARY_DIR}
  NO_DEFAULT_PATH)
if(NOT CUDART_LIB)
  find_library(CUDART_LIB cudart REQUIRED)
endif()
get_filename_component(CUDA_DIR "${CUDA_RUNTIME_API_INCLUDE_DIR}/.." ABSOLUTE)
set(CUDA_DIR
    "${CUDA_DIR}"
    CACHE PATH "CUDA toolkit root (TensorRT-Edge-LLM / CUDART)" FORCE)
set(CUDA_RUNTIME_API_INCLUDE_DIR
    "${CUDA_RUNTIME_API_INCLUDE_DIR}"
    CACHE PATH "CUDA runtime API headers" FORCE)
set(CUDART_LIB
    "${CUDART_LIB}"
    CACHE FILEPATH "libcudart" FORCE)

find_library(
  CUDA_DRIVER_LIB cuda
  HINTS "${CUDA_DIR}/lib/stubs" "${CUDA_DIR}/lib64/stubs" "${CUDA_DIR}/lib64" "${CUDA_DIR}/lib"
        ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES stubs lib lib64
  NO_DEFAULT_PATH)
if(NOT CUDA_DRIVER_LIB)
  find_library(CUDA_DRIVER_LIB cuda REQUIRED)
endif()
set(CUDA_DRIVER_LIB "${CUDA_DRIVER_LIB}" CACHE FILEPATH "libcuda driver stub or .so" FORCE)

find_path(
  CURAND_KERNEL_INCLUDE_DIR curand_kernel.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} "${CUDA_DIR}/include"
  PATH_SUFFIXES include
  REQUIRED)
set(CURAND_KERNEL_INCLUDE_DIR "${CURAND_KERNEL_INCLUDE_DIR}" CACHE PATH "curand_kernel.h" FORCE)

option(LLMONEDGE_BUILD_OPENAI_SERVER "Build OpenAI-compatible HTTP server (llm_openai_server)" ON)
option(LLMONEDGE_BUILD_OPENAI_SERVER_TESTS "Build gtest for openai_server JSON mapping" OFF)

macro(llm_onedge_cross_link target)
  if(DEFINED AARCH64_BUILD)
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib")
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib/stubs")
  endif()
endmacro()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT-Edge-LLM/CMakeLists.txt")
  message(
    FATAL_ERROR
      "TensorRT-Edge-LLM missing. Initialize: git submodule update --init --recursive third_party/TensorRT-Edge-LLM")
endif()

add_subdirectory(third_party/TensorRT-Edge-LLM)

if(LLMONEDGE_BUILD_OPENAI_SERVER)
  add_subdirectory(src/llm_input_parse)
  add_subdirectory(src/openai_server)
endif()
