/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace llm_on_edge::memory::detail
{

inline void throw_cuda_error(cudaError_t e, char const* expr, char const* file, int line)
{
    std::ostringstream oss;
    oss << file << ":" << line << " CUDA " << expr << " failed: " << cudaGetErrorString(e) << " (" << int(e) << ")";
    throw std::runtime_error(oss.str());
}

} // namespace llm_on_edge::memory::detail

#define LLMONEDGE_CUDA_CHECK(expr)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const _llmonedge_cuda_err = (expr);                                                                \
        if (_llmonedge_cuda_err != cudaSuccess)                                                                        \
        {                                                                                                              \
            llm_on_edge::memory::detail::throw_cuda_error(_llmonedge_cuda_err, #expr, __FILE__, __LINE__);               \
        }                                                                                                              \
    } while (0)
