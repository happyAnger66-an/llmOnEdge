/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * End-to-end: BufferManager + Tensor(shape,type) + H2D/D2H + stats (requires CUDA).
 */

#include "llm_on_edge/memory/buffer_manager.h"
#include "llm_on_edge/memory/cuda_check.h"
#include "llm_on_edge/memory/tensor.h"

#include <cuda_runtime.h>

#include <iostream>

using namespace llm_on_edge::memory;

int main()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        std::cout << "[edge_memory_e2e] SKIP: no CUDA device (" << cudaGetErrorString(err) << ")\n";
        return 0;
    }

    try
    {
        auto stats = std::make_shared<MemoryStats>();
        BufferManager mgr(stats, 0);

        std::cout << "[edge_memory_e2e] " << stats->to_string() << "\n";

        constexpr int64_t rows = 2;
        constexpr int64_t cols = 4;

        Tensor t_host(mgr, {rows, cols}, ElementType::kFloat32, MemoryType::kPINNED);
        float* h = static_cast<float*>(t_host.buffer()->data());
        for (int i = 0; i < rows * cols; ++i)
        {
            h[i] = static_cast<float>(i);
        }

        if (t_host.num_elements() != static_cast<std::size_t>(rows * cols))
        {
            std::cerr << "[edge_memory_e2e] FAIL: tensor elements\n";
            return 2;
        }

        Tensor t_dev(mgr, {rows, cols}, ElementType::kFloat32, MemoryType::kGPU);
        Tensor t_out(mgr, {rows, cols}, ElementType::kFloat32, MemoryType::kPINNED);

        mgr.copy(*t_host.buffer(), *t_dev.buffer(), nullptr);
        mgr.copy(*t_dev.buffer(), *t_out.buffer(), nullptr);
        LLMONEDGE_CUDA_CHECK(cudaDeviceSynchronize());

        float const* out = static_cast<float const*>(t_out.buffer()->data());
        for (int i = 0; i < rows * cols; ++i)
        {
            if (out[i] != h[i])
            {
                std::cerr << "[edge_memory_e2e] FAIL: mismatch at " << i << "\n";
                return 3;
            }
        }

        std::cout << "[edge_memory_e2e] " << stats->to_string() << "\n";
        std::cout << "[edge_memory_e2e] OK\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "[edge_memory_e2e] FAIL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
