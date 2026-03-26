/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llm_on_edge/memory/tensor.h"

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

namespace llm_on_edge::engine
{

/// Minimal TensorRT runner: load engine + config json, bind tensors, execute enqueueV3.
class EngineRunner
{
public:
    EngineRunner(std::string engine_path, std::string config_json_path);
    ~EngineRunner();

    EngineRunner(EngineRunner const&) = delete;
    EngineRunner& operator=(EngineRunner const&) = delete;
    EngineRunner(EngineRunner&&) = delete;
    EngineRunner& operator=(EngineRunner&&) = delete;

    [[nodiscard]] std::string const& config_text() const noexcept { return m_config_text; }
    [[nodiscard]] std::vector<std::string> const& io_tensor_names() const noexcept { return m_io_tensor_names; }

    /// Set runtime input shape for dynamic input tensors.
    void set_input_shape(std::string const& tensor_name, std::vector<int32_t> const& dims);

    /// Bind raw pointer to an engine I/O tensor.
    void set_tensor_address(std::string const& tensor_name, void* ptr);

    /// Bind llm_on_edge tensor buffer to an engine I/O tensor.
    void set_tensor_address(std::string const& tensor_name, memory::Tensor const& tensor);

    /// Execute TensorRT context->enqueueV3(stream).
    bool execute(cudaStream_t stream);

    /// Helper for external validation / tests.
    static std::string load_config_text(std::string const& path);

private:
    class TrtLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, char const* msg) noexcept override;
    };

    template <typename T>
    struct TrtDeleter
    {
        void operator()(T* p) const noexcept
        {
            delete p;
        }
    };

    static std::vector<char> read_binary_file(std::string const& path);

    TrtLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> m_context;
    std::string m_config_text;
    std::vector<std::string> m_io_tensor_names;
};

} // namespace llm_on_edge::engine

