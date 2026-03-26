/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/engine/engine_runner.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace llm_on_edge::engine
{

void EngineRunner::TrtLogger::log(Severity severity, char const* msg) noexcept
{
    if (severity == Severity::kVERBOSE)
    {
        return;
    }
    std::cerr << "[EngineRunner][TRT] " << msg << "\n";
}

std::vector<char> EngineRunner::read_binary_file(std::string const& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("EngineRunner: failed to open engine file: " + path);
    }
    std::streamsize const size = f.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("EngineRunner: engine file is empty: " + path);
    }
    std::vector<char> data(static_cast<std::size_t>(size));
    f.seekg(0, std::ios::beg);
    if (!f.read(data.data(), size))
    {
        throw std::runtime_error("EngineRunner: failed to read engine file: " + path);
    }
    return data;
}

std::string EngineRunner::load_config_text(std::string const& path)
{
    std::ifstream f(path);
    if (!f)
    {
        throw std::runtime_error("EngineRunner: failed to open config json: " + path);
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    std::string const txt = oss.str();
    if (txt.empty())
    {
        throw std::runtime_error("EngineRunner: config json is empty: " + path);
    }
    return txt;
}

EngineRunner::EngineRunner(std::string engine_path, std::string config_json_path)
    : m_config_text(load_config_text(config_json_path))
{
    std::vector<char> engine_data = read_binary_file(engine_path);

    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime)
    {
        throw std::runtime_error("EngineRunner: createInferRuntime failed");
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!m_engine)
    {
        throw std::runtime_error("EngineRunner: deserializeCudaEngine failed");
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context)
    {
        throw std::runtime_error("EngineRunner: createExecutionContext failed");
    }

    int const n = m_engine->getNbIOTensors();
    m_io_tensor_names.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        char const* name = m_engine->getIOTensorName(i);
        if (name)
        {
            m_io_tensor_names.emplace_back(name);
        }
    }
}

EngineRunner::~EngineRunner() = default;

void EngineRunner::set_input_shape(std::string const& tensor_name, std::vector<int32_t> const& dims)
{
    nvinfer1::Dims d;
    d.nbDims = static_cast<int32_t>(dims.size());
    if (d.nbDims > nvinfer1::Dims::MAX_DIMS)
    {
        throw std::invalid_argument("EngineRunner::set_input_shape: too many dims");
    }
    for (int i = 0; i < d.nbDims; ++i)
    {
        d.d[i] = dims[static_cast<std::size_t>(i)];
    }
    if (!m_context->setInputShape(tensor_name.c_str(), d))
    {
        throw std::runtime_error("EngineRunner::set_input_shape failed for tensor: " + tensor_name);
    }
}

void EngineRunner::set_tensor_address(std::string const& tensor_name, void* ptr)
{
    if (!ptr)
    {
        throw std::invalid_argument("EngineRunner::set_tensor_address: null pointer");
    }
    if (!m_context->setTensorAddress(tensor_name.c_str(), ptr))
    {
        throw std::runtime_error("EngineRunner::set_tensor_address failed for tensor: " + tensor_name);
    }
}

void EngineRunner::set_tensor_address(std::string const& tensor_name, memory::Tensor const& tensor)
{
    auto const b = tensor.buffer();
    if (!b || !b->data())
    {
        throw std::invalid_argument("EngineRunner::set_tensor_address: tensor buffer is null");
    }
    set_tensor_address(tensor_name, b->data());
}

bool EngineRunner::execute(cudaStream_t stream)
{
    if (stream == nullptr)
    {
        throw std::invalid_argument("EngineRunner::execute: stream must not be null");
    }
    return m_context->enqueueV3(stream);
}

} // namespace llm_on_edge::engine

