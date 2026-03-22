# OpenAI 兼容 HTTP 服务（`llm_openai_server`）

本目录为 **llmOnEdge 项目 `src/` 下** 的 C++ 实现，使用 **`llm_on_edge::openai`** 命名空间：HTTP 形态对齐 OpenAI Chat Completions / vLLM，推理走与上游 `llm_inference` 相同的 **`LLMInferenceRuntime::handleRequest`**。

## 构建

在 **llmOnEdge 仓库根目录**：

```bash
cmake -S . -B build -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DLLMONEDGE_BUILD_OPENAI_SERVER=ON \
  -DLLMONEDGE_BUILD_OPENAI_SERVER_TESTS=ON
cmake --build build -j
ctest --test-dir build -R openai_mapping_test   # 可选
```

产物路径形如：`build/src/openai_server/llm_openai_server`。

## 运行

```bash
./llm_openai_server --engine-dir /path/to/engine \
  [--multimodal-engine-dir /path/to/mm] \
  [--host 0.0.0.0] [--port 8000] [--model my-model-id]
```

## 依赖说明

- **JSON 解析**：`../llm_input_parse/`（`llm_on_edge::parseInputFromJson`）与上游 `llm_inference` 输入 schema 一致。
- **HTTP**：CMake `FetchContent` 拉取 [cpp-httplib](https://github.com/yhirose/cpp-httplib)。
- **TensorRT-Edge-LLM**：由 `third_party` 子模块提供 `edgellmCore` 等库。
