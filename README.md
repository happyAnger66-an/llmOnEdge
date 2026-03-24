# llmOnEdge

端侧推理依赖 [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)，以 **`third_party/TensorRT-Edge-LLM`** 子模块形式引用。**业务代码统一放在 `src/` 目录**，使用 **`llm_on_edge`** 命名空间。

## 目录

| 路径 | 说明 |
|------|------|
| `src/llm_input_parse/` | Edge LLM 批量请求 JSON → `LLMGenerationRequest` |
| `src/openai_server/` | OpenAI 风格 REST API + `llm_openai_server` |
| `third_party/TensorRT-Edge-LLM/` | 上游 SDK（子模块） |
| `docs/` | 笔记文档 |

## 构建

需本机已安装 **CUDA Toolkit**，且 CMake 能通过 **`find_package(CUDAToolkit REQUIRED)`** 找到（与 `nvcc` 一致）。若曾配置失败并留下 **`NOTFOUND`** 的缓存项，请删掉 **`build/`** 后重新 `cmake`。

```bash
git submodule update --init --recursive third_party/TensorRT-Edge-LLM

cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DLLMONEDGE_BUILD_OPENAI_SERVER=ON \
  -DLLMONEDGE_BUILD_OPENAI_SERVER_TESTS=ON

cmake --build build -j
```

HTTP 服务说明见 [`src/openai_server/README.md`](src/openai_server/README.md)。

## 仅构建上游 SDK（不编 llmOnEdge 服务）

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=... \
  -DLLMONEDGE_BUILD_OPENAI_SERVER=OFF
```

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```