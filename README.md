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

## 开启 CuTe DSL FMHA（编译执行 `fmha.py`）

`third_party/TensorRT-Edge-LLM` 中的 `fmha.py` 会在 CMake 构建阶段由
`cmake/CuteDslFMHA.cmake` 自动调用（生成 `fmha_*.o/.h`）。

### 1) 推荐方式：通过 CMake 自动触发

建议先启用 Python 虚拟环境（避免把依赖装到系统 Python）：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

配置与构建：

```bash
cmake -S . -B build-fmha -G Ninja \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_CUTE_DSL_FMHA=ON

cmake --build build-fmha -j
```

说明：
- 该流程会自动检查/安装 `nvidia-cutlass-dsl` 与 `cupy`（版本由 `CuteDslFMHA.cmake` 按 CUDA 主版本选择）。
- 需要在构建机上具备 Blackwell GPU（SM100/SM110）来编译该批内核。

产物目录（自动生成）：
`third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cuteDSLArtifact/`

### 2) 手动单独执行 `fmha.py`（调试单变体）

```bash
cd third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell

python3 fmha.py \
  --q_shape 1,1024,14,128 --k_shape 1,1024,1,128 \
  --is_causal --is_persistent --bottom_right_align \
  --export_only --output_dir ./out \
  --file_name fmha_d128 --function_prefix fmha_d128
```

如需仅做参考正确性检查（不导出 `.o/.h`），去掉 `--export_only` 即可。

### 3) 运行时回退到 FMHA_v2（可选）

```bash
export DISABLE_CUTE_DSL_FMHA=1
```

取消回退：

```bash
unset DISABLE_CUTE_DSL_FMHA
```