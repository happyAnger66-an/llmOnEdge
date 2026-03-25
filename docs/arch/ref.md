# TensorRT-Edge-LLM 与 TensorRT-LLM 实现梳理（参考笔记）

本文档对 **`TensorRT-Edge-LLM`** 与 **`TensorRT-LLM`** 的实现做对照梳理，供自研端侧推理框架时参考。仓库路径以本机为例：`TensorRT-Edge-LLM`、`TensorRT-LLM`（可与 `llmOnEdge/third_party/TensorRT-Edge-LLM` 对照阅读）。

相关设计文档：[`docs/edgeLLM.md`](../edgeLLM.md)。

---

## 一、产品定位差异（决定架构取舍）

| 维度 | TensorRT-Edge-LLM | TensorRT-LLM |
|------|-------------------|--------------|
| 主场景 | 车载 / Jetson / DRIVE 等**端侧**，强调 **C++ 运行时无 Python 依赖** | **数据中心 / 多卡 / 服务化**，Python API + 大规模生态 |
| 推理形态 | 目录加载 engine + JSON 配置，**同步批内 prefill/decode**（`LLMEngineRunner` 注释：**不做 continuous batching**） | **Executor + 调度 / 批处理 / KV 管理**，面向高吞吐与多请求并发 |
| 语言栈 | **C++ 为核心**；Python 主要做导出与建引擎 | **Python 编排 + C++ 执行**；`nanobind` 暴露 `Executor` 等到 Python |
| 规模 | 单机单卡/少卡、插件与 kernel 面向边缘裁剪 | TP/PP/CP、MoE、分离式推理、Triton 等完整链路 |

**结论**：Edge 更接近「**嵌入式一体化 runtime**」；TRT-LLM 更接近「**可扩展推理平台**」。自研端侧库时，Edge 的**分层与资源所有权**更贴近；TRT-LLM 的 **Executor 抽象、异步与 KV 子系统**更值得借鉴思想，不必整盘照搬复杂度。

---

## 二、TensorRT-Edge-LLM：代码结构与数据流

### 2.1 目录与构建

- **`cpp/`**：`common/`（tensor、logger、TRT 工具、文件/safetensors）、`kernels/`（FMHA、embedding、MoE、KV 相关 CUDA 等）、`plugins/`、`multimodal/`、`runtime/`、`tokenizer/`、`sampler/`、`profiling/`。
- **单一大静态库思路**（如 `edgellmCore`）：聚合 `common` / `kernels` / `runtime` 等源；CUDA 上对 FMHA 等按 **SM 裁剪**减少体积（CMake 中 `EXCLUDE_SM_*`），适合端侧控制二进制大小。

### 2.2 运行时分层

1. **`LLMEngineRunner`**（`cpp/runtime/llmEngineRunner.*`）  
   - 持有 **TensorRT IRuntime / Engine / ExecutionContext**。  
   - 持有 **`LinearKVCache`**、RoPE cos/sin 等。  
   - 设计假设：**prefill 与 decode 同步、批内齐步走**，不做服务端 continuous batching。

2. **`LLMInferenceRuntime`**（`llmInferenceRuntime.*`）  
   - **编排层**：组合 `LLMEngineRunner` + 可选 `MultimodalRunner` + `Tokenizer`。  
   - 管理 **大块设备/主机 tensor**（embedding、sampling workspace、input ids、logits、multimodal indices、deepstack 等）。  
   - **系统提示 + LoRA** 维度的 **KV 复用缓存**（哈希表 + `SystemPromptKVCache`）。  
   - **`examineRequest` / `setUpForPrefillExecution`**：校验与张量准备与引擎步进分离。  
   - **指标**分阶段：`LLMPrefillMetrics` / `LLMGenerationMetrics`，多模态另有 metrics。

3. **多模态**  
   - `MultimodalRunner` 与各模型 ViT/Audio 等 runner，由 `LLMInferenceRuntime` 分支调度，与主 LLM 引擎解耦。

4. **其它**  
   - **CUDA Graph**：decode 路径可选 capture，失败则降级。  
   - **插件**：集中一次性加载（如 `loadEdgellmPluginLib` + `std::call_once`）。  
   - **Python** `tensorrt_edgellm/`：导出 ONNX、量化、建引擎——与 C++ runtime **职责分离**。

**可借鉴**：**Runner = 单引擎步进 + KV + TRT 资源**；**InferenceRuntime = 请求语义 + 张量池 + tokenizer + 多模态 + 缓存策略**。

---

## 三、TensorRT-LLM：代码结构与数据流

### 3.1 目录概览

- **`tensorrt_llm/`（Python）**  
  - **`models/` / `layers/`**：模型与算子，对接 builder 产 engine。  
  - **`executor/`**：`GenerationExecutor` ABC、`submit` / `abort_request`、`GenerationRequest`/`GenerationResult`；`Proxy`/`Worker`、RPC、Ray 等多进程与分离式路径。  
  - **`runtime/`**：`model_runner_cpp.py` 等封装 C++ `Executor`；`kv_cache_manager_v2` 等 **KV 子系统**。  
  - **`llmapi/`**：高层 LLM API、参数类型。  
  - **`serve/`、`triton_backend/`**：服务与 Triton。  
  - **`_torch/`**：PyTorch/PyExecutor、auto_deploy 等第二条执行栈（体量大，属平台级扩展）。

- **`cpp/tensorrt_llm/`**  
  - **`runtime/`**：`tllmRuntime.cpp`、`gptDecoder`、`decoderState`、`bufferManager`、`loraManager`、**`worldConfig`**（并行拓扑）等。  
  - **`plugins/`**：按算子拆 CMake 子目录。  
  - **`kernels/`**：Cutlass/CUTE、decoder attention、MoE 等。  
  - **`executor_worker/`、`nanobind/`**：绑定与 worker 通信。

### 3.2 Python 与 C++ 边界

- `tensorrt_llm/executor/executor.py`：`from ..bindings import executor as tllm`，C++ **Executor** 在 Python 侧统一为 `submit(request) -> GenerationResult`。  
- **`CppExecutorError`**：包装 C++ 异常与栈信息，便于排障。  
- **`ModelRunnerCpp`**：包装 Executor，提供 generate 系列 API，承接 batch、KV 比例、speculative / lookahead 等**策略参数**。

**可借鉴**：**稳定绑定层 + 薄 Python 适配**，避免调度逻辑散落在脚本里。

### 3.3 调度与生命周期（与 Edge 对比）

- **多进程 / 队列**：`IterationResultQueue`、`FusedIpcQueue` / `IntraProcessQueue`，传递迭代结果与统计。  
- **`atexit` + `weakref` 注册 shutdown**：降低异常退出时资源悬挂风险。  
- **KV**：v2 管理器 + block reuse、chunked context 等，面向**高并发与显存利用率**；Edge 的 `LinearKVCache` 更简单、确定性强。

---

## 四、两库共通实践（可映射到自研 edgeLLM）

1. **严格分层**  
   - 底层：**引擎执行 + KV + TRT 上下文**（Edge：`LLMEngineRunner`；TRT-LLM：`runtime` + decoder）。  
   - 上层：**请求对象、张量布局、tokenizer、多模态、缓存策略**（Edge：`LLMInferenceRuntime`；TRT-LLM：`GenerationRequest` + `ModelRunner` + Executor 配置）。

2. **配置结构化**  
   - Edge：`LLMEngineRunnerConfig` + JSON 与 engine 同目录。  
   - TRT-LLM：`GptJsonConfig` / `EngineConfig` / `WorldConfig` 等，**构建期与运行期配置分离**。

3. **张量与工作区归一**  
   - 由会话 / runtime 对象持有大 buffer，避免每请求乱分配；Edge 在 `LLMInferenceRuntime` 私有成员中较集中。

4. **指标与调试**  
   - Edge：分阶段 metrics + `profiling/`。  
   - TRT-LLM：`layerProfiler`、`debug_mode`、logger 与 LLM debug 开关联动。

5. **插件 / 自定义算子**  
   - 独立目录、CMake 子工程、运行时 **一次性加载**，与主引擎解耦。

6. **失败可降级**  
   - Edge：CUDA Graph capture 失败则继续无 graph。自研库可对可选优化统一采用同样策略。

7. **类型化的 Request/Response**  
   - TRT-LLM：`GenerationRequest`、`SamplingParams`、LoRA、multimodal、disaggregated 等进入统一请求对象，利于扩展与测试。

---

## 五、自研框架时的取舍建议

- **更贴近 Edge**：目录即部署单元（engine + config）、**C++ 单进程会话**、同步或小批、KV 用 Linear 或 paged **由产品定位决定**。  
- **更贴近 TRT-LLM**：先实现 **Executor 式接口**（`submit` / `abort` / 异步结果），便于以后接队列或多请求；KV 管理独立模块，预留 **block/paged** 扩展。  
- **首版可刻意不引入**：Ray/RPC/分离式、完整 `_torch` 路径，避免平台级复杂度拖垮端侧迭代。

---

## 六、相关链接

- Edge 官方文档：[TensorRT Edge-LLM Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)  
- TRT-LLM 官方文档：[TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)  
- 仓库内：[TensorRT Edge-LLM 项目分析](../trt-edge-llm.md)
