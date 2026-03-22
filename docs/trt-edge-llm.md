# TensorRT-Edge-LLM 项目分析

本文档基于 [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 开源仓库及其开发者指南整理，用于 **llmOnEdge** 的边缘推理引擎设计参考。文中版本与特性以仓库当前主线文档为准（README 标注 release **0.6.0**）。

---

## 1. 产品定位

TensorRT Edge-LLM 是面向 **NVIDIA 边缘与车载平台** 的高性能 **C++ 推理运行时**，主要承载大语言模型（LLM）与视觉语言模型（VLM）的自回归生成。特点包括：

- **端侧部署**：在 Jetson、DRIVE 等资源受限设备上完成 **引擎构建与端到端推理**（导出管线通常在 x86 主机，引擎在边缘设备上编译与运行）。
- **生产形态**：**运行时无 Python 依赖**，面向车载、机器人、工业 IoT、离线对话等场景。
- **完整工具链**：从 HuggingFace → ONNX → TensorRT Engine → C++ 推理的闭环。

官方文档入口：[TensorRT Edge-LLM Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)。

---

## 2. 支持范围概览

### 2.1 硬件与软件平台

| 类别 | 说明 |
|------|------|
| **官方支持** | NVIDIA Jetson Thor（JetPack 7.1）、NVIDIA DRIVE Thor（DriveOS 7，细节见发行文档） |
| **兼容/实验** | Jetson Orin + JetPack 6.2.x（官方后续 JetPack 会正式支持） |
| **其他 GPU** | 文档注明可在部分离散 GPU 等环境做实验，但 **非官方支持** |

### 2.2 模型与精度

- **LLM 家族**：Llama 3.x、Qwen 2/2.5/3、DeepSeek-R1 Distilled 等（完整矩阵见项目 *Supported Models*）。
- **VLM**：Qwen2/2.5/3-VL、InternVL3-1B/2B-hf、Phi-4-Multimodal 等。
- **量化**：FP16、FP8（SM89+）、INT4 AWQ/GPTQ、NVFP4（SM100+）；KV Cache 可单独 **FP8** 以进一步省显存。

### 2.3 运行时与构建阶段能力（功能全景）

下表按 **流水线阶段** 归纳，便于与自研引擎对照。

| 阶段 | 能力 |
|------|------|
| **Python 导出** | HF 加载、可选量化（ModelOpt 等）、ONNX 导出、**图手术（Graph Surgery）**、生成构建配置 |
| **Engine Builder（C++）** | 加载自定义 **TensorRT 插件**、解析 ONNX、**双优化配置（Prefill / Decode）**、编译引擎、输出 tokenizer/EAGLE 映射等 |
| **C++ Runtime** | 分词、可选 ViT 多模态、**Linear KV Cache**、TRT 执行、**CUDA Graph 解码**、采样（greedy / top-k / top-p / 温度）、LoRA 热切换、EAGLE3 投机解码独立运行时 |
| **增值特性** | 系统 Prompt KV 复用、词表裁剪（reduce vocab）、ASR / MoE / TTS 等示例工作流（见 User Guide Examples） |

---

## 3. 端到端架构（三阶段流水线）

整体数据流可概括为：

```text
HuggingFace 模型
    → Python Export Pipeline（量化 + ONNX + 图优化 + 配置）
    → ONNX 与 config
    → Engine Builder（C++，插件 + TensorRT Builder）
    → TensorRT Engine + tokenizer 等产物
    → C++ Runtime（自回归循环 + 显存/KV/图优化）
    → 应用 / Examples
```

与 **llmOnEdge** 的对照意义：Edge-LLM 把 **「导出 / 建图 / 运行」** 拆得很清楚；自研引擎若不走 TensorRT，仍可借鉴其 **双阶段 Profile、KV 布局、Decode 侧 CUDA Graph、投机解码状态机** 等设计思想。

---

## 4. 核心软件架构

### 4.1 Engine Builder

- **两类构建器**：**LLM Builder**（语言模型主体）、**Visual Encoder Builder**（VLM 视觉塔）。
- **典型 8 步流程**（两类构建器共享骨架，细节不同）：插件加载 → 配置解析 → 网络创建 → 模型类型识别 → ONNX 解析 → **Optimization Profile 设置** → TensorRT 编译 → 产物与目录管理。
- **LLM 优化配置（双阶段）**：
  - **Context / Prefill Profile**：长序列、并行吃显存，batch 往往较小。
  - **Generation / Decode Profile**：每步 1 token（或投机场景下的特殊形状），计算密集、可通过更大 batch 提高吞吐。
- **EAGLE3**：对 **base / draft** 各走一遍构建流程，生成双引擎及 **draft→target 词表映射**（如 `d2t.safetensors`）。
- **版本约束**：文档强调 ONNX 与 Engine **不可跨 TensorRT / Edge-LLM 版本随意迁移**，升级需重新导出与编译。

### 4.2 C++ Runtime：两套互斥实现

对外都提供 **`handleRequest`** 风格 API，但内部实现二选一：

| 运行时 | 适用场景 | 核心对象 |
|--------|----------|----------|
| **LLM Inference Runtime** | 标准自回归 + VLM | 单个 `LLMEngineRunner` |
| **LLM Inference SpecDecode Runtime** | EAGLE3 投机解码 | `LLMEngineRunner`（base）+ `EagleDraftEngineRunner`（draft）+ EAGLE 工具核 |

### 4.3 标准 LLM 路径的关键组件

- **LLMEngineRunner**：持有 **单一 TensorRT execution context**，在 Prefill 与 Generation 间 **切换 optimization profile**；管理 **Linear KV Cache**；产出 logits 供采样；负责 **CUDA Graph**（见下节）。
- **Tokenizer**：偏向 HuggingFace 兼容的 BPE 流程（GPT / Llama / Qwen 等词表与特殊符号）。
- **Multimodal Runner**：Qwen-VL / InternVL 等 ViT 推理，输出 **图像 token / embedding** 供 LLM 消费。
- **Linear KV Cache**：按 batch 的线性布局，配合变长序列；与 Attention 插件约定的 KV 张量布局一致（见 §5.2）。
- **Sampling Kernels**：GPU 侧 logits → 分布 → 采样，由 **Runtime** 调用而非 EngineRunner 内部完成，职责分离清晰。

### 4.4 推理时序（概念）

1. **Prefill**：整段 prompt（+ 可选 ViT）一次性前向，填满 KV，采第一个 token。
2. **Decode 循环**：每步用上一 token 更新 KV，前向得到 logits，再采样；直至 EOS / 最大长度等。

文档给出**可选 CUDA Graph**：在 **Decode** 阶段用预捕获图降低 **kernel launch** 开销，量级上描述为约 **10%–30%** 延迟收益（随平台与模型变化）。

### 4.5 EAGLE SpecDecode 路径（摘要）

- **Prefill**：先 **仅 base** 做标准 prefill；用 base 的 hidden states 等在 **首轮** 再 **prefill draft**；之后 draft 侧多用 **accept-token** 路径而非重复全量 prefill。
- **Generation**：draft 建 **候选 token 树**（多轮 proposal、top-k 选枝等），base 做 **整树并行验证**；**最终输出 token 始终来自 base**，draft 仅提供假设；文档说明可对 **draft proposal / draft accept / base verify / base 普通 decode** 等路径做 **CUDA Graph**。
- **工程要点**：**双 KV**、隐藏态同步、**动态 batch 与完成序列驱逐（compaction）**——与标准单引擎运行时相比状态机更复杂。

---

## 5. 关键技术原理（与自研引擎相关的「为什么这么做」）

### 5.1 TensorRT 与自定义插件

- 计算图在 **TensorRT** 内做 **算子融合、精度选择、内核筛选**，边缘 SoC 上由 Builder 针对 **SM / 带宽** 做适配。
- **Attention**、**INT4 分组量化 GEMM** 等以 **TensorRT Plugin** 形式实现（文档说明正向 **IPluginV3** 迁移）。插件在 **导出阶段** 通过 ONNX 中的插件节点与 **建引擎阶段** 的注册 creator 衔接。

### 5.2 AttentionPlugin 与 KV / 解码形态（文档级约定）

- 覆盖 **MHA / GQA（含 MQA 场景）**、**RoPE**、**线性 KV Cache**、**chunked prefill**、普通 decode 与 **EAGLE 树注意力**。
- **KV 数据类型**：FP16 或 **FP8（CUDA ≥ 11.8）**，与 FP8 KV Cache 用户指南一致。
- **张量角色示例**（摘自设计文档）：`PackedQKV` 布局 `[B, S, H, D]`；`KVCache` 布局 `[B, 2, H, S, D]`（`S` 为容量维度）；可选树结构的 mask / position id 用于投机解码。

### 5.3 CUDA Graph 在工程中的用法

- **动机**：Decode 每步调用链短但频率高，**重复 launch** 成为瓶颈；CUDA Graph 把可静态化的一段 GPU 工作 **录制为单图**，用 `cudaGraphLaunch` **降低 CPU 侧开销**。
- **Edge-LLM 实现要点**（从源码命名可印证）：按 **输入形状 / batch / LoRA 权重路径** 等组合维护 **多张图的哈希表**；切换 LoRA 时需 **重新捕获** 对应图；若捕获失败会回退到普通 `enqueue` 路径。

### 5.4 显存与 KV 相关策略汇总

- **maxKVCacheCapacity**：建引擎与运行时的核心 **容量旋钮**，文档建议与 `maxInputLen` + 预期生成长度一起规划。
- **FP8 KV**：在导出/量化与 ONNX 侧打开 FP8 KV 后，由 Attention 插件与引擎构建 **贯通**；EAGLE 场景下文档说明 **通常仅 base 使用 FP8 KV**，draft 因缓存小为精度考虑可保持非 FP8。
- **System Prompt KV Cache**：相同系统指令跨请求复用 KV，降低 **TTFT**。

### 5.5 投机解码（EAGLE3）原理对齐

- **Draft 小模型**快速扩展 **token 树**；**Base 大模型**一次性对树节点做 **验证**，按匹配前缀接受 token，分歧处丢弃后续 draft 分支。
- 与 **llmOnEdge** 相关：**双引擎调度、双 KV、验证核、batch 压缩** 均是独立模块，可单独借鉴。

### 5.6 Python 导出与量化

- 工具链包含：`quantize-llm`、`export-llm`、`export-visual`、`export-draft`、`quantize-draft`、`insert-lora`、`process-lora` 等，对应 **单塔 LLM / 视觉塔 / EAGLE draft / LoRA 权重规范**。
- 量化常借助 **NVIDIA Model Optimizer（ModelOpt）**；GPTQ 等可走预量化权重或第三方工具链。

---

## 6. 与 llmOnEdge 的直接可借鉴点

| 方向 | Edge-LLM 做法 | 对 llmOnEdge 的启发 |
|------|----------------|----------------------|
| **GPU 显存** | 双 Profile 分离 prefill/decode；KV 容量显式上限；FP8 KV；INT4 权重量化 | 分阶段调优与 **可预测的峰值显存** |
| **KV Cache** | Linear 布局、与注意力内核/插件同构；chunked prefill；系统 prompt 级复用 | 指定 **layout 与容量策略**，与内核一体设计 |
| **CUDA Graph** | Decode 路径按配置哈希多图缓存；LoRA 变更触发重捕获 | 先保证 **形状稳定子图**，再上图优化 |
| **运行时结构** | EngineRunner vs Runtime 分层；采样在 Runtime | 清晰的 **执行 / 策略** 边界 |
| **投机解码** | 独立 SpecDecode 运行时 + 专用 CUDA utils | 避免把投机逻辑塞进标准路径导致不可维护 |

---

## 7. 已知限制（0.6.0 摘录）

以下内容来自项目 *Limitations*，仅作集成预期管理：

- 部分模型在 **TensorRT 10.15** 上使用 **NVFP4** 可能出现精度问题，JetPack 7.1 建议使用配套 **TRT 10.13.3.9**。
- **maxBatchSize=1** 可能导致部分模型精度或建引擎失败，**设为 2** 或可规避。
- **CuTe DSL** 相关内核在 **运行时 batch>1** 时可能挂死，故部分内核仅在 **batch=1** 启用。

---

## 8. 参考链接与子模块

本仓库通过 **Git submodule** 引用上游源码，路径为 **`third_party/TensorRT-Edge-LLM`**。克隆后需执行：

```bash
git submodule update --init --recursive third_party/TensorRT-Edge-LLM
```

- 上游仓库：<https://github.com/NVIDIA/TensorRT-Edge-LLM>
- 文档站点：<https://nvidia.github.io/TensorRT-Edge-LLM/>
- 下文及 `docs/` 中凡写 **`TensorRT-Edge-LLM/...`** 的路径，均相对于 `third_party/TensorRT-Edge-LLM`。
- 开发者指南中的软件设计：`TensorRT-Edge-LLM/docs/source/developer_guide/software-design/`
  - `python-export-pipeline.md`
  - `engine-builder.md`
  - `cpp-runtime-overview.md`
  - `llm-inference-runtime.md`
  - `llm-inference-specdecode-runtime.md`
- 插件与内核：`TensorRT-Edge-LLM/docs/source/developer_guide/customization/tensorrt-plugins.md`，以及同目录下 `kernelSrcs/fmha_v2`、`kernelSrcs/xqa` 等说明。
- 本仓库对 **`cpp/plugins`** 的归纳说明见 [trt-plugin.md](trt-plugin.md)。
