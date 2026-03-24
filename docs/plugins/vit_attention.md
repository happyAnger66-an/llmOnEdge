# ViTAttentionPlugin 实现说明

本文分析 **TensorRT-Edge-LLM** 中 **`ViTAttentionPlugin`**（源码：`third_party/TensorRT-Edge-LLM/cpp/plugins/vitAttentionPlugin/`）的计算语义、与 Python/ONNX 的衔接方式，以及相对朴素实现的优化点。路径相对于 **llmOnEdge** 仓库根目录。

---

## 1. 插件在系统中的角色

- **用途**：视觉编码器（ViT）中的 **多头自注意力**，面向 **多图/变长序列打包**（batch 内多条序列拼成 `total_S`，用前缀和描述边界）等场景。
- **与 `AttentionPlugin`（LLM）的差异**：
  - 无 **KV Cache**、无 **RoPE 在插件内**（RoPE 在导出图里、进入插件前已算好 Q/K）。
  - 输入为 **Q/K/V 三份独立张量**，布局为 **`[total_S, num_heads, head_size]`**（head-major）。
  - 掩码类型为 **PADDING**（全连接注意力，仅在 **有效 token** 内计算；跨序列 padding 区域被屏蔽），而非 **因果因果**。

ONNX 侧通过 **`trt::ViTAttentionPlugin`** 节点接入；Python 封装见 `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/llm_models/layers/attention_plugin.py`（`vit_attention_plugin` / `symbolic_vit_attention_plugin`），视觉模型如 `qwen*_vl_model.py` 在 FP16 下调用该插件。

---

## 2. 接口与数据布局

### 2.1 插件注册名与版本

- **类型字符串**：`ViTAttentionPlugin`（`REGISTER_TENSORRT_PLUGIN(ViTAttentionPluginCreator)`）。
- **TensorRT 接口**：`IPluginV2DynamicExt`（动态形状）。
- **序列化字段**：`num_heads`、`head_size`（各 32 位整数）。

### 2.2 输入 / 输出

| 索引 | 名称 | 形状 / 类型 | 说明 |
|------|------|--------------|------|
| 0 | Q | `[total_S, H, D]`，FP16，LINEAR | Query |
| 1 | K | 同 Q | Key |
| 2 | V | 同 Q | Value |
| 3 | `cu_seqlens` | `[B+1]`，`int32` | 各序列长度 **前缀和**，用于变长 batch |
| 4 | `max_seqlen_carrier` | `[max_seqlen]`，`int32` | **仅用于携带运行时 `max_seqlen` 的维度**；元素值可忽略 |

输出：

| 索引 | 形状 |
|------|------|
| 0 | 与 Q 相同：`[total_S, H, D]` |

运行时从 `cu_seqlens` 长度推导 **`batch_size = len(cu_seqlens) - 1`**；从 `max_seqlen_carrier` 的第 0 维得到 **`runtimeMaxSeqLen`**，供 FMHA 启动参数使用。

### 2.3 语义要点

- **`total_S`**：当前步内所有序列 token 的总数（紧凑存储，非按最大长度 pad 到 `[B, S_max, ...]` 的 dense 布局）。
- **PADDING 掩码 + `cu_seqlens`**：内核按 **序列块** 做注意力，避免在无效位置做完整计算（与 FlashAttention 类变长接口一致）。
- 若导出与插件约定不一致（例如非 head-major），需在 Python 侧保证与注释一致（见 `supportsFormatCombination` 中的说明）。

---

## 3. 执行路径：两层内核二选一

`enqueue` 中根据编译选项与硬件选择实现：

### 3.1 路径 A：CuTe DSL FMHA（可选）

- **条件**：编译定义 **`CUTE_DSL_FMHA_ENABLED`**，且 **`mUseCuteDslFMHA`** 为真（默认在头文件中 **SM100+** 且未设置环境变量 **`DISABLE_CUTE_DSL_FMHA=1`** 时启用）；并满足 **`CuteDslFMHARunner::canImplementViT(head_dim, SM)`**（实现要求 **SM ≥ 100**），且 **`loadViTKernelModule()`** 成功。
- **行为**：调用 `CuteDslFMHARunner::run(...)`，对 Q/K/V/O 与 `cu_seqlens` 填 **动态形状与步长**，按 **head_dim** 分派到预编译的 **`vit_fmha_d64` / `d72` / `d80` / `d128`** 等模块。
- **适用**：新一代 GPU（Blackwell 等）上，CuTe DSL 生成的 **专用 FMHA**。

### 3.2 路径 B：FMHA v2（ContextFMHARunner）

- **条件**：CuTe 路径不可用或关闭时，使用 **`ContextFMHARunner`**：
  - `AttentionInputLayout::SEPARATE_Q_K_V`
  - `ContextAttentionMaskType::PADDING`
  - **`is_s_padded = false`**（**非** `[B, S, H, D]` 的 padding 稠密布局，而是 **由 `cu_seqlens` 驱动的 ragged/紧凑** 布局）
- **行为**：构造 `FusedMultiheadAttentionParamsV2`，设置 `q_ptr`、`k_ptr`、`v_ptr`、`o_ptr`，以及 **`cu_q_seqlens` 与 `cu_kv_seqlens`**（ViT 中二者指向同一 `cu_seqlens`），再 **`dispatchFMHAKernel`**。

构建期在 `ViTAttentionPlugin` 构造函数中会 **`canImplement`** 并 **预加载 kernel**（CuTe 或 `loadContextFMHAKernels`）；若 **无法为当前 SM / head_dim / FP16 组合实现**，则 **构造阶段抛异常**。

---

## 4. 实现原理（计算视角）

对 **标准缩放点积注意力**：

\[
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
\]

插件在 **设备端** 将 **QKᵀ、softmax、与 V 的乘法** 融合为 **FMHA** 内核（或等价 CuTe 实现），避免：

- 多次写出大张量中间结果到全局内存；
- 朴素实现中 **GEMM → softmax → GEMM** 的带宽与延迟。

**变长 batch**：通过 `cu_seqlens` 指定每个子序列在 `total_S` 中的范围，内核在 **序列边界** 内做全连接注意力，并对 **padding / 无效位置** 做掩蔽（PADDING 类型），与 `ContextFMHARunner` 里对 `is_s_padded` 与 `cu_*_seqlens` 的约定一致。

---

## 5. 主要优化点归纳

| 方向 | 说明 |
|------|------|
| **融合 FMHA** | 单内核（或紧耦合流水线）完成注意力主要阶段，降低 HBM 读写与 kernel launch 次数。 |
| **FMHA v2 侧** | 使用 **Flash Attention 风格**的 kernel 元数据（`contextFMHARunner.cpp` 中在支持的 SM 上 `flash_attention = true`）；根据 **序列长度与 head 维** 在 **granular tiling** 与 **非 tiled** 间选择，缓解短序列下的 **tile 量化损失** 或发挥大 head 时的优势。 |
| **CuTe DSL 路径** | 在 **SM100+** 上可选用 **按 head_dim 特化**的 ViT FMHA 模块（d64/d72/d80/d128），与通用 FMHA v2 形成 **双路径**，便于在新架构上拿到更优内核。 |
| **变长紧凑存储** | 使用 **`cu_seqlens` + `is_s_padded = false`**，避免按 `max_seqlen` 对整 batch 做无效计算；`max_seqlen` 仍通过 tensor 的 **shape** 传入以支持 **动态图** 下的正确调度。 |
| **无额外 workspace** | `getWorkspaceSize` 返回 **0**（工作区由内核内部策略或寄存器/共享内存解决，不额外向 TensorRT 申请 workspace）。 |
| **仅 FP16** | 当前实现固定 **FP16** 路径，与视觉分支导出时的 `float16` 转换一致，利于选用 **半精度 FMHA** 实现。 |

**限制与注意**：

- **Head 维**：`ContextFMHARunner::canImplement` 对通用路径支持 **64/72/80/128**（72/80 需 **SEPARATE_Q_K_V + PADDING**，与 ViT 插件一致）；CuTe ViT 路径同样覆盖 **64/72/80/128**（但 **SM 下限为 100**）。
- **回退**：CuTe 模块加载失败或 `canImplementViT` 失败时，自动 **回退到 FMHA v2**；若两者皆不可用，则 **插件创建失败**。

---

## 6. 与文档 `docs/trt-plugin.md` 的关系

`docs/trt-plugin.md` 概述了插件库整体构建与加载方式；本文聚焦 **ViTAttentionPlugin** 的 **数据布局、双路径 FMHA、变长语义与优化**。实现细节以 **`third_party/TensorRT-Edge-LLM/cpp/plugins/vitAttentionPlugin/vitAttentionPlugin.cpp`** 与 **`cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp`**（ViT `run`）为准。
