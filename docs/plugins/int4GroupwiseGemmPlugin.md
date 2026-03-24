# Int4GroupwiseGemmPlugin 实现说明

本文分析 **TensorRT-Edge-LLM** 中 **`Int4GroupwiseGemmPlugin`**（源码：`third_party/TensorRT-Edge-LLM/cpp/plugins/int4GroupwiseGemmPlugin/`）的计算语义、张量约定、CUDA 内核路径及相对「先反量化再 GEMM」的优化点。路径相对于 **llmOnEdge** 仓库根目录。

---

## 1. 插件在系统中的角色

- **用途**：**INT4 分组（group-wise）权重量化**下的线性层 / GEMM，常见于 **GPTQ、AWQ** 等 W4A16（权重 4bit、激活 FP16）推理。
- **核心行为**：在 GPU 上计算  
  **Y = X · W**，其中 **W 以 INT4 压缩存储**，并按 **group_size** 配有 **FP16 缩放因子**；实现上通过专用内核在计算过程中 **融合反量化与乘加**，避免先把完整 FP16 权重 materialize 到全局内存。
- **TensorRT 接口**：**`IPluginV3`**（同时实现 `IPluginV3OneCore` / `IPluginV3OneBuild` / `IPluginV3OneRuntime`），与 **`ViTAttentionPlugin` 使用的 `IPluginV2DynamicExt`** 不同代际，Creator 为 **`IPluginCreatorV3One`**。

Python / ONNX 侧见 `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/llm_models/layers/int4_gemm_plugin.py`（自定义算子 **`trt::int4_gemm_plugin`**，导出为 **`trt::Int4GroupwiseGemmPlugin`**）；AWQ 等场景下还可通过 **`int4_dq_gemm_to_plugin`**（`onnx_graphsurgeon`）把图中的 DQ+GEMM 模式替换为该插件节点。

---

## 2. 插件属性与张量形状

### 2.1 序列化 / 创建字段

| 字段 | 含义 |
|------|------|
| `gemm_n` | 输出特征维 **N**（对应权重列数） |
| `gemm_k` | 输入特征维 **K**（对应权重行数） |
| `group_size` | 分组量化步长；沿 **K** 方向每 **group_size** 个元素共享标量尺度 |

### 2.2 输入 / 输出（`supportsFormatCombination`）

约定 **batch 维合并为二维视图**：逻辑上 **M = 第 0 维 × 第 1 维**，最后一维为 **K / N**。

| 位置 | 张量 | 类型 | 形状约束 |
|------|------|------|----------|
| 0 | 输入激活 | FP16，LINEAR | **3D**：`[..., ..., K]`，且 **最后一维 = `mGemmK`** |
| 1 | 量化权重 `qweight` | INT8，LINEAR | **2D**：**`[N/2, K]`**（每字节存 **2 个 int4**，故行数为 **N/2**） |
| 2 | 缩放 `scales` | FP16，LINEAR | **2D**：**`[K / group_size, N]`** |
| 3（输出） | 输出 | FP16，LINEAR | **3D**：前两维与输入相同，**最后一维 = N（= `mGemmN`）** |

`getOutputShapes` 将输出写为：`out[d0]=in[d0]`，`out[d1]=in[d1]`，`out[d2]=constant(mGemmN)`。

### 2.3 计算语义（直观）

对每组沿 **K** 的 **group_size** 个输入通道，使用对应 **scale** 将 INT4 权重还原到 FP16 参与点积；数学上等价于 **分组反量化后的矩阵乘**，具体 INT4 打包与 **device 端 dequantize** 与 `cpp/kernels/int4GroupwiseGemmKernels/dequantize.cuh` 中 **`dequantize_s4_to_fp16x2`** 等逻辑一致（映射到 **[-8, 7]** 等约定，见该文件注释）。

---

## 3. `enqueue` 与内核分派

```text
M = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1]
```

- **`M <= 6`**：调用 **`gemv_forward_cuda_new`**（小 batch / 窄矩阵乘向量风格路径）。
- **`M > 6`**：调用 **`gemm_forward_cuda_new`**（通用 GEMM 路径）。

二者函数签名见 `cpp/kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h`。注意：头文件对 GEMV 的说明侧重 **M 约 1～4** 的优化场景，而 **插件阈值固定为 6**，以仓库中 **`int4GroupwiseGemmPlugin.cpp`** 为准。

**Workspace**：`getWorkspaceSize` 恒为 **0**。

---

## 4. CUDA 内核侧优化要点（实现原理）

内核代码位于 **`cpp/kernels/int4GroupwiseGemmKernels/`**，思路与 **MIT HAN Lab AWQ** 参考实现同源（见 `int4WoqGemmCuda.cu` / `int4WoQGemvCuda.cu` 文件头引用）。

| 方向 | 说明 |
|------|------|
| **融合反量化 + GEMM/GEMV** | 在 kernel 内对 packed INT4 做 **dequantize**（PTX/位技巧，`dequantize.cuh`），与 **MMA/累加** 流水线结合，避免写出完整 **N×K** FP16 权重。 |
| **Tensor Core 风格块算**（GEMM） | `int4WoqGemmCuda.cu` 使用 **`mma_m16n8k16`**、**`ldmatrix`**、**`cp.async`** 等，配合 **共享内存** 与 **warp 级分块**（如 `OP_M/OP_N/OP_K`、`INTRIN_*` 等宏），提升访存与计算重叠。 |
| **L2 cache hint** | 在支持的 CUDA 版本下对 **`cp.async`** 使用 **L2 cache hint** 宏，减轻全局内存压力。 |
| **块索引映射** | `get_block_idx_mapping` / `get_log_tile` 等用于 **tile 划分与调度**，改善大 **N** 时的占用与并行度。 |
| **GEMV 路径** | `int4WoQGemvCuda.cu` 针对 **极小 M** 使用 **warp reduce**（`__shfl_xor_sync` 等）等，降低小批量时的无效开销。 |

---

## 5. 与朴素实现的对比（为何要插件）

| 朴素流程 | 本插件路径 |
|----------|------------|
| INT4 → 展开为 FP16 权重张量（显存 **×8** 量级相对 4bit） | 权重保持 **INT8 打包**，按需 **在寄存器/共享内存中** 解包 |
| 调用通用 cuBLAS GEMM | **融合内核**，减少带宽与 kernel 次数 |
| 小 **M** 仍走大方阵 GEMM | **`M` 较小时走 GEMV** 特化（阈值见上） |

---

## 6. 与其它文档的关系

- 插件库总览、构建与加载：**`docs/trt-plugin.md`**。
- 本文聚焦 **Int4GroupwiseGemmPlugin** 的 **IPluginV3 接口、张量布局、GEMV/GEMM 分派与内核优化**；导出与图变换细节以 **`int4_gemm_plugin.py`** 及 **`onnx_export/onnx_utils.py`** 为准。

---

## 7. 实现文件索引

| 路径 | 作用 |
|------|------|
| `cpp/plugins/int4GroupwiseGemmPlugin/int4GroupwiseGemmPlugin.cpp` | 插件 `enqueue`、形状与类型约束、Creator |
| `cpp/kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h` | `gemm_forward_cuda_new` / `gemv_forward_cuda_new` 声明 |
| `cpp/kernels/int4GroupwiseGemmKernels/int4WoqGemmCuda.cu` | GEMM 主内核 |
| `cpp/kernels/int4GroupwiseGemmKernels/int4WoQGemvCuda.cu` | GEMV 主内核 |
| `cpp/kernels/int4GroupwiseGemmKernels/dequantize.cuh` | INT4 → FP16 解包 |
