# INT4 Group-wise GEMM / GEMV（`int4GroupwiseGemmKernels`）

本文总结 **TensorRT-Edge-LLM** 中 `cpp/kernels/int4GroupwiseGemmKernels/` 的实现思路、与 **W4A16（权重量化、激活 FP16）** 的对应关系，以及在工程里的**调用链与硬约束**。源码路径相对于 **`third_party/TensorRT-Edge-LLM`**（或独立克隆的 TensorRT-Edge-LLM 仓库根目录）。

---

## 1. 在系统里做什么

该目录实现 **按组缩放的 INT4 线性层** 的 GPU 计算，数学上等价于：

\[
\text{out}_{m,n} = \sum_{k} \text{in}_{m,k} \cdot \bigl(\text{dequant}(\text{W\_int4}_{n,k})\bigr),
\]

其中权重以 **4 bit 有符号整数** 存储，**每个长度为 `group_size` 的 K 维分组共享一个 FP16 scale**（与常见 AWQ / W4G128 类格式一致）。激活 **`in`** 为 **FP16**。

对外 C++ 接口见 `int4GroupwiseGemm.h`：

| 符号 | 含义 |
|------|------|
| `gemv_forward_cuda_new` | 小 batch 的 **GEMV / 窄 GEMM**（偏推理里 `M` 很小） |
| `gemm_forward_cuda_new` | 较大 batch 的 **GEMM** |

**TensorRT 侧**：`cpp/plugins/int4GroupwiseGemmPlugin/int4GroupwiseGemmPlugin.cpp` 在 `enqueue` 里根据 **`M`** 分支调用上述二者（`M ≤ 6` 走 GEMV，否则走 GEMM）。插件与加载方式见 [trt-plugin.md](./trt-plugin.md)。

**上游参考**：实现标注改编自 MIT HAN Lab **[llm-awq](https://github.com/mit-han-lab/llm-awq)** 的 CUDA 量化内核（`quantization_new` 路径）。

---

## 2. 张量布局（与头文件注释一致）

- **输入激活** `in_feats`：`[M, K]`，**row-major**，**`half`**。
- **量化权重** `kernel`：逻辑为 **`[N, K]` 的 INT4**，在内存中以 **`int8_t` 打包**：**每字节 2 个 int4**，故 leading 维长度为 **`N/2`**，即总大小对应 **`[N/2, K]`**。  
  GEMM 主机端把 `int8_t*` **`reinterpret_cast` 为 `half const*`**，按 **16 bit 一块** 与共享内存 / `ldmatrix` 路径对齐读取（权重视作「packed 位型」载荷，而非语义上的 FP16 矩阵）。
- **缩放** `scaling_factors`：**FP16**，形状 **`[K / group_size, N]`**（按 K 分组、每组每输出通道一个 scale）。
- **输出** `out_feats`：`[M, N]`，**`half`**。

---

## 3. 反量化核心：`dequantize.cuh`

设备函数 **`dequantize_s4_to_fp16x2`** 将 **打包的 8 个 int4**（经 `half2` / `uint32_t` 视图）展开为 **8 个 FP16**。

- **数值域**：注释写明映射到 **\([-8, 7]\]** 的整数格点再进到 FP16（对称 4 bit）。
- **实现手段**：用 **`lop3.b32`** 与 **`sub.f16x2` / `fma.rn.f16x2`** 等 **PTX**，在寄存器里并行处理多路 nibble，避免朴素移位与分支，贴近 **AWQ 原版内核** 的位运算技巧。

后续在 GEMV / GEMM 里会对展开后的 FP16 权重再 **`__hmul2`（或等价）乘上对应组的 `scale`**，完成 group-wise 反量化。

---

## 4. GEMV 路径：`int4WoQGemvCuda.cu`

**目标场景**：**`M` 很小**（decode / 小 batch），按 **输出通道块** 并行，在线程内做 **K 维累加**。

要点：

- **`warp_reduce`**：warp 内 **`__shfl_xor_sync`** 树形归约，再把部分和写到 **`__shared__`**，由块内线程写回全局 **`outputs`**。
- **权重加载**：`uint32_t` / `float4` 粒度加载 packed 权重，再 **`dequantize_s4_to_fp16x2`**；与 **`local_scale`** 相乘后，与激活做 **`__hfma2`** 累加。
- **模板参数**：`N_PER_BLOCK = 2`、`K_INTERLEAVE = 4`、`BLOCK_SIZE = 256`、`GroupSize = 128`（实例化时写死 **128**）。

---

## 5. GEMM 路径：`int4WoqGemmCuda.cu`

**目标场景**：**`M` 较大**，采用典型 **Tensor Core 友好** 的块 GEMM 结构。

要点：

- **Tile**：如 **`CTA_M=64, CTA_N=128, CTA_K=64`**，warp 级 **`mma.sync.aligned.m16n8k16.row.col.f16`**；**`ldmatrix`** 从 shared 取矩阵片。
- **全局 → 共享**：**`cp.async.cg`**（CUDA 11.4+ 可带 **L2 cache hint**）双缓冲 / 流水线（**`STAGES=4`**），隐藏访存。
- **B 侧（权重）**：`ldmatrix` 读出 packed 片后，在 **`share_to_reg_one_stage_B_T2`** 里调用 **`dequantize_s4_to_fp16x2`**，再乘 **shared 中的 scale**，得到参与 MMA 的 **FP16 权重片**。
- **Group 维**：核函数模板 **`G = 128`** 写死在 `gemm_forward_cuda_new` 调用的实例化里；**`global_to_share_one_stage_scales_T2`** 按 **`G / CTA_K`** 决定 scale 装入节奏。

启动配置里对 **动态共享内存** 有上界 **`static_assert`（&lt; 99 KiB）**，依赖具体 GPU 的 shared / dynamic shared 上限。

---

## 6. 双路径如何选择（与插件一致）

`Int4GroupwiseGemmPlugin::enqueue` 中：

- **`M ≤ 6`** → **`gemv_forward_cuda_new`**
- **`M > 6`** → **`gemm_forward_cuda_new`**

其中 **`M`** 来自输入张量描述：前两维展平后的 batch 规模（与插件里 `inputDesc[0]` 的维度约定一致）。

---

## 7. 使用场景与限制（重要）

以下为阅读源码得到的**硬约束或强假设**；若与 ONNX / 引擎配置不一致，会出现 **异常、错误结果或未定义行为**。

### 7.1 `group_size`

- **GEMV**：`gemv_forward_cuda_new` 内 **`group_size != 128`** 会 **`throw std::runtime_error`**。
- **GEMM**：`gemm_forward_cuda_new` 虽接收 **`group_size` 参数**，但内核模板 **`G` 固定为 128**，**未使用该参数**。因此 GEMM 路径在效果上同样只支持 **K 维按 128 分组** 的 scale 布局。  
- **插件**虽序列化 **`group_size`**，若配置为非 128，**与当前内核不匹配**。

### 7.2 GEMV 对 batch `M`

- 仅 **`M ∈ {1, 2, 3, 4, 5, 6}`** 有 **`switch` 分支**；否则 **`throw std::runtime_error`**。

### 7.3 维度对齐（由 grid / 循环推导）

- **GEMV**：`num_blocks = n / N_PER_BLOCK / K_INTERLEAVE`，即 **`n` 需能被 `2 × 4 = 8` 整除**（否则 launch 维度与循环假设不成立）。
- **GEMM**：`j_factors1 = n / CTA_N`（`CTA_N = 128`），**隐含 `n` 宜为 128 的倍数**；`m` 由核内 `write_row < M` 等守卫处理边界，但非对齐时性能与正确性依赖实现细节，**工程上应按插件构建时的固定 N、K 使用**。

### 7.4 数值与硬件

- **激活 / scale / 输出为 FP16**；累加路径中 GEMV 归约使用 **FP32 中间量** 再写回 half，仍属 **近似计算**。
- 依赖 **Ampere 及以后** 上常见的 **`mma`、`ldmatrix`、`cp.async`** 等能力；具体 SM 版本需与项目整体 **`CMAKE_CUDA_ARCHITECTURES`** 一致。

### 7.5 错误处理

- **`gemv_forward_cuda_new`**：非法参数抛 C++ 异常；插件 **`enqueue` 用 try/catch**，捕获后返回 **-1**。
- **`gemm_forward_cuda_new`**：声明 **`noexcept`**，**不抛异常**；CUDA 错误需依赖调用方或异步检查。

---

## 8. 相关文件一览

| 路径 | 作用 |
|------|------|
| `cpp/kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h` | 对外 API |
| `cpp/kernels/int4GroupwiseGemmKernels/dequantize.cuh` | int4 → FP16 打包反量化 |
| `cpp/kernels/int4GroupwiseGemmKernels/int4WoQGemvCuda.cu` | 小 M GEMV |
| `cpp/kernels/int4GroupwiseGemmKernels/int4WoqGemmCuda.cu` | 大 M GEMM |
| `cpp/plugins/int4GroupwiseGemmPlugin/int4GroupwiseGemmPlugin.cpp` | TRT 插件 `enqueue` 分发 |
| `unittests/woqInt4GemvTest.cu` / `woqInt4GemmTest.cu` | 单元测试入口 |

---

## 9. 与 llmOnEdge 的关系

llmOnEdge 通过子模块依赖 TensorRT-Edge-LLM；**不重复实现**上述内核。若在端侧改 **量化格式、group size 或 M/N/K**，需要同时核对 **插件序列化字段** 与 **本目录内核是否仍匹配**，必要时扩展模板参数或新增 kernel 实例。

更多插件加载与构建说明见 **[trt-plugin.md](./trt-plugin.md)**。
