# CuTe DSL FMHA（`fmha.py`）与 AOT 构建在 TensorRT-Edge-LLM 中的角色

本文汇总 `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` 的代码结构、技术要点，及其在 **TensorRT-Edge-LLM** 工程中的集成方式。源码以本仓库子模块为准：

- `third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py`
- 更完整的构建与集成说明见同目录 [`README.md`](../third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/README.md)

---

## 1. 文件定位

`fmha.py` 是用 **NVIDIA CuTe DSL（Python）** 编写的 **Blackwell（SM100+）融合多头注意力（FMHA）** 内核描述：在 **单个 kernel** 中串联 **QKᵀ → softmax（含因果 / 滑动窗等掩码）→ PV**，并针对 **Tensor Core（tcgen05）+ TMA + warp 专化持久化调度** 做优化。

与仓库中 **FMHA_v2（预编译 cubin）** 不同：CuTe DSL 路径在 **CMake 构建阶段** 于 **带 Blackwell GPU 的构建机** 上执行 `fmha.py --export_only`，**AOT 生成** `.o` + `.h`，再被 C++ Attention 插件链接调用。

文件头注释概括了技术栈与运行方式（节选）：

```54:61:third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py
"""
A fused multi-head attention (FMHA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.
```

**上游来源**：CUTLASS 中 `examples/python/CuTeDSL/blackwell/fmha.py` 等（具体 commit 见 `README.md`）。Edge-LLM 相对上游的增量以 **`fmha.patch`** 记录。

---

## 2. 工程内关键改动（相对上游 CUTLASS 示例）

`README.md` 与 patch 说明中的要点（与自研引擎设计相关）：

| 方向 | 说明 |
|------|------|
| **运行时形状** | 将 batch、序列长度、头数等改为 **运行时动态维度**（`mark_*_dynamic`），避免为每种形状单独编译 |
| **滑动窗注意力（SWA）** | 支持 `window_size`；无 SWA 时在编译期去掉左侧窗逻辑以提升性能 |
| **Prefix / KV** | **融合 KV cache** 布局 `[B, 2, H_kv, S, D]`，减少临时 KV 分配与 layout 转换 |
| **依赖** | 用 **CuPy / NumPy** 替代 PyTorch，便于独立 C++ 链路 |
| **AOT 导出** | `--export_only`、`--output_dir`、`--file_name`、`--function_prefix` → `export_to_c()` |
| **ViT 模式** | `--vit_mode`：packed varlen Q/K/V + 双向注意力，与 LLM 因果路径分离 |

---

## 3. 代码结构（约 3800 行）

| 部分 | 作用 |
|------|------|
| **`BlackwellFusedMultiHeadAttentionForward`** | 占绝大部分行数：用 CuTe DSL **描述** FMHA 的访存与计算；warp 分工（softmax0/1、correction、MMA、load、epilogue 等）、persistent CTA、`MaskEnum`、**`is_causal` / `use_sliding_window`**、`actual_head_dim` 与 MMA tile 对齐（TMA zfill、OOB 等） |
| **`run(...)`** | **驱动入口**：校验 shape/dtype、实例化上述类、准备 CuPy → `cute.Tensor`、**动态维度标记**、`cute.compile`；`export_only` 时 `export_to_c` 后返回；否则参考验证与 benchmark |
| **`__main__` + `argparse`** | `--q_shape` / `--k_shape`、`--is_causal`、`--bottom_right_align`、`--window_size`、`--vit_mode`、`--export_only` 等，最终调用 `run(...)` |

类构造中体现了掩码与滑动窗在 **编译期** 的分叉策略：

```108:182:third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py
class BlackwellFusedMultiHeadAttentionForward:
    WINDOW_NO_LIMIT = 1 << 30

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
        is_causal: bool = False,
        use_sliding_window: bool = False,
        actual_head_dim: Optional[int] = None,
    ):
        # ... qk_acc_dtype, mma_tiler, is_persistent, mask_type ...
        self.is_causal = is_causal
        self.use_sliding_window = use_sliding_window
```

---

## 4. `run()`：LLM 路径与 ViT 路径

### 4.1 LLM（默认）

- **Q**：`[B, S, H, D]`
- **KV**：融合 cache **`[B, 2, H_kv, S, D]`**（与 Attention 插件、RoPE 写 KV 布局一致）
- **`cu_kv_seqlens`**、滑动窗宽度 `_wsl` 等传入编译后的 kernel
- 使用 `mark_bshd_dynamic`、`mark_kv_cache_dynamic` 标记动态维

```2992:3012:third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py
    else:
        # LLM: batched Q [B,S,H,D] + combined KV cache [B,2,H,Cap,D]
        q_dyn = mark_bshd_dynamic(q_tensor)
        kv_dyn = mark_kv_cache_dynamic(kvcache_tensor)
        o_dyn = mark_bshd_dynamic(o_tensor)
        # ...
        compiled_fmha = cute.compile(
            fmha,
            q_dyn, kv_dyn, o_dyn, cu_kv_seqlens, _wsl,
            scale_softmax_log2, scale_softmax, scale_output,
            current_stream,
        )
```

### 4.2 ViT（`--vit_mode`）

- Packed **`[total_S, H, D]`** 的 Q、K、V + **`cu_seqlens`**
- **双向**注意力；编译目标为 **`fmha.__call_vit__`**

```2955:2991:third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py
    if vit_mode:
        # ViT: packed [total_S, H, D] with cu_seqlens for ragged batching.
        # ...
        compiled_fmha = cute.compile(
            fmha.__call_vit__,
            q_dyn, k_dyn, v_dyn, o_dyn, cu_dyn, _max_seqlen,
            scale_softmax_log2, scale_softmax, scale_output,
            current_stream,
        )
```

### 4.3 AOT 导出（构建系统使用）

`--export_only` 时跳过参考检查与 benchmark，仅写出 `.h` / `.o`：

```3017:3025:third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py
    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_fmha.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None
```

---

## 5. 构建产物：八个变体

开启 **`ENABLE_CUTE_DSL_FMHA=ON`** 时，`cmake/CuteDslFMHA.cmake` 会 **多次调用** `fmha.py`（各变体不同 CLI），典型包括：

| 变体 | Head Dim | SWA | 模式 | 因果 |
|------|----------|-----|------|------|
| `fmha_d64` / `fmha_d128` | 64 / 128 | 否 | LLM | 是 |
| `fmha_d64_sw` / `fmha_d128_sw` | 64 / 128 | 是 | LLM | 是 |
| `vit_fmha_d64` / `d72` / `d80` / `d128` | 64/72/80/128 | — | ViT | 否 |

产物目录一般为：`cpp/kernels/contextAttentionKernels/cuteDSLArtifact/`，并定义 **`CUTE_DSL_FMHA_ENABLED`** 供 C++ 条件编译。

依赖版本（由 CMake 校验/安装）见 `README.md`：`nvidia-cutlass-dsl`、对应 CUDA 版本的 `cupy` 等。

---

## 6. 运行时集成链路

```text
fmha.py (构建期, --export_only)
    → .o + .h (cuteDSLArtifact/)
    → 链接进 Attention 插件 + cute_dsl_runtime
    → CuteDslFMHARunner::run(...) 从 C++ 调度
    → attentionPlugin 在 Blackwell 上优先选用；失败或未编译则 FMHA_v2
```

- **C++ 封装**：`third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.{h,cpp}`  
  - LLM：`run(qPtr, kvPtr, oPtr, cuKVSeqLens, stream, slidingWindowSize)`  
  - ViT：packed Q/K/V + `cuSeqLens` 等  
- **插件**：`cpp/plugins/attentionPlugin/attentionPlugin.cpp` — 构造时检查编译宏与 `canImplement()`（如 SM ≥ 100、head dim 64/128），尝试加载模块并可选回退 **FMHA_v2**。  
- **RoPE 与 KV 布局**：`cpp/kernels/posEncoding/applyRopeWriteKV.cu` — CuTe DSL 路径下 RoPE **直接写入** 融合 KV `[B, 2, H_kv, S, D]`。  
- **强制 v2**：环境变量 **`DISABLE_CUTE_DSL_FMHA=1`**（进程启动前设置，插件创建时读取）。

---

## 7. 与 llmOnEdge 的关联

- **`fmha.py` 不参与在线推理 Python 路径**；仅 **构建期 AOT** + 开发期可本地跑精度/benchmark。  
- 若自研边缘引擎要对齐 Edge-LLM：**Prefill/Context 注意力**、**融合 KV 布局**、**动态 batch/seq/head**、**SWA + bottom-right 对齐** 等设计可直接对照本文与 `README.md`。  
- 更细的内核数学与 warp 流水线需直接阅读 `BlackwellFusedMultiHeadAttentionForward` 主体内 DSL 代码；与 **TensorRT 插件 I/O 张量名、形状** 的对应关系以 `attentionPlugin.cpp` 为准。

---

## 8. 相关文件索引（子模块内）

| 路径 | 说明 |
|------|------|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL 主源码 |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | 辅助（掩码等） |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | 相对 CUTLASS 的 diff |
| `cmake/CuteDslFMHA.cmake` | 构建期调用 `fmha.py` |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.*` | C++ 加载与 dispatch |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT 插件集成与回退 |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | CuTe DSL KV 布局的 RoPE |

---

*文档随子模块版本变化；升级 `third_party/TensorRT-Edge-LLM` 后请对照上游 `README.md` 与 `limitations` 更新行为说明。*
