# CuTe DSL FMHA in Blackwell（实现原理与优化要点）

本文总结 `TensorRT-Edge-LLM` 中基于 CuTe DSL 的 FMHA（Fused Multi-Head Attention）在 Blackwell(SM100+) 平台上的实现机制与性能优化思路。

相关代码入口：
- 内核生成源：`third_party/TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py`
- 运行时包装：`third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp`
- 上层调度：`third_party/TensorRT-Edge-LLM/cpp/plugins/attentionPlugin/attentionPlugin.cpp`

---

## 1. 设计目标

CuTe DSL FMHA 的核心目标是：

1. 把 `Q*K^T`、`Softmax`、`P*V` 融合在一个 kernel 中执行；
2. 减少中间结果落地（global memory round-trip）；
3. 利用 Blackwell 的 TMA + TensorCore（tcgen05）能力提升吞吐和降低时延。

---

## 2. 代码分层与职责

## 2.1 `fmha.py`（算法/内核生成）

`BlackwellFusedMultiHeadAttentionForward` 定义了：
- tile 形状与布局映射
- warp specialization（不同 warp 负责不同阶段）
- 多 pipeline + barrier 的执行编排
- causal/sliding-window mask 行为

它不是普通 Python 逻辑，而是 CuTe DSL 的 kernel 描述与生成入口。

## 2.2 `cuteDslFMHARunner.cpp`（AOT 模块调用）

Runner 负责：
- 加载 AOT kernel module（d64/d128 及 sliding-window 变体）
- 打包动态 shape/stride
- 计算缩放参数（`softmaxScale`, `scaleSoftmaxLog2`）
- 调用 `cute_dsl_*_wrapper(...)`

可理解为“把 TensorRT runtime 的张量描述转成 CuTe kernel ABI”。

## 2.3 `attentionPlugin.cpp`（路径选择）

`AttentionPlugin` 在 prefill 路径上判定是否使用 CuTe DSL FMHA。当前逻辑较保守（例如常见条件包含 batch 约束、FP8 KV cache 约束）；不满足时会回退 FMHA_v2。

---

## 3. 核心执行原理

## 3.1 融合计算主线

单 kernel 完成：
1. 读取 Q/K/V（或 Q + KV cache）
2. 计算 attention score（QK）
3. softmax 归一化（含数值稳定修正）
4. 与 V 相乘得到输出 O
5. 写回输出

通过融合减少中间矩阵的存取开销。

## 3.2 Warp Specialization

在 `fmha.py` 中同一 CTA 的 warp 被固定分工（示意）：
- `softmax0` warps
- `softmax1` warps
- `correction` warps
- `mma` warp
- `load` warp
- `epilogue` warp
- `empty/sync` warp

收益是不同阶段并行推进，降低“所有 warp 同步做同一件事”的空转。

## 3.3 Pipeline + Barrier 编排

内核创建了多条并行 pipeline（例如 load->mma、mma->softmax、softmax->correction、correction->epilogue），并通过 `NamedBarrier/mbarrier` 协调数据依赖。

特点：
- load warp 持续喂数据（TMA）
- mma warp 持续消费并计算
- softmax/correction warp 处理归一化和数值修正
- epilogue warp 执行写回

形成“分工流水线”而不是串行阶段。

## 3.4 TMA 与 TensorCore 协同

关键硬件点：
- TMA 用于高效 G2S/S2G tile 传输
- tcgen05 MMA tile（典型 128x128）做主算力
- SMEM/TMEM 分级缓存用于阶段间缓冲

这也是该实现在 Blackwell 上收益显著的根本原因。

## 3.5 Mask 策略（编译期裁剪）

- Causal：右窗口固定为 0（禁止看未来）
- Sliding window：可选左窗口
- 当关闭 sliding window 时，相关逻辑可在编译期裁剪，减少分支与指令路径

---

## 4. 数据布局与形状要点

LLM 路径常见张量：
- Q: `[B, S_q, H_q, D]`
- KV cache: `[B, 2, H_kv, Cap, D]`
- O: `[B, S_q, H_q, D]`
- `cum_seqlen_k`: `[B+1]`（每 batch 的有效 KV 长度）

内核内部会重映射为适合 MMA 的 layout，并根据 `H_q/H_kv` 的分组关系 (`h_r`) 做广播或分块。

---

## 5. 当前实现约束（工程侧）

1. 仅支持 Blackwell SM100+。
2. LLM CuTe DSL 变体在 runner 中主要是 `head_dim=64/128`（ViT 有 64/72/80/128 变体）。
3. 上层插件策略不是所有场景都命中 CuTe DSL，未命中则回退到 FMHA_v2。
4. 某些组合（如 FP8 KV cache + 特定 prefill 形态）会限制 CuTe 路径覆盖。

---

## 6. 为什么快（性能来源）

1. **融合**：减少中间结果写回/再读取；
2. **流水并行**：load/mma/softmax/correction/epilogue 同时推进；
3. **硬件对齐**：tile 与 Blackwell TensorCore/TMA 协同；
4. **编译期特化**：mask/window 逻辑按配置裁剪。

---

## 7. 性能优化建议

## 7.1 短期（配置级）

1. 尽量让请求形态命中 CuTe DSL 路径（避免不必要回退）。
2. 对服务流量分桶（prefill-heavy 与 decode-heavy），分别调 profile。
3. 若不需要 sliding window，关闭以获得更短指令路径。

## 7.2 中期（实现级）

1. 扩展 CuTe 变体覆盖（更多 head_dim、更多 batch 形态）。
2. 减少回退路径中的 layout 转换开销，提升“非命中场景”性能。
3. 结合实际流量重设 tile/pipeline stage 参数，优化 occupancy 与带宽平衡。

## 7.3 长期（架构级）

1. 完善 end-to-end 自适应调度：根据 `B/S/H/D` 与 KV dtype 自动选择最优 kernel 家族。
2. 构建稳定 benchmark 矩阵（B,S,H,D,mask,window,dtype）+ Nsight 指标基线，持续回归优化。

---

## 8. 建议观测指标（Profiling）

建议用 Nsight Systems + Nsight Compute 重点看：
- kernel 时间占比（是否命中 CuTe DSL）
- DRAM 吞吐与 L2 hit（是否带宽受限）
- SM occupancy 与 warp stall 原因
- TensorCore 利用率
- barrier wait 比例（流水是否被同步点卡住）

---

## 9. 一句话总结

CuTe DSL FMHA 的本质是：**在 Blackwell 上用 TMA + TensorCore + warp 专职流水，把 attention 三段计算融合成一个持久化高吞吐 kernel**。调优重点不是“改一行公式”，而是“提高命中率 + 减少回退/搬运 + 按真实流量做特化”。
