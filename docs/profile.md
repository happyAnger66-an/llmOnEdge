# CuTe DSL FMHA（Blackwell）资源用量：结论与精确测量

本文说明 **head 维度从 128 提到 256** 时，CuTe DSL FMHA 内核在 **寄存器** 与 **共享内存（SMEM）** 上的压力变化，以及如何在本仓库构建出 `.o` 后用工具读出 **compiler 报告的精确值**。

## 结论（定性 + 为何不能仅靠公式）

1. **`D` 进入核内主要与工作集大小成比例**  
   FMHA 在 tile 内需要驻留 **Q / K / V** 的片段，以及 **softmax 刻度、累加器、写回缓冲** 等。head 维度 `D` 变大时，这些张量在寄存器与 SMEM 中的占用近似按 **O(D)** 或 **O(D × tile)** 增长（具体由 `fmha.py` 里选的 tile shape、warp 分工和流水阶段数决定）。

2. **`D=256` 相较 `D=128` 通常会同时推高 REG 与 SMEM**  
   - **SMEM**：双缓冲/多 stage 的 **K/V（及中间结果）** 往往在 SMEM 里开窗口；`D` 翻倍，单 stage 的 KV tile 字节数也接近翻倍，**静态 SMEM** 或 **编译期可知的 SMEM 上限** 往往明显上升。  
   - **寄存器**：每个 warp 持有的 **片段寄存器、地址/stride、流水线状态** 增加；在极端情况下 compiler 会 **溢出到 LOCAL**，表现为资源报告里 **LOCAL > 0** 或占用率下降。

3. **精确数字必须以当前工具链 + 架构 + `fmha.py` 参数为准**  
   资源用量由 **CUTLASS/CuTe DSL 版本、`fmha.py` 的 Q/K shape、是否 sliding window、`--is_persistent` 等** 共同决定；手算只能估计量级，**不能替代** `cuobjdump` / Profiler 对具体 cubin 的报告。

4. **与 `D=128` 对比时建议固定其它因素**  
   在同一台机器、同一 CUDA/CUTLASS DSL 版本下，对 **`fmha_d128.o` 与 `fmha_d256.o`**（或 `_sw` 变体）分别 `cuobjdump`，对比 **REG / SMEM / LOCAL** 最有参考价值。

## 如何拿到精确值（推荐流程）

### 1. 先构建生成 cubin 对象文件

CuTe DSL FMHA 在配置 TensorRT-Edge-LLM / llmOnEdge 工程并完成 CuTe DSL 依赖后，由 CMake 调用 `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` 生成对象文件，默认输出目录为：

`third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cuteDSLArtifact/`

成功构建后应能看到例如：

- `fmha_d128.o` / `fmha_d256.o`
- `fmha_d128_sw.o` / `fmha_d256_sw.o`

变体与 shape 定义见 `third_party/TensorRT-Edge-LLM/cmake/CuteDslFMHA.cmake`（例如 LLM 变体当前使用 `Q: 1,1024,14,D` 与 `K: 1,1024,1,D`）。

### 2. 使用 `cuobjdump` 读取每内核资源

在已安装 CUDA Toolkit 的机器上（需与编译所用 `cuobjdump` 版本匹配或至少能解析该 cubin）：

```bash
# 查看该 .o 中所有内核及资源摘要
cuobjdump -res-usage /path/to/cuteDSLArtifact/fmha_d256.o
```

对 `fmha_d128.o` 重复一次，便于逐项对比 **Registers**、**Shared Memory**、**Local Memory**。

若工程链接的是合并后的 fatbin，也可对最终含设备的二进制执行同样命令，但 **直接对 `fmha_d*.o` 最干净**，避免与其它内核混在一起。

### 3. 可选：NVIDIA Nsight Compute（运行时核对）

`cuobjdump` 给出的是 **静态资源**。若在特定 grid/block 下仍存在动态共享内存或驱动侧限制，可用 Nsight Compute 在真实 launch 上核对 **Achieved Occupancy、Spills、Registers/SMEM 瓶颈**。

## 一行对比示例（生成 Markdown 表可自行改编）

```bash
for k fmha_d128 fmha_d256 fmha_d128_sw fmha_d256_sw; do
  echo "=== $k ==="
  cuobjdump -res-usage third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cuteDSLArtifact/${k}.o
done
```

将输出中的 **Registers per thread**、**Shared Memory bytes per block**（以及若存在 **Local Memory**）记入表格即可得到「精确结论」；表格数值会随 **CUDA / CUTLASS DSL / `fmha.py` 提交** 变化，宜在发布或调优分支上固定版本后记录一份 baseline。

## 说明

- 本文不给出固定数字占位，避免在本地未构建或工具链不一致时产生误导。  
- 若 `cuobjdump` 报无法解析对象，请确认该 `.o` 确为 **含设备代码** 的 CUDA 对象，并与本机 `cuobjdump` 来自同一套足够新的 Toolkit。
