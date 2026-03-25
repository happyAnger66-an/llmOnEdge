# Tensor 与内存：底层分配接口设计（llmOnEdge）

本文档描述 **llmOnEdge** 自研推理框架中 **张量与内存管理** 的分层设计与接口约定。实现参考 **TensorRT-LLM**（`tensorrt_llm::runtime`）中的成熟拆分，但不绑定其 TensorRT 类型与工程依赖，便于独立演进。

参考源码位置（本地克隆，仅作阅读）：

- `TensorRT-LLM/cpp/include/tensorrt_llm/runtime/iBuffer.h` — `MemoryType`、`IBuffer`
- `TensorRT-LLM/cpp/include/tensorrt_llm/runtime/iTensor.h` — `ITensor`
- `TensorRT-LLM/cpp/include/tensorrt_llm/runtime/bufferManager.h` — `BufferManager`
- `TensorRT-LLM/cpp/include/tensorrt_llm/runtime/memoryCounters.h` — `MemoryCounters`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/cudaMemPool.*` — 设备侧内存池（可选）

---

## 1. 设计目标

1. **统一抽象**：按 **逻辑内存类型**（GPU / 页锁定主机 / 普通主机）分配与释放，上层不直接散落 `cudaMalloc` / `cudaHostAlloc` / `malloc`。
2. **可统计**：对各类内存分别记录 **当前占用量**（及可选 **最近一次分配/释放的增量**），便于调试与 OOM 分析。
3. **分层清晰**：底层只负责 **字节缓冲 + 元数据**；**张量** 在缓冲之上增加 **形状与元素类型**，与 TRT-LLM 中 **IBuffer → ITensor** 的关系一致。
4. **可替换**：具体分配策略（同步/异步 GPU、`cudaMemPool`、是否使用 pinned pool）通过 **可注入的分配器** 切换，不污染业务张量代码。
5. **线程安全**：全局/进程级统计使用原子操作；单路 stream 上的 GPU 分配行为由调用方保证与 TRT-LLM 相同契约（见下文）。

---

## 2. TRT-LLM 中的优秀实践（摘要）

| 概念 | 作用 |
|------|------|
| **`enum class MemoryType`** | 区分 `kGPU`、`kCPU`、`kPINNED`、`kUVM`、`kPINNEDPOOL` 等，所有 buffer 携带类型信息。 |
| **`IBuffer`** | 一维逻辑存储：`data()`、`getSize()`（元素个数）、`getDataType()`、`getMemoryType()`、`resize`/`release`，以及 `slice`/`view` 等视图。 |
| **`ITensor` : `IBuffer`** | 增加 `nvinfer1::Dims` 形状、`reshape` 等；元素个数与体积一致时与 buffer 统一。 |
| **`BufferManager`** | 集中提供 `gpu` / `cpu` / `pinned` / `managed` 等工厂方法，并统一 **拷贝** 与 **stream**；GPU 侧可配合 **异步分配**（如 `cudaMallocAsync`）与 **设备内存池**。 |
| **`MemoryCounters`** | 单例式全局计数，按类型 `allocate`/`deallocate` 更新；提供 **当前量** 与 **最近一次 diff**，便于日志打印。 |

llmOnEdge 首版可只实现 **GPU + Pinned + CPU** 三类的统计与分配；**UVM**、**PinnedPool**、**IPC/NVLS** 等可作为后续扩展，接口上预留 `MemoryType` 枚举值即可。

---

## 3. 概念模型

### 3.1 内存类型 `MemoryType`

建议使用 **强类型枚举**（与 TRT-LLM 对齐命名习惯，便于对照阅读）：

| 值 | 含义 | 典型 API |
|----|------|----------|
| `kGPU` | 设备全局内存 | `cudaMalloc` / `cudaMallocAsync` |
| `kCPU` | 可分页主机内存 | `std::aligned_alloc` / `malloc`（按对齐需求封装） |
| `kPINNED` | 页锁定主机内存（含可映射到设备的标志按需组合） | `cudaHostAlloc` |

可选扩展：`kUVM`（`cudaMallocManaged`）、`kPINNEDPOOL`（主机侧池化 pinned，对应 TRT-LLM 的 `pinnedPool`）。

**设备号**：`kGPU` 缓冲应携带 **`device_id`**（默认 0），多 GPU 时统计与释放必须按设备区分（可在 `MemoryStats` 中按 `(MemoryType, device_id)` 分桶，首版也可仅支持单卡全局桶 + 断言单卡）。

### 3.2 分层：`Buffer` → `Tensor`

```
┌─────────────────────────────────────────────────────────┐
│  Tensor（张量）                                           │
│  - shape / strides（或仅 contiguous + dims）               │
│  - element DType（fp16 / int32 / …）                     │
│  - 持有或引用底层 Buffer                                  │
└───────────────────────────┬─────────────────────────────┘
                            │ 1:1 或 view
┌───────────────────────────▼─────────────────────────────┐
│  IBuffer（缓冲）                                         │
│  - memory_type, device_id?                                │
│  - size_bytes 或 size_elements + dtype                   │
│  - data pointer                                           │
│  - capacity vs size（支持 resize 不立刻收缩）             │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  IAllocator / MemoryResource（每类内存一种或策略对象）     │
│  - allocate(size, alignment) → 原始指针 + deleter         │
│  - 可选：与 CudaStream 绑定（仅 GPU 异步路径）             │
└─────────────────────────────────────────────────────────┘
```

与 TRT-LLM 一致：**张量是带形状的缓冲**；视图（slice）共享同一设备存储，由引用计数或 `shared_ptr` 管理生命周期，避免重复释放。

---

## 4. 统计模块 `MemoryStats`（对应 `MemoryCounters`）

### 4.1 建议接口

- **按类型计数**（字节数，原子）：`current_gpu()`、`current_cpu()`、`current_pinned()`（与 TRT-LLM 的 `getGpu()` / `getCpu()` / `getPinned()` 对应）。
- **最近一次操作增量**（可选，便于打一条 log）：`last_diff_gpu()` 等，在每次 `allocate`/`deallocate` 后更新。
- **非单例 vs 单例**  
  - **单例**（TRT-LLM 风格）：全进程一处 `getInstance()`，实现简单。  
  - **可注入 `MemoryStats&`**：每个 `Session`/`Runtime` 独立统计，便于单测与多实例隔离；**推荐**在 llmOnEdge 中采用 **可注入 + 默认进程级实例**，兼顾测试与全局调试。

### 4.2 更新时机

在 **底层唯一分配/释放路径** 中调用：

- `on_allocate(MemoryType, bytes)`  
- `on_deallocate(MemoryType, bytes)`  

`Buffer` 析构、`unique_ptr` 自定义 deleter、或 **池化归还** 时都必须触发 `on_deallocate`，避免统计漂移。

### 4.3 辅助

- `bytes_to_string(uint64_t)`：人类可读（与 TRT-LLM `MemoryCounters::bytesToString` 同类）。
- `to_string()`：导出各线当前值，用于 `LOG_INFO` 或调试端点。

---

## 5. 分配器接口

### 5.1 最小接口 `IAllocator`（按类型分派）

两种等价组织方式，二选一即可：

**方案 A — 按类型分派（贴近 BufferManager）**

```text
struct AllocationDesc {
  MemoryType type;
  int device_id = 0;      // 仅 GPU
  size_t size_bytes;
  size_t alignment = 0;   // 可选，默认 max_align_t
};

class IAllocator {
 public:
  virtual ~IAllocator() = default;
  // 返回唯一所有权；析构时释放并更新 MemoryStats
  virtual std::unique_ptr<void, DeleterFn> allocate(AllocationDesc const&) = 0;
};
```

实际实现中更常见的是 **typed unique_ptr 包装为 `Buffer` 对象**，而不是裸 `void*`。

**方案 B — C++ PMR 风格 `memory_resource`（每类型一个）**

- `DeviceMemoryResource`（cuda）  
- `PinnedMemoryResource`  
- `HostMemoryResource`  

上层 `BufferManager` 持有三者指针 + 共享的 `MemoryStats*`，与 TRT-LLM 的 `BufferManager::gpu/cpu/pinned` 工厂一一对应。

### 5.2 `BufferManager`（门面，可选命名 `MemoryService`）

职责与 TRT-LLM `BufferManager` 对齐：

- `gpu(size_bytes, dtype, stream?)`  
- `cpu(...)`  
- `pinned(...)`  
- `allocate(MemoryType, ...)` 统一入口  
- 后续：`copy(src, dst)`（遵守 `cudaMemcpyAsync` 与 stream 同步规则）

**GPU 与 stream**：若使用 `cudaMallocAsync` / 内存池，构造时注入 **`cudaStream_t` 或封装对象**，与 TRT-LLM 中 `BufferManager(CudaStreamPtr stream, ...)` 一致。

---

## 6. `Buffer` 对象设计要点

对齐 `IBuffer` 的核心能力（实现可精简）：

| 能力 | 说明 |
|------|------|
| `data()` / `size_bytes()` / `capacity()` | 基本属性 |
| `memory_type()` / `device_id()` | 元数据 |
| `resize(n)` | 小于等于 capacity 时可能原地；超过则 **重新分配 + 拷贝 + 更新统计** |
| `release()` | 提前释放并清零指针 |
| 禁止拷贝 | 赋值/拷贝构造 delete，避免双重释放 |

**元素类型**：首版可用 `enum class ElementType` 或 `std::size_t element_size`，不必依赖 `nvinfer1::DataType`；与 TensorRT 对接层再映射。

---

## 7. `Tensor` 对象设计要点

对齐 `ITensor`：

- 持有 `std::shared_ptr<Buffer>` 或 `Buffer` 的 `unique_ptr`（若独占）。
- `shape()`：`std::vector<int64_t>` 或固定 `MaxDims` 的小数组（与 Edge 侧 `tensor` 模块一致即可）。
- `reshape`：在 **体积不变** 或 **在 capacity 允许范围内** 调整解释方式；体积变大则通过底层 buffer `resize`。
- **视图**：`slice`/`narrow` 共享存储，引用计数由 `Buffer` 共享持有。

---

## 8. 线程安全与并发

- **统计**：`MemoryStats` 内部 `std::atomic<uint64_t>`（按类型分字段）。  
- **GPU 分配**：若使用默认 stream 或同步 API，多线程同时分配需 **外部互斥** 或与 TRT-LLM 一样由 **单线程 executor** 调用；异步分配必须与 **同一 stream 序** 一致。  
- **CPU/Pinned**：若使用普通 `malloc`，注意对齐；多线程并发分配一般安全，统计仍用原子。

---

## 9. 可选扩展（非首版必选）

| 扩展 | 说明 |
|------|------|
| **CUDA Memory Pool** | 参考 `CudaMemPool::getPrimaryPoolForDevice`，`cudaMallocFromPoolAsync`，析构时 `trim` |
| **Pinned 子池** | 减少频繁 `cudaHostAlloc` 开销 |
| **UVM** | 统一虚拟内存，简化 CPU/GPU 双份，latency 与策略需单独文档 |
| **调试分配器** | 包裹真实分配器：填充 canary、检测越界 |

---

## 10. llmOnEdge 落地顺序建议

1. **`MemoryType` + `MemoryStats`（可注入）**  
2. **三类具体分配函数**（同步 GPU 即可）+ 单元测试（分配/释放后统计为 0）  
3. **`Buffer` 类** + `unique_ptr`/`shared_ptr` 工厂  
4. **`Tensor`**（形状 + `Buffer`）  
5. **`BufferManager` 门面** + `copy` 辅助  
6. （按需）`cudaMallocAsync`、device memory pool、多 GPU 分桶统计  

---

## 11. 与仓库其他文档的关系

- 顶层推理库结构：[`docs/edgeLLM.md`](../edgeLLM.md)  
- TRT-Edge-LLM / TRT-LLM 对照：[`docs/arch/ref.md`](../arch/ref.md)  

---

## 12. 小结

- **TRT-LLM 最值得复用的结构**是：**内存类型枚举 + 缓冲接口 + 张量扩展 + 集中 BufferManager + 全局/可注入统计**。  
- llmOnEdge 应用 **同一分层**，但 **元素类型与 TRT 解耦**，便于无 TensorRT 环境下单测与裁剪。  
- 实现时保证 **所有分配走统一入口、析构统一减计数**，统计即可长期可信。

---

## 13. 当前实现（最小版）

代码位置：`src/edgeLLM/`。

| 组件 | 路径 |
|------|------|
| `MemoryType` / `memory_type_name` | `include/llm_on_edge/memory/memory_types.h`，`src/memory_types.cpp` |
| `MemoryStats` | `include/llm_on_edge/memory/memory_stats.h`，`src/memory_stats.cpp` |
| `ElementType` / `element_size` | `include/llm_on_edge/memory/element_type.h`，`src/element_type.cpp` |
| `Buffer` | `include/llm_on_edge/memory/buffer.h`，`src/buffer.cpp` |
| `BufferManager` | `include/llm_on_edge/memory/buffer_manager.h`，`src/buffer_manager.cpp` |
| `Tensor` | `include/llm_on_edge/memory/tensor.h`，`src/tensor.cpp` |
| CUDA 错误宏 | `include/llm_on_edge/memory/cuda_check.h` |

CMake 目标：**`llmOnEdgeMemory`**（静态库）。测试：`edge_memory_unit_tests`（GoogleTest）、**`edge_memory_e2e`**（端到端 H2D/D2H + `Tensor`）。

构建示例（不编 TensorRT-Edge-LLM 子模块时）：

```bash
cmake -S . -B build -DLLMONEDGE_ENABLE_TENSORRT=OFF
cmake --build build --target edge_memory_unit_tests edge_memory_e2e
ctest -R edge_memory --output-on-failure
```

完整工程仍需要 `-DTRT_PACKAGE_DIR=...` 并开启默认的 `LLMONEDGE_ENABLE_TENSORRT=ON`。
