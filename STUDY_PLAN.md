# OpenXLA Study Plan: JAX → XLA → TPU Backend

**Goal:** Become an expert in the OpenXLA ML compiler stack, focusing exclusively on the JAX frontend, XLA compiler core, and TPU backend.

**Duration:** 8 weeks (2 months)
**Time commitment:** ~2 hours/day on weekdays, ~4 hours on weekends (~18 hrs/week)

---

## Prerequisites

Before starting, ensure you have:
- Strong C++ reading ability (the compiler is C++17)
- Familiarity with Python (JAX is Python-first)
- Basic compiler concepts (IR, passes, lowering, code generation)
- Basic understanding of ML workloads (matmul, convolution, attention, training loops)
- Access to a TPU VM on Google Cloud (for hands-on exercises in later weeks)

---

## Week 1: JAX Fundamentals and the Compilation Boundary

**Objective:** Understand what JAX produces and where XLA takes over.

### Topics
- JAX's tracing model: `jax.jit`, `jax.grad`, `jax.vmap` and how they trace Python into a functional IR
- `jax.make_jaxpr` — inspect the Jaxpr (JAX's internal trace representation)
- How Jaxpr lowers to StableHLO MLIR via `jax.export`
- The `jax.lax` primitive layer and its 1:1 mapping to HLO ops

### Reading
- [JAX internals docs](https://jax.readthedocs.io/en/latest/jaxpr.html) — Jaxpr reference
- [JAX key concepts](https://jax.readthedocs.io/en/latest/key-concepts.html)
- [How JAX primitives work](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_Primitives_Work.html)

### Source Code
- `xla/python/xla_client.py` — the Python-facing XLA client JAX calls into
- `xla/python/xla_builder.cc` — nanobind glue exposing `XlaBuilder` to Python
- `xla/python/ops.cc` — Python bindings for individual HLO op construction

### Exercises
1. Write a simple matmul in JAX, use `jax.make_jaxpr` to see the Jaxpr
2. Use `jax.export` to emit a StableHLO module, inspect the MLIR text
3. Use `jax.jit(f).lower(x).as_text()` to see the HLO text representation
4. Compare the Jaxpr, StableHLO, and HLO for the same function

---

## Week 2: HLO — The Core Intermediate Representation

**Objective:** Deeply understand HLO IR: its structure, opcodes, and semantics.

### Topics
- HLO module / computation / instruction hierarchy
- The ~100 HLO opcodes (`xla/hlo/ir/hlo_opcode.h`) — focus on: `dot`, `convolution`, `reduce`, `broadcast`, `reshape`, `transpose`, `slice`, `dynamic-slice`, `while`, `conditional`, `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`
- Shapes, layouts, and element types (`xla/xla_data.proto`)
- HLO text format and the parser
- Sharding annotations (`HloSharding`) and how they attach to instructions

### Source Code (read in this order)
1. `xla/hlo/ir/hlo_opcode.h` — enum of all opcodes
2. `xla/hlo/ir/hlo_instruction.h` — the core `HloInstruction` class (focus on public API, skip internal details)
3. `xla/hlo/ir/hlo_instructions.h` — derived classes: `HloDotInstruction`, `HloConvolutionInstruction`, `HloCollectiveInstruction`, etc.
4. `xla/hlo/ir/hlo_computation.h` — computation (a graph of instructions with a root)
5. `xla/hlo/ir/hlo_module.h` — module (entry computation + called computations)
6. `xla/hlo/ir/hlo_sharding.h` — sharding metadata
7. `xla/hlo/parser/hlo_parser.h` — text format parser
8. `xla/xla_data.proto` — `PrimitiveType`, `ShapeProto`, `LayoutProto`, `OpSharding`
9. `xla/service/hlo.proto` — `HloModuleProto`, `HloInstructionProto`

### Exercises
1. Read 5 different `.hlo` test files from `xla/tests/` or `xla/service/` to get comfortable with the text format
2. Write a simple HLO module by hand (matmul + add + relu) and verify it parses with the HLO parser tests
3. For a JAX model (e.g., a 2-layer MLP), dump the HLO and trace every instruction back to the Python source

---

## Week 3: The XLA Compiler Pipeline — Pass Infrastructure and Key Optimizations

**Objective:** Understand how HLO is transformed from input to optimized form.

### Topics
- Pass infrastructure: `HloPassInterface`, `HloPassPipeline`, `HloPassFix`
- Platform-independent optimization passes (run for all backends)
- How passes compose into a pipeline (order matters)

### Source Code — Pass Infrastructure
1. `xla/hlo/pass/hlo_pass_interface.h` — base class for all passes
2. `xla/hlo/pass/hlo_pass_pipeline.h` — ordered pipeline of passes
3. `xla/hlo/pass/hlo_pass_fix.h` — fixed-point iteration wrapper

### Source Code — Key Optimization Passes (read roughly in pipeline order)
| Pass | File | What it does |
|------|------|-------------|
| HLO Verifier | `xla/service/hlo_verifier.h` | Shape/type invariant checker, runs before and after passes |
| Algebraic Simplifier | `xla/hlo/transforms/simplifiers/algebraic_simplifier.h` | `a + 0 → a`, reshape folding, broadcast canonicalization |
| CSE | `xla/service/hlo_cse.h` | Common subexpression elimination |
| DCE | `xla/hlo/transforms/simplifiers/hlo_dce.h` | Dead code elimination |
| Instruction Fusion | `xla/service/instruction_fusion.h` | Merge element-wise ops into fusion clusters |
| Dot Decomposer | `xla/service/dot_decomposer.h` | Decomposes complex dots into simpler forms |
| Batch Normalizer Expander | `xla/service/batchnorm_expander.h` | Expands batch norm into primitive ops |
| AllReduce Combiner | `xla/hlo/transforms/collectives/all_reduce_combiner.h` | Fuse adjacent collectives |
| Async Collective Creator | `xla/hlo/transforms/collectives/async_collective_creator.h` | Convert sync collectives to async |

### Exercises
1. Pick `algebraic_simplifier.cc` — read 10 rewrite rules end-to-end, understand the pattern match + transform
2. Use `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump` when running a JAX program. Inspect the dumped HLO at each pass stage
3. Count how many passes run for a simple matmul JAX program by examining the dump directory

---

## Week 4: Layout Assignment, Fusion, and Memory Optimization

**Objective:** Understand the backend-sensitive passes that determine performance.

### Topics
- Layout assignment: why physical data layout matters for hardware (row-major, column-major, tiled layouts)
- Fusion: element-wise fusion, reducing memory traffic
- Copy insertion: resolving layout conflicts and aliasing
- Buffer assignment: mapping logical buffers to physical memory
- Rematerialization: recomputation to save memory
- Memory space assignment: HBM vs VMEM on TPU

### Source Code
1. `xla/service/layout_assignment.h` / `.cc` — assigns physical layouts; backends override `AddBackendConstraints`
2. `xla/service/instruction_fusion.h` / `.cc` — base fusion logic (subclassed by backends)
3. `xla/service/copy_insertion.h` — inserts copies after layout assignment
4. `xla/service/buffer_assignment.h` / `.cc` — heap-based buffer allocation
5. `xla/hlo/transforms/simplifiers/hlo_rematerialization.h` / `.cc` — recomputation trade-off
6. `xla/service/memory_space_assignment/memory_space_assignment.h` / `.cc` — multi-tier memory optimization (critical for TPU VMEM)

### Analysis Passes (supporting the above)
7. `xla/hlo/analysis/hlo_dataflow_analysis.h` — def-use chains
8. `xla/hlo/analysis/hlo_alias_analysis.h` — buffer aliasing
9. `xla/hlo/analysis/hlo_ordering.h` — instruction ordering for liveness

### Exercises
1. For a conv-bn-relu JAX model, dump HLO before and after fusion — see which ops get fused
2. Compare the buffer assignment dump for a model with and without rematerialization enabled
3. Read the memory space assignment code and understand the prefetch/evict copy insertion for a 2-tier memory system (HBM + VMEM)

---

## Week 5: SPMD, Sharding, and Distributed Compilation

**Objective:** Master how XLA partitions programs across multiple TPU chips.

### Topics
- SPMD (Single Program Multiple Data) model in XLA
- Sharding annotations: replicated, tiled, partially replicated, manual
- The SPMD partitioner: how it lowers a sharded HLO module into per-device programs
- Collectives inserted by the partitioner: AllGather, ReduceScatter, AllToAll, CollectivePermute
- The latency-hiding scheduler: overlapping compute and communication
- Shardy: the next-generation sharding propagation framework
- Cost analysis and the HLO cost model

### Source Code
1. `xla/hlo/ir/hlo_sharding.h` — sharding representation
2. `xla/service/spmd/spmd_partitioner.h` / `.cc` — the main SPMD partitioner pass
3. `xla/service/spmd/shardy/shardy_xla_pass.h` — Shardy integration
4. `xla/service/latency_hiding_scheduler.h` / `.cc` — scheduler that overlaps compute/comms
5. `xla/service/hlo_cost_analysis.h` / `.cc` — FLOP and bandwidth cost model

### JAX-side context
- `jax.sharding` API: `NamedSharding`, `PositionalSharding`, `SingleDeviceSharding`
- `jax.jit` with `in_shardings` / `out_shardings`
- `jax.experimental.mesh_utils` and `jax.experimental.multihost_utils`

### Exercises
1. Write a JAX program with `jax.sharding.NamedSharding` on a 2×2 TPU mesh, dump the HLO before and after SPMD partitioning
2. Identify the collectives inserted by the partitioner for a sharded matmul
3. Experiment with different sharding strategies and observe the impact on inserted collectives
4. Use `jax.jit(f).lower(x).compile().cost_analysis()` to get FLOP/bandwidth estimates

---

## Week 6: The TPU Backend — Architecture and Execution

**Objective:** Understand the TPU-specific parts of the XLA stack.

### Topics
- TPU hardware architecture: MXU (matrix multiply unit), VPU (vector processing unit), HBM, VMEM, ICI (inter-chip interconnect)
- The TPU plugin model: how the proprietary compiler is loaded via the PJRT C API
- TPU-specific compilation: what happens behind the C API boundary
- TPU executor: streams, memory management, transfers
- TPU computation placement: replica × partition → chip mapping
- TPU topology and mesh shapes

### Source Code
| File | Purpose |
|------|---------|
| `xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h` | `GetTpuClient()` — entry point for TPU PJRT client |
| `xla/pjrt/plugin/xla_tpu/tpu_dynamic_registration.cc` | dlopen-based plugin loading |
| `xla/stream_executor/tpu/tpu_on_demand_compiler.cc` | `Compiler` impl that delegates to the C API |
| `xla/stream_executor/tpu/tpu_executor_c_api.h` | Stable C ABI for TPU operations |
| `xla/stream_executor/tpu/tpu_platform.h` | TPU `Platform` in StreamExecutor |
| `xla/stream_executor/tpu/tpu_executable.h` | Wraps compiled TPU binary |
| `xla/stream_executor/tpu/tpu_transfer_manager.h` | Host ↔ TPU HBM transfers |
| `xla/service/tpu_computation_placer.h` | Device assignment for TPU |
| `xla/pjrt/c/pjrt_c_api_tpu_internal.h` | TPU-specific PJRT C API extensions |

### External Resources
- [Google Cloud TPU Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) — TPU v4/v5e hardware docs
- [TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide) — understanding MXU utilization, padding, tiling

### Exercises
1. On a TPU VM, run a JAX program and trace the call stack from `jax.jit(f)(x)` down to `TpuExecutable::ExecuteAsyncOnStream`
2. Inspect TPU topology with `jax.devices()` and `jax.local_devices()`
3. Use `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump` on a TPU and study the TPU-specific HLO passes
4. Profile a matmul on TPU using `jax.profiler` and TensorBoard — identify MXU utilization

---

## Week 7: PJRT, IFRT, and the Runtime Stack

**Objective:** Understand the runtime that connects compiled executables to hardware.

### Topics
- PJRT architecture: Client → Device → Buffer → Executable
- The PJRT C API plugin system and how TPU implements it
- IFRT: the sharding-aware runtime layer above PJRT
- IFRT arrays: multi-device arrays with sharding metadata
- IFRT proxy: remote execution over gRPC
- StableHLO → MHLO → HLO translation pipeline
- Async execution and futures in PJRT/IFRT

### Source Code — PJRT
1. `xla/pjrt/pjrt_client.h` — `PjRtClient` interface (Compile, BufferFromHostBuffer, devices)
2. `xla/pjrt/pjrt_executable.h` — `PjRtLoadedExecutable` (Execute)
3. `xla/pjrt/pjrt_future.h` — async result type
4. `xla/pjrt/pjrt_compiler.h` — AOT compilation interface
5. `xla/pjrt/pjrt_stream_executor_client.h` — concrete SE-based PJRT client
6. `xla/pjrt/c/pjrt_c_api.h` — the stable C ABI (function pointer table)
7. `xla/pjrt/c_api_client/pjrt_c_api_client.h` — C++ wrapper around C API plugins
8. `xla/pjrt/distributed/client.h` / `service.h` — multi-host coordination

### Source Code — IFRT
9. `xla/python/ifrt/client.h` — `ifrt::Client`
10. `xla/python/ifrt/array.h` — `ifrt::Array` (sharded, multi-device)
11. `xla/python/ifrt/executable.h` — `ifrt::LoadedExecutable`
12. `xla/python/ifrt/sharding.h` — `ifrt::Sharding`
13. `xla/python/pjrt_ifrt/pjrt_client.h` — IFRT→PJRT bridge (what JAX uses)

### Source Code — MLIR Translation
14. `xla/pjrt/mlir_to_hlo.h` — entry point: MLIR string → XlaComputation
15. `xla/hlo/translate/stablehlo_to_hlo/translate.h` — StableHLO → HLO
16. `xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h` — MHLO → HLO

### Exercises
1. Write a Python script that directly uses `xla_client.py` (bypassing JAX) to compile and run an HLO module on TPU
2. Trace the full call stack: `jax.jit(f)(x)` → IFRT → PJRT → StreamExecutor → TPU C API
3. Study how `jax.export` serializes a StableHLO module and how it gets deserialized back through the MLIR translation pipeline

---

## Week 8: Advanced Topics, Performance Tuning, and Putting It All Together

**Objective:** Integrate everything. Work on real-world performance problems.

### Topics
- Autotuning: how XLA selects algorithm variants (relevant for both GPU and TPU)
- XLA debug options and flags that affect compilation
- Profiling TPU workloads: TensorBoard profiler, trace viewer, MXU utilization analysis
- Custom calls and FFI: extending XLA with custom TPU kernels
- Multi-slice TPU: compilation for TPU pods (v4-128, v5e-256, etc.)
- Megascale: multi-pod training infrastructure
- XLA:TPU performance patterns: padding to MXU-friendly shapes, bf16 usage, collective optimization

### Source Code
1. `xla/autotuning.proto` / `xla/autotune_results.proto` — autotuning configuration
2. `xla/debug_options_flags.h` / `.cc` — all XLA debug flags
3. `xla/ffi/` — Foreign Function Interface for custom operations
4. `xla/megascale/` — multi-pod coordination
5. `xla/service/compiler.h` — revisit the full `RunHloPasses` + `RunBackend` pipeline with fresh eyes

### Capstone Project
Build an end-to-end understanding by doing ONE of the following:

**Option A: Compiler Pass Analysis**
- Pick a real model (e.g., a Transformer layer) in JAX
- Dump HLO at every pass stage on TPU
- Write a detailed analysis of: which passes fire, what each changes, how the final HLO maps to TPU hardware
- Measure compilation time breakdown across passes

**Option B: SPMD Performance Study**
- Take a medium-sized model (e.g., GPT-2 scale)
- Shard it across a TPU v4-32 (or similar) with different strategies
- Analyze the partitioned HLO: which collectives are inserted, what's the communication volume
- Profile and compare throughput across sharding strategies
- Write up findings with HLO snippets and profiler screenshots

**Option C: Custom Compiler Pass**
- Write a new HLO optimization pass (e.g., a custom pattern rewriter)
- Register it in the pass pipeline
- Write tests using the HLO text format
- Benchmark the impact on a real workload

---

## Appendix A: Key XLA Flags for Debugging and Learning

```bash
# Dump HLO at every compilation stage
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"

# Dump HLO before and after specific passes
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re='.*'"

# Verbose logging for specific components
TF_CPP_VMODULE="hlo_pass_pipeline=3,algebraic_simplifier=2"

# Disable specific optimizations (to see their effect)
XLA_FLAGS="--xla_disable_hlo_passes=algsimp,dce"
```

## Appendix B: Essential JAX Debugging Commands

```python
import jax

# See the Jaxpr (JAX's internal trace)
jax.make_jaxpr(f)(x)

# See the HLO text
jax.jit(f).lower(x).as_text()

# See the optimized HLO (after compiler passes)
jax.jit(f).lower(x).compile().as_text()

# Get cost analysis
jax.jit(f).lower(x).compile().cost_analysis()

# Get memory analysis
jax.jit(f).lower(x).compile().memory_analysis()

# Export as StableHLO
jax.export.export(f)(x)

# Profile
jax.profiler.start_trace("/tmp/jax_trace")
result = f(x)
jax.profiler.stop_trace()
```

## Appendix C: Full Compilation Pipeline (Data Flow)

```
JAX Python (jax.jit / jax.export)
  │
  ▼  emit StableHLO MLIR  or  XlaBuilder ops
xla/python/xla_builder.cc  +  xla/python/ops.cc
  │
  ▼
xla/hlo/builder/xla_builder.h  ──→  XlaComputation (HloModuleProto)
  │  OR
xla/pjrt/mlir_to_hlo.h  ─────────→  StableHLO → MHLO → HLO translation
  │
  ▼
ifrt::Client::Compile()  →  PjRtClient::Compile()
  │
  ▼
service::Compiler::RunHloPasses()   ← platform-independent optimizations
  │  algebraic_simplifier, cse, dce, fusion,
  │  spmd_partitioner, layout_assignment,
  │  copy_insertion, memory_space_assignment,
  │  latency_hiding_scheduler, rematerialization ...
  ▼
service::Compiler::RunBackend()     ← TPU-specific compilation
  │
  └──[TPU]→  tpu_on_demand_compiler.cc
               └─ TpuCompiler C API → proprietary libtftpu.so
  │
  ▼
PjRtLoadedExecutable  →  TpuExecutable
  │
  ▼  Execute()
ifrt::Array results  →  PjRtBuffer on TPU HBM
  │
  ▼  transfer to host
JAX NumPy arrays
```

## Appendix D: Recommended External Resources

| Resource | URL | When to Read |
|----------|-----|-------------|
| XLA Architecture Overview | https://openxla.org/xla/architecture | Week 1 |
| HLO Operation Semantics | https://openxla.org/xla/operation_semantics | Week 2 |
| StableHLO Spec | https://github.com/openxla/stablehlo/blob/main/docs/spec.md | Week 2 |
| JAX Sharding Guide | https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html | Week 5 |
| GSPMD Paper (arXiv:2105.04663) | https://arxiv.org/abs/2105.04663 | Week 5 |
| TPU System Architecture | https://cloud.google.com/tpu/docs/system-architecture-tpu-vm | Week 6 |
| PJRT Design Doc | https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md | Week 7 |
| XLA Autotuning | https://openxla.org/xla/persisted_autotuning | Week 8 |

## Appendix E: Weekly Checkpoint Questions

Use these to gauge your understanding at the end of each week.

**Week 1:** Can you explain what happens between `jax.jit(f)(x)` and XLA receiving an HloModuleProto?

**Week 2:** Given an HLO text dump, can you identify the computation structure, data types, layouts, and which ops dominate?

**Week 3:** Can you name 10 optimization passes, explain what each does, and why their ordering matters?

**Week 4:** Can you explain why layout assignment is backend-specific and how memory space assignment affects TPU performance?

**Week 5:** Given a sharded JAX program, can you predict which collectives the SPMD partitioner will insert?

**Week 6:** Can you draw the full TPU software stack from PJRT C API down to hardware, and explain what is open source vs proprietary?

**Week 7:** Can you trace a buffer from Python creation through IFRT → PJRT → StreamExecutor → TPU HBM?

**Week 8:** Given a slow TPU workload, can you use XLA dumps and profiling to diagnose the bottleneck?
