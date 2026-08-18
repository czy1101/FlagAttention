# Hy3 HPC-Ops FP8 Decode Attention with Triton + TLE

本项目提供面向 NVIDIA Hopper GPU 的 MTP=1 Hy3 HPC-Ops FP8 decode attention 实现，
计算内核基于 Triton Language Extensions（TLE），并提供 GPU 任务调度、测试和
性能基准。

## 支持范围

- NVIDIA Hopper 架构（SM90，已在 NVIDIA H20 上验证）
- `num_seq_q = 1`
- `head_dim = 128`
- `heads_per_group <= 8`
- FP8 E4M3 Q/K/V 与 KV per-tensor scaling
- Paged KV cache，并保留 NHD/HND 的实际物理 stride
- Base-2 online softmax
- GPU-resident cluster task scheduling
- Cluster size 2、cluster size 4 和 cluster size 8 的 DSM reduction
- 通过 TLE raw CUDA dialect 完成跨 cluster Split-K finalization

## 目录

```text
src/flag_attn/hpcops_decode_attention/
  __init__.py                  公开 Python API
  compute_kernel.py           TLE decode compute kernel
  task_scheduler_kernel.py    GPU cluster task-map assignment kernels
  dynamic_splitk_finalize.cu  跨 cluster Split-K finalizer
  runtime.py                  配置选择、workspace、descriptor 和公开运行入口
benchmark/
  hpcops_decode_benchmark.py  CUDA/TLE compute 和可选 assign + compute benchmark
tests/flag_attn/
  test_hpcops_decode.py       数值正确性和 GPU task-map 协议测试
```

## 调用结构

```text
调用方
  -> flag_attn.hpcops_decode_attention
     -> hpcops_decode_attention/runtime.py
        -> select_decode_config
        -> prepare_decode_workspace
           -> task_scheduler_kernel.allocate_cluster_task_map
              -> assign_cluster_task_prefix_kernel
              -> assign_cluster_task_records_compact_kernel
        -> make_paged_kv_descriptors
        -> compute_kernel.fp8_kvpertensor_decode_kernel
           -> fused LDSM/PRMT/register-source WGMMA
           -> cluster size 2/4/8 DSM reduction
           -> dynamic_splitk_finalize.cu
```

## 运行时配置选择

运行时根据 KV 序列长度分布选择实测性能最优的 cluster size 和每个任务处理的
token 数量：

| 输入分布 | 配置 |
|---|---|
| `max_seq_kv <= 512` | `cluster2token512` |
| 所有序列长度相同且 `max_seq_kv <= 4096` | `cluster4token1024` |
| `max_seq_kv >= 128K` | `cluster8token1024` |
| 至少两个长度不小于 32K 的序列 | `cluster4token1024` |
| 存在长度不小于 64K 的序列且 `num_batch <= 16` | `cluster8token512` |
| 存在长度不小于 64K 的序列且 `num_batch > 16` | `cluster4token1024` |
| 其他中短序列或非均匀分布 | `cluster8token512` |

## 测试覆盖

主要数值正确性测试覆盖以下笛卡尔积，共 24 个配置：

```text
num_batch                  = 1, 16, 200
max_seq_kv                 = 1024, 4096
(num_head_kv, num_head_q)  = (1, 8), (4, 32)
layout                     = NHD, HND
```

此外还验证：

- GPU task map 与测试专用 CPU reference implementation 逐元素一致；
- task scheduler 在 `cluster_size ∈ {2, 4, 8}`、`chunk_tokens ∈ {512, 1024}`
  下的 task-map 协议；
- completion counters 在每次 decode 后归零；
- 输出不存在 NaN/Inf。

## Benchmark

Benchmark 默认与 HPC-Ops 官方 benchmark 的计时范围保持一致，只计时预构建
task map 后的 compute kernel：

- 官方 CUDA decode compute；
- TLE decode compute。

延迟结果统一以微秒（`us`）输出。
默认使用与原始 cluster/token sweep 相同的 CUDA Graph replay + CUDA event
计时，并使用 `min_process_len=512` 构造官方 CUDA task map。可通过
`--no-graph` 切换为 eager event timing，通过 `--min-process-len` 覆盖 CUDA
task-map 阈值。

只有显式传入官方同名参数 `--include-taskmap`（兼容别名
`--include-assign`）时，才对两边都计时各自的 GPU assign kernel + compute
kernel。

## 依赖

- Python 3.12
- PyTorch（CUDA + FP8）
- Triton + `triton.experimental.tle`
- 支持 `tle.raw.cuda` 的 FlagTree/TLE 环境
- Clang（用于编译 `dynamic_splitk_finalize.cu`）
- CUDA Toolkit 及 CUDA headers
- 已安装并包含编译扩展的 `hpc` Python package（仅官方 CUDA benchmark
  baseline 需要）

## 环境配置

以下命令均从仓库根目录执行。将占位符替换为本机工具链的实际安装位置：

```bash
export PYTHONPATH="${PWD}/src"
export CUDA_HOME="<CUDA_TOOLKIT_ROOT>"
export CLANG="<CLANG_EXECUTABLE>"
export CLANG_FLAGS="-I<CUDA_INCLUDE_DIRECTORY>"
```

FlagAttention 使用 `src` package layout，因此源码方式运行时应将 `${PWD}/src`
加入模块搜索路径。也可以先执行 `python3 -m pip install -e .`，然后省略
`PYTHONPATH` 设置。不要加入未编译的 HPC-Ops 源码目录，以免遮蔽当前 Python
环境中已安装的 `hpc` package。项目代码不依赖仓库所在目录的绝对路径。

`compute_kernel.py` 通过 `@dialect(name="cuda", ...)` 编译
`dynamic_splitk_finalize.cu`，因此 `CLANG` 必须指向可执行文件，
`CLANG_FLAGS` 必须包含 CUDA headers。CUDA headers 位于
`${CUDA_HOME}/include` 时，例如：

```bash
export CLANG_FLAGS="-I${CUDA_HOME}/include"
```

为避免调试选项影响性能数据，运行 benchmark 前建议清理相关环境变量，并使用
独立的 Triton cache：

```bash
unset CUDA_LAUNCH_BLOCKING \
      TRITON_ALWAYS_COMPILE \
      TRITON_REPRODUCER_PATH \
      MLIR_ENABLE_DUMP \
      TRITON_KERNEL_DUMP \
      TRITON_DUMP_DIR

export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/fp8-decode-tle-cache"
mkdir -p "${TRITON_CACHE_DIR}"
```

## 运行测试

测试需要 NVIDIA Hopper/SM90 GPU：

```bash
python3 -m pytest tests/flag_attn/test_hpcops_decode.py -v
```

## 运行 Benchmark

Benchmark 的官方 CUDA baseline 需要包含 `_C*.so` 编译扩展的 `hpc`
package。运行前可验证实际导入位置：

```bash
python3 -c "import hpc; print(hpc.__file__); print(hpc.__version__)"
```

如果导入失败或提示 `Expected one _C*.so file, found 0`，应按照 HPC-Ops
官方安装说明构建并安装 wheel，然后重新执行上述验证。不要通过
`PYTHONPATH` 直接导入未编译的 HPC-Ops 源码目录。

默认运行 NHD/HND、全部九个分布，并且只计时预构建 task map 后的 compute
kernel：

```bash
python3 benchmark/hpcops_decode_benchmark.py \
  --check \
  --warmup 100 \
  --iters 300 \
  --repeat 5
```

显式加入官方 `--include-taskmap` 参数后，计时范围变为 assign kernel +
compute kernel：

```bash
python3 benchmark/hpcops_decode_benchmark.py \
  --include-taskmap \
  --check \
  --warmup 100 \
  --iters 300 \
  --repeat 5
```

可通过 `--layout` 和 `--cases` 选择布局或输入分布。使用 `--help` 查看全部参数：

```bash
python3 benchmark/hpcops_decode_benchmark.py --help
```
