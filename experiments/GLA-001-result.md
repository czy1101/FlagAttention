# GLA-001：Enflame FP16 状态计算优化

基线提交 `3912bd75af73ce258267ecac3c508d918f63ee1f`。测试容器 `zwj-enflame-vllm`，主机 GCU 5（容器 `gcu:1`），Python 3.12.13、torch 2.10.0+cpu、torch_gcu 2.10.0+3.7.20260408、Triton 3.6.0。源码目录为本任务独立克隆；`PYTHONPATH` 包含该克隆的 `src` 和 `/opt/flagtree`。

## 瓶颈与改动

`B=1,T=512,H=4,K=V=64` 的 FP16 前向约 6.92 ms。分阶段计时显示 `chunk_fwd_h` 状态计算约 6.62 ms，而 A 矩阵阶段约 0.22 ms；原先怀疑的 FP16 A block 大小并非主要瓶颈。BF16 同形状的状态计算约 0.10 ms。源码中 BF16 使用并行的 local state + scan 路径，FP16 则进入顺序 fallback。

仅对 `K=V=64` 的 FP16 启用已有的并行状态路径。其余 FP16 维度、BF16 和变长路径保持原分支选择。没有改动算子公式。

## 稳态性能

测试脚本 `experiments/gla_ab.py` 与 `experiments/gla_ab_bwd.py` 在同一进程、相同输入上交替测 fallback 和候选路径。每个结果为 `do_bench` 的中位数；每个路径重复三轮，以下取三轮中位数。预热 10 ms，前向采样预算 50 ms，前向加反向采样预算 200 ms。数字包含 Python 调用与分配；不含首次编译。测量期间另一张 GCU 有 GDN2 任务运行，因此后续可再独占复测。

| 形状与阶段 | fallback | 候选 | 加速比 |
|---|---:|---:|---:|
| B1,T512,H4,D64，FP16 前向 | 6.915 ms | 0.404 ms | 17.1× |
| B1,T512,H4,D64，FP16 前向+反向 | 28.935 ms | 16.211 ms | 1.78× |
| B1,T64,H1,D64，FP16 前向 | 0.444 ms | 0.307 ms | 1.45× |

原始三轮数据在 `experiments/ab.log`、`experiments/ab-bwd.log`、`experiments/ab-small.log`。分阶段结果在 `experiments/fp16.log`、`experiments/fp16-candidate.log` 和 `experiments/bf16.log`。

## 正确性

- 仓库 `tests/flag_attn/test_chunk_gla.py` 中 `B1-T63-H1-D64` FP16 用例：**1 passed**，覆盖前向、反向、初始状态和最终状态。
- `experiments/gla_compare.py --T 512 --grad --signed-gate --gate-normalizer 1`：同输入对照原 fallback，输出及 `q/k/v` 梯度的相对 L2 为 0；`g` 梯度相对 L2 为 `6.94e-8`，最大绝对差 `1.91e-6`，两边均无 NaN。
- 极端负门控 `--gate-normalizer 0.1`：原 fallback 与候选路径都在输出中出现 384 个 NaN。这是**已有的数值问题**，不能把该形状算作正确性通过；候选没有增加 NaN 数。此问题应另列修复任务。

## 限制

只验证了 FP16 64×64 状态维度及上述形状。更大 K/V、变长和其他使用 `chunk_fwd_h` 的算子未验证，因此改动限定在 64×64。性能数值来自这台设备和这些输入，不外推到其他形状。没有声称修复极端负门控的 NaN。
