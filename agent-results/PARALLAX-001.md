# PARALLAX-001：Enflame 反向冷编译定位

环境：`zwj-enflame-vllm`，主机 GCU 5（容器 `gcu:1`），基线提交 `3912bd75af73ce258267ecac3c508d918f63ee1f`。目标形状 `B=2,T=256,H=2,HQ=8,D=64`，BF16。使用 `experiments/parallax_target.py` 对公共 API 做一次前向、反向和 FP32 PyTorch 参考比较，每次进程限时 180 秒。首次编译耗时与稳态延迟分开。

## 结果

| 尝试 | 前向冷编译 | 反向 | 结论 |
|---|---:|---|---|
| 原始基线（先前运行） | 前向可完成 | 180 秒内未完成 | 没有目标形状反向正确性/性能数据 |
| 仅固定 DKV 64×64 tile | 65.3 秒 | 180 秒内未完成 | 固定 DKV 不足以解决超时 |
| 临时阶段标记 | 3.74 秒（缓存后） | 输出 `PARALLAX_DQR_START`，120 秒内无 `PARALLAX_DQR_DONE` | 阻塞在融合 DQR 的首次编译或启动 |
| 分离 DQR + 固定 DQR/DKV tile | 3.72 秒（缓存后） | 180 秒内未完成 | 该替代路径也未通过限时 |

所有尝试均在独立分支进行；**未获得通过数值参考的目标形状反向结果，也未保留任何 kernel 改动**。分离 DQR 尝试的失败说明仅缩减 autotune 配置并不够；需要继续检查 Triton/FlagTree 编译阶段，或针对该形状设计更简单的反向 kernel。不能把“运行时间上限内未完成”当作数值失败，也不能宣称性能改善。

## 复现与下一步

`experiments/parallax_target.py` 使用 CPU 生成后复制到 `gcu:1` 的相同输入，并以仓库测试里的 `_reference_parallax` 为独立数值参考。运行：

```bash
cd /workspace/codex-workspaces/parallax-backward
PYTHONPATH="$PWD/src:$PYTHONPATH" timeout 180 python experiments/parallax_target.py --dtype bf16
```

在下一轮中先把 DQR 编译与执行分开记录，抓取编译器阶段及耗时；然后只对该 kernel 做一次最小变更，并在 180 秒内得到目标形状的 `output/q/r/k/v` 误差。通过后再测稳态延迟。小形状与 decode 的既有基线通过，不能外推到目标形状反向。
