# PARALLAX-002: BF16 D64/T256/G4 dense backward path

## Change

The existing Enflame pure-Triton dense-aligned DQR/DKV backward kernels were previously selected only for K=128 and more than 16 query tiles. For the validated BF16 target shape B=2, T=256, H=2, HQ=8, D=64 (G=4), route to those fixed kernels too. The new condition is restricted to BF16, D64, T256, G4, dense causal input. Arithmetic in the kernels is unchanged. The original K128 selection remains unchanged.

## Evidence

Environment: `zwj-enflame-vllm` on container GCU 1 (host GCU 5), FlagAttention baseline `3912bd75af73ce258267ecac3c508d918f63ee1f` plus the branch change. `experiments/parallax_target.py` compares the public API and all gradients with `_reference_parallax` on the same inputs.

- Original BF16 target backward did not finish within a 180 s process limit (`PARALLAX-001.md`). Merely fixing DKV autotune and trying split DQR also timed out.
- New path completed first BF16 target backward in 73.75 s after the forward had been compiled. Output relative RMS error was 0.002999; q/r/k/v gradient errors were 0.003224/0.002776/0.002890/0.002614. All were finite. Raw output: `experiments/dense64.log`.
- Rerunning the exact final condition with compiled kernels finished backward in 8.49 s (includes process and dispatch overhead); errors matched the first run. Raw output: `experiments/dense64-final-bf16.log`.
- Repeated same-process BF16 calls in `experiments/parallax_steady.py`: after the first compilation/initialization run and one settling run, repeats 2–4 had forward wall times 0.504–0.525 ms and backward wall times 0.972–1.002 ms, with finite gradients. These are Python wall times with explicit GCU synchronization, not device-event kernel timings. Raw output: `experiments/steady.log`.
- Existing T63 dense FP16/BF16 forward+backward pytest cases: **2 passed**, 30 deselected. Raw output: `experiments/small-regression.log`.

## FP16 scope decision

A provisional version of the same path also produced numerically matching FP16 target gradients (relative RMS <= 0.000402), but its first 180 s attempt timed out and a second attempt still spent 203.97 s in backward after the forward cache was warm. Therefore the committed selector only changes BF16 for this shape. FP16 attempts are preserved in `experiments/dense64-fp16.log` and `experiments/dense64-fp16-rerun.log`.

## Limits

The old target BF16 backward has no completed device-event result, so a steady-state speedup ratio cannot be computed. The selected shape has a strict B/H-independent condition on T, D, and G; other larger shapes, varlen, windows, and FP16 still require separate compilation and numerical evidence. Cold BF16 compilation remains expensive, though it now finishes inside the 180 s target limit.
