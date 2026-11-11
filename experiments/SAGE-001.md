# SAGE-001: Enflame FP16 SageAttention versus SDPA

## Method

- Image: `harbor.baai.ac.cn/flagos-app/vllm0.24.0-enflame-topsl.9.10:2.3.0-0.3.0`; container GCU 1 (host GCU 5).
- Source: FlagAttention `3912bd75af73ce258267ecac3c508d918f63ee1f`.
- Input `[B,H,T,D]=[1,2,T,64]`, FP16 Q/K/V and FP16 output, T from 128 to 4096.
- For each shape, same inputs for Sage and PyTorch SDPA. Sage Q/K are quantized per block to INT8. Correctness is compared to SDPA using the repository's metric.
- Three repeats of device-event timing per shape; each repeats 10 ms warmup and 50 ms measurement. Values below are medians of the three repeats.
- `Sage pure` times the attention call on prequantized inputs. `Sage total` includes `per_block_int8` Q/K quantization. SDPA receives the original FP16 Q/K/V.

| T | SDPA (ms) | Sage pure (ms) | Sage total (ms) | cosine | relative L2 |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0401 | 0.0908 | 0.1602 | 0.999916 | 0.01298 |
| 512 | 0.0479 | 0.1330 | 0.2109 | 0.999921 | 0.01258 |
| 1024 | 0.0537 | 0.1945 | 0.3393 | 0.999923 | 0.01239 |
| 2048 | 0.1109 | 0.3197 | 0.5970 | 0.999920 | 0.01263 |
| 4096 | 0.2626 | 1.0214 | 1.6013 | 0.999914 | 0.01311 |

The repository accuracy thresholds (cosine >= 0.99; relative L2 <= 0.05) hold for all five shapes. Sage is slower than SDPA in both timing scopes at these shapes. For T4096, quantization adds roughly 0.58 ms to the full call, while the prequantized attention is still about 3.9 times slower than SDPA. Avoid claiming a speedup or changing the default attention path based on these data.

## Next optimization experiment

Profile the Enflame Sage kernel's launch geometry and Q/K quantization separately. Test any changes against all five shapes and both timing scopes, then retain only changes that preserve the stated accuracy and improve the full path. The harness and per-run raw output are in `experiments/`.
