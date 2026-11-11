"""Independent tiny PyTorch/FP64 reference and GCU comparison for GDN2 (GDN2-001).

Semantics assumed by this reference (token recurrence, per head, per token t):

    h_t = (I - k_t (b_t * k_t)^T) @ (exp(g_t) * h_{t-1}) + k_t (w_t * v_t)^T
    o_t = scale * (q_t @ h_t)

i.e. the per-channel decay is applied first, then the delta-rule erase/write
correction.  ``o`` uses the post-update state (h_t), and ``ht`` is h_T.
This assumption is fixed here so a mismatch is reported as a mismatch, not
silently absorbed.

Run:  PYTHONPATH="$PWD/src:$PYTHONPATH" python tests/flag_attn/gdn2_tiny_reference.py
Exit code 0 only when every case is within tolerance.
"""

import math
import sys

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# FP16/BF16 dot products accumulate in FP32 on the device, so a few 1e-2
# relative RMS is expected for a 64-wide reduction; 2e-2 is the gate.
RTOL = 2e-2
ATOL = 1e-3


def reference(q, k, v, g, b, w, scale, h0):
    """FP64 token recurrence; q/k/v/g/b/w [B,T,H,D], h0 [B,H,K,V]."""
    q, k, v, g, b, w = (x.double() for x in (q, k, v, g, b, w))
    h0 = h0.double()
    B, T, H, K = q.shape
    V = v.shape[-1]
    h = h0.clone()
    outs = []
    for t in range(T):
        kt, bt, gt, vt, wt = k[:, t], b[:, t], g[:, t], v[:, t], w[:, t]
        hd = h * torch.exp(gt)[..., :, None]
        # delta-rule erase: hd - k_t * ((b_t * k_t)^T @ hd), then write.
        erased = torch.einsum("bhk,bhkv->bhv", bt * kt, hd)
        h = hd - torch.einsum("bhk,bhv->bhkv", kt, erased)
        h = h + torch.einsum("bhk,bhv->bhkv", kt, wt * vt)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], h) * scale)
    o = torch.stack(outs, dim=1)
    return o, h


def metrics(expected, actual):
    e, a = expected.double().flatten(), actual.double().flatten()
    diff = (e - a).abs()
    rms = diff.square().mean().sqrt().item()
    base = e.square().mean().sqrt().item()
    return diff.max().item(), rms / (base + 1e-12), base, a.square().mean().sqrt().item()


def run_case(T, dtype_name, seed=0, use_initial_state=True):
    from flag_attn import chunk_gdn2

    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    B, H, K, V = 1, 1, 64, 64
    dtype = DTYPES[dtype_name]
    dev = torch.device("gcu:0")
    scale = K**-0.5
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    q, k, v, g, b, w, h0 = [x.to(dev) for x in (q, k, v, g, b, w, h0)]
    initial_state = h0 if use_initial_state else None
    o, ht = chunk_gdn2(q, k, v, g, b, w, scale=scale, initial_state=initial_state,
                       output_final_state=True, chunk_size=64)
    reference_h0 = h0.cpu() if use_initial_state else torch.zeros_like(h0.cpu())
    eo, eh = reference(q.cpu(), k.cpu(), v.cpu(), g.cpu(), b.cpu(), w.cpu(), scale, reference_h0)
    om, orr, obase, orms = metrics(eo, o.cpu())
    hm, hrr, hbase, hrms = metrics(eh, ht.cpu())
    ok = (om <= ATOL + RTOL * obase and hm <= ATOL + RTOL * hbase
          and orr <= RTOL and hrr <= RTOL)
    print(f"T={T:2d} dtype={dtype_name:4s} initial={use_initial_state} "
          f"shape=B{B},H{H},K{K},V{V} seed={seed}")
    print(f"  o : max_abs={om:.6e} rms_rel={orr:.6e} ref_rms={obase:.6e} act_rms={orms:.6e}")
    print(f"  ht: max_abs={hm:.6e} rms_rel={hrr:.6e} ref_rms={hbase:.6e} act_rms={hrms:.6e}")
    verdict = "PASS" if ok else "FAIL"
    print(f"  gate atol={ATOL:.1e} rtol={RTOL:.1e} -> {verdict}")
    return ok


if __name__ == "__main__":
    # T=1 and T=2 exercise the single-token path; T=65 forces two BT=64 chunks
    # and a 1-token tail, which is the smallest chunk-boundary case.
    lengths = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "65"])]
    results = []
    for T in lengths:
        for name in ("fp16", "bf16"):
            for use_initial_state in (True, False):
                results.append(run_case(T, name, use_initial_state=use_initial_state))
    print("SUMMARY:", "PASS" if all(results) else "FAIL", results)
    sys.exit(0 if all(results) else 1)
