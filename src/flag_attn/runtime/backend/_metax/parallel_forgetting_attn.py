import math

import torch
import triton
import triton.language as tl
from triton.language import core as tlcore
from collections import OrderedDict
from triton import knobs as _knobs
from triton.runtime import jit as _runtime_jit


@triton.jit
def _c550_starts(
    P, T, S, M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, BN: tl.constexpr, QM: tl.constexpr, NK: tl.constexpr
):
    h, b, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = group * 16 + tl.arange(0, 16)
    j = tl.arange(0, NK)
    pp = P + (b * H + h) * N
    a = tl.load(pp + N - M + r * QM, r < tl.cdiv(M, QM), other=0)
    end = tl.minimum(j * BN + BN - 1, N - 1)
    e = tl.load(pp + end, j < tl.cdiv(N, BN), other=0)
    threshold = T
    old = (j[None, :] < tl.cdiv(N, BN)) & ((a[:, None] - e[None, :]) < threshold)
    starts = tl.sum(old.to(tl.int32), 1) * BN
    tl.store(S + (b * H + h) * tl.cdiv(M, QM) + r, starts, r < tl.cdiv(M, QM))


@triton.jit
def _c550_attention(
    Q,
    K,
    V,
    P,
    S,
    O,
    sm_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    REAL_D: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QM: tl.constexpr,
):
    h, b, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = h // (H // HK)
    rm = block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    qptr = Q + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    q = tl.load(qptr, ((M % BM == 0) | (rm[:, None] < M)) & (rd[None, :] < REAL_D), other=0)
    pp = P + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, (M % BM == 0) | (rm < M), other=0)
    lo = tl.load(S + (b * H + h) * tl.cdiv(M, QM) + block * BM // QM)
    hi = tl.minimum(N, N - M + (block + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    diagonal = ((N - M + block * BM + 1) // BN) * BN


    for start in range(lo, hi, BN):
        kn = start + rn
        kp = K + (b * N * HK + kn[None, :] * HK + hk) * REAL_D + rd[:, None]
        vp = V + (b * N * HK + kn[:, None] * HK + hk) * REAL_D + rd[None, :]
        k = tl.load(kp, ((N % BN == 0) | (kn[None, :] < N)) & (rd[:, None] < REAL_D), other=0)
        v = tl.load(vp, ((N % BN == 0) | (kn[:, None] < N)) & (rd[None, :] < REAL_D), other=0)
        gk = tl.load(pp + kn, (N % BN == 0) | (kn < N), other=0)
        score = tl.dot(q, k, input_precision="ieee") * scale
        score += (gq[:, None] - gk[None, :]) * log2e
        if N % BN != 0:
            score = tl.where(kn[None, :] < N, score, -float("inf"))
        if M >= 1024 and N % BN == 0:

            score = tl.where((N - M + rm[:, None]) >= kn[None, :], score, -float("inf"))
        elif start >= diagonal:
            score = tl.where((N - M + rm[:, None]) >= kn[None, :], score, -float("inf"))
        mnew = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mnew)
        prob = tl.exp2(score - mnew[:, None])
        psum = tl.sum(prob, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(prob.to(Q.dtype.element_ty), v, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    out = (acc * (1.0 / li[:, None])).to(Q.dtype.element_ty)
    op = O + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    tl.store(op, out, ((M % BM == 0) | (rm[:, None] < M)) & (rd[None, :] < REAL_D))


@triton.jit
def _c550_attention_d128(
    Q,
    K,
    V,
    P,
    S,
    O,
    sm_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    REAL_D: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QM: tl.constexpr,
    BATCH_OUTER: tl.constexpr,
):
    h = tl.program_id(0)
    if BATCH_OUTER:
        block = tl.program_id(1)
        b = tl.program_id(2)
    else:
        b = tl.program_id(1)
        block = tl.program_id(2)
    hk = h // (H // HK)
    rm = block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    qptr = Q + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    q = tl.load(qptr, ((M % BM == 0) | (rm[:, None] < M)) & (rd[None, :] < REAL_D), other=0)



    q = tl.reshape(q, (BM, D // 32, 4, 2, 4)).permute(0, 1, 3, 2, 4).reshape(BM, D)
    pp = P + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, (M % BM == 0) | (rm < M), other=0)
    lo = tl.load(S + (b * H + h) * tl.cdiv(M, QM) + block * BM // QM)
    hi = tl.minimum(N, N - M + (block + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    diagonal = ((N - M + block * BM + 1) // BN) * BN


    for start in range(lo, hi, BN):
        kn = start + rn
        kp = K + (b * N * HK + kn[None, :] * HK + hk) * REAL_D + rd[:, None]
        vp = V + (b * N * HK + kn[:, None] * HK + hk) * REAL_D + rd[None, :]
        k = tl.load(kp, ((N % BN == 0) | (kn[None, :] < N)) & (rd[:, None] < REAL_D), other=0)
        k = tl.reshape(k, (D // 32, 4, 2, 4, BN)).permute(0, 2, 1, 3, 4).reshape(D, BN)
        v = tl.load(vp, ((N % BN == 0) | (kn[:, None] < N)) & (rd[None, :] < REAL_D), other=0)
        gk = tl.load(pp + kn, (N % BN == 0) | (kn < N), other=0)
        score = tl.dot(q, k, input_precision="ieee") * scale
        score += (gq[:, None] - gk[None, :]) * log2e
        if N % BN != 0:
            score = tl.where(kn[None, :] < N, score, -float("inf"))
        if start >= diagonal:
            score = tl.where((N - M + rm[:, None]) >= kn[None, :], score, -float("inf"))
        mnew = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mnew)
        prob = tl.exp2(score - mnew[:, None])


        halves = tl.sum(tl.reshape(prob, (BM, 2, 16)), 2)
        psum = tl.sum(halves, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(prob.to(Q.dtype.element_ty), v, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    out = (acc * (1.0 / li[:, None])).to(Q.dtype.element_ty)
    op = O + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    tl.store(op, out, ((M % BM == 0) | (rm[:, None] < M)) & (rd[None, :] < REAL_D))


@triton.jit
def _exact_prepare(
    X,
    L,
    S,
    THR,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    M: tl.constexpr,
    BN: tl.constexpr,
    QM: tl.constexpr,
):
    h = tl.program_id(0)
    b = tl.program_id(1)
    PAD: tl.constexpr = C if C >= 256 else 256
    i = tl.arange(0, PAD)
    qi = tl.arange(0, NQ)
    ki = tl.arange(0, NK)
    anchor = tl.full((NQ,), 0, tl.float32)
    ends = tl.full((NK,), 0, tl.float32)
    carry = tl.full((), 0, tl.float32)
    pos_q = N - M + qi * QM
    pos_k = tl.minimum(ki * BN + BN - 1, N - 1)
    for chunk in range(tl.cdiv(N, C)):
        t = chunk * C + i
        x = tl.load(X + b * N * H + h + t * H, (t < N) & (i < C), other=0)
        x = tl.where(i == 0, x + carry, x)
        for level in tl.static_range(LOGC):
            src = (i // (2 << level)) * (2 << level) + (1 << level) - 1
            left = tl.gather(x, src, axis=0)
            x = tl.where((i & (1 << level)) != 0, x + left, x)
        tl.store(L + (b * H + h) * N + t, x, (t < N) & (i < C))
        carry = tl.sum(tl.where(i == C - 1, x, 0), 0)
        av = tl.gather(x, pos_q % C, axis=0)
        ev = tl.gather(x, pos_k % C, axis=0)
        anchor = tl.where(pos_q // C == chunk, av, anchor)
        ends = tl.where(pos_k // C == chunk, ev, ends)
    old = (ki[None, :] < tl.cdiv(N, BN)) & ((anchor[:, None] - ends[None, :]) < THR)
    starts = tl.sum(old.to(tl.int32), 1) * BN
    tl.store(S + (b * H + h) * tl.cdiv(M, QM) + qi, starts, qi < tl.cdiv(M, QM))


@triton.jit
def _c550_decode_short(
    Q,
    K,
    V,
    G,
    O,
    scale,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
):
    h, b = tl.program_id(0), tl.program_id(1)
    hk = h // (H // HK)
    PAD: tl.constexpr = max(C, 256)
    i = tl.arange(0, PAD)
    carry = tl.full((), 0, tl.float32)
    prefix = tl.full((PAD,), 0, tl.float32)
    for chunk in range(tl.cdiv(N, C)):
        ti = chunk * C + i
        x = tl.load(G + (b * N + ti) * H + h, (ti < N) & (i < C), other=0)
        x = tl.where(i == 0, x + carry, x)
        for level in tl.static_range(LOGC):
            src = (i // (2 << level)) * (2 << level) + (1 << level) - 1
            left = tl.gather(x, src, axis=0)
            x = tl.where((i & (1 << level)) != 0, x + left, x)
        carry = tl.sum(tl.where(i == C - 1, x, 0), 0)
        prefix = tl.where(i // C == chunk, tl.gather(x, i % C, axis=0), prefix)
    last = tl.sum(tl.where(i == N - 1, prefix, 0), 0)
    gk = prefix
    for level in tl.static_range((PAD // BN).bit_length() - 1):
        gk, unused = tl.split(tl.reshape(gk, (2, PAD // (2 << level))).T)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    q = tl.load(Q + (b * H + h) * R + rd[None, :], rd[None, :] < R, other=0)
    k = tl.load(
        K + (b * N * HK + rn[None, :] * HK + hk) * R + rd[:, None], (rn[None, :] < N) & (rd[:, None] < R), other=0
    )
    v = tl.load(
        V + (b * N * HK + rn[:, None] * HK + hk) * R + rd[None, :], (rn[:, None] < N) & (rd[None, :] < R), other=0
    )
    log2e: tl.constexpr = 1.4426950408889634
    score = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * (scale * log2e)
    score += (last - gk[None, :]) * log2e
    score = tl.where(rn[None, :] < N, score, -float("inf"))
    mi = tl.full((1,), -float("inf"), tl.float32)
    li = tl.zeros((1,), tl.float32)
    acc = tl.zeros((1, D), tl.float32)
    mnew = tl.maximum(mi, tl.max(score, 1))
    alpha = tl.exp2(mi - mnew)
    p = tl.exp2(score - mnew[:, None])
    psum = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    acc += tl.sum(p.T * v, axis=0, keep_dims=True)
    li = li * alpha + psum
    out = (acc * (1.0 / li[:, None])).to(Q.dtype.element_ty)
    tl.store(O + (b * H + h) * R + rd[None, :], out, rd[None, :] < R)


@tlcore.extern
def _c550_bpermute(index, value, _semantic=None):
    return tlcore.extern_elementwise(
        "",
        "",
        [index, value],
        {(tl.int32, tl.int32): ("llvm.mxc.bsm.bpermute", tl.int32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _c550_lane_read(x, index):
    bits = _c550_bpermute((index * 4).to(tl.int32), x.to(tl.int32, bitcast=True))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _c550_segment_tail(values, SEG: tl.constexpr, LEVEL: tl.constexpr, lane):
    SRC: tl.constexpr = (SEG // (1 << (LEVEL - 5))) * (1 << (LEVEL - 5)) + (1 << (LEVEL - 6)) - 1
    return _c550_lane_read(values[SRC], tl.full((64,), 63, tl.int32))


@triton.jit
def _c550_load_scan_segments(X, base, chunk, lane, H: tl.constexpr, N: tl.constexpr, C: tl.constexpr):
    values = ()
    for segment in tl.static_range(max(1, C // 64)):
        t = chunk * C + segment * 64 + lane
        values += (tl.load(X + base + t * H, (t < N) & (segment * 64 + lane < C), other=0),)
    return values


@triton.jit
def _c550_prepare_native(
    X,
    L,
    S,
    THR,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    M: tl.constexpr,
    BN: tl.constexpr,
    QM: tl.constexpr,
):



    h = tl.program_id(0)
    b = tl.program_id(1)
    lane = tl.arange(0, 64)
    carry = tl.full((64,), 0, tl.float32)
    data = _c550_load_scan_segments(X, b * N * H + h, 0, lane, H, N, C)
    for chunk in range(tl.cdiv(N, C)):
        following = _c550_load_scan_segments(X, b * N * H + h, chunk + 1, lane, H, N, C)
        values = (tl.where(lane == 0, data[0] + carry, data[0]),)
        for segment in tl.static_range(1, max(1, C // 64)):
            values += (data[segment],)

        for level in tl.static_range(min(LOGC, 6)):
            src = (lane // (2 << level)) * (2 << level) + (1 << level) - 1
            updated = ()
            for segment in tl.static_range(max(1, C // 64)):
                x = values[segment]
                left = _c550_lane_read(x, src)
                updated += (tl.where((lane & (1 << level)) != 0, x + left, x),)
            values = updated
        for level in tl.static_range(6, LOGC):
            updated = ()
            for segment in tl.static_range(C // 64):
                if segment & (1 << (level - 6)):
                    tail = _c550_segment_tail(values, segment, level, lane)
                    updated += (values[segment] + tail,)
                else:
                    updated += (values[segment],)
            values = updated
        for segment in tl.static_range(max(1, C // 64)):
            t = chunk * C + segment * 64 + lane
            tl.store(L + (b * H + h) * N + t, values[segment], (t < N) & (segment * 64 + lane < C))
        carry = _c550_lane_read(values[max(1, C // 64) - 1], tl.full((64,), min(C, 64) - 1, tl.int32))
        data = following


    tl.debug_barrier()
    qi = tl.arange(0, NQ)
    anchor = tl.load(L + (b * H + h) * N + N - M + qi * QM, qi < tl.cdiv(M, QM), other=0)
    low = tl.full((NQ,), 0, tl.int32)
    high = tl.full((NQ,), (N + BN - 1) // BN, tl.int32)
    for step in tl.static_range(((N + BN - 1) // BN).bit_length()):
        mid = (low + high) // 2
        pos = tl.minimum(mid * BN + BN - 1, N - 1)
        end = tl.load(L + (b * H + h) * N + pos)
        old = (mid < tl.cdiv(N, BN)) & ((anchor - end) < THR)
        low = tl.where(old, mid + 1, low)
        high = tl.where(old, high, mid)
    tl.store(S + (b * H + h) * tl.cdiv(M, QM) + qi, low * BN, qi < tl.cdiv(M, QM))


class _Launch:

    def __init__(self, fn):
        self.fn, self.plans = fn, OrderedDict()
        self.dist_context = getattr(_runtime_jit, "DistributedRtContext", None)

    def __call__(self, key, grid, args, kwargs):
        key = (key, tuple(grid), tuple(sorted(kwargs.items())))
        allowed = not self.fn.pre_run_hooks and not _knobs.runtime.debug and not _knobs.compilation.instrumentation_mode
        if self.dist_context is not None and self.dist_context().is_lite_mode:
            allowed = False
        plan = self.plans.get(key) if allowed else None
        if plan is None:
            compiled = self.fn[grid](*args, **kwargs)
            if allowed:
                self.plans[key] = compiled[tuple(grid) + (1,) * (3 - len(grid))]
                if len(self.plans) > 256:
                    self.plans.popitem(last=False)
        else:
            plan(*args)


_starts_launch = _Launch(_c550_starts)
_prepare_launch = _Launch(_exact_prepare)
_native_prepare_launch = _Launch(_c550_prepare_native)
_attention_launch = _Launch(_c550_attention)
_large_attention_launch = _Launch(_c550_attention_d128)
_short_decode_launch = _Launch(_c550_decode_short)
_reference_launch = None


def prepare(gate, m, qm, bn, threshold):
    b, n, h = gate.shape
    nq, nk = triton.cdiv(m, qm), triton.cdiv(n, bn)
    starts = torch.empty((b, h, nq), device=gate.device, dtype=torch.int32)


    if b * h > 1 and n <= 4096 and torch.__version__.startswith("2.8."):
        prefix = torch.empty((b, h, n), device=gate.device, dtype=torch.float32)
        ln, lr = (n - 1).bit_length(), (b * h - 1).bit_length()
        chunk = 2 * (1 << min(9, max(4, (9 + ln - lr) // 2)))
        args = (
            gate,
            prefix,
            starts,
            threshold,
            h,
            n,
            chunk,
            chunk.bit_length() - 1,
            triton.next_power_of_2(nq),
            triton.next_power_of_2(nk),
            m,
            bn,
            qm,
        )
        key = (gate.device.index, gate.data_ptr() % 16, b, threshold, args[4:])
        if n >= 256:
            _native_prepare_launch(key, (h, b), args, dict(num_warps=1, num_stages=1))
        else:
            _prepare_launch(key, (h, b), args, dict(num_warps=4))
    else:
        prefix = torch.cumsum(gate.transpose(1, 2), -1, dtype=torch.float32)
        args = (prefix, threshold, starts, m, n, h, bn, qm, triton.next_power_of_2(nk))
        key = (gate.device.index, b, threshold, args[3:])
        _starts_launch(key, (h, b, triton.cdiv(nq, 16)), args, dict(num_warps=4))
    return prefix, starts


def c550_forward(q, k, v, gate, scale, threshold):
    b, m, h, real_d = q.shape
    n, hk = k.shape[1:3]
    d = triton.next_power_of_2(real_d)
    qm = 1 if m == 1 else (64 if d <= 64 else 128)
    bn = min(128, max(16, triton.next_power_of_2(n))) if m == 1 else 32
    out = torch.empty_like(q)
    key = (q.device.index, q.dtype, b, m, n, h, hk, real_d, type(scale), float(scale))




    if m == 1 and n <= 128 and real_d == d and b * h > 1 and threshold <= 0 and torch.__version__.startswith("2.8."):
        chunk = 2 * (1 << min(9, max(4, (9 + (n - 1).bit_length() - (b * h - 1).bit_length()) // 2)))
        args = (q, k, v, gate, out, scale, n, h, hk, real_d, d, bn, chunk, chunk.bit_length() - 1)
        _short_decode_launch(key, (h, b), args, dict(num_warps=4, num_stages=1))
        return out
    prefix, starts = prepare(gate, m, qm, bn, threshold)
    if m > 1 and d > 64:


        bm = 32 if real_d != d else 64


        batch_outer = real_d == d and m >= 2048 and b > 1
        grid = (h, triton.cdiv(m, bm), b) if batch_outer else (h, b, triton.cdiv(m, bm))
        args = (q, k, v, prefix, starts, out, scale, m, n, h, hk, real_d, d, bm, bn, qm, batch_outer)
        _large_attention_launch(key, grid, args, dict(num_warps=4, num_stages=1, scenario="mla" if real_d == d else ""))
    elif m > 1:
        warps = 2 if m >= 1024 and (m == 1024 or b * h * m >= 131072) else 4



        bm = 32 if m >= 1024 and warps == 4 and hk <= 8 else qm
        if bm == 32:
            warps = 2
        args = (q, k, v, prefix, starts, out, scale, m, n, h, hk, real_d, d, bm, bn, qm)
        stages = 1 if m >= 1024 else 3
        scenario = "mla" if warps == 2 else ("flashattn-fwd" if stages == 3 else "")
        _attention_launch(
            key, (h, b, triton.cdiv(m, bm)), args, dict(num_warps=warps, num_stages=stages, scenario=scenario)
        )
    else:
        from flag_attn.forgetting_attention.naive import _fwd_kernel

        global _reference_launch
        if _reference_launch is None:
            _reference_launch = _Launch(_fwd_kernel)
        if real_d != d:
            q, k, v = (torch.nn.functional.pad(t, (0, d - real_d)) for t in (q, k, v))
            out = torch.empty((b, m, h, d), device=q.device, dtype=q.dtype)
        norm = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
        nq = triton.cdiv(m, qm)
        args = (
            q,
            k,
            v,
            prefix,
            starts,
            starts,
            scale,
            norm,
            out,
            m * h * d,
            d,
            h * d,
            1,
            n * hk * d,
            d,
            hk * d,
            1,
            n * hk * d,
            d,
            hk * d,
            1,
            h * n,
            n,
            1,
            h * nq,
            nq,
            1,
            m * h * d,
            d,
            h * d,
            1,
            b,
            h,
            m,
            n,
            n - m,
            h // hk,
            qm,
            d,
            bn,
            True,
            False,
            False,
            True,
            m % qm == 0,
            n % bn == 0,
        )
        _reference_launch(
            key,
            (nq, h, b),
            args,
            dict(
                num_warps=4,
                num_stages=3 if m > 1 and d <= 64 else 1,
                scenario="reduceSmemUsage;flashattn-fwd" if m > 1 and d > 64 else "",
            ),
        )
        if real_d != d:
            out = out[..., :real_d].contiguous()
    return out


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None, sm_scale=None, adaptive_threshold=None
):
    if head_first or seq_start is not None or (not torch.is_inference_mode_enabled()):
        raise NotImplementedError("general profile supports BTHD inference without seq_start")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("require 4D Q/K/V and matching K/V shapes")
    b, m, h, d = q.shape
    n, hk = k.shape[1:3]
    if d not in (16, 32, 60, 64, 100, 128) or k.shape != (b, n, hk, d) or hk <= 0 or h % hk:
        raise NotImplementedError("require supported D and Q heads divisible by KV heads")
    if not (0 < m <= n and b > 0 and (h > 0)):
        raise ValueError("require 0 < M <= N and positive batch/heads")
    if not (
        q.dtype in (torch.bfloat16, torch.float16)
        and q.is_cuda
        and all(
            (
                t.dtype == q.dtype and t.device == q.device and t.is_contiguous() and (t.data_ptr() % 16 == 0)
                for t in (q, k, v)
            )
        )
    ):
        raise NotImplementedError("require contiguous, 16-byte-aligned CUDA BF16/FP16 inputs")
    if (
        log_fgate.shape != (b, n, h)
        or log_fgate.dtype != torch.float32
        or log_fgate.device != q.device
        or (not log_fgate.is_contiguous())
    ):
        raise NotImplementedError("gate must be contiguous FP32 BNH")
    if (
        not isinstance(adaptive_threshold, (int, float))
        or not math.isfinite(adaptive_threshold)
        or adaptive_threshold > 0
    ):
        raise NotImplementedError("require a finite nonpositive scalar ACP threshold")
    scale = 1 / math.sqrt(d) if sm_scale is None else sm_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite scalar")
    with torch.cuda.device(q.device):
        return c550_forward(q, k, v, log_fgate, scale, float(adaptive_threshold))


__all__ = ["parallel_forgetting_attn"]
