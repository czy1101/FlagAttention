# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BI-V150 paged Hy3 forward kernels, with no Hopper/TLE instructions.

FP8 operands are exactly representable in BF16. FP8 PV uses a separate
global-softmax pass so probability rounding matches the reference even when
the KV axis is split. Public decode calls accept the upstream workspaces.
"""

from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def _round_probability_to_bf16(x):
    # Normalized probabilities scaled by 256 lie in [0,256]. Round their
    # FP32 significands directly to E4M3 precision, retaining BF16 storage.
    bits = x.to(tl.uint32, bitcast=True)
    normal = (bits + 0x7ffff + ((bits >> 20) & 1)) & 0xfff00000
    # E4M3 subnormals are multiples of 2^-9; FP32 addition rounds to that grid.
    subnormal = ((x + 16384.0) - 16384.0).to(tl.uint32, bitcast=True)
    rounded = tl.where(x < 0.015625, subnormal, normal)
    # Preserve signed zero and the installed CoreX software conversion's
    # NaN/Inf behavior. This is a probability helper, not a general FP8 cast.
    magnitude = bits & 0x7fffffff
    special = tl.where((bits & 0x7fffff) != 0,0x43f0,0x43e0)
    encoded = tl.where(magnitude >= 0x7f800000,special,rounded >> 16)
    encoded = encoded | ((bits >> 16) & 0x8000)
    return encoded.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _convert_value_cache_kernel(
    V, VB, N:tl.constexpr, PAGE:tl.constexpr, HK:tl.constexpr,
    V0:tl.constexpr, V1:tl.constexpr, V2:tl.constexpr, V3:tl.constexpr,
    BLOCK:tl.constexpr,
):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    page=i//(PAGE*HK*128)
    token=(i//(HK*128))%PAGE
    head=(i//128)%HK
    d=i%128
    value=tl.load(V+page*V0+token*V1+head*V2+d*V3,i<N,0.0).to(tl.bfloat16)
    tl.store(VB+i,value,i<N)


def _value_cache_for_call(v):
    # Canonical NHD storage also avoids repeating HND operand redistribution.
    # Refill on every call, including graph replay; never reuse stale contents.
    result=torch.empty(v.shape,dtype=torch.bfloat16,device=v.device)
    with torch.cuda.device(v.device):
        _convert_value_cache_kernel[(triton.cdiv(v.numel(),1024),)](
            v,result,v.numel(),v.shape[1],v.shape[2],*v.stride(),1024,num_warps=4)
    return result


@triton.jit
def _decode_kernel(
    Q, K, V, IDS, LENGTHS, QS, KS, VS, PART, MAXIMUM, DENOM, GLOBAL_M, GLOBAL_L,
    HQ: tl.constexpr, GROUP: tl.constexpr, MTP: tl.constexpr,
    SPLITS: tl.constexpr, PAGE: tl.constexpr, MAX_PAGES: tl.constexpr,
    QUANT: tl.constexpr, PHASE: tl.constexpr,
    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
    I0: tl.constexpr, I1: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, FAST_P_ROUND: tl.constexpr,
):
    b, hk, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tl.arange(0, BM)
    query = rows // GROUP
    head = hk * GROUP + rows % GROUP
    valid_r = rows < MTP * GROUP
    d = tl.arange(0, 128)
    n = tl.arange(0, BN)
    q = tl.load(Q + (b*MTP+query[:, None])*Q0 + head[:, None]*Q1 + d[None, :]*Q2,
                valid_r[:, None], 0.0).to(tl.bfloat16)
    if QUANT >= 0:
        qs = tl.load(QS + (b*MTP+query)*QS0 + head*QS1, valid_r, 0).to(tl.float32)
        if QUANT == 1:
            ks_scalar = tl.load(KS).to(tl.float32)
        vs = tl.load(VS + (hk if QUANT == 0 else 0)).to(tl.float32) / 256.0
    length = tl.load(LENGTHS+b)
    tiles = tl.cdiv(length, BN)
    per_split = tl.cdiv(tiles, SPLITS)
    begin = split * per_split * BN
    end = tl.minimum(length, begin + per_split * BN)
    causal_end = length - MTP + query
    m = tl.full((BM,), -float('inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    stat_offset = ((b*MTP+query)*HQ+head)
    if QUANT >= 0 and PHASE == 1:
        m = tl.load(GLOBAL_M + stat_offset, valid_r, 0)
    for start in range(begin, end, BN):
        positions = start + n
        valid_n = positions < end
        if BN <= PAGE:
            # A tile stays within one page; all lanes share its physical ID.
            page = tl.load(IDS+b*I0+(start//PAGE)*I1)
            physical = page + tl.zeros((BN,), tl.int32)
            pos = (start % PAGE) + n
        else:
            physical = tl.load(IDS+b*I0+(positions//PAGE)*I1, valid_n, 0)
            pos = positions % PAGE
        k = tl.load(K + physical[:, None]*K0 + pos[:, None]*K1 + hk*K2 + d[None, :]*K3,
                    valid_n[:, None], 0.0).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * 0.08838834764831845
        if QUANT == 0:
            scale_offset = physical*KS0 + (pos//32)*KS1 + hk*KS2 + (pos%32)*KS3
            keyscale = tl.load(KS+scale_offset, valid_n, 0.0)
            scores *= qs[:, None]*keyscale[None, :]
        elif QUANT == 1:
            scores *= qs[:, None]*ks_scalar
        valid = valid_r[:, None] & valid_n[None, :] & (positions[None, :] <= causal_end[:, None])
        scores = tl.where(valid, scores, -float('inf'))
        if QUANT >= 0 and PHASE == 1:
            p = tl.where(valid, tl.exp(scores-m[:, None]), 0)
            den += tl.sum(p, 1)
            if FAST_P_ROUND:
                p = _round_probability_to_bf16(p*256.0)
            else:
                p = (p*256.0).to(tl.float8e4nv).to(tl.bfloat16)
            v = tl.load(V + physical[:, None]*V0 + pos[:, None]*V1 + hk*V2 + d[None, :]*V3,
                        valid_n[:, None], 0.0).to(tl.bfloat16)
            acc += tl.dot(p, v, out_dtype=tl.float32)
        elif QUANT >= 0:
            # Only the global maximum is needed before probability rounding.
            m = tl.maximum(m, tl.max(scores, 1))
        else:
            new_m = tl.maximum(m, tl.max(scores, 1))
            safe_m = tl.where(new_m == -float('inf'), 0.0, new_m)
            alpha = tl.where(m == -float('inf'), 0.0, tl.exp(m-safe_m))
            p = tl.where(valid, tl.exp(scores-safe_m[:, None]), 0.0)
            den = den*alpha + tl.sum(p, 1)
            if QUANT == -1:
                v = tl.load(V + physical[:, None]*V0 + pos[:, None]*V1 + hk*V2 + d[None, :]*V3,
                            valid_n[:, None], 0.0).to(tl.bfloat16)
                acc = acc*alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m = new_m
    split_stat = ((b*SPLITS+split)*MTP+query)*HQ+head
    if PHASE == 0:
        tl.store(MAXIMUM+split_stat, m, valid_r)
        if QUANT == -1:
            tl.store(DENOM+split_stat, den, valid_r)
    if QUANT == -1:
        partial = tl.where(den[:, None]>0, acc/tl.maximum(den[:, None], 1.e-30), 0.0)
        tl.store(PART+split_stat[:, None]*128+d[None, :], partial, valid_r[:, None])
    elif PHASE == 1:
        # Normalize once after summing the split-local unquantized denominators.
        tl.store(DENOM+split_stat, den, valid_r)
        partial = acc*vs
        tl.store(PART+split_stat[:, None]*128+d[None, :], partial, valid_r[:, None])


@triton.jit
def _reduce_stats(MAXIMUM, GLOBAL_M, HQ: tl.constexpr,
                  MTP: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0)
    row = tl.program_id(1)
    splits = tl.arange(0, BS)
    offset = (b*SPLITS+splits)*MTP*HQ+row
    m = tl.load(MAXIMUM+offset, splits<SPLITS, -float('inf'))
    maximum = tl.max(m, 0)
    safe = tl.where(maximum == -float('inf'), 0.0, maximum)
    tl.store(GLOBAL_M+b*MTP*HQ+row, safe)


@triton.jit
def _combine(PART, MAXIMUM, DENOM, OUT, HQ: tl.constexpr, MTP: tl.constexpr,
             SPLITS: tl.constexpr, BS: tl.constexpr, FP8: tl.constexpr):
    b = tl.program_id(0)
    row = tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    offset = (b*SPLITS+s)*MTP*HQ+row
    part = tl.load(PART+offset[:, None]*128+d[None, :], (s<SPLITS)[:, None], 0)
    if FP8:
        den = tl.load(DENOM+offset, s<SPLITS, 0.0)
        out = tl.sum(part, 0)/tl.maximum(tl.sum(den, 0), 1.e-30)
    else:
        m = tl.load(MAXIMUM+offset, s<SPLITS, -float('inf'))
        den = tl.load(DENOM+offset, s<SPLITS, 0)
        maximum = tl.max(m, 0)
        safe = tl.where(maximum == -float('inf'), 0.0, maximum)
        weight = den*tl.exp(m-safe)
        out = tl.sum(part*weight[:, None], 0)/tl.maximum(tl.sum(weight, 0), 1.e-30)
    tl.store(OUT+(b*MTP*HQ+row)*128+d, out)


@dataclass
class DecodeWorkspace:
    out: torch.Tensor
    partial: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor
    global_maximum: torch.Tensor
    global_denominator: torch.Tensor
    signature: tuple
    splits: int
    mtp: int
    route: str = 'iluvatar-bf16-mma-paged-split-k'


def _signature(inputs, quant):
    tensors = [inputs.q, inputs.k_cache, inputs.v_cache, inputs.block_ids, inputs.kv_lens]
    if quant >= 0:
        tensors += [inputs.q_scale, inputs.k_scale, inputs.v_scale]
    return (quant, tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors))


def prepare_decode_workspace(inputs, quant=-1):
    q, k, v = inputs.q, inputs.k_cache, inputs.v_cache
    batch = inputs.kv_lens.numel()
    if not q.is_cuda or batch <= 0 or q.ndim != 3 or q.shape[0] % batch:
        raise ValueError('Q must be CUDA [batch*MTP,Hq,128] with a nonempty batch')
    mtp, hq, dim = q.shape[0]//batch, q.shape[1], q.shape[2]
    supported = (1, 2, 3) if quant == -1 else (1, 2, 4)
    if mtp not in supported or dim != 128 or hq % k.shape[2]:
        raise ValueError('unsupported MTP, head dimension or GQA grouping')
    if k.ndim != 4 or k.shape[1] != 64 or k.shape[-1] != 128 or v.shape != k.shape:
        raise ValueError('decode K/V must have logical shape [page,64,Hkv,128]')
    expected = torch.bfloat16 if quant == -1 else torch.float8_e4m3fn
    if any(t.dtype != expected for t in (q, k, v)):
        raise TypeError('Q/K/V dtype does not match the selected variant')
    if inputs.block_ids.dtype != torch.int32 or inputs.kv_lens.dtype != torch.int32:
        raise TypeError('page table and lengths must be int32')
    if inputs.block_ids.shape[0] != batch:
        raise ValueError('page table batch differs from query batch')
    if any(t.device != q.device for t in (k,v,inputs.block_ids,inputs.kv_lens)):
        raise ValueError('all inputs must be on the Q device')
    if quant >= 0 and (inputs.q_scale.shape != q.shape[:2] or inputs.q_scale.dtype != torch.float32):
        raise ValueError('Q scale must be FP32 [batch*MTP,Hq]')
    if quant >= 0:
        if any(t.device != q.device for t in (inputs.q_scale, inputs.k_scale, inputs.v_scale)):
            raise ValueError('quantization scales must be on the Q device')
        if inputs.v_scale.dtype != torch.float32:
            raise TypeError('V scale must be FP32')
        if quant == 1 and (inputs.k_scale.dtype != torch.float32 or inputs.k_scale.numel() != 1 or inputs.v_scale.numel() != 1):
            raise ValueError('per-tensor K/V requires one FP32 scale each')
        if quant == 0 and (inputs.k_scale.ndim != 4 or inputs.k_scale.element_size() not in (1, 4) or inputs.v_scale.numel() != k.shape[2]):
            raise ValueError('per-token K requires packed FP32 scales and per-head V scales')
    max_tokens = inputs.block_ids.shape[1]*64
    # Keep enough independent programs for small batches, targeting at least
    # roughly 256 tokens per task at capacity and retaining V1's 32-split bound.
    # Shape-only selection keeps workspace and graph replay stable as lengths
    # change on device. FP8's two passes benefit from more parallel programs.
    target_programs = 64 if quant == -1 else 128
    parallel_splits = min(triton.cdiv(max_tokens, 256),
                          triton.cdiv(target_programs, batch*k.shape[2]))
    splits = min(32, max(1, triton.cdiv(max_tokens, 2048), parallel_splits))
    stats_shape = (batch, splits, mtp, hq)
    kw = dict(dtype=torch.float32, device=q.device)
    return DecodeWorkspace(
        torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
        torch.empty((*stats_shape, 128), **kw),
        torch.empty(stats_shape, **kw), torch.empty(stats_shape, **kw),
        torch.empty((batch, mtp, hq), **kw), torch.empty((batch, mtp, hq), **kw),
        _signature(inputs, quant), splits, mtp,
    )


def _workspace(inputs, workspace, quant):
    if isinstance(workspace, DecodeWorkspace):
        result = workspace
    else:
        # Upstream tests own preparation. Add private state to their object;
        # never replace or patch an upstream function/module.
        result = getattr(workspace, '_iluvatar_hy3', None)
        if result is None or result.signature != _signature(inputs, quant):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm up Hy3 before CUDA graph capture')
            result = prepare_decode_workspace(inputs, quant)
            workspace._iluvatar_hy3 = result
    if result.signature != _signature(inputs, quant):
        raise ValueError('workspace shape/layout/dtype changed; prepare a new workspace')
    return result


def _decode(inputs, workspace, quant):
    ws = _workspace(inputs, workspace, quant)
    q, k, v, ids = inputs.q, inputs.k_cache, inputs.v_cache, inputs.block_ids
    batch, mtp, hq, hk = inputs.kv_lens.numel(), ws.mtp, q.shape[1], k.shape[2]
    group = hq//hk
    qs = inputs.q_scale if quant >= 0 else q
    ks = inputs.k_scale if quant >= 0 else q
    if quant == 0 and ks.element_size() == 1:
        # Reinterpret packed metadata bytes; never numerically cast them.
        ks = ks.view(torch.float32)
    vs = inputs.v_scale if quant >= 0 else q
    ks_strides = tuple(ks.stride()) if quant == 0 else (0,0,0,0)
    # Short split-local loops do not amortize a narrower CTA on BF16. FP8's
    # conversion-heavy path benefits earlier. Shape-only selection is graph safe.
    capacity = ids.shape[1]*64
    # With one long request and the rest short, retain enough warps even when
    # only one split per short request runs. The BI-V150 has 16 compute units.
    enough_tasks = hk*(batch+ws.splits-1) >= 32
    use_two_warps = mtp*group <= 32 and enough_tasks and (
        capacity >= 2048 if quant >= 0 else
        mtp*group != 16 and triton.cdiv(capacity,ws.splits) >= 1024)
    decode_warps = 2 if use_two_warps else 4
    common = dict(FAST_P_ROUND=use_two_warps or capacity <= 1024,
                  HQ=hq, GROUP=group, MTP=mtp, SPLITS=ws.splits, PAGE=64,
                  MAX_PAGES=ids.shape[1], QUANT=quant,
                  Q0=q.stride(0),Q1=q.stride(1),Q2=q.stride(2),
                  K0=k.stride(0),K1=k.stride(1),K2=k.stride(2),K3=k.stride(3),
                  V0=v.stride(0),V1=v.stride(1),V2=v.stride(2),V3=v.stride(3),
                  I0=ids.stride(0),I1=ids.stride(1),
                  QS0=qs.stride(0),QS1=qs.stride(1),
                  KS0=ks_strides[0],KS1=ks_strides[1],KS2=ks_strides[2],KS3=ks_strides[3],
                  BM=max(32,triton.next_power_of_2(mtp*group)),BN=32 if quant == -1 else 64,
                  num_warps=decode_warps,num_stages=1)
    args=(q,k,v,ids,inputs.kv_lens,qs,ks,vs,ws.partial,ws.maximum,
          ws.denominator,ws.global_maximum,ws.global_denominator)
    with torch.cuda.device(q.device):
        _decode_kernel[(batch,hk,ws.splits)](*args,PHASE=0,**common)
        if quant >= 0:
            _reduce_stats[(batch,mtp*hq)](ws.maximum,ws.global_maximum,
                                         HQ=hq,MTP=mtp,SPLITS=ws.splits,
                                         BS=triton.next_power_of_2(ws.splits),num_warps=decode_warps)
            _decode_kernel[(batch,hk,ws.splits)](*args,PHASE=1,**common)
        _combine[(batch,mtp*hq)](ws.partial,ws.maximum,ws.denominator,ws.out,HQ=hq,MTP=mtp,
                                SPLITS=ws.splits,BS=triton.next_power_of_2(ws.splits),
                                FP8=quant>=0,num_warps=decode_warps)
    return ws.out


def attention_decode_bf16_static(inputs, workspace):
    return _decode(inputs, workspace, -1)


def attention_decode_bf16_dynamic(inputs, workspace):
    return _decode(inputs, workspace, -1)


def attention_decode_fp8_qk(inputs, workspace):
    return _decode(inputs, workspace, 0)


def attention_decode_fp8_kv(inputs, workspace):
    return _decode(inputs, workspace, 1)


@triton.jit
def _prefill_kernel(
    Q,K,V,QS,KS,VS,CU,IDS,LENGTHS,MASK,OUT,
    HQ:tl.constexpr,HK:tl.constexpr,PAGE:tl.constexpr,MAX_K:tl.constexpr,
    Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
    K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
    V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
    QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
    KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
    I0:tl.constexpr,I1:tl.constexpr,
    MS0:tl.constexpr,MS1:tl.constexpr,MS2:tl.constexpr,MS3:tl.constexpr,
    O0:tl.constexpr,O1:tl.constexpr,O2:tl.constexpr,
    QUANT:tl.constexpr,HAS_MASK:tl.constexpr,MASK_K:tl.constexpr,
    BM:tl.constexpr,BN:tl.constexpr,
):
    tile,head,b=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    qb=tl.load(CU+b)
    qe=tl.load(CU+b+1)
    qlen=qe-qb
    kvlen=tl.load(LENGTHS+b)
    hk=head//(HQ//HK)
    rows=tile*BM+tl.arange(0,BM)
    valid_r=rows<qlen
    d=tl.arange(0,128)
    n=tl.arange(0,BN)
    q=tl.load(Q+(qb+rows[:,None])*Q0+head*Q1+d[None,:]*Q2,valid_r[:,None],0.0).to(tl.bfloat16)
    qs=tl.load(QS+b*QS0+head*QS1+rows*QS2,valid_r,0)
    qpositions=kvlen-qlen+rows
    end=tl.minimum(kvlen,kvlen-qlen+tl.minimum((tile+1)*BM,qlen))
    m=tl.full((BM,),-float('inf'),tl.float32)
    den=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,128),tl.float32)
    # Two passes preserve reference FP8 rounding against the global maximum.
    for phase in tl.static_range(0,2):
        for start in range(0,end,BN):
            active=tile*BM<qlen
            if HAS_MASK:
                kt=start//128
                selected=tl.load(MASK+b*MS0+head*MS1+(tile*BM//128)*MS2+kt*MS3,kt<MASK_K,0)
                active &= (selected!=0) | (kt==MASK_K)
            if active:
                positions=start+n
                valid_n=positions<kvlen
                if BN <= PAGE:
                    page=tl.load(IDS+b*I0+(start//PAGE)*I1)
                    physical=page+tl.zeros((BN,),tl.int32)
                    pos=(start%PAGE)+n
                else:
                    physical=tl.load(IDS+b*I0+(positions//PAGE)*I1,valid_n,0)
                    pos=positions%PAGE
                k=tl.load(K+physical[:,None]*K0+pos[:,None]*K1+hk*K2+d[None,:]*K3,valid_n[:,None],0.0).to(tl.bfloat16)
                scores=tl.dot(q,tl.trans(k),out_dtype=tl.float32)*0.08838834764831845
                if QUANT==0:
                    ks=tl.load(KS+physical*KS0+(pos//32)*KS1+hk*KS2+(pos%32)*KS3,valid_n,0)
                    scores*=qs[:,None]*ks[None,:]
                else:
                    scores*=qs[:,None]*tl.load(KS)
                valid=valid_r[:,None]&valid_n[None,:]&(positions[None,:]<=qpositions[:,None])
                scores=tl.where(valid,scores,-float('inf'))
                if phase==0:
                    m=tl.maximum(m,tl.max(scores,1))
                else:
                    safe=tl.where(m==-float('inf'),0.0,m)
                    p=tl.where(valid,tl.exp(scores-safe[:,None]),0.0)
                    den+=tl.sum(p,1)
                    p=_round_probability_to_bf16(p*256.0)
                    v=tl.load(V+physical[:,None]*V0+pos[:,None]*V1+hk*V2+d[None,:]*V3,valid_n[:,None],0.0).to(tl.bfloat16)
                    acc+=tl.dot(p,v,out_dtype=tl.float32)
    vs=tl.load(VS+(hk if QUANT==0 else 0))
    # Empty sparse rows intentionally retain the upstream NaN contract.
    output=acc*(vs/256.0)/den[:,None]
    tl.store(OUT+(qb+rows[:,None])*O0+head*O1+d[None,:]*O2,output,valid_r[:,None])


def attention_with_kvcache_blocksparse_prefill_fp8(
    q,kcache,vcache,qscale,kscale,vscale,cu_seqlens_q,block_ids,seqlens_kvcache,
    max_seqlens_q,quant_type=1,block_mask=None,output=None,*,sparsity_bucket=None,
):
    from flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8 import _check_inputs, _quant_type_value
    quant=_quant_type_value(quant_type)
    _check_inputs(q,kcache,vcache,qscale,kscale,vscale,cu_seqlens_q,block_ids,
                  seqlens_kvcache,max_seqlens_q,quant,block_mask,output)
    if output is None:
        output=torch.empty(q.shape,device=q.device,dtype=torch.bfloat16)
    # Amortize V conversion across long queries while bounding temporary data.
    if max_seqlens_q >= 256 and vcache.numel()*2 <= 128*1024*1024:
        vcache=_value_cache_for_call(vcache)
    if quant==0 and kscale.element_size()==1:
        kscale=kscale.view(torch.float32)
    ks_strides=kscale.stride() if quant==0 else (0,0,0,0)
    ms=block_mask.stride() if block_mask is not None else (0,0,0,0)
    # Query tiles divide the public 128-token sparse block. Long queries reuse
    # K/V across 128 rows; short queries avoid a large, mostly empty CTA.
    bm = 128 if max_seqlens_q >= 256 else 32
    # A 64-token KV tile also divides the sparse mask's 128-token blocks.
    bn = 32 if max_seqlens_q < 256 else 64
    warps = 8 if bm == 128 else 2
    with torch.cuda.device(q.device):
        _prefill_kernel[(triton.cdiv(max_seqlens_q,bm),q.shape[1],block_ids.shape[0])](
            q,kcache,vcache,qscale,kscale,vscale,cu_seqlens_q,block_ids,seqlens_kvcache,
            block_mask if block_mask is not None else q,output,
            HQ=q.shape[1],HK=kcache.shape[2],PAGE=kcache.shape[1],MAX_K=block_ids.shape[1]*kcache.shape[1],
            Q0=q.stride(0),Q1=q.stride(1),Q2=q.stride(2),
            K0=kcache.stride(0),K1=kcache.stride(1),K2=kcache.stride(2),K3=kcache.stride(3),
            V0=vcache.stride(0),V1=vcache.stride(1),V2=vcache.stride(2),V3=vcache.stride(3),
            QS0=qscale.stride(0),QS1=qscale.stride(1),QS2=qscale.stride(2),
            KS0=ks_strides[0],KS1=ks_strides[1],KS2=ks_strides[2],KS3=ks_strides[3],
            I0=block_ids.stride(0),I1=block_ids.stride(1),
            MS0=ms[0],MS1=ms[1],MS2=ms[2],MS3=ms[3],
            O0=output.stride(0),O1=output.stride(1),O2=output.stride(2),
            QUANT=quant,HAS_MASK=block_mask is not None,MASK_K=block_mask.shape[3] if block_mask is not None else 0,
            BM=bm,BN=bn,num_warps=warps,num_stages=1,
        )
    return output
