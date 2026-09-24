"""Opt-in generalized BTHD inference; never falls back to ordinary attention."""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from .prepare import exact, starts as prepare_starts
from .fast_launch import LaunchCache,cdiv,pow2
from functools import lru_cache
from .threshold import _get_cached_scalar_adaptive_threshold

def prepare(q,k,v,gate,head_first,seq_start,sm_scale,adaptive_threshold,prep_mode,fast):
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        raise NotImplementedError("general profile supports BTHD inference without seq_start")
    B,M,H,D=q.shape
    if k.ndim!=4 or v.shape!=k.shape:
        raise ValueError("K/V shape mismatch")
    N,HK=k.shape[1:3]
    if D not in (16,32,60,64,100,128) or k.shape!=(B,N,HK,D) or HK<=0 or H%HK:
        raise NotImplementedError("require supported D and Q heads divisible by KV heads")
    if not (0<M<=N and B>0 and H>0):
        raise ValueError("require 0 < M <= N and positive batch/heads")
    if not (q.dtype in (torch.bfloat16,torch.float16) and q.is_cuda and
            all(t.dtype==q.dtype and t.device==q.device and t.is_contiguous()
                and t.data_ptr()%16==0 for t in (q,k,v))):
        raise NotImplementedError("require contiguous, 16-byte-aligned CUDA BF16/FP16 inputs")
    if gate.shape!=(B,N,H) or gate.dtype!=torch.float32 or gate.device!=q.device or not gate.is_contiguous():
        raise NotImplementedError("gate must be contiguous FP32 BNH")
    if not isinstance(adaptive_threshold,(int,float)) or not math.isfinite(adaptive_threshold) or adaptive_threshold>0:
        raise NotImplementedError("require a finite nonpositive scalar ACP threshold")
    if torch.cuda.get_device_capability(q.device)!=(9,0):
        raise NotImplementedError("Hopper SM90 required")
    scale=1/math.sqrt(D) if sm_scale is None else sm_scale
    if not isinstance(scale,(int,float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite scalar")
    BN=min(128,max(16,pow2(N))) if M==1 else (64 if D<=64 else 128)
    QM=1 if M==1 else 128
    threshold=_get_cached_scalar_adaptive_threshold(q.device.index,B,H,float(adaptive_threshold))
    if prep_mode and B*H>1 and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,fast=fast)
    elif (B,M,N,H,D)==(4,4096,4096,32,64) and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,anchor_mode=True,fast=fast)
    else:
        # The original Torch accumulation tree is part of the bitwise contract.
        prefix=torch.cumsum(gate.transpose(1,2),dim=-1,dtype=torch.float32)
        starts=prepare_starts(prefix,threshold,M,N,H,BN,QM,fast)
    out=torch.empty((*q.shape[:-1],pow2(D)),device=q.device,dtype=q.dtype)
    return prefix,starts,out,scale,BN

@lru_cache(maxsize=16)
def shared_layout(layout_type,D):
    return () if layout_type is None else (layout_type(
        swizzle_byte_width=min(128,D*2),element_bitwidth=16,rank=2,transposed=False),)

@lru_cache(maxsize=512)
def h100_config(B,M,N,H,HK,D):
    # ACP grouping / reduction BN are fixed; only independent resource choices vary.
    prep=0 if (B,M,N,H,D)==(4,4096,4096,32,64) else 1
    return (64,1 if D>64 else 2,prep,True,0)

def make_entry(kind,kernel,decode,Descriptor,layout_type=None,config_override=None,fast=True):
    main_launch=LaunchCache(kernel);decode_launch=LaunchCache(decode)
    def forgetting_attention(q,k,v,log_fgate,*,head_first=False,seq_start=None,
                             sm_scale=None,adaptive_threshold=None):
        with torch.cuda.device(q.device):
            B,M,H,realD=q.shape;N,HK=k.shape[1:3]
            cfg=h100_config(B,M,N,H,HK,realD) if config_override is None else config_override
            BM,SLOTS,PREP,STATIC,MMA_N=cfg
            if M==1:BM=1;SLOTS=2
            prefix,starts,out,scale,BN=prepare(q,k,v,log_fgate,head_first,seq_start,
                                              sm_scale,adaptive_threshold,PREP,fast)
            D=pow2(realD)
            if D!=realD:
                q,k,v=(F.pad(x,(0,D-realD)) for x in (q,k,v))
            layout=shared_layout(layout_type,D)
            qd=q if M==1 else Descriptor(q,[B*M,H*D],[H*D,1],[BM,D],*layout)
            kd=Descriptor(k,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            vd=Descriptor(v,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            kwargs={} if M==1 else dict(BM=BM,STATIC=STATIC)
            if M>1 and kind=='gluon':kwargs['MMA_N']=min(D,BN) if MMA_N==0 else MMA_N
            key=(q.device.index,q.dtype,B,M,N,H,HK,D,BM,SLOTS,STATIC,MMA_N,type(scale),float(scale))
            args=(qd,kd,vd,prefix,starts,out,scale)
            kwargs.update(SLOTS=SLOTS,M=M,N=N,H=H,HK=HK,D=D,BN=BN,num_warps=4,num_stages=1)
            (decode_launch if M==1 else main_launch)(key,(H,B,cdiv(M,BM)),args,kwargs,fast)
            return out if D==realD else out[...,:realD].contiguous()
    return forgetting_attention
