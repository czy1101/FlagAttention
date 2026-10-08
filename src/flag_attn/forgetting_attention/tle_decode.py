"""M=1 exact vector specialization: explicit TMA, intentionally no WGMMA."""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.types import BlockEncoding as BlockedLayout

PV_LAYOUT=tl.constexpr(BlockedLayout([1,8],[4,8],[4,1],[1,0]))

@triton.jit
def _tle_decode_kernel(QDESC,KDESC,VDESC,PREFIX,STARTS,O,sm_scale,
                       M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
                       BN:tl.constexpr,SLOTS:tl.constexpr):
    dtype=O.dtype.element_ty
    h,b=tl.program_id(0),tl.program_id(1)
    hk=h//(H//HK)
    ks=tle.gpu.alloc([SLOTS,BN,D],dtype)
    vs=tle.gpu.alloc([SLOTS,BN,D],dtype)
    kb=tle.gpu.alloc_barriers(SLOTS,expect_bytes=BN*D*2)
    vb=tle.gpu.alloc_barriers(SLOTS,expect_bytes=BN*D*2)
    lo=tl.load(STARTS+b*H+h)
    tle.gpu.copy(KDESC,ks.slot(0),[BN,D],[b*N+lo,hk*D],barrier=kb[0])
    tle.gpu.copy(VDESC,vs.slot(0),[BN,D],[b*N+lo,hk*D],barrier=vb[0])
    # Distinct range expression prevents TLE hint conflicts when BN == D.
    rn_pair=tl.reshape(tl.arange(0,2*BN),(BN,2))
    rn=tl.sum(rn_pair,1)//4
    rd=tl.arange(0,D)
    pp=PREFIX+(b*H+h)*N
    gq=tl.load(pp+N-1)
    mi=tl.full((),-float("inf"),tl.float32)
    li=tl.zeros((),tl.float32)
    acc=tl.zeros([D],tl.float32)
    log2e:tl.constexpr=1.4426950408889634
    scale=sm_scale*log2e
    q=tl.load(QDESC+(b*H+h)*D+rd)
    for start in range(lo,N,BN):
        it=(start-lo)//BN
        slot=it%SLOTS;phase=it//SLOTS
        tle.gpu.barrier_wait(kb[slot],phaseIdx=phase)
        k=tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(ks.slot(slot)),[1,8]),[1,8]))
        if start+BN<N:
            nxt=(it+1)%SLOTS
            tle.gpu.copy(KDESC,ks.slot(nxt),[BN,D],[b*N+start+BN,hk*D],barrier=kb[nxt])
            tle.gpu.copy(VDESC,vs.slot(nxt),[BN,D],[b*N+start+BN,hk*D],barrier=vb[nxt])
        product=tle.gpu.set_layout((q[None,:]*k).to(dtype).to(tl.float32),PV_LAYOUT)
        score=tl.sum(product,1)*scale
        # A nonzero masked filler avoids CSE with the D-wide zero accumulator.
        # Invalid logits are replaced by -inf below, so this filler is unobservable.
        gk=tl.load(pp+start+rn,start+rn<N,other=-10000)
        score=tl.fma(gq-gk,log2e,score)
        score=tl.where(start+rn<N,score,-float("inf"))
        mnew=tl.maximum(mi,tl.max(score,0))
        alpha=tl.exp2(mi-mnew)
        p=tl.exp2(score-mnew)
        psum=tl.sum(p,0)
        tle.gpu.barrier_wait(vb[slot],phaseIdx=phase)
        v=tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(vs.slot(slot)),[1,8]),[1,8]))
        product_v=tle.gpu.set_layout(p[:,None]*v,PV_LAYOUT)
        acc=acc*alpha+tl.sum(product_v,0)
        li=li*alpha+psum
        mi=mnew
    tl.store(O+(b*H+h)*D+rd,(acc*(1.0/li)).to(dtype))
