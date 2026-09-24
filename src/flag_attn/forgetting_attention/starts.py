"""Unchanged general ACP boundary kernel, extracted from V7.5."""
import triton
import triton.language as tl

@triton.jit
def _general_starts(P, T, S, M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                    BN:tl.constexpr,QM:tl.constexpr,NK:tl.constexpr):
    h,b,group=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    r=group*16+tl.arange(0,16)
    j=tl.arange(0,NK)
    pp=P+(b*H+h)*N
    a=tl.load(pp+N-M+r*QM,r<tl.cdiv(M,QM),other=0)
    end=tl.minimum(j*BN+BN-1,N-1)
    e=tl.load(pp+end,j<tl.cdiv(N,BN),other=0)
    threshold=tl.load(T)
    old=(j[None,:]<tl.cdiv(N,BN))&((a[:,None]-e[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+r,starts,r<tl.cdiv(M,QM))
