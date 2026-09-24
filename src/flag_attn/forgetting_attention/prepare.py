"""Same GPU kernels as v7.5; lighter scalar setup and compiled-launch reuse."""
import torch
from .h100_prepare import _h100_exact_prepare
from .exact_prepare import _exact_prepare
from .starts import _general_starts
from .fast_launch import LaunchCache,cdiv,pow2
general=LaunchCache(_h100_exact_prepare)
anchor=LaunchCache(_exact_prepare)
starts_launch=LaunchCache(_general_starts)
def exact(gate,threshold,m,bn,qm,anchor_mode=False,fast=True):
    b,h,n=gate.shape
    ln=(n-1).bit_length();lr=(b*h-1).bit_length()
    lx=min(9,max(4,(9+ln-lr)//2));chunk=2*(1<<lx)
    prefix=torch.empty((b,h,n),device=gate.device,dtype=torch.float32)
    starts=torch.empty((b,h,cdiv(m,qm)),device=gate.device,dtype=torch.int32)
    args=(gate,prefix,starts,threshold,*gate.stride(),*threshold.stride(),
          h,n,chunk,chunk.bit_length()-1,pow2(cdiv(m,qm)),pow2(cdiv(n,bn)))
    if not anchor_mode:args+= (m,bn,qm)
    # Every pointer except gate is an aligned fresh allocation or validated scalar buffer.
    key=(gate.device.index,b,gate.data_ptr()%16,args[4:])
    (anchor if anchor_mode else general)(key,(h,b),args,dict(num_warps=4),fast)
    return prefix,starts
def starts(prefix,threshold,M,N,H,BN,QM,fast=True):
    B=prefix.shape[0];NQ=cdiv(M,QM)
    out=torch.empty((B,H,NQ),device=prefix.device,dtype=torch.int32)
    args=(prefix,threshold,out,M,N,H,BN,QM,pow2(cdiv(N,BN)))
    starts_launch((prefix.device.index,B,args[3:]),(H,B,cdiv(NQ,16)),args,dict(num_warps=4),fast)
    return out
