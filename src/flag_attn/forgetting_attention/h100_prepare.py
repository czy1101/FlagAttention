"""Pure Triton, installed Torch 2.10 innermost scan tree, with fused ACP starts."""
import torch
import triton
import triton.language as tl

@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",constraints="=f,f,f",
        args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _h100_exact_prepare(X,L,S,THR, XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,
                   TB:tl.constexpr,TH:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
                   C:tl.constexpr,LOGC:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,M:tl.constexpr,
                   BN:tl.constexpr,QM:tl.constexpr):
    h=tl.program_id(0);b=tl.program_id(1)
    i=tl.arange(0,C);qi=tl.arange(0,NQ);ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32);ends=tl.full((NK,),0,tl.float32)
    carry=tl.full((),0,tl.float32)
    pos_q=N-M+qi*QM
    pos_k=tl.minimum(ki*BN+BN-1,N-1)
    for chunk in range(tl.cdiv(N,C)):
        t=chunk*C+i
        x=tl.load(X+b*XB+h*XH+t*XT,t<N,other=0)
        x=tl.where(i==0,_add(x,carry),x)
        for m in tl.static_range(LOGC):
            src=(i//(2<<m))*(2<<m)+(1<<m)-1
            left=tl.gather(x,src,axis=0)
            x=tl.where((i&(1<<m))!=0,_add(x,left),x)
        tl.store(L+(b*H+h)*N+t,x,t<N)
        carry=tl.sum(tl.where(i==C-1,x,0),0)
        av=tl.gather(x,pos_q%C,axis=0)
        ev=tl.gather(x,pos_k%C,axis=0)
        anchor=tl.where(pos_q//C==chunk,av,anchor)
        ends=tl.where(pos_k//C==chunk,ev,ends)
    threshold=tl.load(THR+b*TB+h*TH)
    old=(ki[None,:]<tl.cdiv(N,BN))&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+qi,starts,qi<tl.cdiv(M,QM))

def h100_exact_prepare(gate,threshold,m,bn,qm,index_dtype=torch.int32):
    b,h,n=gate.shape
    assert b*h>1, 'Single-row Torch prefix uses CUB and must keep its original path'
    ln=(n-1).bit_length();lr=(b*h-1).bit_length()
    lx=min(9,max(4,(9+ln-lr)//2));chunk=2*(1<<lx)
    prefix=torch.empty((b,h,n),device=gate.device,dtype=torch.float32)
    starts=torch.empty((b,h,triton.cdiv(m,qm)),device=gate.device,dtype=index_dtype)
    _h100_exact_prepare[(h,b)](gate,prefix,starts,threshold,*gate.stride(),
        *threshold.stride(),h,n,chunk,chunk.bit_length()-1,
        triton.next_power_of_2(triton.cdiv(m,qm)),triton.next_power_of_2(triton.cdiv(n,bn)),m,bn,qm,num_warps=4)
    return prefix,starts
