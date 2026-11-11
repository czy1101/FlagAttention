import os, sys, json, time, argparse
import torch, torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument("--dtype",default="fp16"); p.add_argument("--T",type=int,default=512)
p.add_argument("--B",type=int,default=1); p.add_argument("--H",type=int,default=4)
p.add_argument("--K",type=int,default=64); p.add_argument("--V",type=int,default=64)
p.add_argument("--rounds",type=int,default=3); p.add_argument("--warmup",type=int,default=3)
a=p.parse_args()
from flag_attn import chunk_gla
dev=torch.device("gcu:0"); torch.manual_seed(42)
dt=torch.float16 if a.dtype=="fp16" else torch.bfloat16
B,T,H,K,V=a.B,a.T,a.H,a.K,a.V
q=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
k=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
v=torch.rand(B,T,H,V,dtype=dt,device=dev).requires_grad_()
g=F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev)).requires_grad_()
h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev).requires_grad_()
do=torch.randn_like(v); dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)

def step():
    o,ht=chunk_gla(q=q,k=k,v=v,g=g,scale=K**-0.5,
        initial_state=h0,output_final_state=True,cu_seqlens=None)
    ((o*do).sum()+(ht*dht).sum().to(do.dtype)).backward()
    q.grad=k.grad=v.grad=g.grad=h0.grad=None

for _ in range(a.warmup): step()
torch.gcu.synchronize()
ts=[]
for _ in range(a.rounds):
    torch.gcu.synchronize(); t0=time.perf_counter(); step(); torch.gcu.synchronize()
    ts.append((time.perf_counter()-t0)*1000)
print(json.dumps({"dtype":a.dtype,"T":T,"K":K,"V":V,"ms":ts,"median":sorted(ts)[len(ts)//2]}))
