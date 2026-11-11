import os, json, time, argparse, sys
import torch, torch.nn.functional as F
from flag_attn import chunk_gla
import flag_attn.runtime.backend._enflame.FLA.chunk_h as M
p=argparse.ArgumentParser()
p.add_argument("--dtype",required=True); p.add_argument("--T",type=int,required=True)
p.add_argument("--K",type=int,required=True); p.add_argument("--V",type=int,required=True)
p.add_argument("--B",type=int,default=1); p.add_argument("--H",type=int,default=4)
p.add_argument("--rounds",type=int,default=3)
a=p.parse_args()
dev=torch.device("gcu:0"); torch.manual_seed(42)
dt=torch.float16 if a.dtype=="fp16" else torch.bfloat16
B,T,H,K,V=a.B,a.T,a.H,a.K,a.V
q=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
k=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
v=torch.rand(B,T,H,V,dtype=dt,device=dev).requires_grad_()
g=F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev)).requires_grad_()
h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev).requires_grad_()
do=torch.randn_like(v); dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)
def fwd_bwd():
    o,ht=chunk_gla(q=q,k=k,v=v,g=g,scale=K**-0.5,initial_state=h0,
                   output_final_state=True,cu_seqlens=None)
    ((o*do).sum()+(ht*dht).sum().to(do.dtype)).backward()
    q.grad=k.grad=v.grad=g.grad=h0.grad=None
def grab():
    outs={}
    def go():
        o,ht=chunk_gla(q=q,k=k,v=v,g=g,scale=K**-0.5,initial_state=h0,
                       output_final_state=True,cu_seqlens=None)
        return o.detach().float(),ht.detach().float()
    return go()
def timed():
    torch.gcu.synchronize(); t0=time.perf_counter(); fwd_bwd(); torch.gcu.synchronize()
    return (time.perf_counter()-t0)*1000
def med(x): s=sorted(x); return s[len(s)//2]
out={"dtype":a.dtype,"T":T,"K":K,"V":V}
try:
    M.PARALLEL_BWD_STATE_ENABLED=True
    oc,hc=grab(); fwd_bwd()
    M.PARALLEL_BWD_STATE_ENABLED=False
    of,hf=grab(); fwd_bwd()
    res={"cand":[],"fall":[]}
    for _ in range(a.rounds):
        M.PARALLEL_BWD_STATE_ENABLED=True;  res["cand"].append(timed())
        M.PARALLEL_BWD_STATE_ENABLED=False; res["fall"].append(timed())
    M.PARALLEL_BWD_STATE_ENABLED=True
    out.update(cand_ms=res["cand"],fall_ms=res["fall"],
      cand_median=med(res["cand"]),fall_median=med(res["fall"]),
      speedup=med(res["fall"])/med(res["cand"]),
      o_max_abs=float((oc-of).abs().max()),o_rel=float(((oc-of).square().mean().sqrt()/oc.square().mean().sqrt().clamp_min(1e-8))),
      ht_max_abs=float((hc-hf).abs().max()),
      nan=bool(torch.isnan(oc).any() or torch.isinf(oc).any()))
    out["status"]="ok"
except Exception as e:
    out["status"]="error"; out["error"]=repr(e)[:300]
print("RESULT "+json.dumps(out))
