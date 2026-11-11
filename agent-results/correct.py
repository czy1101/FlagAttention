import os, sys, json, argparse
import torch, torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument("--dtype",default="fp16")
p.add_argument("--B",type=int,default=1)
p.add_argument("--T",type=int,default=512)
p.add_argument("--H",type=int,default=4)
p.add_argument("--K",type=int,default=64)
p.add_argument("--V",type=int,default=64)
p.add_argument("--gln",type=float,default=1.0)
p.add_argument("--out",default=None)
p.add_argument("--compare",default=None)
a=p.parse_args()

def run_case():
    from flag_attn import chunk_gla
    dev=torch.device("gcu:0")
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"]="ieee"
    dt=torch.float16 if a.dtype=="fp16" else torch.bfloat16
    B,T,H,K,V=a.B,a.T,a.H,a.K,a.V
    q=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
    k=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
    v=torch.rand(B,T,H,V,dtype=dt,device=dev).requires_grad_()
    g=(F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev))/a.gln).requires_grad_()
    h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev).requires_grad_()
    do=torch.randn_like(v); dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)
    o,ht=chunk_gla(q=q,k=k,v=v,g=g,scale=K**-0.5,initial_state=h0,
                   output_final_state=True,cu_seqlens=None)
    ((o*do).sum()+(ht*dht).sum().to(do.dtype)).backward()
    return {n:t.detach().float().cpu() for n,t in
            dict(o=o,ht=ht,dq=q.grad,dk=k.grad,dv=v.grad,dg=g.grad,dh0=h0.grad).items()}

if a.out:
    r=run_case()
    torch.save(r,a.out)
    print(json.dumps({k:float(v.square().mean().sqrt()) for k,v in r.items()}))
if a.compare:
    cur=run_case()
    ref=torch.load(a.compare,map_location="cpu")
    ratio=0.004 if a.dtype=="fp16" else 0.004
    res=[]
    for n in ["o","ht","dq","dk","dv","dg","dh0"]:
        A,Bt=cur[n],ref[n]
        d=(A-Bt).abs().max().item()
        rel=((A-Bt).square().mean().sqrt()/A.square().mean().sqrt().clamp_min(1e-8)).item()
        nan=bool(torch.isnan(A).any() or torch.isinf(A).any())
        ok=(d<=1e-6) or (rel<ratio and not nan)
        res.append(dict(name=n,max_abs=d,rel_rms=rel,nan_inf=nan,ok=ok))
    print(json.dumps(res,indent=1))
    print("ALL_PASS",all(x["ok"] for x in res))
