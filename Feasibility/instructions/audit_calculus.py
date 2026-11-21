"""Additional recorded finite-difference errors and CUDA microbatch validation at trained weights."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
from common import torch,np,deterministic,attempt_dir,write_json,seed_all
from test_correctness import fixture_model
from measurement import mean_gradients,summarize
from torch.func import functional_call
from torch.nn.functional import cross_entropy

def main():
    deterministic()
    model,x,y=fixture_model("B")
    params=dict(model.named_parameters())
    gradients,_,_=mean_gradients(model,x,y,1)
    records=[]
    for name,index in (("conv1.weight",(0,0,1,1)),("conv2.weight",(0,0,1,1)),("conv3.weight",(0,0,1,1)),("head.weight",(0,0))):
        mask=torch.zeros_like(params[name]); mask[index]=1
        def loss(gate):
            pp=dict(params); pp[name]=params[name]*(1+mask*(gate-1))
            return cross_entropy(functional_call(model,pp,(x,)),y)
        gate=torch.ones((),dtype=torch.float64,requires_grad=True)
        gradcheck=bool(torch.autograd.gradcheck(loss,(gate,),eps=1e-6,atol=1e-5,rtol=1e-3))
        autograd=float(torch.autograd.grad(loss(gate),gate)[0])
        numeric=float((loss(gate.detach()+1e-6)-loss(gate.detach()-1e-6))/(2e-6))
        assert gradcheck and abs(autograd-numeric)<1e-9+1e-7*abs(autograd)
        records.append({"name":name,"index":index,"autograd":autograd,"central_difference":numeric,
                        "absolute_error":abs(autograd-numeric),"epsilon":1e-6,"gradcheck_passed":gradcheck})
    cuda=[]
    if torch.cuda.is_available():
        from data import load,panel_indices
        from model import CNNA
        import json
        f=json.loads((attempt_dir()/"frozen_config.json").read_text())
        for condition in ("independent","correlated"):
            d=load(condition); ix=panel_indices(d,"probe_A",64)
            m=CNNA().cuda(); cp=torch.load(attempt_dir("checkpoints")/(condition+"_s11")/f"step_{f['endpoint']:04d}.pt",weights_only=False,map_location="cpu")
            m.load_state_dict(cp["model"])
            base,_,_=mean_gradients(m,d["x"][ix],d["y"][ix],640)
            b=summarize(m,base)
            for micro in (64,128,192):
                g,_,_=mean_gradients(m,d["x"][ix],d["y"][ix],micro)
                err=max((g[n]-base[n]).abs().max().item() for n in g)
                for n in g: torch.testing.assert_close(g[n],base[n],atol=1e-6,rtol=1e-4)
                s=summarize(m,g)
                for block in ("shallow","middle","deep"):
                    np.testing.assert_allclose(s[block+"_total"],b[block+"_total"],atol=1e-6,rtol=1e-4)
                qerr=max(abs(s[block+"_share"]-b[block+"_share"]) for block in ("shallow","middle","deep"))
                assert qerr<1e-5
                cuda.append({"condition":condition,"seed":11,"step":f["endpoint"],"probe_size":640,"microbatch":micro,
                             "max_gradient_error":err,"max_block_share_error":qerr,"total_absolute_error":abs(s["total"]-b["total"])})
            del d,m
    result={"passed":True,"finite_differences":records,"trained_checkpoint_cuda_microbatch":cuda}
    write_json(attempt_dir()/"additional_calculus_audit.json",result)
    print("ADDITIONAL CALCULUS AUDIT PASSED",result)

if __name__=="__main__":main()
