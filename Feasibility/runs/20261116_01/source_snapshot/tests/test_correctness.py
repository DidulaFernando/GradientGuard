import copy
import pickle
import pytest
from common import torch, np, config, seed_all, rng_state, attempt_dir, write_json
from model import CNNA
from measurement import measure, mean_gradients, summarize
from torch.nn.functional import cross_entropy, max_pool2d

RESULTS = {}

def record(name, value):
    RESULTS[name] = value
    write_json(attempt_dir()/"correctness_metrics.json", RESULTS)

def fixture_model(state="A"):
    m = CNNA().double()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        for layer in (m.conv1, m.conv2, m.conv3):
            layer.weight[0,0,1,1] = 1
            layer.bias.fill_(-10)
            layer.bias[0] = 0
        m.head.weight[0,0] = 1
        if state == "B":
            m.conv1.bias[0] = .5
            m.conv2.bias[0] = -.75
            m.head.bias[0] = .25
    # Strictly positive, globally unique values => no pooling ties on the active path.
    x = torch.linspace(.8, 1.2, 28*28, dtype=torch.float64).reshape(1,1,28,28)
    x /= max_pool2d(max_pool2d(x,2),2).mean()
    x = x.repeat(1,3,1,1)
    return m, x, torch.tensor([1])

@pytest.mark.parametrize("state,expected,shares", [("A", [1,1,1,1], [.25,.25,.5]), ("B", [1,1.5,.75,.75], [.25,.375,.375])])
def test_analytical(state, expected, shares):
    m,x,y = fixture_model(state)
    z = m(x)
    assert abs(z[0,0].item()-1) < 1e-9
    assert torch.equal(z[0,1:], torch.zeros(9,dtype=torch.float64))
    r = torch.autograd.grad(cross_entropy(z,y),z)[0][0,0].item()
    g,_,_ = mean_gradients(m,x,y,1)
    s = summarize(m,g)
    actual = np.array([s[l+"_abs"] for l in ("conv1","conv2","conv3","head")]) / abs(r)
    np.testing.assert_allclose(actual, expected, atol=1e-9,rtol=1e-7)
    np.testing.assert_allclose(s["total"],4*abs(r),atol=1e-9,rtol=1e-7)
    stabilized = np.array(shares)*(4*abs(r))/(4*abs(r)+config()["measurement"]["epsilon"])
    actual_q = [s[b+"_share"] for b in ("shallow","middle","deep")]
    np.testing.assert_allclose(actual_q,stabilized,atol=1e-9,rtol=1e-7)
    record("analytical_"+state, {"logit": z[0,0].item(), "r": r, "normalized_sensitivities": actual.tolist(),
                               "shares": actual_q, "max_error": float(np.max(abs(actual-expected)))})

def test_gates_and_signed_distinction():
    from torch.func import functional_call
    seed_all(741)
    m = CNNA().double()
    # Keep the final active channel away from ReLU boundaries and eliminate pooling ties using random input.
    x = torch.rand(3,3,28,28,dtype=torch.float64)+.3
    y = torch.tensor([1,4,8])
    params = dict(m.named_parameters())
    g,_,_ = mean_gradients(m,x,y,2)
    selected = [("conv1.weight", (0,0,1,1)), ("conv2.weight", (0,0,1,1)),
                ("conv3.weight", (0,0,1,1)), ("head.weight", (0,0))]
    errors=[]
    for name, index in selected:
        mask = torch.zeros_like(params[name]); mask[index] = 1
        def gate_loss(gate):
            pp = dict(params)
            pp[name] = params[name]*(1+mask*(gate-1))
            return cross_entropy(functional_call(m,pp,(x,)),y)
        gate = torch.ones((),dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(gate_loss,(gate,),eps=1e-6,atol=1e-5,rtol=1e-3)
        dg = torch.autograd.grad(gate_loss(gate),gate)[0].item()
        expected = (params[name]*g[name])[index].item()
        errors.append(abs(dg-expected))
        np.testing.assert_allclose(dg,expected,atol=1e-9,rtol=1e-7)
    name="head.weight"
    def block_loss(gate):
        pp=dict(params); pp[name]=params[name]*gate
        return cross_entropy(functional_call(m,pp,(x,)),y)
    gate=torch.ones((),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(block_loss,(gate,),eps=1e-6,atol=1e-5,rtol=1e-3)
    dg=torch.autograd.grad(block_loss(gate),gate)[0].item()
    z=params[name]*g[name]
    np.testing.assert_allclose(dg,z.sum().item(),atol=1e-9,rtol=1e-7)
    assert z.abs().sum().item()-abs(dg)>1e-5
    record("gates",{"max_chain_rule_error":max(errors),"common_gate":dg,"signed_sum":z.sum().item(),"absolute_sum":z.abs().sum().item(),"gradcheck":{"eps":1e-6,"atol":1e-5,"rtol":1e-3}})

@pytest.mark.parametrize("dtype",[torch.float64,torch.float32])
def test_microbatch(dtype):
    seed_all(103)
    m=CNNA().to(dtype=dtype)
    x=torch.rand(13,3,28,28,dtype=dtype); y=torch.arange(13)%10
    base,_,_=mean_gradients(m,x,y,13)
    b=summarize(m,base)
    worst=0.0; qerr=0.0; totalerr=0.0
    for micro in (1,4,7):
        gs,_,_=mean_gradients(m,x,y,micro)
        for name in base:
            err=(base[name]-gs[name]).abs().max().item(); worst=max(worst,err)
            torch.testing.assert_close(base[name],gs[name],atol=1e-9 if dtype==torch.float64 else 1e-6,rtol=1e-7 if dtype==torch.float64 else 1e-4)
        s=summarize(m,gs)
        totalerr=max(totalerr,abs(s["total"]-b["total"]))
        np.testing.assert_allclose(s["total"],b["total"],atol=1e-9 if dtype==torch.float64 else 1e-6,rtol=1e-7 if dtype==torch.float64 else 1e-4)
        for block in ("shallow","middle","deep"):
            qerr=max(qerr,abs(s[block+"_share"]-b[block+"_share"]))
    assert qerr<1e-5
    record("microbatch_"+str(dtype),{"max_gradient_error":worst,"max_share_error":qerr,"max_total_error":totalerr,"n":13,"microbatches":[13,1,4,7]})

def test_restricted_rescaling():
    seed_all(715)
    m=CNNA().double(); n=copy.deepcopy(m)
    x=torch.rand(8,3,28,28,dtype=torch.float64); y=torch.arange(8)
    with torch.no_grad():
        n.conv2.weight[3]*=2.3
        n.conv2.bias[3]*=2.3
        n.conv3.weight[:,3]/=2.3
    out_err=(m(x)-n(x)).abs().max().item()
    torch.testing.assert_close(m(x),n(x),atol=1e-9,rtol=1e-7)
    a=measure(m,x,y,3); b=measure(n,x,y,3)
    errors={k:abs(a[k]-b[k]) for k in a if (k.endswith("_abs") or k.endswith("_share"))}
    for k in errors:
        np.testing.assert_allclose(a[k],b[k],atol=1e-9,rtol=1e-7)
    record("rescaling",{"max_output_error":out_err,"max_sensitivity_error":max(errors.values())})

def equal_nested(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b

@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_repeated_isolation(dev):
    if dev=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seed_all(777)
    m=CNNA().to(dev); m.train()
    opt=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
    x=torch.rand(9,3,28,28,device=dev); y=torch.arange(9,device=dev)
    cross_entropy(m(x),y).backward(); opt.step()
    params=copy.deepcopy(m.state_dict()); optim=copy.deepcopy(opt.state_dict())
    grads={n:p.grad.clone() for n,p in m.named_parameters()}; rng=rng_state()
    results=[measure(m,x,y,4) for _ in range(3)]
    assert m.training and all(mod.training for mod in m.modules())
    assert equal_nested(params,m.state_dict()) and equal_nested(optim,opt.state_dict())
    assert equal_nested(grads,{n:p.grad for n,p in m.named_parameters()}) and equal_nested(rng,rng_state())
    keys=[k for k in results[0] if k.endswith("_share") or k.endswith("_abs")]
    err=max(abs(a[k]-results[0][k]) for a in results for k in keys)
    assert err==0
    record("repeated_isolation_"+dev,{"maximum_difference":err,"parameters_optimizer_rng_grad_modes_unchanged":True})

@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_training_neutrality(dev):
    if dev=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    seed_all(909)
    a=CNNA().to(dev); b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False)
    ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
    x=torch.rand(32,3,28,28,device=dev); y=torch.arange(32,device=dev)%10
    for i in range(5):
        for m,o in ((a,oa),(b,ob)):
            m.train(); o.zero_grad(set_to_none=True); cross_entropy(m(x),y).backward(); o.step()
        measure(b,x[:13],y[:13],4)
    err=max((a.state_dict()[k]-b.state_dict()[k]).abs().max().item() for k in a.state_dict())
    assert equal_nested(a.state_dict(),b.state_dict()) and equal_nested(oa.state_dict(),ob.state_dict())
    record("training_neutrality_"+dev,{"max_parameter_error":err,"optimizer_equal":True,"updates":5})

def test_zero_invalid():
    m,x,y=fixture_model()
    z={n:torch.zeros_like(p) for n,p in m.named_parameters()}
    out=summarize(m,z)
    assert not out["valid"] and out["shallow_share"] is None
    record("invalid_zero",{"passed":True})

@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_checkpoint_resume(tmp_path,dev):
    if dev=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    from common import save_torch,restore_rng
    seed_all(832)
    a=CNNA().to(dev); b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False)
    ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
    x=torch.rand(17,3,28,28,device=dev); y=torch.arange(17,device=dev)%10
    def step(m,o):
        o.zero_grad(set_to_none=True); cross_entropy(m(x),y).backward(); o.step()
    for _ in range(5): step(a,oa)
    for _ in range(3): step(b,ob)
    save_torch(tmp_path/"resume.pt",{"model":b.state_dict(),"optimizer":ob.state_dict(),"rng":rng_state()})
    saved=torch.load(tmp_path/"resume.pt",weights_only=False,map_location="cpu")
    resumed=CNNA().to(dev); resumed.load_state_dict(saved["model"])
    ores=torch.optim.Adam(resumed.parameters(),lr=.001,foreach=False,fused=False)
    ores.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
    for _ in range(2): step(resumed,ores)
    assert equal_nested(a.state_dict(),resumed.state_dict()) and equal_nested(oa.state_dict(),ores.state_dict())
    record("resume_"+dev,{"bitwise_model_optimizer_match":True,"split_updates":[3,2]})
