"""Additional structural measurements from the very same mean-loss gradients."""
from core import *
LAYERS=['conv1','conv2','conv3','head']
def structural(model,g):
    result={};us=[]
    for l in LAYERS:
        w=getattr(model,l).weight.detach().cpu().double().flatten()
        grad=g[l+'.weight'].detach().cpu().double().flatten()
        for kind,v in [('w',w),('g',grad)]:
            a=v.abs()
            result.update({f'{l}_{kind}_l2':v.norm().item(),f'{l}_{kind}_absmean':a.mean().item(),f'{l}_{kind}_std':v.std(unbiased=False).item(),f'{l}_{kind}_absmedian':torch.quantile(a,.5).item(),f'{l}_{kind}_absq90':torch.quantile(a,.9).item()})
        u=w.norm().item()*grad.norm().item();us.append(u)
        result[l+'_U']=u
        s=(w*grad).abs().sum().item()
        assert s<=u+1e-10*max(1,u)
        result[l+'_co_location']=s/u if u>0 else None
    den=sum(us)+cfg()['epsilon']
    result['proxy_shallow']=us[0]/den;result['proxy_deep']=(us[2]+us[3])/den
    return result
def observe(model,x,y):
    dev=next(model.parameters()).device
    if dev.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();base=torch.cuda.memory_allocated() if dev.type=='cuda' else 0
    g,loss,accuracy=mean_gradients(model,x,y,cfg()['microbatch'])
    r=summarize(model,g,epsilon=cfg()['epsilon']);r.update(structural(model,g));del g
    if dev.type=='cuda':torch.cuda.synchronize()
    r.update(probe_loss=loss,probe_accuracy=accuracy,seconds=time.perf_counter()-start,gpu_peak_allocated=torch.cuda.max_memory_allocated() if dev.type=='cuda' else None,gpu_incremental=torch.cuda.max_memory_allocated()-base if dev.type=='cuda' else None)
    return r
def panels(spec):
    """Reconstruct ONLY the original held-out source probes; no outcome panels."""
    from torchvision.datasets import MNIST
    from spuco.datasets import SpuCoMNIST,SpuriousFeatureDifficulty
    folder=OLD/'data'/f"f{spec['family']:02d}_c{spec['construction']}"
    path=folder/'compact.npz';assert sha(path)==read(folder/'verification.json')['sha256']
    with np.load(path) as z:d={k:z[k] for k in z.files}
    assert len(set(d['ids']))==len(d['ids'])
    raw=MNIST(str(LEGACY/'data/mnist'),train=True,download=False)
    masks=torch.stack([SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,spec['difficulty']),c) for c in d['palette']])
    result={}
    for panel,role in [('A',1),('B',2)]:
        ix=np.flatnonzero(d['roles']==role);ids=d['ids'][ix]
        assert np.bincount(d['y'][ix],minlength=10).tolist()==[256]*10
        assert np.array_equal(raw.targets[ids].numpy(),d['y'][ix])
        gray=raw.data[ids].float().div(255).unsqueeze(1).expand(-1,3,-1,-1)
        x=(gray+masks[d['bg'][ix]]*(gray==0)).contiguous()
        assert torch.equal(x[gray!=0],gray[gray!=0])
        result[panel]=(x.cuda(),torch.tensor(d['y'][ix],device='cuda'))
    sets=[set(d['ids'][d['roles']==r]) for r in range(4)]
    assert all(not sets[a]&sets[b] for a in range(4) for b in range(a))
    return result
def extract():
    freeze=require();deterministic();start=time.perf_counter();count=0;maxerr=0.
    specs=read(OLD/'protocol.json')['runs'];allrows=[]
    import itertools
    for key,group in itertools.groupby(specs,key=lambda r:(r['family'],r['construction'])):
        group=list(group);data=None
        for spec in group:
            folder=OLD/'classifiers'/spec['run'];plan=read(folder/'plan.json');legacy=pd.read_csv(folder/'early.csv')
            for step in plan['steps']:
                for panel in cfg()['panels']:
                    checkpoint=folder/f'early_{step}.pt';dest=out()/'measurements'/f"{spec['run']}_{step}_{panel}.json"
                    identity={'run':spec['run'],'step':step,'panel':panel,'checkpoint_sha256':freeze['inputs'][str(checkpoint.relative_to(REPO))],'protocol_sha256':sha(out()/'protocol_freeze.json')}
                    if dest.exists():
                        row=read(dest);assert all(row[k]==v for k,v in identity.items())
                    else:
                        if data is None:data=panels(spec)
                        model=CNNA().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.train()
                        before={n:v.clone() for n,v in model.state_dict().items()}
                        r=observe(model,*data[panel])
                        assert model.training and all(torch.equal(v,before[n]) for n,v in model.state_dict().items())
                        old=legacy[(legacy.step==step)&(legacy.panel==panel)].iloc[0]
                        errors={}
                        for col in ['total','shallow_share','middle_share','deep_share','conv1_abs','conv2_abs','conv3_abs','head_abs','bias_norm','bias_sensitivity']:
                            errors[col]=abs(float(r[col])-float(old[col]));assert np.isclose(r[col],old[col],atol=cfg()['atol'],rtol=cfg()['rtol']),(identity,col,r[col],old[col])
                            if col.endswith('_share'):assert errors[col]<=cfg()['share_atol']
                        assert r['valid'];row=identity|{'fraction':step/plan['horizon'],'family':spec['family'],'split':spec['split'],'max_parity_error':max(errors.values()),**r};write(dest,row)
                        del model,before
                    maxerr=max(maxerr,row['max_parity_error']);allrows.append(row);count+=1
            event('checkpoint_run_complete',run=spec['run'],measurements=count,total=1152,elapsed_seconds=round(time.perf_counter()-start,2))
        del data;torch.cuda.empty_cache()
    pd.DataFrame(allrows).to_csv(out()/'measurements.csv',index=False)
    write(out()/'measurement_audit.json',{'records':count,'expected':1152,'max_parity_error':maxerr,'seconds_this_invocation':time.perf_counter()-start,'no_optimizer_or_training':True})
