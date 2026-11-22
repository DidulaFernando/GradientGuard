"""Adapted verified helpers; no legacy command-module imports or writes."""
from .core import *
from torchvision.datasets import MNIST
from spuco.datasets import SpuCoMNIST,SpuriousFeatureDifficulty
from spuco.utils import set_seed
from spuco.datasets.spuco_mnist_config import config as splits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def source(spec):
    require()
    if spec['split']=='evaluation':allow_evaluation_training()
    folder=O/'data'/f"f{spec['family']:02d}_c{spec['construction']}"; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'compact.npz'
    raw=MNIST(str(ROOT/'data/mnist'),train=True,download=False)
    if not path.exists():
        set_seed(spec['data_seed']); seed_all(spec['data_seed'])
        ds=SpuCoMNIST(root=str(ROOT/'data/mnist'),spurious_feature_difficulty=getattr(SpuriousFeatureDifficulty,spec['difficulty']),classes=[[i] for i in range(10)],spurious_correlation_strength=spec['alignment'],label_noise=0.,core_feature_noise=0.,split='train',download=False)
        ds.initialize(); ids=np.array(ds.mnist.indices); y=np.array(ds.labels); bg=np.array(ds.spurious)
        roles=np.zeros(len(ids),dtype=np.int8); ranks=np.full(len(ids),-1,dtype=np.int16); rng=np.random.default_rng(spec['reservation_seed'])
        for label in range(10):
            ix=rng.permutation(np.flatnonzero(y==label)); offset=0
            for role,n in [(1,256),(2,256),(3,128)]:
                chosen=ix[offset:offset+n]; roles[chosen]=role;ranks[chosen]=np.arange(n);offset+=n
        palette=np.array(ds.colors[:10],dtype=np.float32)
        masks=torch.stack([SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,spec['difficulty']),c) for c in palette])
        error=0.
        for start in range(0,len(ids),512):
            gray=raw.data[ids[start:start+512]].float().div(255).unsqueeze(1).expand(-1,3,-1,-1)
            expected=gray+masks[bg[start:start+512]]*(gray==0)
            actual=torch.stack(ds.data.X[start:start+512]); error=max(error,(expected-actual).abs().max().item())
        assert error==0 and np.array_equal(y,raw.targets[ids].numpy()) and not set(ids)&set(splits['val'])
        for a,b in itertools.combinations(range(4),2):assert not set(ids[roles==a])&set(ids[roles==b])
        for role,n in [(1,256),(2,256),(3,128)]:assert np.bincount(y[roles==role],minlength=10).tolist()==[n]*10
        np.savez_compressed(path,ids=ids,y=y,bg=bg,roles=roles,ranks=ranks,palette=palette)
        pd.DataFrame({'original_id':ids,'class':y,'role':roles,'rank':ranks}).to_csv(folder/'manifest.csv',index=False)
        pd.DataFrame({'original_id':ids,'background':bg}).to_csv(folder/'offline_backgrounds.csv',index=False)
        pd.crosstab(y,bg).to_csv(folder/'contingency.csv')
        write_json(folder/'verification.json',{'passed':True,'foreground_and_background_error':error,'sha256':sha(path),'difficulty':spec['difficulty'],'medium_mask_nonzero_pixels_per_channel':int((masks[0,0]!=0).sum()),'source_n':len(ids),'train_n':int((roles==0).sum())})
        del ds
    compact=np.load(path); assert sha(path)==read(folder/'verification.json')['sha256']
    masks=torch.stack([SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,spec['difficulty']),c) for c in compact['palette']])
    gray=raw.data[compact['ids']].float().div(255).unsqueeze(1).expand(-1,3,-1,-1)
    x=(gray+masks[compact['bg']]*(gray==0)).contiguous()
    return {k:compact[k] for k in compact.files}|{'x':x,'y':torch.tensor(compact['y']), 'data_hash':sha(path)}

def outcome_panel(difficulty,split):
    require()
    if split=='test':allow_test_outcomes()
    key=('val_' if split=='val' else 'test_')+('large' if difficulty.endswith('LARGE') else 'medium')
    seed=read(O/'protocol.json')['outcome_seeds'][key]
    raw=MNIST(str(ROOT/'data/mnist'),train=split=='val',download=False)
    ids=np.array(splits['val']) if split=='val' else np.arange(len(raw))
    y=raw.targets[ids].numpy(); rng=np.random.default_rng(seed); bg=np.zeros(len(ids),dtype=int)
    for c in range(10):
        ix=np.flatnonzero(y==c); assignment=np.arange(len(ix))%10; bg[ix]=rng.permutation(assignment)
    # Outcome assignments balanced within each class, with all original official identities retained.
    generator=SpuCoMNIST(root=str(ROOT/'data/mnist'),spurious_feature_difficulty=getattr(SpuriousFeatureDifficulty,difficulty),classes=[[i] for i in range(10)],spurious_correlation_strength=.1,download=False)
    masks=torch.stack([generator.create_background(getattr(SpuriousFeatureDifficulty,difficulty),c) for c in generator.colors[:10]])
    gray=raw.data[ids].float().div(255).unsqueeze(1).expand(-1,3,-1,-1); x=(gray+masks[bg]*(gray==0)).contiguous()
    assert torch.equal(x[gray!=0],gray[gray!=0])
    folder=O/'outcome_panels';folder.mkdir(exist_ok=True)
    frame=pd.DataFrame({'official_id':ids,'class':y,'background':bg,'source':split})
    dest=folder/(key+'.csv')
    if dest.exists():pd.testing.assert_frame_equal(pd.read_csv(dest),frame)
    else:frame.to_csv(dest,index=False)
    assert (pd.crosstab(y,bg).to_numpy()>0).all()
    return x,torch.tensor(y),bg

def coordinate_indices(model):
    result={}; generator=torch.Generator().manual_seed(cfg()['coordinate_seed'])
    weights=[(n,p) for n,p in model.named_parameters() if n.endswith('weight')]
    allocations=[min(p.numel(),1024) for n,p in weights]
    left=4096-sum(allocations)
    for i,(n,p) in enumerate(weights):
        add=min(left,p.numel()-allocations[i]);allocations[i]+=add;left-=add
    for (n,p),count in zip(weights,allocations):result[n]=torch.randperm(p.numel(),generator=generator)[:count]
    return result

def observe(model,x,y,initial_norm,coords,saturation=False):
    sync(next(model.parameters()).device);start=time.perf_counter();dev=next(model.parameters()).device; named=dict(model.named_parameters());base=torch.cuda.memory_allocated() if dev.type=='cuda' else 0
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats()
    acc={n:torch.zeros_like(p) for n,p in named.items()}; vectors=[]; probs=[]; losses=[]; sat=[0,0]; handles=[]
    def hook(mod,inp,z):sat[0]+=(z<=0).sum().item();sat[1]+=z.numel()
    if saturation:
        handles=[getattr(model,l).register_forward_hook(hook) for l in ['conv1','conv2','conv3']]
    try:
        with isolated(model),torch.enable_grad():
            for i in range(0,len(y),128):
                xb=x[i:i+128].to(dev);yb=y[i:i+128].to(dev);z=model(xb);loss=cross_entropy(z,yb)
                gradients=torch.autograd.grad(loss,tuple(named.values()))
                for (n,p),g in zip(named.items(),gradients):acc[n].add_(g.detach(),alpha=len(yb)/len(y))
                vectors.append(torch.cat([g.flatten()[coords[n].to(dev)] for (n,p),g in zip(named.items(),gradients) if n in coords]).detach().cpu())
                probs.append(z.softmax(1).detach().cpu());losses.append(loss.item()*len(yb))
        r=summarize(model,acc);r.update(structural(model,acc));gnorm=sum(g.double().square().sum().item() for g in acc.values())**.5
    finally:
        for h in handles:h.remove()
    probabilities=torch.cat(probs).numpy();labels=y.cpu().numpy();pred=probabilities.argmax(1);correct=pred==labels;confidence=probabilities.max(1)
    v=torch.stack(vectors).double();gns=(v.mean(0).abs()/(v.std(0,unbiased=False)+1e-12)).mean().item()
    wnorm=sum(p.detach().double().square().sum().item() for n,p in named.items() if n.endswith('weight'))**.5
    sep=[]
    for c in range(10):
        ps=probabilities[labels==c]; cc=correct[labels==c];best=(0.,1.,0.)
        for k in (2,3):
            cluster=KMeans(n_clusters=k,n_init=10,max_iter=100,random_state=cfg()['cluster_seed']).fit_predict(ps)
            counts=np.bincount(cluster,minlength=k)
            if counts.min()<5:continue
            score=silhouette_score(ps,cluster)
            if score>best[0]:best=(float(score),float(counts.min()/len(ps)),float(np.ptp([cc[cluster==j].mean() for j in range(k)])))
        sep.append(best)
    r.update(loss=sum(losses)/len(y),accuracy=float(correct.mean()),worst_class=float(min(correct[labels==c].mean() for c in range(10))),confidence=float(confidence.mean()),correct_confidence=float(confidence[correct].mean()) if correct.any() else None,
             entropy=float(-(probabilities*np.log(probabilities+1e-12)).sum(1).mean()),gradient_norm=gnorm,growth=wnorm/initial_norm-1,gns=gns,
             separation=float(np.mean([v[0] for v in sep])),smallest_cluster=float(np.mean([v[1] for v in sep])),cluster_gap=float(max(v[2] for v in sep)),
             saturation=sat[0]/sat[1] if sat[1] else None,seconds=time.perf_counter()-start)
    sync(dev)
    r.update(seconds=time.perf_counter()-start,gpu_peak_allocated=torch.cuda.max_memory_allocated() if dev.type=='cuda' else None,gpu_incremental=torch.cuda.max_memory_allocated()-base if dev.type=='cuda' else None)
    return r

