from run import *
from torchvision.datasets import MNIST
from spuco.datasets import SpuCoMNIST,SpuriousFeatureDifficulty
from spuco.utils import set_seed
from spuco.datasets.spuco_mnist_config import config as splits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def source(spec):
    folder=O/'data'/f"f{spec['family']:02d}_c{spec['construction']}"; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'compact.npz'
    raw=MNIST(str(ROOT/'data/mnist'),train=True,download=False)
    if not path.exists():
        set_seed(spec['data_seed']); seed_all(spec['data_seed'])
        ds=SpuCoMNIST(root=str(ROOT/'data/mnist'),spurious_feature_difficulty=getattr(SpuriousFeatureDifficulty,spec['difficulty']),classes=[[i] for i in range(10)],spurious_correlation_strength=spec['alignment'],split='train',download=False)
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
    result={}; generator=torch.Generator().manual_seed(130000)
    weights=[(n,p) for n,p in model.named_parameters() if n.endswith('weight')]
    allocations=[min(p.numel(),1024) for n,p in weights]
    left=4096-sum(allocations)
    for i,(n,p) in enumerate(weights):
        add=min(left,p.numel()-allocations[i]);allocations[i]+=add;left-=add
    for (n,p),count in zip(weights,allocations):result[n]=torch.randperm(p.numel(),generator=generator)[:count]
    return result

def observe(model,x,y,initial_norm,coords,saturation=False):
    start=time.perf_counter();dev=next(model.parameters()).device; named=dict(model.named_parameters())
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
        r=summarize(model,acc);gnorm=sum(g.double().square().sum().item() for g in acc.values())**.5
    finally:
        for h in handles:h.remove()
    probabilities=torch.cat(probs).numpy();labels=y.cpu().numpy();pred=probabilities.argmax(1);correct=pred==labels;confidence=probabilities.max(1)
    v=torch.stack(vectors).double();gns=(v.mean(0).abs()/(v.std(0,unbiased=False)+1e-12)).mean().item()
    wnorm=sum(p.detach().double().square().sum().item() for n,p in named.items() if n.endswith('weight'))**.5
    sep=[]
    for c in range(10):
        ps=probabilities[labels==c]; cc=correct[labels==c];best=(0.,1.,0.)
        for k in (2,3):
            cluster=KMeans(n_clusters=k,n_init=10,max_iter=100,random_state=130001).fit_predict(ps)
            counts=np.bincount(cluster,minlength=k)
            if counts.min()<5:continue
            score=silhouette_score(ps,cluster)
            if score>best[0]:best=(float(score),float(counts.min()/len(ps)),float(np.ptp([cc[cluster==j].mean() for j in range(k)])))
        sep.append(best)
    r.update(loss=sum(losses)/len(y),accuracy=float(correct.mean()),worst_class=float(min(correct[labels==c].mean() for c in range(10))),confidence=float(confidence.mean()),correct_confidence=float(confidence[correct].mean()) if correct.any() else None,
             entropy=float(-(probabilities*np.log(probabilities+1e-12)).sum(1).mean()),gradient_norm=gnorm,growth=wnorm/initial_norm-1,gns=gns,
             separation=float(np.mean([v[0] for v in sep])),smallest_cluster=float(np.mean([v[1] for v in sep])),cluster_gap=float(max(v[2] for v in sep)),
             saturation=sat[0]/sat[1] if sat[1] else None,seconds=time.perf_counter()-start)
    return r

def train(spec,d):
    folder=O/'classifiers'/spec['run'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    seed_all(spec['training_seed']); model=CNNA().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=spec['lr'],foreach=False,fused=False)
    initial={n:p.detach().clone() for n,p in model.state_dict().items()};norm=sum(p.double().square().sum().item() for n,p in initial.items() if n.endswith('weight'))**.5
    coords=coordinate_indices(model);write_json(folder/'gradient_coordinates.json',{k:v.tolist() for k,v in coords.items()})
    ix=torch.tensor(np.flatnonzero(d['roles']==0)); per=(len(ix)+127)//128;T=60*per;steps=[round(f*T) for f in read(O/'protocol.json')['fractions']]
    plan={'horizon':T,'epochs':60,'updates_per_epoch':per,'steps':steps,'absolute_epochs':[s/per for s in steps]}
    write_json(folder/'plan.json',plan);step=0;rows=[];timing={'training':0.,'monitor_A':0.,'monitor_B':0.,'health':0.};wall=time.perf_counter()
    recovery=folder/'recovery.pt'
    if recovery.exists():
        cp=torch.load(recovery,weights_only=False,map_location='cpu');assert cp['protocol']==sha(O/'protocol.json') and cp['source']==read(O/'source_freeze.json')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_rng(cp['rng']);step=cp['step'];rows=cp['rows'];timing=cp['timing']
    else:save_torch(folder/'initial.pt',initial)
    if not recovery.exists():
        for panel,role in [('A',1),('B',2)]:
            pi=torch.tensor(np.flatnonzero(d['roles']==role));r=observe(model,d['x'][pi],d['y'][pi],norm,coords,False);r.update(step=0,fraction=0.,panel=panel);rows.append(r)
        pd.DataFrame(rows).to_csv(folder/'early.csv',index=False)
    # Whole verified RGB construction fits available GPU RAM and avoids repeated disk copies; precision and sample order unchanged.
    x=d['x'].cuda();y=d['y'].cuda()
    for epoch in range(step//per,60):
        order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(spec['training_seed']*100000+epoch))]
        for within in range(step%per if epoch==step//per else 0,per):
            current=epoch*per+within
            if current<step:continue
            batch=order[within*128:(within+1)*128].cuda();torch.cuda.synchronize();ts=time.perf_counter()
            model.train();optimizer.zero_grad(set_to_none=True);loss=cross_entropy(model(x[batch]),y[batch])
            if not torch.isfinite(loss):
                write_json(folder/'diverged.json',{'step':step});return
            loss.backward();optimizer.step();torch.cuda.synchronize();timing['training']+=time.perf_counter()-ts;step=current+1
            if step in steps:
                for panel,role in [('A',1),('B',2)]:
                    pi=torch.tensor(np.flatnonzero(d['roles']==role),device='cuda');r=observe(model,x[pi],y[pi],norm,coords,step==steps[0]);timing['monitor_'+panel]+=r['seconds'];r.update(step=step,fraction=step/T,panel=panel);rows.append(r)
                hi=torch.tensor(np.flatnonzero(d['roles']==3),device='cuda');h,_=health(model,x[hi],y[hi]);timing['health']+=h['seconds']
                write_json(folder/f'health_{step}.json',h)
                save_torch(folder/f'early_{step}.pt',model.state_dict())
                pd.DataFrame(rows).to_csv(folder/'early.csv',index=False)
            if step%per==0:
                save_torch(recovery,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'step':step,'rows':rows,'timing':timing,'protocol':sha(O/'protocol.json'),'source':read(O/'source_freeze.json')})
        if epoch%10==9:event('classifier_progress',run=spec['run'],epochs=epoch+1,total=60)
    hi=torch.tensor(np.flatnonzero(d['roles']==3),device='cuda');final_health,_=health(model,x[hi],y[hi]);save_torch(folder/'final.pt',model.state_dict())
    write_json(folder/'complete.json',{'run':spec['run'],'updates':step,'planned_updates':T,'full_60_epochs':step==T,'timing':timing,'wall_this_invocation':time.perf_counter()-wall,'final_health':final_health,'peak_gpu_allocated':torch.cuda.max_memory_allocated()})
    event('classifier_complete',run=spec['run'],updates=step,seconds=round(time.perf_counter()-wall,2))

def outcomes(spec,split):
    if split=='test':assert (O/'predictor_freeze.json').exists() and (O/'evaluation_predictions.csv').exists()
    folder=O/'outcomes'/split;folder.mkdir(parents=True,exist_ok=True);p=folder/(spec['run']+'.json')
    if p.exists():return
    x,y,bg=outcome_panel(spec['difficulty'],split);model=CNNA().cuda();model.load_state_dict(torch.load(O/'classifiers'/spec['run']/'final.pt',weights_only=True))
    _,probs=health(model,x,y);correct=(probs.argmax(1)==y).numpy();groups=[]
    for c,b in itertools.product(range(10),range(10)):
        mask=(y.numpy()==c)&(bg==b);n=int(mask.sum());k=int(correct[mask].sum());assert n>0
        ph=k/n;z=1.959963984540054;den=1+z*z/n;center=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        groups.append({'class':c,'background':b,'n':n,'correct':k,'accuracy':ph,'wilson_low':center-half,'wilson_high':center+half})
    wga=min(g['accuracy'] for g in groups)
    write_json(p,{'run':spec['run'],'wga':wga,'failure':wga<.75,'aggregate_accuracy':float(correct.mean()),'groups':groups,'cutoff_near':abs(wga-.75)<=.05,'split':split})

def execute(split):
    manifest=read(O/'protocol.json')['runs'];selected=[r for r in manifest if r['split']==split]
    if split=='evaluation':assert (O/'predictor_freeze.json').exists()
    for key,group in itertools.groupby(selected,key=lambda r:(r['family'],r['construction'])):
        specs=list(group);d=source(specs[0])
        for spec in specs:
            train(spec,d)
            if split=='development' and (O/'classifiers'/spec['run']/'complete.json').exists():outcomes(spec,'val')
        del d;torch.cuda.empty_cache()
