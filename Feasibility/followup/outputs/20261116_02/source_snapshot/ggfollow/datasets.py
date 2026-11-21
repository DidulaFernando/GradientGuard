"""Parameterised official SpuCo construction, using only its source-training split."""
from .core import *
from .partitions import validate_matched
import pandas as pd
import gc

def build(name,data_seed,reservation_seed,conditions):
    require_protocol()
    folder=out()/"data"/name; folder.mkdir(parents=True,exist_ok=True)
    done=folder/"verification.json"
    if done.exists():
        v=read_json(done)
        for condition,item in v["conditions"].items(): assert sha(folder/(condition+".pt"))==item["sha256"]
        return folder
    from spuco.datasets import SpuCoMNIST,SpuriousFeatureDifficulty
    from spuco.utils import set_seed
    from spuco.datasets.spuco_mnist_config import config as mnist_split
    old=read_json(legacy("runs")/"frozen_config.json")["configuration"]["dataset"]
    rows={}; reference=None; grids=[]; start=time.perf_counter()
    for condition,strength in conditions.items():
        destination=folder/(condition+".pt")
        if destination.exists():
            d=torch.load(destination,map_location="cpu",weights_only=False)
            assert d["generation_seeds"]==[data_seed,reservation_seed]
            if reference is None: reference={k:d[k] for k in ("ids","y","roles","ranks","palette")}
            else: validate_matched(reference,d)
            rows[condition]={"sha256":sha(destination),"n":len(d["y"]),"train_n":d["roles"].count("train"),"reused_verified_partial":True}
            continue
        set_seed(data_seed); seed_all(data_seed)
        # download=False avoids touching the protected original MNIST archives; no val/test Dataset is created.
        ds=SpuCoMNIST(root=str(ROOT/"data/mnist"),spurious_feature_difficulty=SpuriousFeatureDifficulty.MAGNITUDE_LARGE,
                       classes=old["classes"],spurious_correlation_strength=strength,label_noise=0.,core_feature_noise=0.,
                       split="train",verbose=False,download=False)
        ds.initialize()
        ids=np.asarray(ds.mnist.indices,dtype=np.int64)
        raw=ds.mnist.dataset.data[torch.from_numpy(ids)].clone()
        labels=ds.mnist.dataset.targets[torch.from_numpy(ids)]
        y=torch.tensor(ds.labels); bg=torch.tensor(ds.spurious); x=torch.stack(ds.data.X); palette=np.asarray(ds.colors[:10])
        assert torch.equal(y,labels) and not ds.is_noisy_label.any() and not ds.data.core_feature_noise.any()
        assert len(set(ids))==len(ids) and not set(ids)&set(mnist_split["val"])
        assert x.shape[1:]==(3,28,28)
        foreground_error=background_error=0.
        for i in range(0,len(x),512):
            rr=raw[i:i+512].float().div(255).unsqueeze(1).expand(-1,3,-1,-1); xx=x[i:i+512]
            mask=rr!=0; cc=torch.tensor(palette[bg[i:i+512]],dtype=x.dtype)[:,:,None,None].expand_as(xx)
            foreground_error=max(foreground_error,(xx[mask]-rr[mask]).abs().max().item())
            background_error=max(background_error,(xx[~mask]-cc[~mask]).abs().max().item())
            assert torch.isfinite(xx).all() and xx.min()>=0 and xx.max()<=1
        assert foreground_error==0 and background_error==0
        if reference is None:
            roles=np.full(len(y),"train",dtype="U8"); ranks=np.full(len(y),-1,dtype=np.int64)
            rng=np.random.default_rng(reservation_seed)
            for label in range(10):
                order=rng.permutation(np.flatnonzero(y.numpy()==label)); offset=0
                for role,n in (("probe_A",256),("probe_B",256),("health",128)):
                    idx=order[offset:offset+n]; assert len(idx)==n
                    roles[idx]=role; ranks[idx]=np.arange(n); offset+=n
            reference={"ids":ids,"y":y,"roles":roles.tolist(),"ranks":ranks.tolist(),"palette":palette}
        assert np.array_equal(ids,reference["ids"]) and torch.equal(y,reference["y"]) and np.array_equal(palette,reference["palette"])
        roles=np.asarray(reference["roles"]); ranks=np.asarray(reference["ranks"])
        d={"x":x,"y":y,"raw":raw,"ids":ids,"roles":roles.tolist(),"ranks":ranks.tolist(),"background":bg,"palette":palette,
           "generation_seeds":[data_seed,reservation_seed],"condition":condition,"protocol_hash":sha(PROTOCOL_PATH)}
        manifest=pd.DataFrame({"original_id":ids,"class":y.numpy(),"role":roles,"rank":ranks})
        manifest.to_csv(folder/(condition+"_manifest.csv"),index=False)
        manifest.groupby(["role","class"]).size().unstack(fill_value=0).to_csv(folder/(condition+"_class_counts.csv"))
        offline=folder/"offline"; offline.mkdir(exist_ok=True)
        pd.DataFrame({"original_id":ids,"class":y.numpy(),"background":bg.numpy()}).to_csv(offline/(condition+"_backgrounds.csv"),index=False)
        align={}
        for role in ("all","train","probe_A","probe_B","health"):
            mask=np.ones(len(y),dtype=bool) if role=="all" else roles==role
            contingency=pd.crosstab(y.numpy()[mask],bg.numpy()[mask]).reindex(index=range(10),columns=range(10),fill_value=0)
            contingency.to_csv(offline/(condition+"_"+role+"_contingency.csv"))
            align[role]=float((y.numpy()[mask]==bg.numpy()[mask]).mean())
            if role=="all":
                assert np.max(abs(np.diag(contingency)/contingency.sum(axis=1)-strength))<.001
                if strength==.1: assert np.max(abs(contingency.div(contingency.sum(axis=1),axis=0).to_numpy()-.1))<.02
        sets={r:set(ids[roles==r]) for r in ("train","probe_A","probe_B","health")}
        assert all(not sets[a]&sets[b] for a in sets for b in sets if a<b)
        for role,n in (("probe_A",256),("probe_B",256),("health",128)):
            assert torch.bincount(y[torch.from_numpy(np.flatnonzero(roles==role))],minlength=10).tolist()==[n]*10
        save_torch(destination,d)
        rows[condition]={"sha256":sha(destination),"n":len(y),"train_n":int((roles=="train").sum()),"alignment":align,
                         "foreground_error":foreground_error,"background_error":background_error,"overlap_count":0,"validation_overlap":0}
        chosen=[int(torch.nonzero(y==label)[0]) for label in range(10)]
        grids.append((condition,x[chosen].clone()))
        del d,x,ds,raw; gc.collect()
        log("dataset_condition_verified",construction=name,condition=condition)
    if grids:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(grids),10,figsize=(15,2*len(grids)),squeeze=False)
        for i,(condition,grid) in enumerate(grids):
            for j in range(10): axes[i,j].imshow(grid[j].permute(1,2,0)); axes[i,j].axis("off"); axes[i,j].set_title(f"{condition[:4]} {j}",fontsize=8)
        fig.tight_layout(); fig.savefig(folder/"matched_images.png",dpi=120); plt.close(fig)
    result={"passed":True,"data_seed":data_seed,"reservation_seed":reservation_seed,"split":"official source train only",
            "official_validation_test_instantiated":False,"conditions":rows,"seconds":time.perf_counter()-start,"protocol_hash":sha(PROTOCOL_PATH)}
    write_json(done,result)
    return folder

def load_new(folder,condition): return torch.load(Path(folder)/(condition+".pt"),map_location="cpu",weights_only=False)

def schedule(train_n,epochs):
    updates=(train_n+cfg()["training"]["batch_size"]-1)//cfg()["training"]["batch_size"]
    horizon=updates*epochs; steps=[round(f*horizon) for f in cfg()["measurement"]["fractions"]]
    assert len(set(steps))==len(steps) and min(steps)>0
    return {"train_n":train_n,"epochs":epochs,"updates_per_epoch":updates,"horizon":horizon,"steps":steps,"endpoint":steps[-1],
            "actual_fractions":[s/horizon for s in steps],"absolute_epoch_equivalents":[s/updates for s in steps]}
