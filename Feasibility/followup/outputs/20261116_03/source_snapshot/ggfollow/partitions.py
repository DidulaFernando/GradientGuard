from .core import *
import pandas as pd

def validate_matched(a,b):
    assert np.array_equal(a["ids"],b["ids"]) and torch.equal(a["y"],b["y"])
    assert a["roles"]==b["roles"] and a["ranks"]==b["ranks"]

def partition(data,seed):
    roles=np.asarray(data["roles"]); labels=data["y"].numpy()
    union=np.flatnonzero(np.isin(roles,["probe_A","probe_B"]))
    assert len(set(data["ids"][union].tolist()))==5120
    assert not set(union)&set(np.flatnonzero(np.isin(roles,["train","health"])))
    rng=np.random.default_rng(seed); rows=[]
    for label in range(10):
        ids=union[labels[union]==label]; assert len(ids)==512
        shuffled=rng.permutation(ids)
        for panel,indices in (("A",shuffled[:256]),("B",shuffled[256:])):
            for rank,index in enumerate(indices):
                rows.append({"partition":seed,"class":label,"panel":panel,"rank":rank,"index":int(index),"original_id":int(data["ids"][index])})
    result=pd.DataFrame(rows)
    validate_partition(result,data)
    return result

def validate_partition(frame,data):
    roles=np.asarray(data["roles"])
    assert len(frame)==5120 and frame.original_id.nunique()==5120
    assert set(frame.original_id)==set(data["ids"][np.flatnonzero(np.isin(roles,["probe_A","probe_B"]))])
    assert np.isin(roles[frame["index"]],["probe_A","probe_B"]).all()
    assert np.array_equal(data["ids"][frame["index"]],frame.original_id)
    assert np.array_equal(data["y"].numpy()[frame["index"]],frame["class"])
    for (_,panel),group in frame.groupby(["class","panel"]):
        assert len(group)==256 and sorted(group["rank"])==list(range(256))
        nested=[set(group.loc[group["rank"]<n,"original_id"]) for n in (64,128,256)]
        assert nested[0]<nested[1]<nested[2]
    assert not set(frame.loc[frame.panel=="A","original_id"])&set(frame.loc[frame.panel=="B","original_id"])

def make_manifests(data,folder,seeds):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    result={}
    for seed in seeds:
        path=folder/f"partition_{seed}.csv"
        frame=partition(data,seed)
        if path.exists(): pd.testing.assert_frame_equal(pd.read_csv(path),frame,check_dtype=False)
        else: frame.to_csv(path,index=False)
        result[str(seed)]={"sha256":sha(path),"file":str(path),"unique_id_count":5120}
    write_once(folder/"manifest_hashes.json",result)
    return result

def indices(data,panel,budget,manifest=None):
    if manifest is None:
        roles=np.asarray(data["roles"]); rank=np.asarray(data["ranks"])
        return torch.tensor(np.flatnonzero((roles=="probe_"+panel)&(rank<budget)),dtype=torch.long)
    # Preserve original data-index order, matching the legacy panel extraction convention.
    ix=manifest.loc[(manifest.panel==panel)&(manifest["rank"]<budget),"index"].to_numpy()
    return torch.tensor(np.sort(ix),dtype=torch.long)
