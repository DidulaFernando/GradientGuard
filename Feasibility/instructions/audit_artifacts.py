"""Independent post-run artifact integrity audit. No model training or result filtering."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common import torch,np,config,attempt_dir,sha,write_json,deterministic
from data import load,panel_indices,verify
from measurement import measure
from model import CNNA
import pandas as pd
import json
import inspect

def main():
    deterministic(); c=config(); folder=attempt_dir()
    frozen=json.loads((folder/"frozen_config.json").read_text())
    provenance=json.loads((folder/"provenance.json").read_text())
    sources=provenance["project_source_hashes"]
    # Reconstruct a missing snapshot only when current bytes equal the frozen hashes.
    for rel,digest in sources.items():
        target=folder/"source_snapshot"/rel
        if not target.exists():
            assert sha(ROOT/rel)==digest,rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/rel).read_bytes())
    for rel,digest in sources.items():
        assert sha(folder/"source_snapshot"/rel)==digest,rel
        assert sha(ROOT/rel)==digest,rel
    verify()
    a,b=load("independent"),load("correlated")
    assert np.array_equal(a["ids"],b["ids"]) and torch.equal(a["y"],b["y"])
    assert a["roles"]==b["roles"] and a["ranks"]==b["ranks"]
    out={"snapshot_matches_frozen_sources":True,"current_sources_match_frozen_sources":True,
         "matched_ids_labels_roles_nested_ranks":True,"runs":{},"nested_probe_counts":{},"training_order_hashes":{}}
    for role in ("probe_A","probe_B","health"):
        for n in ([64,128,256] if role!="health" else [32,128]):
            ix=panel_indices(a,role,n)
            assert torch.bincount(a["y"][ix],minlength=10).tolist()==[n]*10
            out["nested_probe_counts"][role+"_"+str(n)]=len(ix)
    train_ix=panel_indices(a,"train")
    import hashlib
    for seed in c["training"]["seeds"]:
        hashes=[]
        for epoch in range(2):
            generator=torch.Generator().manual_seed(seed*100000+epoch)
            order=train_ix[torch.randperm(len(train_ix),generator=generator)]
            hashes.append(hashlib.sha256(a["ids"][order].tobytes()).hexdigest())
        out["training_order_hashes"][str(seed)]=hashes
    for condition in c["dataset"]["conditions"]:
        d=a if condition=="independent" else b
        for seed in c["training"]["seeds"]:
            run=f"{condition}_s{seed}"
            cpdir=attempt_dir("checkpoints")/run
            final=torch.load(cpdir/f"step_{frozen['endpoint']:04d}.pt",weights_only=False,map_location="cpu")
            initial=torch.load(attempt_dir("checkpoints")/f"initial_seed_{seed}.pt",weights_only=True)
            cp0=torch.load(cpdir/"step_0000.pt",weights_only=False,map_location="cpu")
            assert all(torch.equal(initial[k],cp0["model"][k]) for k in initial)
            assert final["step"]==frozen["endpoint"]
            assert all(float(s["step"])==frozen["endpoint"] for s in final["optimizer"]["state"].values())
            assert all(v.dtype==torch.float32 for v in final["model"].values())
            assert any(not torch.equal(initial[k],final["model"][k]) for k in initial)
            for step in [0]+frozen["checkpoint_updates"]:
                cp=torch.load(cpdir/f"step_{step:04d}.pt",weights_only=False,map_location="cpu")
                assert cp["source_hashes"]==sources and cp["config_hash"]==frozen["configuration_sha256"]
            measurements=pd.read_csv(folder/run/"measurements_64.csv")
            assert len(measurements)==14 and measurements.valid.all()
            assert set(measurements.step)==set([0]+frozen["checkpoint_updates"])
            assert measurements.groupby(["step","panel"]).size().eq(1).all()
            assert (measurements.epsilon_fraction<.01).all()
            maximum_share_sum_error=float(np.abs(measurements[["shallow_share","middle_share","deep_share"]].sum(axis=1)-1).max())
            assert maximum_share_sum_error<1e-8
            out["runs"][run]={"updates_verified_from_Adam_state":frozen["endpoint"],"initial_state_matches_paired_seed":True,
                              "monitoring_points_verified":7,"max_share_sum_error":maximum_share_sum_error}
    # Installed official generator source must equal the pinned checkout, not just have the same API.
    from spuco.datasets import SpuCoMNIST
    installed=Path(inspect.getfile(SpuCoMNIST)).parents[1]
    vendor=ROOT/"tools/SpuCo/src/spuco"
    installed_count=0
    for p in vendor.rglob("*.py"):
        target=installed/p.relative_to(vendor)
        assert target.exists() and sha(target)==sha(p),str(p)
        installed_count+=1
    out["installed_spuco_source_files_verified"]=installed_count
    out["model_parameter_counts"]={n:p.numel() for n,p in CNNA().named_parameters()}
    out["passed"]=True
    write_json(folder/"artifact_audit.json",out)
    print("ARTIFACT AUDIT PASSED",json.dumps(out,indent=2))

if __name__=="__main__":main()
