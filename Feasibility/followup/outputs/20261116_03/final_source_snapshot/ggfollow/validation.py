"""Independent end-product checks without training or modifying legacy outputs."""
from .core import *
from .partitions import validate_partition
import pandas as pd
import re
import subprocess
import inspect

def same_rng(a,b):
    return a["python"]==b["python"] and a["numpy"][0]==b["numpy"][0] and np.array_equal(a["numpy"][1],b["numpy"][1]) and a["numpy"][2:]==b["numpy"][2:] and torch.equal(a["torch"],b["torch"]) and all(torch.equal(x,y) for x,y in zip(a["cuda"],b["cuda"]))

def validate():
    require_protocol(); t=out()/"reports/tables"; checks={}
    frame=pd.read_csv(out()/"saved/checkpoint_measurements.csv")
    assert len(frame)==8640 and frame.cache_identity.nunique()==8640
    assert set(frame.partition)==set(range(41000,41020)) and set(frame.budget)=={64,128,256}
    steps=read_json(legacy("runs")/"frozen_config.json")["checkpoint_updates"]
    assert set(frame.step)==set(steps) and frame.valid.all()
    assert frame.groupby(["run","step","partition","budget"]).size().eq(4).all()
    assert frame[frame.panel=="A"].groupby(["run","step","partition","budget"]).size().eq(3).all()
    assert frame[frame.panel=="B"]["repeat"].eq(0).all()
    checks["complete_saved_factorial_design_rows"]=len(frame)
    d=torch.load(legacy("data")/"independent.pt",map_location="cpu",weights_only=False)
    for seed in range(41000,41020):
        p=out()/"saved/manifests"/f"partition_{seed}.csv"
        validate_partition(pd.read_csv(p),d)
        assert frame.loc[frame.partition==seed,"manifest_sha256"].eq(sha(p)).all()
    checks["partition_manifests_verified"]=20
    original_ids=read_json(out()/"preflight/checkpoint_hashes.json")
    cpmap={(r["run"],r["step"]):r["sha256"] for r in original_ids}
    assert all(cpmap[(r.run,r.step)]==r.checkpoint_sha256 for r in frame[["run","step","checkpoint_sha256"]].drop_duplicates().itertuples(index=False))
    checks["checkpoint_cache_keys_verified"]=36
    ft=pd.read_csv(t/"saved_eight_features.csv")
    assert len(ft)==3024
    err=0.
    for seed in (41000,41009,41019):
        for (run,budget,panel),group in frame[(frame.partition==seed)&(frame["repeat"]==0)].groupby(["run","budget","panel"]):
            for prefix in (.05,.1):
                x=group[(group.step>0)&(group.fraction<=prefix+1e-12)].sort_values("step")
                row=ft[(ft.run==run)&(ft.budget==budget)&(ft.panel==panel)&(ft.partition==seed)&(ft["repeat"]==0)&(ft.prefix==prefix)].iloc[0]
                for block in ("shallow","deep"):
                    v=x[block+"_share"].to_numpy(); xx=x.fraction.to_numpy()
                    expected=[v[-1],sum(v)/len(v),np.sqrt(sum((v-v.mean())**2)/len(v)),np.polyfit(xx,v,1)[0]]
                    for stat,value in zip(("last","mean","std","slope"),expected): err=max(err,abs(row[block+"_"+stat]-value))
    assert err<1e-9; checks["independent_feature_recalculation_max_error"]=float(err)
    summary=pd.read_csv(t/"saved_partition_distributions.csv")
    assert len(summary)==48 and summary.actual_partitions.eq(20).all() and summary.valid_partitions.eq(20).all()
    target=summary[(summary.budget==256)&(summary.prefix==.1)&(summary.feature=="deep_slope")].iloc[0]
    decision=read_json(out()/"saved/decision.json")
    assert int(target.passing_partitions)==decision["target_summary"]["passing_partitions"]
    checks["primary_target_passes"]=int(target.passing_partitions)
    h=pd.read_csv(out()/"calibration/health_trajectories.csv")
    assert len(h)==9 and set(h.seed)=={101,102,103} and set(h.actual_epochs)=={2.,4.,6.}
    qualifying=[]
    for horizon,epoch in ((20,2),(40,4),(60,6)):
        hh=h[h.actual_epochs==epoch]
        if len(hh)==3 and (hh.health_accuracy>=.8).all() and (hh.health_loss<=np.log(10)).all() and hh.finite_weights.all() and hh.required_telemetry_finite.all(): qualifying.append(horizon)
    selected=min(qualifying) if qualifying else None
    assert selected==read_json(out()/"calibration/selection.json")["selected_full_epochs"]
    checks["independent_health_rule_selection"]=selected
    for seed in (101,102,103):
        cp=torch.load(out()/"checkpoints/calibration"/f"cal_s{seed}"/"step_1956.pt",weights_only=False,map_location="cpu")
        assert cp["step"]==1956 and all(float(v["step"])==1956 for v in cp["optimizer"]["state"].values())
        assert all(torch.isfinite(v).all() for v in cp["model"].values())
    checks["calibration_models_completed_six_epochs"]=3
    if selected is None:
        assert not read_json(out()/"fresh/frozen.json")["launched"]
        assert not list((out()/"checkpoints/fresh").rglob("*.pt"))
        planned=pd.read_csv(t/"fresh_not_launched_prefixes.csv")
        assert len(planned)==18 and planned.launched.eq(False).all() and planned.health_accuracy.isna().all()
        checks["predefined_fresh_gate_respected"]=True
    # Directly test the new checkpoint loader, then the unchanged probe helper, without constructing any optimizer.
    from .measuring import load_model
    from .partitions import indices
    from measurement import measure
    seed_all(73019); before=rng_state()
    model=load_model(legacy("checkpoints")/"independent_s11/step_0652.pt")
    assert same_rng(before,rng_state())
    params={k:v.clone() for k,v in model.state_dict().items()}; modes=[m.training for m in model.modules()]
    ix=indices(d,"A",256); observations=[measure(model,d["x"][ix],d["y"][ix],128) for _ in range(3)]
    assert same_rng(before,rng_state()) and modes==[m.training for m in model.modules()]
    assert all(torch.equal(params[k],v) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    error=max(abs(row[k]-observations[0][k]) for row in observations for k in ("shallow_share","middle_share","deep_share","total"))
    assert error==0.; checks["new_loader_and_probe_isolation_verified"]=True; checks["direct_repeat_max_error"]=error
    from spuco.datasets import SpuCoMNIST
    installed=Path(inspect.getfile(SpuCoMNIST)).parents[1]
    old=read_json(legacy("runs")/"provenance.json")
    for rel,hsh in old["spuco_relevant_hashes"].items():
        p=installed/Path(rel).relative_to(Path("src/spuco"))
        assert sha(p)==hsh,str(p)
    checks["installed_spuco_sources_unchanged"]=len(old["spuco_relevant_hashes"])
    current=subprocess.check_output([sys.executable,"-m","pip","freeze","--all"],text=True)
    assert set(current.splitlines())==set((legacy("runs")/"requirements-lock.txt").read_text().splitlines())
    checks["dependency_lock_unchanged"]=True
    review=read_json(out()/"reports/structural_review.json")
    assert all(sha(t/name)==value for name,value in review["evidence_hashes"].items())
    checks["structural_review_evidence_hashes_match"]=True
    path=out()/"reports/REPORT.md"
    for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
        assert target.startswith("https://") or (path.parent/target).exists(),target
    checks["report_links_valid"]=True
    from .artifacts import protect
    count=protect(after=True); checks["legacy_files_unchanged"]=count
    write_json(out()/"reports/validation_audit.json",{"passed":True,"checks":checks,"precision_settings":{"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
        "matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32}})
    log("output_validation_passed",checks=len(checks),legacy_files=count)
