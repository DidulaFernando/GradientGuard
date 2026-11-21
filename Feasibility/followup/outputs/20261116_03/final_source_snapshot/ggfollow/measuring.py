"""Checkpoint-only work. This module never constructs an optimiser or performs updates."""
from .core import *
from .metrics import enrich,features,reliability
from .partitions import make_manifests,validate_matched,indices
from model import CNNA
from measurement import measure
import pandas as pd
import sqlite3

class ScalarCache:
    def __init__(self,name):
        folder=out()/name; folder.mkdir(parents=True,exist_ok=True)
        self.path=folder/"scalar_cache.sqlite"
        self.db=sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    def get(self,key):
        result=self.db.execute("SELECT payload FROM results WHERE identity=?",(digest(key),)).fetchone()
        return json.loads(result[0]) if result else None
    def put(self,key,row):
        row={**row,"cache_identity":digest(key)}
        self.db.execute("INSERT INTO results VALUES (?,?)",(digest(key),json.dumps(row,allow_nan=False)))
        self.db.commit()
        return row
    def frame(self):
        return pd.DataFrame([json.loads(r[0]) for r in self.db.execute("SELECT payload FROM results ORDER BY identity")])
    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); self.db.close()

def numerical_identity():
    f=require_protocol()
    return digest({k:v for k,v in f["extension_source_hashes"].items() if k in ("ggfollow/core.py","ggfollow/metrics.py","ggfollow/partitions.py","ggfollow/measuring.py")})

def load_model(path):
    state=rng_state()
    try:
        cp=torch.load(path,map_location="cpu",weights_only=False)
        model=CNNA().to(device()); model.load_state_dict(cp["model"])
        return model
    finally: restore_rng(state)

def measured(cache,model,d,ix,key):
    existing=cache.get(key)
    if existing is not None: return existing
    start=time.perf_counter()
    xb,yb=d["x"][ix],d["y"][ix]
    # Legacy helper recomputes signed, sample-weighted mean gradients; scalars are not recycled across panels.
    row=enrich(measure(model,xb,yb,cfg()["measurement"]["microbatch"]))
    sync(next(model.parameters()).device)
    row["including_indexing_seconds"]=time.perf_counter()-start
    row.update(key)
    return cache.put(key,row)

def original_schedule():
    f=read_json(legacy("runs")/"frozen_config.json")
    return {"steps":f["checkpoint_updates"],"horizon":f["full_planned_updates"],"endpoint":f["endpoint"],"updates_per_epoch":f["updates_per_epoch"]}

def legacy_table(size):
    return pd.read_csv(legacy("runs")/"reliability"/f"measurements_{size}.csv").rename(columns={"probe_per_class":"budget"})

def reproduce_arithmetic():
    from analysis import feature_table as legacy_features,reliability as legacy_reliability
    folder=out()/"reproduction"; folder.mkdir(exist_ok=True)
    s=original_schedule(); maximum=0.; ratios=[]
    for budget in (64,128,256):
        t=legacy_table(budget)
        oldft=legacy_features(t.rename(columns={"budget":"probe_per_class"}))
        newft=features(t,s)
        names=[b+"_"+x for b in ("shallow","deep") for x in ("last","mean","std","slope")]
        keys=["run","panel","repeat","prefix"]
        merged=newft.merge(oldft,on=keys,suffixes=("_new","_old"),validate="one_to_one")
        for n in names:
            error=np.max(abs(merged[n+"_new"]-merged[n+"_old"])); maximum=max(maximum,float(error))
            np.testing.assert_allclose(merged[n+"_new"],merged[n+"_old"],atol=1e-12,rtol=1e-12)
        rel=reliability(newft,s,sorted(t.run.unique()))
        original=legacy_reliability(oldft)
        for column_old,column_new in (("noise","noise"),("spread","spread"),("screen_ratio","screen_ratio"),("numerical_floor","numerical_floor")):
            a=original.sort_values(["prefix","feature"])[column_old]
            b=rel.sort_values(["prefix","feature"])[column_new]
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
        saved=pd.read_csv(legacy("reports")/"tables"/f"reliability_{budget}.csv").sort_values(["prefix","feature"])
        np.testing.assert_allclose(saved.screen_ratio,rel.sort_values(["prefix","feature"]).screen_ratio,atol=1e-12,rtol=1e-12)
        newft.to_csv(folder/f"recalculated_features_{budget}.csv",index=False)
        rel["budget"]=budget; ratios.append(rel)
    result=pd.concat(ratios,ignore_index=True); result.to_csv(folder/"recalculated_reliability.csv",index=False)
    ratio=float(result[(result.budget==256)&(result.prefix==.1)&(result.feature=="deep_slope")].screen_ratio.iloc[0])
    assert abs(ratio-3.0501409990602664)<1e-10
    return {"ratio":ratio,"maximum_feature_formula_error":maximum}

def reproduce():
    require_protocol(); assert read_json(out()/"checks/status.json")["passed"]
    with cost("original_result_reproduction"):
        arithmetic=reproduce_arithmetic(); s=original_schedule(); cache=ScalarCache("reproduction")
        numeric=numerical_identity(); env=digest(read_json(out()/"preflight/environment.json")["hardware"])
        errors=[]; start=time.perf_counter()
        for condition in ("independent","correlated"):
            d=torch.load(legacy("data")/(condition+".pt"),weights_only=False,map_location="cpu")
            datahash=sha(legacy("data")/(condition+".pt"))
            for seed in (11,22,33):
                run=f"{condition}_s{seed}"
                # Selected parity checkpoints first, then the remaining frozen monitoring points only.
                for step in [s["steps"][0],s["steps"][-1]]+[x for x in s["steps"][1:-1]]:
                    cp=legacy("checkpoints")/run/f"step_{step:04d}.pt"; model=load_model(cp); cp_hash=sha(cp)
                    for budget in (64,128,256):
                        old=legacy_table(budget)
                        for panel in ("A","B"):
                            ix=indices(d,panel,budget)
                            for rep in range(cfg()["measurement"]["repeats_A"] if panel=="A" else 1):
                                key={"stage":"reproduction","realisation":"legacy","condition":condition,"seed":seed,"run":run,"step":step,
                                     "fraction":step/s["horizon"],"partition":0,"budget":budget,"panel":panel,"repeat":rep,
                                     "checkpoint_sha256":cp_hash,"data_sha256":datahash,"manifest_sha256":digest(d["ids"][ix].tolist()),
                                     "protocol_sha256":sha(PROTOCOL_PATH),"numerical_source_sha256":numeric,"environment_sha256":env}
                                row=measured(cache,model,d,ix,key)
                                reference=old[(old.run==run)&(old.step==step)&(old.panel==panel)&(old["repeat"]==0)].iloc[0]
                                names=[n for n in reference.index if n.endswith("_share") or n.endswith("_abs") or n.endswith("_signed") or n.endswith("_bias_sensitivity")]+["total","probe_loss","probe_accuracy"]
                                for name in names:
                                    np.testing.assert_allclose(row[name],reference[name],atol=1e-6,rtol=1e-4)
                                    if name in ("shallow_share","middle_share","deep_share"): assert abs(row[name]-reference[name])<=1e-5
                                errors.append({"run":run,"step":step,"budget":budget,"panel":panel,"repeat":rep,
                                               "max_primary_share_error":max(abs(row[n]-reference[n]) for n in ("shallow_share","middle_share","deep_share")),
                                               "max_total_error":abs(row["total"]-reference["total"])})
                    del model
                log("reproduced_run",run=run)
            del d
        frame=cache.frame(); cache.close(); frame.to_csv(out()/"reproduction/remeasured_checkpoints.csv",index=False)
        ft=features(frame,s); ft.to_csv(out()/"reproduction/remeasured_eight_features.csv",index=False)
        rerel=[]; max_feature_error=0.
        for budget,group in ft.groupby("budget"):
            ref=features(legacy_table(budget),s)
            merged=group.merge(ref,on=["run","panel","repeat","prefix"],suffixes=("_new","_old"))
            for name in [b+"_"+x for b in ("shallow","deep") for x in ("last","mean","std","slope")]:
                np.testing.assert_allclose(merged[name+"_new"],merged[name+"_old"],atol=1e-6,rtol=1e-4)
                max_feature_error=max(max_feature_error,float(np.max(abs(merged[name+"_new"]-merged[name+"_old"]))))
            rel=reliability(group,s,sorted(frame.run.unique())); rel["budget"]=budget; rerel.append(rel)
        re=pd.concat(rerel); re.to_csv(out()/"reproduction/remeasured_reliability.csv",index=False)
        ratio=float(re[(re.budget==256)&(re.prefix==.1)&(re.feature=="deep_slope")].screen_ratio.iloc[0])
        np.testing.assert_allclose(ratio,arithmetic["ratio"],atol=1e-6,rtol=1e-4)
        pd.DataFrame(errors).to_csv(out()/"reproduction/measurement_parity_errors.csv",index=False)
        write_json(out()/"reproduction/audit.json",{"passed":True,**arithmetic,"remeasured_ratio":ratio,"maximum_remeasured_feature_error":max_feature_error,
                    "maximum_primary_share_error":max(r["max_primary_share_error"] for r in errors),"measurement_rows":len(frame),"seconds":time.perf_counter()-start,
                    "checkpoints":s["steps"],"optimizer_constructed":False,"training_updates":0})
        log("original_reproduction_passed",original_ratio=arithmetic["ratio"],remeasured_ratio=ratio)

def measure_partitions(stage,data_pairs,run_specs,manifest_folders,budgets,seeds,schedules):
    require_reproduction(); cache=ScalarCache(stage); numeric=numerical_identity()
    environment=digest(read_json(out()/"preflight/environment.json")["hardware"])
    start=time.perf_counter(); initial=len(cache.frame())
    for realisation,condition,path in data_pairs:
        d=torch.load(path,weights_only=False,map_location="cpu"); datahash=sha(path)
        manifests={seed:(None if seed==0 else pd.read_csv(manifest_folders[realisation]/f"partition_{seed}.csv")) for seed in seeds}
        memberships={(seed,budget,panel):indices(d,panel,budget,manifests[seed]) for seed in seeds for budget in budgets for panel in ("A","B")}
        manifest_hashes={seed:(digest({"ids":d["ids"].tolist(),"roles":d["roles"],"ranks":d["ranks"]}) if seed==0 else sha(manifest_folders[realisation]/f"partition_{seed}.csv")) for seed in seeds}
        for spec in [r for r in run_specs if r["realisation"]==realisation and r["condition"]==condition]:
            schedule=schedules[realisation]
            for step in schedule["steps"]:
                cp=Path(spec["checkpoint_dir"])/f"step_{step:04d}.pt"; model=load_model(cp); cp_hash=sha(cp)
                for seed in seeds:
                    for budget in budgets:
                        for panel in ("A","B"):
                            ix=memberships[(seed,budget,panel)]
                            for rep in range(cfg()["measurement"]["repeats_A"] if panel=="A" else 1):
                                key={"stage":stage,"realisation":realisation,"condition":condition,"seed":spec["seed"],"run":spec["run"],
                                     "step":step,"fraction":step/schedule["horizon"],"absolute_epochs":step/schedule["updates_per_epoch"],
                                     "partition":seed,"budget":budget,"panel":panel,"repeat":rep,
                                     "checkpoint_sha256":cp_hash,"data_sha256":datahash,"manifest_sha256":manifest_hashes[seed],
                                     "protocol_sha256":sha(PROTOCOL_PATH),"numerical_source_sha256":numeric,"environment_sha256":environment}
                                measured(cache,model,d,ix,key)
                count=cache.db.execute("SELECT count(*) FROM results").fetchone()[0]
                elapsed=time.perf_counter()-start
                log(stage+"_checkpoint",run=spec["run"],step=step,rows=count,elapsed_seconds=round(elapsed,2),seconds_per_new_row=round(elapsed/max(count-initial,1),5))
                del model
        del d
    frame=cache.frame(); cache.close()
    frame.to_csv(out()/stage/"checkpoint_measurements.csv",index=False)
    write_json(out()/stage/"complete.json",{"completed":True,"rows":len(frame),"seconds_this_invocation":time.perf_counter()-start,"partition_seeds":list(seeds),"budgets":list(budgets)})
    return frame

def saved_partitions(manifests_only=False):
    require_reproduction()
    with cost("saved_partitions_manifests" if manifests_only else "saved_checkpoint_partitions"):
        a=torch.load(legacy("data")/"independent.pt",weights_only=False,map_location="cpu")
        b=torch.load(legacy("data")/"correlated.pt",weights_only=False,map_location="cpu"); validate_matched(a,b)
        folder=out()/"saved/manifests"; seeds=cfg()["stage_A"]["partition_seeds"]
        make_manifests(a,folder,seeds); del a,b
        if manifests_only: return
        specs=[{"run":f"{condition}_s{seed}","condition":condition,"seed":seed,"realisation":"legacy","checkpoint_dir":str(legacy("checkpoints")/f"{condition}_s{seed}")}
               for condition in ("independent","correlated") for seed in (11,22,33)]
        pairs=[("legacy",c,legacy("data")/(c+".pt")) for c in ("independent","correlated")]
        measure_partitions("saved",pairs,specs,{"legacy":folder},cfg()["measurement"]["budgets"],seeds,{"legacy":original_schedule()})

def fresh_partitions():
    require_reproduction()
    plan=read_json(out()/"fresh/frozen.json")
    if not plan["launched"]:
        log("fresh_measurement_skipped_calibration_gate"); return
    with cost("fresh_checkpoint_partitions"):
        specs=plan["runs"]; pairs=[]; folders={}; schedules={}
        for item in plan["realisations"]:
            name=item["name"]; path=out()/"data"/name
            assert all((out()/"fresh/runs"/r["run"]/"complete.json").exists() for r in specs if r["realisation"]==name)
            a=torch.load(path/"independent.pt",weights_only=False,map_location="cpu"); b=torch.load(path/"correlated.pt",weights_only=False,map_location="cpu")
            validate_matched(a,b); folder=out()/"fresh/manifests"/name
            make_manifests(a,folder,cfg()["fresh"]["partition_seeds"]); del a,b
            folders[name]=folder; schedules[name]=item["schedule"]
            pairs.extend([(name,c,path/(c+".pt")) for c in ("independent","correlated")])
        measure_partitions("fresh_validation",pairs,specs,folders,[256],cfg()["fresh"]["partition_seeds"],schedules)
        with cost("fresh_original_panel_context_validation"):
            context=measure_partitions("fresh_original",pairs,specs,folders,cfg()["fresh"]["original_panel_budgets"],[0],schedules)
            parity=[]
            for spec in specs:
                for path in (out()/"fresh/runs"/spec["run"]).glob("primary_step_*.json"):
                    for row in read_json(path):
                        if row["step"]==0: continue
                        r=context[(context.run==spec["run"])&(context.step==row["step"])&(context.budget==row["budget"])&(context.panel=="A")&(context["repeat"]==0)].iloc[0]
                        error=max(abs(r[n]-row[n]) for n in ("shallow_share","middle_share","deep_share"))
                        assert error<1e-5
                        parity.append({"run":spec["run"],"step":row["step"],"budget":row["budget"],"max_online_offline_share_error":error})
            pd.DataFrame(parity).to_csv(out()/"fresh_original/online_offline_parity.csv",index=False)
