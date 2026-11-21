"""Parameterised, deterministic new training. Never writes a legacy attempt."""
from .core import *
from .datasets import build,load_new,schedule
from .partitions import indices,make_manifests,validate_matched
from .metrics import enrich
from model import CNNA
from measurement import health,measure
from torch.nn.functional import cross_entropy
import pandas as pd
import math

def train_indices(data): return torch.tensor(np.flatnonzero(np.asarray(data["roles"])=="train"),dtype=torch.long)
def epoch_order(data,seed,epoch):
    ix=train_indices(data); g=torch.Generator(device="cpu").manual_seed(seed*100000+epoch)
    return ix[torch.randperm(len(ix),generator=g)]
def make_optimizer(model):
    c=cfg()["training"]
    return torch.optim.Adam(model.parameters(),lr=c["lr"],betas=tuple(c["betas"]),weight_decay=c["weight_decay"],foreach=False,fused=False)
def update(model,optimizer,x,y):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss=cross_entropy(model(x),y)
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training CE")
    loss.backward(); optimizer.step()
    return loss.item()

def source_health(model,data,step,updates_per_epoch,last_loss):
    ix=torch.tensor(np.flatnonzero(np.asarray(data["roles"])=="health"),dtype=torch.long)
    h,p=health(model,data["x"][ix],data["y"][ix])
    weights_finite=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    finite=bool(weights_finite and np.isfinite([h["loss"],h["accuracy"],last_loss]).all() and torch.isfinite(p).all())
    return {"step":step,"actual_epochs":step/updates_per_epoch,"health_loss":h["loss"],"health_accuracy":h["accuracy"],
            "health_seconds":h["seconds"],"finite_weights":weights_finite,"required_telemetry_finite":finite,
            "eligible":bool(finite and h["accuracy"]>=.8 and h["loss"]<=math.log(10))}

def initial(seed):
    path=out()/"checkpoints/initial"/f"seed_{seed}.pt"
    if not path.exists():
        state=rng_state()
        try:
            seed_all(seed); save_torch(path,CNNA().state_dict())
        finally: restore_rng(state)
    return torch.load(path,weights_only=True,map_location="cpu"),sha(path)

def checkpoint(path,model,optimizer,step,stats,health_rows,identity):
    save_torch(path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"rng":rng_state(),
                    "stats":stats,"health_rows":health_rows,"identity":identity})

def recolour(model,d):
    roles=np.asarray(d["roles"]); rank=np.asarray(d["ranks"])
    ix=torch.tensor(np.flatnonzero((roles=="health")&(rank<32)),dtype=torch.long)
    raw=d["raw"][ix].float().div(255).unsqueeze(1).expand(-1,3,-1,-1); x=d["x"][ix]; y=d["y"][ix]
    bg=(d["background"][ix]+1)%10; colors=torch.tensor(d["palette"][bg],dtype=x.dtype)[:,:,None,None].expand_as(x)
    alt=torch.where(raw==0,colors,raw); assert torch.equal(x[raw!=0],alt[raw!=0])
    start=time.perf_counter(); h,p=health(model,x,y); h2,p2=health(model,alt,y)
    return {"n":len(ix),"foreground_preserved":True,"rule":"original background+1 mod10; original nonzero foreground and labels unchanged",
            "flip_rate":(p.argmax(1)!=p2.argmax(1)).float().mean().item(),
            "true_class_probability_change":(p2[torch.arange(len(y)),y]-p[torch.arange(len(y)),y]).mean().item(),
            "original_accuracy":h["accuracy"],"alternative_accuracy":h2["accuracy"],"seconds":time.perf_counter()-start}

def train_run(data,stage,run,seed,plan,health_steps,monitor,plan_hash,data_hash):
    require_protocol(); dev=device(); folder=out()/stage/"runs"/run; folder.mkdir(parents=True,exist_ok=True)
    cpdir=out()/"checkpoints"/stage/run; cpdir.mkdir(parents=True,exist_ok=True)
    numerical=digest({k:v for k,v in require_protocol()["extension_source_hashes"].items() if k in ("ggfollow/core.py","ggfollow/training.py","ggfollow/datasets.py")})
    identity={"protocol_sha256":sha(PROTOCOL_PATH),"plan_sha256":plan_hash,"numerical_sha256":numerical,"data_sha256":data_hash,"seed":seed}
    if (folder/"complete.json").exists():
        done=read_json(folder/"complete.json"); assert done["identity"]==identity; return done
    seed_all(seed); model=CNNA().to(dev); init,init_hash=initial(seed); model.load_state_dict(init); optimizer=make_optimizer(model)
    step=0; rows=[]; stats={"training_seconds":0.,"primary_panel_monitor_seconds":0.,"health_seconds":0.,"recolour_seconds":0.,
                           "checkpoint_seconds":0.,"warmup_seconds":0.,"max_gpu_allocated":0,"max_gpu_reserved":0}
    previous=sorted(cpdir.glob("step_*.pt"),key=lambda p:int(p.stem.split("_")[1]))
    if previous:
        state=torch.load(previous[-1],weights_only=False,map_location="cpu"); assert state["identity"]==identity,"New-run resume guard mismatch"
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step=state["step"]; rows=state["health_rows"]; stats=state["stats"]; restore_rng(state["rng"])
        log("training_resumed",run=run,step=step)
    else:
        # Warm-up performs no update and uses the same isolation semantics as ordinary probes.
        ix=train_indices(data)[:128]; start=time.perf_counter()
        measure(model,data["x"][ix],data["y"][ix],128); sync(dev); stats["warmup_seconds"]=time.perf_counter()-start
    def observe(update_number,last_loss):
        if update_number in health_steps:
            h=source_health(model,data,update_number,plan["updates_per_epoch"],last_loss)
            stats["health_seconds"]+=h["health_seconds"]; rows.append(h)
            write_json(folder/"health.json",rows)
        if monitor and (update_number==0 or update_number in plan["steps"]):
            records=[]
            for budget in cfg()["fresh"]["original_panel_budgets"]:
                ix=indices(data,"A",budget); start=time.perf_counter()
                r=enrich(measure(model,data["x"][ix],data["y"][ix],128)); sync(dev)
                stats["primary_panel_monitor_seconds"]+=time.perf_counter()-start
                r.update(run=run,step=update_number,budget=budget,fraction=update_number/plan["horizon"],panel="A",repeat=0)
                records.append(r)
            path=folder/f"primary_step_{update_number:04d}.json"; write_json(path,records)
        log("training_checkpoint",experiment_stage=stage,run=run,step=update_number,endpoint=plan["endpoint"],
            actual_epochs=round(update_number/plan["updates_per_epoch"],4),training_seconds=round(stats["training_seconds"],3),
            health_accuracy=rows[-1]["health_accuracy"] if rows else None,
            estimated_training_seconds_remaining=stats["training_seconds"]*max(plan["endpoint"]-update_number,0)/max(update_number,1))
        write_json(folder/"status.json",{"state":"running","run":run,"step":update_number,"endpoint":plan["endpoint"],"identity":identity})
    if not previous:
        observe(0,math.log(10)); checkpoint(cpdir/"step_0000.pt",model,optimizer,0,stats,rows,identity)
    last_epoch=-1; order=None; wall_start=time.perf_counter()
    while step<plan["endpoint"]:
        epoch=step//plan["updates_per_epoch"]; within=step%plan["updates_per_epoch"]
        if epoch!=last_epoch: order=epoch_order(data,seed,epoch); last_epoch=epoch
        ix=order[within*128:(within+1)*128]
        sync(dev)
        if dev.type=="cuda": torch.cuda.reset_peak_memory_stats(dev)
        start=time.perf_counter(); xb=data["x"][ix].to(dev); yb=data["y"][ix].to(dev)
        loss=update(model,optimizer,xb,yb); sync(dev); stats["training_seconds"]+=time.perf_counter()-start
        if dev.type=="cuda":
            stats["max_gpu_allocated"]=max(stats["max_gpu_allocated"],torch.cuda.max_memory_allocated(dev))
            stats["max_gpu_reserved"]=max(stats["max_gpu_reserved"],torch.cuda.max_memory_reserved(dev))
        step+=1
        monitored=(step in health_steps) or (monitor and step in plan["steps"])
        if monitored: observe(step,loss)
        if monitored or step%cfg()["training"]["save_every_updates"]==0 or step==plan["endpoint"]:
            start=time.perf_counter(); checkpoint(cpdir/f"step_{step:04d}.pt",model,optimizer,step,stats,rows,identity)
            stats["checkpoint_seconds"]+=time.perf_counter()-start
    if monitor:
        r=recolour(model,data); write_json(folder/"offline_recolour.json",r); stats["recolour_seconds"]=r["seconds"]
    stats["training_loop_wall_seconds_this_invocation"]=time.perf_counter()-wall_start
    done={"completed":True,"stage":stage,"run":run,"seed":seed,"endpoint":step,"schedule":plan,"identity":identity,
          "initial_state_sha256":init_hash,"checkpoint_dir":str(cpdir),"health":rows,"timing":stats}
    write_json(folder/"complete.json",done); write_json(folder/"status.json",{"state":"completed","run":run,"step":step,"endpoint":step})
    log("training_run_completed",experiment_stage=stage,run=run,endpoint=step)
    return done

def select_horizon(health_rows):
    c=cfg()["calibration"]; selected=None; checks=[]
    for full_epochs,actual_epochs in zip(c["full_horizon_candidates"],c["health_epochs"]):
        for seed in c["seeds"]:
            found=[r for r in health_rows if r["seed"]==seed and r["actual_epochs"]==actual_epochs]
            good=bool(len(found)==1 and found[0]["health_accuracy"]>=c["health_accuracy"] and found[0]["health_loss"]<=math.log(10)
                      and found[0]["required_telemetry_finite"] and found[0]["finite_weights"])
            checks.append({"full_epochs":full_epochs,"actual_epochs":actual_epochs,"seed":seed,"passes":good})
        if selected is None and all(r["passes"] for r in checks if r["full_epochs"]==full_epochs): selected=full_epochs
    return selected,checks

def calibrate():
    require_reproduction()
    with cost("source_health_calibration"):
        c=cfg()["calibration"]
        data_folder=build("calibration",c["data_seed"],c["reservation_seed"],{"independent":.1})
        d=load_new(data_folder,"independent")
        plan=schedule(len(train_indices(d)),60)
        plan["endpoint"]=plan["updates_per_epoch"]*c["max_actual_epochs"]
        plan["label"]="development calibration trajectory; NOT completed 60-epoch training"
        write_once(out()/"calibration/plan.json",{"schedule":plan,"seeds":c["seeds"],"selection_rule":c["selection"],"source":"protocol frozen before execution"})
        records=[]
        for seed in c["seeds"]:
            result=train_run(d,"calibration",f"cal_s{seed}",seed,plan,[e*plan["updates_per_epoch"] for e in c["health_epochs"]],False,
                             digest(read_json(out()/"calibration/plan.json")),sha(data_folder/"independent.pt"))
            records.extend([{**r,"seed":seed} for r in result["health"]])
        pd.DataFrame(records).to_csv(out()/"calibration/health_trajectories.csv",index=False)
        selected,checks=select_horizon(records)
        write_json(out()/"calibration/selection.json",{"selected_full_epochs":selected,"source_health_coverage_resolved":selected is not None,
                   "checks":checks,"rule":c["selection"],"used_for_selection":["health_accuracy","health_loss","finite_weights","required_telemetry_finite"],
                   "gradient_reliability_used":False,"recolouring_used":False,"candidate_grid_expanded":False})
        log("calibration_completed",selected_full_epochs=selected)

def freeze_fresh():
    require_reproduction(); selection=read_json(out()/"calibration/selection.json")
    selected=selection["selected_full_epochs"]
    if selected is None:
        write_once(out()/"fresh/frozen.json",{"launched":False,"reason":"No frozen horizon candidate passes all three calibration seeds","selection_sha256":sha(out()/"calibration/selection.json")})
        log("fresh_not_launched_calibration_gate"); return False
    expected_n=read_json(legacy("runs")/"frozen_config.json")["train_n"]
    realisations=[]; runs=[]
    for i,(data_seed,reservation_seed) in enumerate(cfg()["fresh"]["construction_pairs"]):
        name=f"realisation_{i+1}"; seeds=[base+1000*i for base in cfg()["fresh"]["seed_bases"]]
        realisations.append({"name":name,"data_seed":data_seed,"reservation_seed":reservation_seed,"training_seeds":seeds,"schedule":schedule(expected_n,selected)})
        for condition in ("independent","correlated"):
            for seed in seeds:
                run=f"r{i+1}_{condition}_s{seed}"
                runs.append({"run":run,"realisation":name,"condition":condition,"seed":seed,"checkpoint_dir":str(out()/"checkpoints/fresh"/run)})
    plan={"launched":True,"full_epochs":selected,"protocol_amendment":selected!=20,"rationale":"Smallest predefined full horizon passing all three calibration seeds at its 10% point, based only on source health.",
          "selection_sha256":sha(out()/"calibration/selection.json"),"legacy_full_epochs":20,"legacy_endpoint_updates":652,
          "realisations":realisations,"runs":runs,"partition_seeds":cfg()["fresh"]["partition_seeds"],"primary_budget":256,
          "expected_all_runs":18,"coverage_rule":"At least 2 of 3 seeds healthy in each condition of every realisation",
          "no_raw_slope_pooling_across_horizons":True,"source_hashes":require_protocol()["extension_source_hashes"]}
    write_once(out()/"fresh/frozen.json",plan)
    log("fresh_protocol_frozen",full_epochs=selected,prefix_actual_epochs=selected*.1,runs=18)
    return True

def fresh():
    require_reproduction(); plan=read_json(out()/"fresh/frozen.json")
    if not plan["launched"]: log("fresh_skipped_calibration_gate"); return
    with cost("fresh_realisation_training"):
        for item in plan["realisations"]:
            name=item["name"]; folder=build(name,item["data_seed"],item["reservation_seed"],{"independent":.1,"correlated":.95})
            a=load_new(folder,"independent"); b=load_new(folder,"correlated"); validate_matched(a,b)
            # Freeze all partition memberships before any model in this realisation is trained/measured.
            make_manifests(a,out()/"fresh/manifests"/name,plan["partition_seeds"]); del a,b
            for condition in ("independent","correlated"):
                d=load_new(folder,condition); assert len(train_indices(d))==item["schedule"]["train_n"]
                for spec in [r for r in plan["runs"] if r["realisation"]==name and r["condition"]==condition]:
                    train_run(d,"fresh",spec["run"],spec["seed"],item["schedule"],item["schedule"]["steps"],True,
                              digest(plan),sha(folder/(condition+".pt")))
                del d
