import copy
import json
import math
import time
from pathlib import Path
import pandas as pd
from common import ROOT, config, attempt_dir, torch, np, seed_all, deterministic, device, rng_state, restore_rng, sync, write_json, save_torch, sha, source_hashes
from model import CNNA
from data import load, panel_indices, data_dir
from measurement import measure, health
from torch.nn.functional import cross_entropy

def frozen():
    return json.loads((attempt_dir()/"frozen_config.json").read_text())

def check_gates():
    status=json.loads((attempt_dir()/"correctness_status.json").read_text())
    assert status["passed"], "Correctness must pass before pilot"
    assert status["source_hashes"] == source_hashes(), "Code changed after correctness; rerun checks"
    smoke=json.loads((attempt_dir()/"smoke/status.json").read_text())
    assert smoke["passed"] and smoke["config_hash"]==sha(ROOT/"configs/pilot.yaml")

def checkpoints_dir(run):
    p=attempt_dir("checkpoints")/run
    p.mkdir(parents=True,exist_ok=True)
    return p

def save_checkpoint(path,model,opt,step,stats):
    save_torch(path,{"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,
                    "rng":rng_state(),"stats":stats,"config_hash":sha(ROOT/"configs/pilot.yaml"),
                    "source_hashes":source_hashes()})

def smoke():
    deterministic(); c=config(); seed_all(c["smoke"]["seed"])
    dev=device(); d=load("independent"); ix=panel_indices(d,"train")[:c["smoke"]["examples"]]
    x,y=d["x"][ix].to(dev),d["y"][ix].to(dev)
    model=CNNA(len(c["dataset"]["classes"])).to(dev)
    reference=copy.deepcopy(model)
    opt=torch.optim.Adam(model.parameters(),lr=.001,foreach=False,fused=False)
    opt_ref=torch.optim.Adam(reference.parameters(),lr=.001,foreach=False,fused=False)
    rows=[]
    for step in range(c["smoke"]["updates"]):
        start=(step*128)%len(y)
        for m,o in ((model,opt),(reference,opt_ref)):
            m.train(); o.zero_grad(set_to_none=True)
            loss=cross_entropy(m(x[start:start+128]),y[start:start+128]); loss.backward(); o.step()
        row=measure(model,x[:73],y[:73],32); row["step"]=step+1; rows.append(row)
    err=max((model.state_dict()[k]-reference.state_dict()[k]).abs().max().item() for k in model.state_dict())
    assert err==0 and all(r["valid"] for r in rows)
    folder=attempt_dir()/"smoke"; folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/"SMOKE_ONLY_measurements.csv",index=False)
    save_torch(folder/"SMOKE_ONLY_final.pt",model.state_dict())
    write_json(folder/"status.json",{"passed":True,"label":"SMOKE ONLY - not feasibility evidence", "updates":len(rows),
                                    "training_neutrality_max_error":err,"config_hash":sha(ROOT/"configs/pilot.yaml"),"device":str(dev)})
    print("SMOKE PASSED: 12 real-data engineering updates; bitwise monitoring neutrality",flush=True)

def initial_state(seed):
    path=attempt_dir("checkpoints")/f"initial_seed_{seed}.pt"
    if not path.exists():
        seed_all(seed)
        save_torch(path,CNNA(len(config()["dataset"]["classes"])).state_dict())
    return torch.load(path,weights_only=True,map_location="cpu")

def pilot(only=None):
    deterministic(); check_gates(); c=config(); f=frozen(); dev=device()
    assert f["configuration_sha256"]==sha(ROOT/"configs/pilot.yaml")
    for seed in c["training"]["seeds"]:
        initial_state(seed)
    for condition in c["dataset"]["conditions"]:
        d=load(condition)
        train_ix=panel_indices(d,"train")
        for seed in c["training"]["seeds"]:
            run=f"{condition}_s{seed}"
            if only and run!=only: continue
            folder=attempt_dir()/run; folder.mkdir(exist_ok=True)
            if (folder/"complete.json").exists():
                print("ALREADY COMPLETE",run,flush=True); continue
            seed_all(seed); model=CNNA(len(c["dataset"]["classes"])).to(dev)
            model.load_state_dict(initial_state(seed))
            opt=torch.optim.Adam(model.parameters(),lr=c["training"]["lr"],betas=tuple(c["training"]["betas"]),
                                 weight_decay=c["training"]["weight_decay"],foreach=False,fused=False)
            cpdir=checkpoints_dir(run)
            existing=sorted(cpdir.glob("step_*.pt"),key=lambda p:int(p.stem.split("_")[1]))
            step=0
            stats={"training_seconds":0.,"primary_A_seconds":0.,"validation_B_seconds":0.,"health_seconds":0.,
                   "warmup_seconds":0.,"gpu_training_peak_allocated":0,"gpu_peak_reserved":0,"training_loss":[],"checkpoint_save_seconds":0.}
            if existing:
                state=torch.load(existing[-1],weights_only=False,map_location="cpu")
                assert state["config_hash"]==f["configuration_sha256"] and state["source_hashes"]==source_hashes(), "Unsafe resume: code/config changed"
                model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
                step=state["step"]; stats=state["stats"]; restore_rng(state["rng"])
                print("RESUMING",run,step,flush=True)
            else:
                start=time.perf_counter()
                # Warm up forward/backward paths without parameter updates and outside monitoring summaries.
                for _ in range(2):
                    measure(model,d["x"][train_ix[:128]],d["y"][train_ix[:128]],128)
                stats["warmup_seconds"]=time.perf_counter()-start
            rows_path=folder/"measurements_64.csv"
            rows=pd.read_csv(rows_path).to_dict("records") if rows_path.exists() else []
            # A restart discards only summary rows beyond the last committed checkpoint; retain a recovery copy.
            if any(r["step"]>step for r in rows):
                pd.DataFrame(rows).to_csv(folder/f"recovered_uncommitted_{time.time_ns()}.csv",index=False)
                rows=[r for r in rows if r["step"]<=step]
            def observe(update):
                for panel,statkey in (("A","primary_A_seconds"),("B","validation_B_seconds")):
                    ix=panel_indices(d,"probe_"+panel,64)
                    out=measure(model,d["x"][ix],d["y"][ix],c["measurement"]["microbatch"])
                    stats[statkey]+=out["seconds"]
                    out.update(run=run,condition=condition,seed=seed,step=update,fraction=update/f["full_planned_updates"],panel=panel,probe_per_class=64,repeat=0)
                    rows.append(out)
                hi=panel_indices(d,"health")
                h,_=health(model,d["x"][hi],d["y"][hi])
                stats["health_seconds"]+=h["seconds"]
                for out in rows[-2:]:
                    out.update(health_loss=h["loss"],health_accuracy=h["accuracy"],
                               eligible=bool(h["accuracy"]>=c["measurement"]["health_accuracy"] and h["loss"]<=math.log(len(c["dataset"]["classes"])) and out["valid"] and math.isfinite(h["loss"])))
                pd.DataFrame(rows).to_csv(rows_path,index=False)
                elapsed=stats["training_seconds"]
                write_json(folder/"status.json",{"state":"running","run":run,"step":update,"endpoint":f["endpoint"],"training_seconds":elapsed,"estimated_training_seconds_remaining":elapsed/max(update,1)*(f["endpoint"]-update)})
                print(f"{run} update={update}/{f['endpoint']} fraction={update/f['full_planned_updates']:.6f} health_acc={h['accuracy']:.4f} health_loss={h['loss']:.4f} train_s={elapsed:.2f}",flush=True)
            if not existing:
                observe(0)
                save_checkpoint(cpdir/"step_0000.pt",model,opt,0,stats)
            last_epoch=-1; order=None
            while step<f["endpoint"]:
                epoch=step//f["updates_per_epoch"]
                within=step%f["updates_per_epoch"]
                if epoch!=last_epoch:
                    generator=torch.Generator(device="cpu").manual_seed(seed*100000+epoch)
                    order=train_ix[torch.randperm(len(train_ix),generator=generator)]
                    last_epoch=epoch
                ix=order[within*c["training"]["batch_size"]:(within+1)*c["training"]["batch_size"]]
                sync(dev)
                if dev.type=="cuda": torch.cuda.reset_peak_memory_stats(dev)
                t=time.perf_counter()
                xb=d["x"][ix].to(dev); yb=d["y"][ix].to(dev)
                model.train(); opt.zero_grad(set_to_none=True)
                loss=cross_entropy(model(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite training loss {run}/{step}")
                loss.backward(); opt.step(); sync(dev)
                stats["training_seconds"]+=time.perf_counter()-t
                if dev.type=="cuda":
                    stats["gpu_training_peak_allocated"]=max(stats["gpu_training_peak_allocated"],torch.cuda.max_memory_allocated(dev))
                    stats["gpu_peak_reserved"]=max(stats["gpu_peak_reserved"],torch.cuda.max_memory_reserved(dev))
                step+=1
                if step%25==0:
                    stats["training_loss"].append([step,loss.item()])
                if step in f["checkpoint_updates"]:
                    observe(step)
                if step in f["checkpoint_updates"] or step%50==0:
                    t=time.perf_counter()
                    save_checkpoint(cpdir/f"step_{step:04d}.pt",model,opt,step,stats)
                    stats["checkpoint_save_seconds"]+=time.perf_counter()-t
            # Recolouring is offline only; not part of the feature export.
            recolor=recolour(model,d)
            write_json(folder/"offline_recolour.json",recolor)
            write_json(folder/"timing.json",stats)
            complete={"state":"completed","run":run,"step":step,"endpoint":f["endpoint"],"device":str(dev),"source_health_eligible":bool(rows[-1]["eligible"]),"config_hash":f["configuration_sha256"]}
            write_json(folder/"complete.json",complete); write_json(folder/"status.json",complete)
            print("COMPLETED",run,flush=True)
        del d

def recolour(model,d):
    c=config()["measurement"]; ix=panel_indices(d,"health",c["recolor_per_class"])
    x=d["x"][ix].clone(); y=d["y"][ix]; raw=d["raw"][ix].float().div(255).unsqueeze(1).expand(-1,3,-1,-1)
    bg=(d["background"][ix]+1)%len(config()["dataset"]["classes"])
    color=torch.tensor(d["palette"][bg],dtype=x.dtype)[:,:,None,None].expand_as(x)
    alt=torch.where(raw==0,color,raw)
    assert torch.equal(alt[raw!=0],x[raw!=0])
    h,p=health(model,x,y); h2,p2=health(model,alt,y)
    true=p[torch.arange(len(y)),y]; true2=p2[torch.arange(len(y)),y]
    return {"label":"OFFLINE MANIPULATION CHECK ONLY","n":len(y),"rule":"background index +1 modulo K; original nonzero foreground and labels preserved", "flip_rate":(p.argmax(1)!=p2.argmax(1)).float().mean().item(),
            "mean_true_class_probability_change":(true2-true).mean().item(),"original_accuracy":h["accuracy"],"alternative_accuracy":h2["accuracy"],"seconds":h["seconds"]+h2["seconds"],"foreground_preserved":True}

def remeasure(sizes=(64,), repeats=3):
    """Only saved checkpoints, no optimiser construction or updates."""
    deterministic(); f=frozen(); c=config(); dev=device()
    outdir=attempt_dir()/"reliability"; outdir.mkdir(exist_ok=True)
    for size in sizes:
        outfile=outdir/f"measurements_{size}.csv"
        rows=pd.read_csv(outfile).to_dict("records") if outfile.exists() else []
        done={(r["run"],int(r["step"]),r["panel"],int(r["repeat"])) for r in rows}
        for condition in c["dataset"]["conditions"]:
            d=load(condition)
            for seed in c["training"]["seeds"]:
                run=f"{condition}_s{seed}"
                if not (attempt_dir()/run/"complete.json").exists(): continue
                model=CNNA(len(c["dataset"]["classes"])).to(dev)
                original=pd.read_csv(attempt_dir()/run/"measurements_64.csv")
                for step in [0]+f["checkpoint_updates"]:
                    cp=torch.load(checkpoints_dir(run)/f"step_{step:04d}.pt",map_location="cpu",weights_only=False)
                    model.load_state_dict(cp["model"])
                    for panel in ("A","B"):
                        # Three repeated A measurements assess numerical jitter; B needs one for sampling.
                        for rep in range(repeats if panel=="A" else 1):
                            key=(run,step,panel,rep)
                            if key in done: continue
                            if size==64 and rep==0:
                                row=original[(original.step==step)&(original.panel==panel)].iloc[0].to_dict()
                            else:
                                ix=panel_indices(d,"probe_"+panel,size)
                                row=measure(model,d["x"][ix],d["y"][ix],c["measurement"]["microbatch"])
                                row.update(run=run,condition=condition,seed=seed,step=step,fraction=step/f["full_planned_updates"],panel=panel,probe_per_class=size,repeat=rep)
                            rows.append(row); done.add(key)
                    pd.DataFrame(rows).to_csv(outfile,index=False)
                print(f"REMEASURED {run} size={size}",flush=True)
            del d
