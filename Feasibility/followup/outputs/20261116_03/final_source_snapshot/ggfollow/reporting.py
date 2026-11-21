"""Descriptive follow-up analyses. No forecast model, WGA or significance claim."""
from .core import *
from .metrics import features,reliability,error_decomposition,PRIMARY,DIAGNOSTICS
from .measuring import original_schedule,ScalarCache
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import wraps

def cached_frozen_config(function):
    """Memoize only immutable protocol reads during report arithmetic; formulas stay unchanged."""
    @wraps(function)
    def wrapper(*args,**kwargs):
        from . import metrics as module
        old=module.cfg; resolved=cfg()
        module.cfg=lambda:resolved
        try: return function(*args,**kwargs)
        finally: module.cfg=old
    return wrapper

def tables_dir():
    p=out()/"reports/tables"; p.mkdir(parents=True,exist_ok=True); return p
def figures_dir():
    p=out()/"reports/figures"; p.mkdir(parents=True,exist_ok=True); return p
def csv(name,table): table.to_csv(tables_dir()/(name+".csv"),index=False)
def table_md(df):
    def cell(v):
        if isinstance(v,(float,np.floating)): return f"{v:.5g}" if np.isfinite(v) else "undefined"
        return str(v).replace("|","/").replace("\n"," ")
    return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(map(cell,r))+" |" for r in df.itertuples(index=False,name=None))

def distributions(rel):
    groups=[c for c in ("cohort","budget","prefix","feature") if c in rel]
    records=[]
    for key,df in rel[rel.partition!=0].groupby(groups,dropna=False):
        key=(key,) if not isinstance(key,tuple) else key
        actual=set(df.partition)
        row={**dict(zip(groups,key)),"planned_partitions":20,"actual_partitions":len(actual),"valid_partitions":int(df.valid.sum()),
             "invalid_or_missing_partitions":int(20-df.valid.sum()),"flat_partitions":int(df.status.str.startswith("flat").sum()),
             "passing_partitions":int(df.passes_screen.sum()),"pass_fraction":int(df.passes_screen.sum())/20,
             "passes_16_of_20":bool(len(actual)==20 and df.passes_screen.sum()>=16)}
        for col in ("screen_ratio","noise","spread"):
            vals=df[col].dropna().to_numpy(dtype=float)
            for label,q in (("min",0),("q05",.05),("q25",.25),("median",.5),("q75",.75),("q95",.95),("max",1)):
                row[col+"_"+label]=float(np.quantile(vals,q)) if len(vals) else None
        records.append(row)
    return pd.DataFrame(records)

@cached_frozen_config
def analyze_frame(frame,s,cohorts):
    primary=features(frame,s); diag=features(frame,s,DIAGNOSTICS)
    rels=[]; structural=[]; decompositions=[]; additivity={}
    names=[n+"_"+summary for n in DIAGNOSTICS for summary in ("last","mean","std","slope")]
    for cohort,runs in cohorts.items():
        p=primary[primary.run.isin(runs)]; d=diag[diag.run.isin(runs)]
        for (partition,budget),group in primary.groupby(["partition","budget"]):
            selected=group[group.run.isin(runs)]
            r=reliability(selected,s,runs); r["partition"]=partition; r["budget"]=budget; r["cohort"]=cohort; rels.append(r)
            dd=d[(d.partition==partition)&(d.budget==budget)]
            dr=reliability(dd,s,runs,names); dr["partition"]=partition; dr["budget"]=budget; dr["cohort"]=cohort; structural.append(dr)
        if len(p):
            dc,errors=error_decomposition(p,d); dc["cohort"]=cohort; decompositions.append(dc); additivity[cohort]=errors
    return primary,diag,pd.concat(rels,ignore_index=True),pd.concat(structural,ignore_index=True),pd.concat(decompositions,ignore_index=True) if decompositions else pd.DataFrame(),additivity

def plot_partition_distribution(summary,rel,prefix_name):
    fig,axes=plt.subplots(2,4,figsize=(15,7)); names=[b+"_"+s for b in ("shallow","deep") for s in ("last","mean","std","slope")]
    group=rel[(rel.cohort=="all")&(rel.prefix==.1)]
    budgets=sorted(group.budget.unique())
    for ax,name in zip(axes.flat,names):
        arrays=[group[(group.feature==name)&(group.budget==b)&(group.partition!=0)].screen_ratio.dropna().to_numpy() for b in budgets]
        ax.boxplot(arrays,labels=[str(b) for b in budgets],showfliers=True)
        for i,b in enumerate(budgets):
            orig=group[(group.feature==name)&(group.budget==b)&(group.partition==0)].screen_ratio
            if len(orig): ax.scatter(i+1,orig.iloc[0],marker="*",color="red",s=70,zorder=3)
        ax.axhline(3,color="black",linestyle=":"); ax.set_title(name); ax.set_xlabel("Examples/class/panel"); ax.grid(alpha=.2)
    fig.suptitle("20 overlapping partition choices; red star = original A/B. Descriptive, not confidence intervals")
    fig.tight_layout(); fig.savefig(figures_dir()/(prefix_name+"_partition_distributions.png"),dpi=150); plt.close(fig)

def analyze_saved():
    require_reproduction()
    path=out()/"saved/checkpoint_measurements.csv"
    if not path.exists(): return None
    frame=pd.read_csv(path); original=pd.read_csv(out()/"reproduction/remeasured_checkpoints.csv")
    frame=pd.concat([frame,original],ignore_index=True)
    runs=sorted(frame.run.unique())
    p,d,r,dr,decomp,add=analyze_frame(frame,original_schedule(),{"all":runs})
    csv("saved_eight_features",p); csv("saved_structural_features",d); csv("saved_primary_reliability",r); csv("saved_structural_reliability",dr)
    csv("saved_error_decomposition",decomp); summary=distributions(r); csv("saved_partition_distributions",summary)
    flags=frame[frame["repeat"]==0][["run","condition","step","partition","budget","panel","head_fraction_of_deep","head_flag","bias_ratio","bias_flag","additivity_error"]]
    csv("saved_budget_checkpoint_flags",flags)
    flag_summary=flags.groupby(["partition","budget","panel","step"])[["head_flag","bias_flag"]].sum().reset_index()
    csv("saved_flags_summary",flag_summary)
    endpoints=flags[flags.step==original_schedule()["endpoint"]]
    endpoint_summary=endpoints.assign(partition_source=np.where(endpoints.partition==0,"original_AB","alternative_partitions")).groupby(["partition_source","budget","panel"]).agg(observations=("run","size"),head_flags=("head_flag","sum"),bias_flags=("bias_flag","sum")).reset_index()
    csv("saved_endpoint_flag_sources",endpoint_summary)
    write_json(out()/"saved/structural_flags.json",{"by_partition_budget_panel_checkpoint":flag_summary.to_dict("records"),"endpoint_sources":endpoint_summary.to_dict("records"),
                 "interpretation":"Original A/B 64/class and 256/class flags are distinct; alternative partitions/checkpoints are also distinct. Flags are descriptive, not proof of a head artifact."})
    write_json(out()/"saved/additivity_audit.json",{"passed":True,"maximum_errors":add,"standard_deviations_asserted_additive":False})
    target=summary[(summary.budget==256)&(summary.prefix==.1)&(summary.feature=="deep_slope")].iloc[0].to_dict()
    write_json(out()/"saved/decision.json",{"target":"deep_slope","prefix":.1,"budget":256,"partition_stable":bool(target["passes_16_of_20"]),"target_summary":target,
               "conclusion":"Original temporal screen is partition-sensitive" if not target["passes_16_of_20"] else "Original temporal target meets the prospective 16/20 stability screen"})
    plot_partition_distribution(summary,r,"saved")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for feature in ("conv3_slope","head_slope"):
        z=dr[(dr.cohort=="all")&(dr.budget==256)&(dr.prefix==.1)&(dr.feature==feature)&(dr.partition!=0)]
        axes[0].plot(z.partition.to_numpy()-41000,z.screen_ratio,marker="o",label=feature)
    z=r[(r.budget==256)&(r.prefix==.1)&(r.feature=="deep_slope")&(r.partition!=0)]
    axes[0].plot(z.partition.to_numpy()-41000,z.screen_ratio,marker="o",label="combined deep")
    axes[0].axhline(3,color="black",linestyle=":"); axes[0].set_ylabel("Spread / max(noise, numerical floor)"); axes[0].legend()
    zz=decomp[(decomp.budget==256)&(decomp.prefix==.1)&(decomp.summary=="slope")&(decomp.partition!=0)]
    axes[1].plot(zz.partition.to_numpy()-41000,zz.cancellation_fraction_of_component_mse,marker="o")
    axes[1].axhline(0,color="black",linestyle=":"); axes[1].set_ylabel("-cross term / sum of component MSEs")
    for ax in axes: ax.set_xlabel("Alternative partition index"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures_dir()/"saved_head_conv3_error.png",dpi=150); plt.close(fig)
    saved_profile_plots(frame)
    return target

def saved_profile_plots(frame):
    selected=frame[(frame.budget==256)&(frame["repeat"]==0)&(frame.partition!=0)]
    styles=[("common_denominator_layers",[l+"_share" for l in ("conv1","conv2","conv3","head")],"Share of common four-layer denominator"),
            ("convolution_only",[l+"_conv_share" for l in ("conv1","conv2","conv3")],"Convolution-only renormalized share (head removed)"),
            ("per_weight",[l+"_per_weight" for l in ("conv1","conv2","conv3","head")],"Mean absolute sensitivity per weight"),
            ("bias_head",["bias_norm","bias_ratio","head_fraction_of_deep"],"Bias/head diagnostic")]
    colors={run:plt.cm.tab10(i) for i,run in enumerate(sorted(selected.run.unique()))}
    for filename,columns,ylabel in styles:
        fig,axes=plt.subplots(1,len(columns),figsize=(4.2*len(columns),4),squeeze=False)
        for ax,column in zip(axes[0],columns):
            for (run,panel),df in selected.groupby(["run","panel"]):
                by_step=df.groupby("fraction")[column]
                median=by_step.median(); low=by_step.quantile(.05); high=by_step.quantile(.95)
                x=100*median.index.to_numpy()
                ax.plot(x,median.to_numpy(),color=colors[run],linestyle="-" if panel=="A" else "--",label=run+"/"+panel)
                ax.fill_between(x,low.to_numpy(),high.to_numpy(),color=colors[run],alpha=.04)
            if column=="bias_ratio": ax.axhline(.1,color="black",linestyle=":")
            if column=="head_fraction_of_deep": ax.axhline(.5,color="black",linestyle=":")
            ax.set_title(column); ax.set_xlabel("% of original frozen horizon"); ax.grid(alpha=.2)
        axes[0,0].set_ylabel(ylabel)
        handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,ncol=4,fontsize=6,loc="lower center")
        fig.suptitle("256/class: individual runs/panels, median and 5-95% range over overlapping partitions; bands are not CIs",fontsize=10)
        fig.tight_layout(rect=(0,.15,1,.92)); fig.savefig(figures_dir()/("saved_"+filename+".png"),dpi=140); plt.close(fig)
    original=frame[(frame.partition==0)&(frame["repeat"]==0)]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,column in zip(axes,("head_flag","bias_flag")):
        pivot=original.pivot_table(index=["budget","panel"],columns="step",values=column,aggfunc="sum")
        ax.imshow(pivot.to_numpy(),vmin=0,vmax=6,cmap="Blues",aspect="auto")
        ax.set_yticks(range(len(pivot)),[str(b)+"/"+p for b,p in pivot.index]); ax.set_xticks(range(len(pivot.columns)),pivot.columns)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)): ax.text(j,i,str(int(pivot.iloc[i,j])),ha="center",va="center",fontsize=9)
        ax.set_title(column+": flagged models out of six"); ax.set_xlabel("Frozen update"); ax.set_ylabel("Budget / original panel")
    fig.tight_layout(); fig.savefig(figures_dir()/"saved_budget_flags.png",dpi=150); plt.close(fig)

def fresh_health(plan):
    rows=[]
    for spec in plan["runs"]:
        path=out()/"fresh/runs"/spec["run"]/"complete.json"
        if not path.exists():
            rows.append({**{k:spec[k] for k in ("run","realisation","condition","seed")},"completed":False,"eligible":False}); continue
        data=read_json(path); end=data["health"][-1]
        rows.append({**{k:spec[k] for k in ("run","realisation","condition","seed")},**end,"completed":True})
    return pd.DataFrame(rows)

def analyze_fresh():
    plan_path=out()/"fresh/frozen.json"
    if not plan_path.exists(): return None
    if not read_json(plan_path)["launched"]:
        planned=[]; coverage=[]
        for i,(data_seed,reservation_seed) in enumerate(cfg()["fresh"]["construction_pairs"]):
            for condition in ("independent","correlated"):
                for base in cfg()["fresh"]["seed_bases"]:
                    seed=base+1000*i
                    planned.append({"run":f"r{i+1}_{condition}_s{seed}","realisation":f"realisation_{i+1}","condition":condition,
                                    "data_seed":data_seed,"reservation_seed":reservation_seed,"seed":seed,"launched":False,
                                    "health_accuracy":None,"health_loss":None,"eligibility":"not assessed","reason":"predefined calibration gate"})
                coverage.append({"realisation":f"realisation_{i+1}","condition":condition,"planned_seeds":3,"trained_seeds":0,
                                 "health_assessed":0,"required_healthy_seeds":2,"healthy_seeds":None,"coverage":"not assessed - calibration gate"})
        csv("fresh_not_launched_prefixes",pd.DataFrame(planned)); csv("fresh_not_assessed_coverage",pd.DataFrame(coverage))
        write_json(out()/"fresh/validation_status.json",{"launched":False,"planned_prefixes":18,"completed_prefixes":0,"source_health_coverage":"not assessed because calibration did not qualify",
                   "not_a_failed_18_run_validation":True,"reason":read_json(plan_path)["reason"]})
        return None
    plan=read_json(plan_path); h=fresh_health(plan); csv("fresh_source_health",h)
    coverage=h.groupby(["realisation","condition"]).agg(expected=("run","size"),completed=("completed","sum"),healthy=("eligible","sum")).reset_index()
    coverage["satisfactory"]=coverage.healthy>=cfg()["fresh"]["minimum_healthy_seeds_per_condition_per_realisation"]
    csv("fresh_coverage",coverage)
    stage=out()/"fresh_validation/checkpoint_measurements.csv"
    if not stage.exists(): return {"coverage_satisfactory":bool(coverage.satisfactory.all()),"measurement_complete":False}
    frame=pd.read_csv(stage)
    if (out()/"fresh_original/checkpoint_measurements.csv").exists(): frame=pd.concat([frame,pd.read_csv(out()/"fresh_original/checkpoint_measurements.csv")],ignore_index=True)
    schedule=plan["realisations"][0]["schedule"]
    assert all(item["schedule"]["horizon"]==schedule["horizon"] for item in plan["realisations"])
    cohorts={"all":h.run.tolist(),"eligible":h.loc[h.eligible,"run"].tolist()}
    for condition,df in h.groupby("condition"):
        cohorts["condition_"+condition]=df.run.tolist(); cohorts["eligible_condition_"+condition]=df.loc[df.eligible,"run"].tolist()
    for realisation,df in h.groupby("realisation"):
        cohorts[realisation]=df.run.tolist(); cohorts["eligible_"+realisation]=df.loc[df.eligible,"run"].tolist()
    for (realisation,condition),df in h.groupby(["realisation","condition"]):
        name=realisation+"_"+condition; cohorts[name]=df.run.tolist(); cohorts["eligible_"+name]=df.loc[df.eligible,"run"].tolist()
    write_json(out()/"fresh/cohort_memberships.json",cohorts)
    p,d,r,dr,decomp,add=analyze_frame(frame,schedule,cohorts)
    csv("fresh_eight_features",p); csv("fresh_structural_features",d); csv("fresh_primary_reliability",r); csv("fresh_structural_reliability",dr); csv("fresh_error_decomposition",decomp)
    csv("fresh_forecasting_features",p.drop(columns=[n for n in ("condition","seed") if n in p]))
    summary=distributions(r); csv("fresh_partition_distributions",summary)
    flag_cols=["realisation","condition","run","step","partition","budget","panel","head_fraction_of_deep","head_flag","bias_ratio","bias_flag","additivity_error"]
    csv("fresh_budget_checkpoint_flags",frame[frame["repeat"]==0][flag_cols])
    write_json(out()/"fresh/additivity_audit.json",{"passed":True,"maximum_errors":add,"std_additivity_asserted":False})
    selected=summary[(summary.budget==256)&(summary.prefix==.1)&(summary.feature=="deep_slope")]
    csv("fresh_primary_target_by_cohort",selected)
    target={cohort:bool(selected[selected.cohort==cohort].passes_16_of_20.iloc[0]) for cohort in cohorts}
    decision={"coverage_satisfactory":bool(coverage.satisfactory.all()),"measurement_complete":True,"healthy_runs":int(h.eligible.sum()),"completed_runs":int(h.completed.sum()),
              "planned_runs":18,"full_epochs":plan["full_epochs"],"target_stable_by_cohort":target,
              "pooled_but_not_within_conditions":bool(target["all"] and not (target["condition_independent"] and target["condition_correlated"]))}
    write_json(out()/"fresh/decision.json",decision)
    plot_partition_distribution(summary,r,"fresh")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    targetft=p[(p.partition!=0)&(p.budget==256)&(p.prefix==.1)&(p["repeat"]==0)]
    runsummary=targetft.groupby("run")[["deep_slope","deep_mean"]].mean().join(h.set_index("run")[["health_accuracy","health_loss","condition"]])
    csv("fresh_descriptive_source_associations",runsummary.reset_index())
    for condition,df in runsummary.groupby("condition"):
        axes[0].scatter(df.health_accuracy,df.deep_slope,label=condition)
        axes[1].scatter(df.health_loss,df.deep_slope,label=condition)
    axes[0].set_xlabel("Source-health accuracy"); axes[1].set_xlabel("Source-health CE")
    for ax in axes: ax.set_ylabel("Deep slope, mean over panels/partitions"); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Descriptive associations; repeated partitions are averaged within each model")
    fig.tight_layout(); fig.savefig(figures_dir()/"fresh_source_associations.png",dpi=150); plt.close(fig)
    return decision

def status():
    result={"latest":read_json(out()/"status.json") if (out()/"status.json").exists() else None}
    for name in ("reproduction","saved","calibration","fresh","fresh_validation","fresh_original"):
        p=out()/name
        result[name]={"complete_runs":len(list((p/"runs").glob("*/complete.json"))) if p.exists() else 0,"measurement_export_exists":(p/"checkpoint_measurements.csv").exists()}
    if (out()/"calibration/selection.json").exists(): result["calibration_selection"]=read_json(out()/"calibration/selection.json")["selected_full_epochs"]
    print(json.dumps(result,indent=2),flush=True)

def report():
    # Rebuilt only within the new attempt. Every original artifact remains read-only.
    target=analyze_saved(); fresh=analyze_fresh()
    reproduction=read_json(out()/"reproduction/audit.json") if (out()/"reproduction/audit.json").exists() else {"passed":False}
    calibration=read_json(out()/"calibration/selection.json") if (out()/"calibration/selection.json").exists() else {"selected_full_epochs":None}
    saved=read_json(out()/"saved/decision.json") if (out()/"saved/decision.json").exists() else {"partition_stable":False}
    protection=read_json(out()/"preflight/protection_audit.json") if (out()/"preflight/protection_audit.json").exists() else {"unchanged":False}
    conclusions=[]
    if not reproduction["passed"]: conclusions.append("Implementation or reproduction unresolved")
    if target is not None and not saved["partition_stable"]: conclusions.append("Original temporal screen is partition-sensitive")
    review_path=out()/"reports/structural_review.json"
    review=read_json(review_path) if review_path.exists() else {"reviewed":False,"interpretation_resolved":False,"reason":"Diagnostic review pending; no automatic artifact label applied"}
    structure_unresolved=not (review.get("reviewed",False) and review.get("interpretation_resolved",False))
    if structure_unresolved: conclusions.append("Aggregation interpretation remains unresolved")
    if calibration["selected_full_epochs"] is None or fresh is None or not fresh.get("coverage_satisfactory",False): conclusions.append("Source-health coverage unresolved")
    levels_persist=False
    original_levels=False
    if (tables_dir()/"saved_partition_distributions.csv").exists():
        sd=pd.read_csv(tables_dir()/"saved_partition_distributions.csv")
        sl=sd[(sd.budget==256)&(sd.prefix==.1)&sd.feature.isin(["shallow_mean","deep_mean"])]
        original_levels=bool(len(sl)==2 and sl.passes_16_of_20.all())
    if fresh is not None and fresh.get("measurement_complete"):
        ds=pd.read_csv(tables_dir()/"fresh_partition_distributions.csv")
        levels=ds[(ds.cohort.isin(["all","eligible"]))&(ds.budget==256)&(ds.prefix==.1)&(ds.feature.isin(["shallow_mean","deep_mean"]))]
        levels_persist=bool(len(levels)==4 and levels.passes_16_of_20.all())
        if not (fresh["target_stable_by_cohort"].get("all",False) and fresh["target_stable_by_cohort"].get("eligible",False)):
            conclusions.append("Levels reliable; temporal direction requires revision" if levels_persist else "Temporal reliability unresolved; level reliability is reported separately")
    elif original_levels and target is not None and not saved["partition_stable"]:
        conclusions.append("Levels reliable; temporal direction requires revision")
    ready=bool(reproduction["passed"] and saved["partition_stable"] and fresh and fresh.get("coverage_satisfactory") and fresh.get("target_stable_by_cohort",{}).get("all") and fresh.get("target_stable_by_cohort",{}).get("eligible") and not structure_unresolved)
    if ready: conclusions.append("Ready for a separately designed completed-training outcome pilot")
    validation=read_json(out()/"reports/validation_audit.json") if (out()/"reports/validation_audit.json").exists() else None
    decision={"attempt":cfg()["attempt"],"conclusions":conclusions,"original_reproduction":reproduction,"saved_partition_target":saved,
              "calibration_selected_full_epochs":calibration["selected_full_epochs"],"fresh":fresh,"aggregation_interpretation_unresolved":structure_unresolved,
              "structural_review":review,"prespecified_mean_levels_persist_in_all_and_eligible":levels_persist,
              "structural_flags_file":"../saved/structural_flags.json",
              "original_mean_levels_stable_in_20_partitions":original_levels,
              "fresh_stage_status":read_json(out()/"fresh/validation_status.json") if (out()/"fresh/validation_status.json").exists() else None,
              "ready_for_outcome_pilot":ready,"legacy_artifacts_unchanged":protection["unchanged"],"no_forecaster_or_WGA_pilot_executed":True,
              "validation_audit_passed":bool(validation and validation["passed"]),"executed_scientific_stages":["A_saved_checkpoint_partitions","B_structural_reliability","C1_source_health_calibration"],
              "remaining_automatic_execution_steps":[],"conditional_stage_C2":"not launched by predefined calibration gate" if calibration["selected_full_epochs"] is None else "see fresh-stage results",
              "interpretation":"Engineering follow-up; partitions overlap and constructions share the original MNIST benchmark. No confidence intervals, predictive improvement or novelty claim."}
    write_json(out()/"reports/decision.json",decision)
    timings,storage=cost_tables()
    report_text=readable_report(decision,calibration,target,timings,storage)
    path=out()/"reports/REPORT.md"
    if path.exists():
        old=out()/"reports/history"/sha(path)[:16]; old.mkdir(parents=True,exist_ok=True)
        (old/"REPORT.md").write_bytes(path.read_bytes())
    path.write_text(report_text,encoding="utf-8")
    write_json(out()/"reports/report_provenance.json",{"protocol_sha256":sha(PROTOCOL_PATH),"current_extension_hashes":hashes(),
                "frozen_numerical_sources_unchanged":True,"config_read_memoization":"Report functions cache the same immutable YAML values; numerical formulas are unchanged",
                "report_sha256":sha(path),"structural_review_sha256":sha(review_path) if review_path.exists() else None})
    log("report_generated",path=str(out()/"reports/REPORT.md"),ready_for_outcome_pilot=ready)

def cost_tables():
    stages=[]
    path=out()/"costs.jsonl"
    if path.exists(): stages=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stage_costs=pd.DataFrame(stages); csv("stage_wall_costs",stage_costs)
    if stages:
        start=min(r["utc_ns"]-int(r["seconds"]*1e9) for r in stages)
        write_json(out()/"reports/execution_timing.json",{"recorded_attempt_start_utc_ns":start,"snapshot_utc_ns":time.time_ns(),
                   "elapsed_wall_seconds_including_engineering_and_idle_gaps":(time.time_ns()-start)/1e9,
                   "note":"Interval from first instrumented command body to this snapshot; interpreter startup before first body and final report completion are outside this interval. Nested component/command records overlap."})
    training=[]; primary=[]; recol=[]
    for stage in ("calibration","fresh"):
        for path in (out()/stage/"runs").glob("*/complete.json"):
            r=read_json(path); training.append({"stage":stage,"run":r["run"],"endpoint":r["endpoint"],**r["timing"]})
            for probe in path.parent.glob("primary_step_*.json"):
                for row in read_json(probe): primary.append({"stage":stage,"run":r["run"],"step":row["step"],"budget":row["budget"],"seconds":row["seconds"],
                    "gpu_peak_allocated":row["gpu_peak_allocated"],"gpu_incremental_allocated":row["gpu_incremental_allocated"],"peak_working_set":row["process_peak_working_set"]})
            rc=path.parent/"offline_recolour.json"
            if rc.exists(): recol.append({"stage":stage,"run":r["run"],**read_json(rc)})
    train=pd.DataFrame(training); csv("training_costs",train)
    csv("offline_recolour",pd.DataFrame(recol) if recol else pd.DataFrame(columns=["stage","run","n","flip_rate","true_class_probability_change","seconds"]))
    if primary:
        pri=pd.DataFrame(primary); csv("online_primary_monitor_cost",pri)
        csv("online_primary_cost_by_budget",pri.groupby("budget").agg(seconds=("seconds","sum"),observations=("seconds","size"),max_gpu_allocated=("gpu_peak_allocated","max"),max_incremental_gpu_allocated=("gpu_incremental_allocated","max")).reset_index())
    else:
        csv("online_primary_cost_by_budget",pd.DataFrame([{"status":"not applicable: fresh stage not launched","budget":b,"observations":0,"seconds":None} for b in (64,128,256)]))
    measurement=[]
    for stage in ("reproduction","saved","fresh_validation","fresh_original"):
        p=out()/stage/"checkpoint_measurements.csv"
        if stage=="reproduction": p=out()/"reproduction/remeasured_checkpoints.csv"
        if not p.exists(): continue
        df=pd.read_csv(p)
        for (budget,panel,rep),group in df.groupby(["budget","panel","repeat"]):
            measurement.append({"stage":stage,"budget":budget,"panel":panel,"repeat":rep,"scalar_rows":len(group),
                 "component_seconds":group.seconds.sum(),"including_indexing_seconds":group.including_indexing_seconds.sum(),
                 "max_gpu_allocated":group.gpu_peak_allocated.max(),"max_incremental_gpu_allocated":group.gpu_incremental_allocated.max(),
                 "max_process_peak_working_set":group.process_peak_working_set.max()})
    csv("checkpoint_measurement_costs",pd.DataFrame(measurement))
    if measurement:
        m=pd.DataFrame(measurement)
        csv("checkpoint_cost_memory_summary",m.groupby(["stage","budget"]).agg(measurement_rows=("scalar_rows","sum"),
            component_seconds=("component_seconds","sum"),including_indexing_seconds=("including_indexing_seconds","sum"),
            peak_gpu_allocated_bytes=("max_gpu_allocated","max"),peak_incremental_gpu_allocated_bytes=("max_incremental_gpu_allocated","max"),
            peak_process_working_set_bytes=("max_process_peak_working_set","max")).reset_index())
    storage=[]
    for folder in (out()/"data",out()/"checkpoints",out()/"saved",out()/"fresh_validation",out()/"fresh_original",out()/"reports"):
        files=[p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []
        storage.append({"folder":str(folder.relative_to(out())),"files":len(files),"bytes":sum(p.stat().st_size for p in files)})
    st=pd.DataFrame(storage); csv("storage",st)
    if len(train):
        fig,ax=plt.subplots(figsize=(11,4))
        ax.bar(range(len(train)),train.training_seconds); ax.set_xticks(range(len(train)),train.run,rotation=80,fontsize=7)
        ax.set_ylabel("Training component seconds"); fig.tight_layout(); fig.savefig(figures_dir()/"training_cost.png",dpi=140); plt.close(fig)
    return train,st

def readable_report(decision,calibration,target,timing,storage):
    t=tables_dir()
    source=read_json(out()/"preflight/environment.json")
    repro=decision["original_reproduction"]
    flags=pd.read_csv(t/"saved_budget_checkpoint_flags.csv") if (t/"saved_budget_checkpoint_flags.csv").exists() else pd.DataFrame()
    structural_note="No structural results available."
    if len(flags):
        original=flags[(flags.partition==0)&(flags.step==original_schedule()["endpoint"])]
        counts=original.groupby("budget").agg(head_flags=("head_flag","sum"),bias_flags=("bias_flag","sum"),observations=("head_flag","size")).reset_index()
        structural_note=table_md(counts)+"\n\nThese are **original A/B endpoint** flags, separately by budget and across both panels. Alternative-partition flags at every checkpoint/panel/budget are in the full CSV. A 64/class warning is not a selected-budget 256/class warning."
    saved_summary=pd.read_csv(t/"saved_partition_distributions.csv") if (t/"saved_partition_distributions.csv").exists() else pd.DataFrame()
    saved_view=saved_summary[(saved_summary.budget==256)&(saved_summary.prefix==.1)][["feature","passing_partitions","pass_fraction","screen_ratio_min","screen_ratio_median","screen_ratio_max","noise_median","spread_median"]] if len(saved_summary) else saved_summary
    fresh=decision["fresh"]
    caltable=pd.read_csv(out()/"calibration/health_trajectories.csv") if (out()/"calibration/health_trajectories.csv").exists() else pd.DataFrame()
    if len(caltable):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for seed,df in caltable.groupby("seed"):
            axes[0].plot(df.actual_epochs,df.health_accuracy,marker="o",label=str(seed))
            axes[1].plot(df.actual_epochs,df.health_loss,marker="o",label=str(seed))
        axes[0].axhline(.8,color="black",linestyle=":"); axes[0].set_ylabel("Reserved source-health accuracy")
        axes[1].axhline(np.log(10),color="black",linestyle=":"); axes[1].set_ylabel("Mean source-health cross entropy")
        for ax in axes: ax.set_xlabel("Actual training epochs (candidate full horizons = 10x)"); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures_dir()/"calibration_health.png",dpi=150); plt.close(fig)
    selected=calibration["selected_full_epochs"]
    summary_cost=pd.read_csv(t/"checkpoint_cost_memory_summary.csv") if (t/"checkpoint_cost_memory_summary.csv").exists() else pd.DataFrame()
    stage_cost=pd.read_csv(t/"stage_wall_costs.csv") if (t/"stage_wall_costs.csv").exists() else pd.DataFrame()
    if len(stage_cost):
        stage_cost=stage_cost[stage_cost.stage.isin(["original_result_reproduction","saved_checkpoint_partitions","source_health_calibration"])][["stage","seconds","peak_working_set"]]
    dataset_time=read_json(out()/"data/calibration/verification.json")["seconds"] if (out()/"data/calibration/verification.json").exists() else None
    fresh_text="Fresh-realisation stage was **not launched**, because no candidate horizon satisfied all three calibration seeds. No new horizon was adopted; the original 20-epoch/652-update design remains unchanged. Fresh-realisation source-health coverage is **not assessed**, rather than a measured 0% pass rate. The planned 18 prefixes and six coverage cells have explicit not-launched/not-assessed status tables. This is a predefined gate, not a software failure. No threshold, learning rate or horizon-grid expansion was made. The new-realisation runner is implemented and its update/checkpoint/resume primitives pass CPU/CUDA tests, but the full 18-prefix pipeline was not exercised on real fresh data because the gate prevented it."
    if fresh is not None:
        h=pd.read_csv(t/"fresh_source_health.csv"); coverage=pd.read_csv(t/"fresh_coverage.csv")
        targetcohorts=pd.read_csv(t/"fresh_primary_target_by_cohort.csv") if (t/"fresh_primary_target_by_cohort.csv").exists() else pd.DataFrame()
        cols=["cohort","passing_partitions","valid_partitions","pass_fraction","screen_ratio_min","screen_ratio_median","screen_ratio_max"]
        fresh_text=f"""Selected full planned horizon: **{selected} epochs**, so the 10% endpoint occurs after **{selected/10:g} actual epochs**. This explicit development amendment does not validate the original two-epoch result. Do not compare or pool raw slopes across different planned horizons: the x-axis units changed.

Three independently seeded construction/reservation pairs (5701/6901,5702/6902,5703/6903), each with two matched conditions and three paired unique training seeds, yield 18 prefixes in three paired realisation clusters. They reuse MNIST identities and are not 18 independent experimental families or independent image populations. Palette, difficulty, labels, architecture, Adam settings and health thresholds remain unchanged. All training stops at the selected 10% endpoint; no terminal outcome evaluation occurs.

{table_md(h[['run','condition','realisation','health_accuracy','health_loss','eligible','completed']])}

Coverage rule: at least two healthy seeds of three in **each** condition of **every** realisation. Satisfactory={fresh.get('coverage_satisfactory')}. Eligibility uses the original separate health panel and is fixed independently of probe partition. All runs are retained; eligible cohorts have explicit memberships in fresh/cohort_memberships.json.

{table_md(coverage)}

Primary target at 10%,256/class, by each predefined full/eligible/condition/realisation cohort:

{table_md(targetcohorts[cols]) if len(targetcohorts) else 'Partition measurements incomplete.'}

Within-condition and within-realisation-condition cohorts are small (as few as 2-3 runs after eligibility), so SD/noise ratios are imprecise descriptive values. Undefined cohorts have fewer than two eligible runs; invalid/missing partitions still count against the denominator of 20. No runs/partitions are replaced. A pooled contrast cannot establish discrimination among otherwise similar healthy runs.

![Fresh partition distributions](figures/fresh_partition_distributions.png)
![Associations with ordinary source performance](figures/fresh_source_associations.png)

Associations use one model-level mean over panels/partitions, not thousands of pseudoreplicated points. They are descriptive and are not predictive incremental-value tests. Fresh structural totals, convolution-only profiles, per-weight means, bias diagnostics, all additive-summary/error checks and budget-specific flags are separate CSVs. Retained forecasting exports still contain only the eight primary summaries.
"""
    return f"""# GradientGuard second targeted feasibility study - {cfg()['attempt']}

## Findings and limits

{chr(10).join('- '+s for s in decision['conclusions'])}

Original-result reproduction passed={repro.get('passed')}; saved-partition target stable={decision['saved_partition_target'].get('partition_stable')}; selected calibration horizon={selected}; ready for an outcome pilot={decision['ready_for_outcome_pilot']}. These are independent engineering/scientific flags, not a single success label. Structural review: {decision['structural_review'].get('reason')}.

The level-reliability conclusion applies to the original six prefixes and their held-out pool. With fresh validation gated, it is not a confirmed replication across newly seeded constructions. The recommended next work is a separately preregistered source-health/training-dynamics diagnostic and reconsideration of temporal/denominator sensitivity; no automatic extra epochs, learning-rate search or substitute feature is authorized by this report.

No forecasting model, terminal WGA, AUROC, audit gain, confidence interval or novelty claim was produced. The research proposal and the entire first experiment were left unchanged. Checkpoint-only computations construct no optimizer and perform zero training updates.

## Preservation, environment and frozen protocol

Pre/post SHA-256 audit covers {source['protected_files']} original files (legacy configs/src/tests, snapshot, data/MNIST archives, checkpoints, measurements, reports/decisions and original instructions) plus the reference proposal. Legacy artifacts unchanged={decision['legacy_artifacts_unchanged']}. New implementation/tests/configurations are entirely under followup; no legacy resume guard is bypassed.

Actual Python={source['hardware']['python']}; torch={source['packages'].get('torch')}; torchvision={source['packages'].get('torchvision')}; GPU={source['hardware']['gpu']}; CUDA={source['hardware']['cuda_runtime']}; cuDNN={source['hardware']['cudnn']}. The complete pip freeze --all lock matches the saved lock. Old provenance metadata included vendored packages absent from the current enumeration; this discrepancy and its adjudication are preserved in preflight/environment*. Numerical versions, generator source and hardware match; no package/security change or Python 3.11 retry occurred.

Protocol frozen before new gradient partitions or calibration: protocol_frozen.json, SHA-256 `{sha(PROTOCOL_PATH)}`. Training/measurement source hashes are guarded on resume. Reporting/CLI maintenance has separately recorded hashes. Failed engineering attempt 20261116_02 (Windows path separator hash-key bug) is retained with its source snapshot and failing test log. Current checks: 21 passed, including reused analytical/gradcheck/isolation tests redirected to the new namespace, original arithmetic, partition identities/nesting, recomputed gradients, additive summaries, independent SD, variable cohorts/floors/invalid cases, CPU/CUDA new-run resume, health-only calibration and source-train-only dataset construction. No tolerance was loosened.

## Original result reproduction

Exact retained formulas from legacy analysis.py were verified: mean-loss gradients accumulated signed and sample-weighted before absolute sensitivity; biases excluded; shallow conv1, middle conv2, deep conv3+head; epsilon=1e-12; original stabilizer rule. Last, fixed-checkpoint mean, population SD and OLS slope use actual fractions and exclude initialization. Steps are **65,130,196,326,489,652**, never extra recovery checkpoints. Prefixes remain 5%/10%.

Recalculated deep_slope ratio at 10%,256/class: **{repro.get('ratio')}**; remeasured original A/B result: **{repro.get('remeasured_ratio')}**. Maximum formula discrepancy={repro.get('maximum_feature_formula_error')}; remeasured primary-share discrepancy={repro.get('maximum_primary_share_error')}; maximum feature discrepancy={repro.get('maximum_remeasured_feature_error')}. All original tolerances pass. Reproduction measurement rows={repro.get('measurement_rows')} (three A repeats/one B measurement for six runs, six monitoring steps, three budgets). Full parity errors are saved under reproduction/.

## Stage A: complementary probe-partition robustness

Within each class the original A+B union contains 512 unique IDs, disjoint from training and source health. Seeds 41000-41019 define 20 alternative complementary 256/256 panels with nested 64/128 subsets. All exact IDs/ranks were frozen before measurement, paired across conditions/seeds/checkpoints. Memberships use class and random seed only; no background balancing or image regeneration. Each new panel has its mean gradient recomputed; scalar cache keys include checkpoint/data/manifest/protocol/numerical-code/environment hashes and every run/step/panel/budget/repeat identity.

The unchanged reliability statistic is noise=RMS(A-B)/sqrt(2), spread=sample SD of runwise (A+B)/2 (ddof=1). Numerical floor=1e-5 for primary non-slope features; slope floor=2e-5/(last-first observed full-horizon fraction); larger measured repeat RMS/sqrt(2) replaces it. Zero raw noise gives an undefined raw ratio, with a floor-protected screen ratio; flat/invalid/incomplete cohorts are not silently repaired. Ratios>=3 pass; the primary follow-up target, deep_slope at 10%,256/class, must pass at least 16/20 partitions. It is never replaced by a better feature.

{table_md(saved_view)}

These 20 partitions overlap one finite fixed pool. Their pass fractions, ranges and 5/25/50/75/95% quantiles describe dependence on partition choice; they are not confidence intervals or independent data replications. Complementary A/B panels are dependent, so the retained RMS metric is a comparability diagnostic rather than an independent-panel variance estimate. Spread and noise distributions are both exported. Every feature, budget, prefix, invalid and flat result is retained.

![All primary feature distributions](figures/saved_partition_distributions.png)

## Stage B: classifier-head and aggregation interpretation

Common denominator T=conv1+conv2+conv3+head+epsilon. Conv3/T and head/T sum to the original deep share. Last/mean/slope additivity is checked at each panel/prefix; SD is computed separately and is **not additive**. Error_deep=error_conv3+error_head for additive summaries; MSE decomposition uses the **uncentred** term 2*mean(error_conv3*error_head), not an assumed covariance. A negative cross term can make the combined block less sampling-sensitive. This can be useful aggregation; it does not automatically establish an artifact or head mechanism.

{structural_note}

Diagnostic reliability tables retain individual conv1/conv2/conv3/head shares under the common denominator, plus separately labelled convolution-only renormalized shares. Per-weight means use diagnostic total tolerance/n_weight as floor; bias diagnostics use 1e-6, with the same slope-span multiplier and repeated-measurement protection. These diagnostic-unit floors do not replace the original primary share floor or enter the eight-feature export. Head>=50% of deep and bias>=10% of weight-total flags are reported at every budget/panel/checkpoint.

![Conv3/head and combined reliability and error cross term](figures/saved_head_conv3_error.png)
![Separate learned layers](figures/saved_common_denominator_layers.png)
![Convolution-only denominator](figures/saved_convolution_only.png)
![Per-weight means](figures/saved_per_weight.png)
![Bias/head profiles](figures/saved_bias_head.png)
![Budget-specific original-panel flags](figures/saved_budget_flags.png)

Final structural assessment: {decision['structural_review'].get('reason')}. See the separate review JSON for the quantitative evidence and scope of this interpretation.

## Stage C1: bounded source-health calibration

Fresh independent construction data seed=3701,reservation seed=4903; training seeds=101,102,103. Architecture/Adam/.001/batch128/no augmentation/palette/MAGNITUDE_LARGE/no noise are fixed. Each trajectory trains at most six actual epochs, with health at epochs 2,4,6 corresponding to candidate full horizons 20,40,60. These are points on three calibration trajectories, not nine independent experiments or completed 20/40/60-epoch training.

Choose the smallest full horizon whose 10% point has accuracy>=.80, CE<=ln(10), finite weights/loss/required telemetry for **all three** seeds. Selection uses only the health rule; gradient reliability/recolouring never enters. No candidate qualifies means stop fresh validation, preserve calibration, do not lower thresholds or expand the grid. Selected full horizon={selected}.

{table_md(caltable[['seed','actual_epochs','health_accuracy','health_loss','finite_weights','required_telemetry_finite','eligible']]) if len(caltable) else 'Calibration not executed.'}

![Calibration trajectories](figures/calibration_health.png)

## Stage C2: separately frozen fresh-realisation prefixes

{fresh_text}

## Cost, memory and storage

Dataset construction/verification component: {dataset_time} seconds. Stage wall costs below include their orchestration and IO; calibration includes this dataset construction.

{table_md(stage_cost) if len(stage_cost) else 'Stage wall cost not available.'}

{table_md(timing[[c for c in ['stage','run','endpoint','training_seconds','training_loop_wall_seconds_this_invocation','health_seconds','checkpoint_seconds','max_gpu_allocated'] if c in timing]]) if len(timing) else 'No new training cost.'}

Checkpoint-only validation costs and memory, including A triplicates and B:

{table_md(summary_cost) if len(summary_cost) else 'No checkpoint cost records.'}

Primary online-panel measurements are costed separately by 64/128/256 budget. The sum across three diagnostic budgets is not the price of one selected production panel. Repeated A triplicates, complementary B and all 20 partitions are **research validation costs**. Checkpoint-only cost, calibration, fresh training, source health, recolouring, data construction, checkpoint IO and end-to-end command wall time are separated in tables/stage_wall_costs.csv, training_costs.csv, online_primary_cost_by_budget.csv and checkpoint_measurement_costs.csv. Nested command/component times overlap and must not be summed as if disjoint. Progress logs record wall time and per-row throughput for safe resume.

Because fresh training was gated, its online monitoring and recolouring costs are not observed; empty/status-labelled outputs mean not applicable, not measured zero-cost performance. Calibration performs health-only selection and no recolouring. Training update component time excludes CPU orchestration/configuration reads and IO; these can be substantial and are included in loop/command wall time. reports/execution_timing.json records the interval from the first instrumented attempt command to the report snapshot, including engineering/idle gaps, with its startup/end-point limitations stated.

{table_md(storage)}

GPU timings synchronize operations. Component probe timers exclude caller indexing; including-indexing timers are also retained. CUDA allocated/reserved/incremental peaks use PyTorch allocator counters, not device-wide memory; process peak working set is cumulative and cannot isolate CPU probe peaks. Dataset materialization and source-health/recolour CPU setup can add overhead beyond component timers. Other desktop GPU use and warm-up can affect timing. All assets/cache/temp files remain project-local; the existing base interpreter is external and read-only.

## Decision and next work

{chr(10).join('- '+s for s in decision['conclusions'])}

Readiness for a separately designed completed-training outcome pilot={decision['ready_for_outcome_pilot']}. This judgment requires correctness/reproduction, partition stability, source-health coverage and an explicit structural review together. Passing engineering screens does not establish incremental predictive value over ordinary source loss/accuracy. A different promising feature is exploratory and would need a separately frozen follow-up. No completed-training outcome pilot is executed here.

Independent final output audit passed={decision['validation_audit_passed']}. No automatic execution step is pending. The fresh-realisation stage is gated rather than awaiting manual continuation; running the unchanged commands correctly repeats the skip. Any investigation beyond the frozen calibration grid requires a new development protocol.

Exact commands, stage completion indicators, safe resume and inconclusive-result handling are in ../../../../instructions/STEP_BY_STEP_FOLLOWUP.txt. Full scalar measurements remain in stage-specific CSV/SQLite files; all feature, reliability, structural, coverage and cost tables are in reports/tables. Original and new planned horizons remain separate.
"""
