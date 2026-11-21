"""Descriptive follow-up analyses. No forecast model, WGA or significance claim."""
from .core import *
from .metrics import features,reliability,error_decomposition,PRIMARY,DIAGNOSTICS
from .measuring import original_schedule,ScalarCache
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    csv("saved_flags_summary",flags.groupby(["partition","budget","panel","step"])[["head_flag","bias_flag"]].sum().reset_index())
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
    return target

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
    if not plan_path.exists() or not read_json(plan_path)["launched"]: return None
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
    # Structural diagnostics are descriptive; a stable combined block is not automatically an artifact.
    structure_unresolved=True
    conclusions.append("Aggregation interpretation remains unresolved")
    if calibration["selected_full_epochs"] is None or fresh is None or not fresh.get("coverage_satisfactory",False): conclusions.append("Source-health coverage unresolved")
    if fresh is not None and fresh.get("measurement_complete") and not (fresh["target_stable_by_cohort"].get("all",False) and fresh["target_stable_by_cohort"].get("eligible",False)):
        conclusions.append("Levels reliable; temporal direction requires revision")
    ready=bool(reproduction["passed"] and saved["partition_stable"] and fresh and fresh.get("coverage_satisfactory") and fresh.get("target_stable_by_cohort",{}).get("all") and fresh.get("target_stable_by_cohort",{}).get("eligible") and not structure_unresolved)
    if ready: conclusions.append("Ready for a separately designed completed-training outcome pilot")
    decision={"attempt":cfg()["attempt"],"conclusions":conclusions,"original_reproduction":reproduction,"saved_partition_target":saved,
              "calibration_selected_full_epochs":calibration["selected_full_epochs"],"fresh":fresh,"aggregation_interpretation_unresolved":structure_unresolved,
              "ready_for_outcome_pilot":ready,"legacy_artifacts_unchanged":protection["unchanged"],"no_forecaster_or_WGA_pilot_executed":True,
              "interpretation":"Engineering follow-up; partitions overlap and constructions share the original MNIST benchmark. No confidence intervals, predictive improvement or novelty claim."}
    write_json(out()/"reports/decision.json",decision)
    # Detailed report is extended below after the numerical stages have completed.
    (out()/"reports/REPORT.md").write_text("# GradientGuard targeted feasibility follow-up\n\n"+"\n".join("- "+s for s in conclusions)+"\n\n"+json.dumps(decision,indent=2),encoding="utf-8")
    log("report_generated",path=str(out()/"reports/REPORT.md"),ready_for_outcome_pilot=ready)
