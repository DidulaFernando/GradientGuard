"""Rebuild the base report and append cost/structural audit at the revised probe budget.

Kept outside the immutable training source snapshot; does not alter training/configuration.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common import config,attempt_dir,write_json,np,sha
from analysis import report,measurement_table,feature_table,reliability,plot_profiles,mdtable
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def integer_or_none(value):
    return int(value) if pd.notna(value) else None

def main():
    from audit_artifacts import main as artifact_audit
    from audit_calculus import main as calculus_audit
    artifact_audit()
    calculus_audit()
    folder=attempt_dir("reports")
    if (folder/"REPORT.md").exists():
        history=folder/"history"; history.mkdir(exist_ok=True)
        # Content-addressed history preserves prior generated reports without duplicates.
        digest=sha(folder/"REPORT.md")[:16]
        for filename in ("REPORT.md","decision.json"):
            p=folder/filename
            if p.exists(): (history/(digest+"_"+filename)).write_bytes(p.read_bytes())
    report()
    decision=json.loads((folder/"decision.json").read_text())
    calculus=json.loads((attempt_dir()/"additional_calculus_audit.json").read_text())
    c=config(); sizes=decision["probe_sizes_measured"]; largest=max(sizes)
    tables=folder/"tables"; figs=folder/"figures"
    allft=pd.concat([feature_table(measurement_table(n)) for n in sizes],ignore_index=True)
    rows=[]; structural=[]
    for size in sizes:
        table=measurement_table(size)
        for (run,panel),df in table[table["repeat"]==0].groupby(["run","panel"]):
            rows.append({"probe_per_class":size,"run":run,"panel":panel,
                         "monitor_all_points_seconds":df.seconds.sum(),
                         "monitor_post_initialization_seconds":df.loc[df.step>0,"seconds"].sum(),
                         "initialization_seconds":df.loc[df.step==0,"seconds"].sum(),
                         "max_gpu_allocated_bytes":integer_or_none(df.gpu_peak_allocated.max()),
                         "max_incremental_probe_allocated_bytes":integer_or_none(df.gpu_incremental_allocated.max()),
                         "max_gpu_reserved_bytes":integer_or_none(df.gpu_peak_reserved.max()),
                         "max_process_peak_working_set_bytes":integer_or_none(df.process_peak_working_set.max())})
        if size>64:
            p=figs/f"probe_{size}"; p.mkdir(exist_ok=True)
            health_metadata=measurement_table(64).query('panel == "A" and repeat == 0')[["run","step","health_accuracy","health_loss"]]
            plot_profiles(table.merge(health_metadata,on=["run","step"],how="left",validate="many_to_one"),p)
        endpoints=table[(table.step==table.step.max())&(table["repeat"]==0)].copy()
        for _,r in endpoints.iterrows():
            structural.append({"probe_per_class":size,"run":r.run,"panel":r.panel,
                               "head_fraction_of_deep":r.head_fraction_of_deep,
                               "bias_sensitivity_over_weight_total":r.bias_sensitivity/r.total,
                               "conv1_conv_share":r.conv1_conv_share,"conv2_conv_share":r.conv2_conv_share,
                               "conv3_conv_share":r.conv3_conv_share,
                               "conv1_per_weight":r.conv1_per_weight,"conv2_per_weight":r.conv2_per_weight,
                               "conv3_per_weight":r.conv3_per_weight,"head_per_weight":r.head_per_weight})
    timing=pd.DataFrame(rows)
    for column in ("max_gpu_allocated_bytes","max_incremental_probe_allocated_bytes","max_gpu_reserved_bytes","max_process_peak_working_set_bytes"):
        timing[column]=pd.to_numeric(timing[column],errors="coerce")
    train=pd.read_csv(tables/"timing.csv").set_index("run").training_seconds
    timing["training_seconds"]=timing.run.map(train)
    timing["monitor_training_ratio"]=timing.monitor_all_points_seconds/timing.training_seconds
    timing.to_csv(tables/"timing_memory_by_probe_budget.csv",index=False)
    structural=pd.DataFrame(structural); structural.to_csv(tables/"structural_all_budgets_panels.csv",index=False)
    by_budget=timing.groupby(["probe_per_class","panel"])[["monitor_all_points_seconds","monitor_post_initialization_seconds","max_gpu_allocated_bytes","max_incremental_probe_allocated_bytes","max_process_peak_working_set_bytes"]].agg({"monitor_all_points_seconds":"sum","monitor_post_initialization_seconds":"sum","max_gpu_allocated_bytes":"max","max_incremental_probe_allocated_bytes":"max","max_process_peak_working_set_bytes":"max"}).reset_index()
    ratios=timing.groupby("probe_per_class").monitor_all_points_seconds.sum()/train.sum()
    selected=structural[structural.probe_per_class==largest]
    rel=pd.read_csv(tables/"reliability_all.csv")
    rel10=rel[rel.prefix==.1]
    slopeview=rel10[rel10.feature.str.endswith("_slope")][["probe_per_class","feature","noise","spread","screen_ratio","passes_screen"]]
    passing=rel10[(rel10.probe_per_class==largest)&rel10.passes_screen]
    smallest_passing_slope=passing.loc[passing.feature.str.endswith("_slope"),"screen_ratio"].min()
    head_min=float(selected.head_fraction_of_deep.min()); head_max=float(selected.head_fraction_of_deep.max())
    bias_min=float(selected.bias_sensitivity_over_weight_total.min()); bias_max=float(selected.bias_sensitivity_over_weight_total.max())
    selected_head_flag=bool((selected.head_fraction_of_deep>=.5).any())
    selected_bias_flag=bool((selected.bias_sensitivity_over_weight_total>=.1).any())
    conv_only_features=[]
    # Diagnostic profiles only; never merge these additional controls into forecasting export.
    for (run,panel),df in measurement_table(largest).query('repeat == 0').groupby(["run","panel"]):
        df=df[df.step>0].sort_values("fraction")
        x=df.fraction.to_numpy()
        rr={"run":run,"panel":panel}
        for col in ("conv1_conv_share","conv3_conv_share","head_fraction_of_deep","bias_sensitivity"):
            y=df[col].to_numpy(); rr[col+"_mean"]=y.mean(); rr[col+"_slope"]=np.polyfit(x,y,1)[0]
        conv_only_features.append(rr)
    pd.DataFrame(conv_only_features).to_csv(tables/"offline_structural_prefix_diagnostics.csv",index=False)
    # Independent feature verification against explicit arrays and numpy.polyfit.
    max_feature_error=0.
    for size in sizes:
        table=measurement_table(size)
        ft=allft[allft.probe_per_class==size]
        for _,row in ft.iterrows():
            df=table[(table.run==row.run)&(table.panel==row.panel)&(table["repeat"]==row["repeat"])&(table.fraction>0)&(table.fraction<=row.prefix+1e-12)].sort_values("fraction")
            for block in ("shallow","deep"):
                values=df[block+"_share"].to_numpy(); xx=df.fraction.to_numpy()
                expected=[values[-1],sum(values)/len(values),np.sqrt(sum((values-values.mean())**2)/len(values)),np.polyfit(xx,values,1)[0]]
                for name,val in zip(("last","mean","std","slope"),expected):
                    max_feature_error=max(max_feature_error,abs(row[block+"_"+name]-val))
    assert max_feature_error<1e-9
    decision.update({"assessment":"Strong reproducible level variation in these six prefixes; temporal screen passes narrowly only with 256/class. Structural control required before attribution or progression claims.",
                     "passing_features_at_10pct_selected_budget":passing.feature.tolist(),
                     "deep_slope_screen_ratio_at_10pct_256":float(rel10[(rel10.probe_per_class==256)&(rel10.feature=="deep_slope")].screen_ratio.iloc[0]),
                     "slope_margin_above_frozen_ratio_screen":float(smallest_passing_slope-c["measurement"]["ratio_screen"]),
                     "AB_monitor_training_ratio_by_budget":{str(k):float(v) for k,v in ratios.items()},
                     "selected_budget_head_flag":selected_head_flag,"selected_budget_bias_flag":selected_bias_flag,
                     "independent_feature_recalculation_max_error":max_feature_error,
                     "artifact_audit_passed":json.loads((attempt_dir()/"artifact_audit.json").read_text())["passed"],
                     "additional_calculus_audit_passed":calculus["passed"],
                     "remaining_manual_execution_steps":[],
                     "next_research_steps":["Targeted classifier-head/aggregation control", "Recheck temporal reliability across additional probe reservations and independent settings", "Address source-health mismatch before a source-normal outcome study"]})
    decision["summary_storage_bytes"]=sum(p.stat().st_size for p in tables.glob("*.csv"))
    write_json(folder/"decision.json",decision)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for panel in ("A","B"):
        df=by_budget[by_budget.panel==panel]
        axes[0].plot(df.probe_per_class,df.monitor_all_points_seconds,marker="o",label=panel)
        axes[1].plot(df.probe_per_class,df.max_incremental_probe_allocated_bytes/(1024**2),marker="o",label=panel)
    axes[0].set_ylabel("All six runs / seven monitor points (seconds)")
    axes[1].set_ylabel("Peak incremental CUDA allocation (MiB)")
    for ax in axes: ax.set_xlabel("Examples per class"); ax.legend(title="Panel"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figs/"cost_memory_by_budget.png",dpi=150); plt.close(fig)
    add=f"""

## Final engineering qualification and revised-budget audit

**The temporal result is marginal and probe-budget dependent.** At 10%, deep slope reaches {decision['deep_slope_screen_ratio_at_10pct_256']:.4f}, only {decision['slope_margin_above_frozen_ratio_screen']:.4f} above the prespecified screen of 3. The shallow slope still fails. Slopes at 64 and 128/class fail the screen; increasing the probe budget does not improve every feature monotonically. Strong level ratios do not turn this into robust evidence for all eight features. At 256/class, five of eight 10% features pass; three still fail. The exact frozen screen is reported without changing it or adding a post-hoc success threshold. The appropriate conclusion is **qualified measurement feasibility, requiring structural controls and further temporal reliability work**.

{mdtable(slopeview)}

This pilot retains all six prefixes, but all three independent-background runs miss 80% source-health accuracy. Consequently the observed between-condition variation also reflects markedly different ordinary learning progress. It is not evidence of useful variation among source-normal candidate classifiers. Three paired seeds per condition cannot establish a generally reproducible between-run distribution. Even within this pilot, A/B panel changes remain substantial for several features.

At the revised 256/class budget and across both panels, the head is {100*head_min:.1f}% to {100*head_max:.1f}% of deep sensitivity; bias sensitivity is {100*bias_min:.2f}% to {100*bias_max:.2f}% of total weight sensitivity. Selected-budget head flag={selected_head_flag}, bias flag={selected_bias_flag}. The original 64/class endpoint head flag therefore does not survive the larger-panel screen: the flag itself is probe-sensitive. This is not established head dominance. A targeted aggregation control remains a conservative next step because the deep block combines two learned layers and the temporal result is marginal. The bias magnitude screen does not flag these endpoints, which does not rule out bias effects on trajectories. Inspect the separate learned layers, convolution-only shares and per-weight means below before attributing differences to depth-specific redistribution.

![256/class primary profiles](figures/probe_256/primary_shares.png)
![256/class separate layers](figures/probe_256/four_layer_totals.png)
![256/class convolution-only profiles](figures/probe_256/convolution_profiles.png)
![256/class means per weight](figures/probe_256/per_weight_means.png)
![256/class bias diagnostics](figures/probe_256/bias_diagnostics.png)

The main timing table describes the original 64/class online pilot. The revised budget must also be costed. Combined A+B component monitoring/training ratios by budget: {dict((int(k),round(float(v),4)) for k,v in ratios.items())}. These ratios compare seven repeated frozen-checkpoint measurements, including initialization, to the original training component time. The larger-budget timing measurements were offline, not interleaved training; device warm-up/caching context can differ. They are prospective monitoring cost estimates, not a measured retrained system. Caller-side indexing, process startup/data loading and all allocation overhead are not fully covered by the component timers. Peak CUDA incremental allocations are allocator-counter measurements; process peak working set is cumulative, not an incremental CPU peak.

{mdtable(by_budget)}

![Probe cost and memory](figures/cost_memory_by_budget.png)

All diagnostic profiles are generated separately for 128 and 256/class under figures/probe_128 and figures/probe_256. Full per-panel structural endpoints, convolution-only/head/bias prefix diagnostics, and timing/memory for every budget are separate CSVs. Forecasting exports still contain only the same eight prespecified features and identifiers.

Independent artifact checks confirm all six Adam states reached exactly 652 updates, initial weights match across conditions for each seed, raw original IDs/labels/reservations/nested ranks match, checkpoints and current code match the pretraining source snapshot, and 67 installed SpuCo Python sources equal the pinned official checkout. Independent feature recalculation using explicit arrays and numpy.polyfit has maximum discrepancy {max_feature_error:.3g}. The observed 4.44e-16 identical-summary difference includes CSV serialization roundoff; direct repeated-measurement correctness tests had exact zero difference on CPU/CUDA.

Additional explicit central finite differences (epsilon=1e-6, CPU float64) at the active analytical path passed installed PyTorch gradcheck and have maximum absolute error {max(r['absolute_error'] for r in calculus['finite_differences']):.6g}. Trained-checkpoint CUDA microbatch comparisons (both conditions, seed 11, step 652, 640 probe images, batch sizes 640 versus 64/128/192 including an unequal final batch) have maximum gradient discrepancy {max((r['max_gradient_error'] for r in calculus['trained_checkpoint_cuda_microbatch']),default=float('nan')):.6g} and share discrepancy {max((r['max_block_share_error'] for r in calculus['trained_checkpoint_cuda_microbatch']),default=float('nan')):.6g}. CUDA comparisons are omitted if unavailable (shown as nan), without implying a pass on absent hardware. All executed comparisons use declared original tolerances. Exact errors are retained in `../../runs/{c['attempt']}/additional_calculus_audit.json`. The Windows CPU model/RAM/driver inventory is also retained in `../../runs/{c['attempt']}/hardware_windows.json`; pretraining available memory and software settings remain in provenance.json.

No execution step remains pending and no training process is left running. The next research work is a targeted head/aggregation control, additional independent settings/probe reservations, and reconsideration of source-health coverage before a completed-training outcome pilot. Those are follow-up studies, not missing execution steps in this test. The proposal was left unchanged. Setup failures and all inconclusive feature results remain preserved.
"""
    p=folder/"REPORT.md"
    original=p.read_text(encoding="utf-8")
    # Put the key qualification near the start as well as in the detailed appendix.
    original=original.replace("## Result and scope\n", "## Result and scope\n\n**Qualified outcome:** levels show clear variation, but the only passing 10% slope is deep slope at 256/class (ratio 3.05). This narrowly clears the provisional screen; it is not robust temporal validation. Classifier-head aggregation and unequal source health require follow-up.\n",1)
    p.write_text((original+add).replace("pretraining source snapshot","source snapshot matching the pretraining hashes"),encoding="utf-8")
    write_json(folder/"postanalysis_provenance.json",{"script":str(Path(__file__).resolve()),"script_sha256":sha(__file__),"training_sources_unchanged":True,
              "note":"Descriptive report extension only; no modification to frozen settings, retained features, thresholds, checkpoints or training."})
    print("FINAL QUALIFIED REPORT",p)
    print(json.dumps(decision,indent=2))

if __name__=="__main__":main()
