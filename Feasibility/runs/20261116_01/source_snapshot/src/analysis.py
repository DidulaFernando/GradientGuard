"""Descriptive engineering analysis; no fitted forecaster or terminal outcomes."""
import json
import math
import time
from common import ROOT, attempt_dir, config, np, write_json, sha
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS=("shallow","deep")
SUMMARIES=("last","mean","std","slope")
FEATURES=[b+"_"+s for b in BLOCKS for s in SUMMARIES]

def frozen():
    return json.loads((attempt_dir()/"frozen_config.json").read_text())

def measurement_table(size):
    path=attempt_dir()/"reliability"/f"measurements_{size}.csv"
    if path.exists(): return pd.read_csv(path)
    if size==64:
        tables=[pd.read_csv(p) for p in attempt_dir().glob("*_s*/measurements_64.csv")]
        return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    return pd.DataFrame()

def feature_table(table):
    if table.empty: return pd.DataFrame()
    f=frozen(); c=config(); records=[]
    for (run,panel,rep),df in table.groupby(["run","panel","repeat"]):
        for prefix in c["measurement"]["feature_prefixes"]:
            endpoint=round(prefix*f["full_planned_updates"])
            points=[s for s in f["checkpoint_updates"] if s<=endpoint]
            observed=df[(df.step>0)&(df.step<=endpoint)].sort_values("step")
            valid=(observed.step.tolist()==points and observed.valid.astype(bool).all()
                   and np.isfinite(observed[[b+"_share" for b in BLOCKS]].to_numpy()).all())
            row={"run":run,"panel":panel,"repeat":rep,"prefix":prefix,"probe_per_class":int(df.probe_per_class.iloc[0]),
                 "observations":len(observed),"valid":bool(valid)}
            for b in BLOCKS:
                values=observed[b+"_share"].to_numpy()
                x=observed.fraction.to_numpy()
                stats=[values[-1],values.mean(),values.std(ddof=0),np.dot(x-x.mean(),values-values.mean())/np.dot(x-x.mean(),x-x.mean())] if valid else [None]*4
                row.update(dict(zip([b+"_"+s for s in SUMMARIES],stats)))
            records.append(row)
    return pd.DataFrame(records)

def reliability(features):
    c=config(); f=frozen(); records=[]
    expected=len(c["dataset"]["conditions"])*len(c["training"]["seeds"])
    for prefix,df in features.groupby("prefix"):
        a=df[(df.panel=="A")&(df["repeat"]==0)].set_index("run")
        b=df[(df.panel=="B")&(df["repeat"]==0)].set_index("run")
        runs=sorted(set(a.index)|set(b.index))
        steps=[s/f["full_planned_updates"] for s in f["checkpoint_updates"] if s<=round(prefix*f["full_planned_updates"])]
        for feature in FEATURES:
            floor=c["measurement"]["numerical_share_floor"]*(2/(steps[-1]-steps[0]) if feature.endswith("_slope") else 1)
            repeat_noise=0.
            repeat_diffs=[]
            for rep in sorted(df["repeat"].unique()):
                if rep==0: continue
                ar=df[(df.panel=="A")&(df["repeat"]==rep)].set_index("run")
                common=a.index.intersection(ar.index)
                repeat_diffs.extend((a.loc[common,feature]-ar.loc[common,feature]).tolist())
            finite_rep=np.asarray(repeat_diffs,dtype=float)
            if len(finite_rep) and np.isfinite(finite_rep).all(): repeat_noise=float(np.sqrt(np.mean(finite_rep**2))/math.sqrt(2))
            floor=max(floor,repeat_noise)
            valid=len(runs)==expected and set(a.index)==set(b.index) and bool(a.valid.all() and b.valid.all())
            av=a.reindex(runs)[feature].to_numpy(dtype=float); bv=b.reindex(runs)[feature].to_numpy(dtype=float)
            valid=bool(valid and np.isfinite(av).all() and np.isfinite(bv).all())
            noise=spread=ratio=screen_ratio=None
            status="invalid observations / incomplete runs"
            if valid:
                noise=float(np.sqrt(np.mean((av-bv)**2))/math.sqrt(2))
                spread=float(np.std((av+bv)/2,ddof=1))
                ratio=spread/noise if noise>0 else None
                screen_ratio=spread/max(noise,floor)
                status="flat / below numerical floor" if spread<=floor else ("zero sampling noise; floor-limited" if noise==0 else "finite")
            records.append({"prefix":prefix,"probe_per_class":int(df.probe_per_class.iloc[0]),"feature":feature,"runs":len(runs),"valid":valid,
                            "noise":noise,"spread":spread,"raw_spread_noise_ratio":ratio,"numerical_repeat_noise":repeat_noise,
                            "numerical_floor":floor,"screen_ratio":screen_ratio,"status":status,
                            "passes_screen":bool(valid and screen_ratio>=c["measurement"]["ratio_screen"] and spread>floor)})
    return pd.DataFrame(records)

def needs_larger_probes():
    ft=feature_table(measurement_table(64))
    if ft.empty: return False
    rr=reliability(ft)
    return not bool(rr.passes_screen.all())

def mdtable(df):
    """Avoid an additional tabulate dependency."""
    def cell(x):
        if isinstance(x,(float,np.floating)): return f"{x:.6g}" if math.isfinite(x) else "undefined"
        return str(x).replace("|","/")
    return "| "+" | ".join(map(str,df.columns))+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(cell(x) for x in row)+" |" for row in df.itertuples(index=False,name=None))

def plot_profiles(table,folder):
    base=table[table["repeat"]==0]
    plots=[("primary_shares",[b+"_share" for b in ("shallow","middle","deep")],"Primary block share"),
           ("four_layer_totals",[l+"_abs" for l in ("conv1","conv2","conv3","head")],"Sum |weight × mean-loss gradient|"),
           ("signed_layer_totals",[l+"_signed" for l in ("conv1","conv2","conv3","head")],"Signed sum weight × gradient"),
           ("convolution_profiles",[l+"_conv_share" for l in ("conv1","conv2","conv3")],"Convolution-only share"),
           ("per_weight_means",[l+"_per_weight" for l in ("conv1","conv2","conv3","head")],"Mean absolute sensitivity per weight"),
           ("bias_diagnostics",["bias_norm","bias_sensitivity","head_fraction_of_deep"],"Separate bias / aggregation diagnostics")]
    colors={"independent":"#2474ab","correlated":"#c45b24"}
    for name,cols,title in plots:
        fig,axes=plt.subplots(1,len(cols),figsize=(4*len(cols),3.7),squeeze=False)
        for ax,col in zip(axes[0],cols):
            for (run,panel),df in base.groupby(["run","panel"]):
                df=df.sort_values("fraction"); cond=df.condition.iloc[0]
                ax.plot(100*df.fraction,df[col],color=colors[cond],linestyle="-" if panel=="A" else "--",alpha=.8,
                        marker="o",markersize=3,label=run+"/"+panel)
            ax.set_title(col); ax.set_xlabel("% of frozen full horizon"); ax.grid(alpha=.2)
        axes[0,0].set_ylabel(title)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=6,bbox_to_anchor=(.5,-.04))
        fig.tight_layout(rect=(0,.12,1,1)); fig.savefig(folder/(name+".png"),dpi=150,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for run,df in base[base.panel=="A"].groupby("run"):
        df=df.sort_values("fraction")
        axes[0].plot(100*df.fraction,df.health_accuracy,marker="o",label=run)
        axes[1].plot(100*df.fraction,df.health_loss,marker="o",label=run)
    axes[0].axhline(.8,color="black",linestyle=":"); axes[1].axhline(math.log(10),color="black",linestyle=":")
    axes[0].set_ylabel("Source-health accuracy"); axes[1].set_ylabel("Mean source-health CE")
    for ax in axes: ax.set_xlabel("% of full horizon"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/"source_health.png",dpi=150); plt.close(fig)

def report():
    start=time.perf_counter(); c=config(); f=frozen()
    folder=attempt_dir("reports"); figures=folder/"figures"; figures.mkdir(exist_ok=True)
    tables=folder/"tables"; tables.mkdir(exist_ok=True)
    rr=[]; ff=[]; measured_sizes=[]
    for size in c["measurement"]["probe_sizes"]:
        table=measurement_table(size)
        if table.empty: continue
        measured_sizes.append(size)
        ft=feature_table(table); rel=reliability(ft)
        # Features deliberately carry no class/background annotations or manipulation outcomes.
        ft.to_csv(tables/f"eight_features_{size}.csv",index=False)
        ft[ft["repeat"]==0].to_csv(tables/f"forecasting_features_{size}.csv",index=False)
        rel.to_csv(tables/f"reliability_{size}.csv",index=False)
        table.to_csv(tables/f"checkpoint_measurements_{size}.csv",index=False)
        rr.append(rel); ff.append(ft)
    assert measured_sizes,"No completed measurements to report"
    combined=pd.concat(rr,ignore_index=True); combined.to_csv(tables/"reliability_all.csv",index=False)
    original=measurement_table(64); plot_profiles(original,figures)
    largest=max(measured_sizes)
    # All eight features at both prespecified prefixes, each plotted for both panels.
    ft=pd.concat(ff,ignore_index=True)
    for prefix in c["measurement"]["feature_prefixes"]:
        fig,axes=plt.subplots(2,4,figsize=(16,7))
        df=ft[(ft.probe_per_class==largest)&(ft.prefix==prefix)&(ft["repeat"]==0)]
        runs=sorted(df.run.unique())
        for ax,feature in zip(axes.flat,FEATURES):
            for panel,offset,color in (("A",-.12,"#2474ab"),("B",.12,"#c45b24")):
                vals=df[df.panel==panel].set_index("run").reindex(runs)[feature]
                ax.scatter(np.arange(len(runs))+offset,vals,label=panel,color=color)
            ax.set_xticks(range(len(runs)),[r.replace("independent","ind").replace("correlated","corr") for r in runs],rotation=40,fontsize=7)
            ax.set_title(feature); ax.grid(alpha=.2)
        axes.flat[0].legend(); fig.suptitle(f"All eight features at {100*prefix:g}% / {largest} examples per class")
        fig.tight_layout(); fig.savefig(figures/f"eight_features_{prefix}.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,prefix in zip(axes,c["measurement"]["feature_prefixes"]):
        for size in measured_sizes:
            z=combined[(combined.prefix==prefix)&(combined.probe_per_class==size)].set_index("feature").reindex(FEATURES)
            ax.plot(range(8),z.screen_ratio,marker="o",label=f"{size}/class")
        ax.axhline(3,color="black",linestyle=":"); ax.set_yscale("symlog",linthresh=1)
        ax.set_xticks(range(8),FEATURES,rotation=55,ha="right",fontsize=8); ax.set_title(f"{100*prefix:g}% prefix")
        ax.set_ylabel("Spread / max(sampling noise, numerical floor)"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures/"reliability_all_features.png",dpi=150); plt.close(fig)
    base=original[(original["repeat"]==0)&(original.panel=="A")]
    endpoints=base[base.step==f["endpoint"]].copy()
    endpoints[["run","health_accuracy","health_loss","eligible"]].to_csv(tables/"source_health_endpoints.csv",index=False)
    time_rows=[]; recolor=[]; completed=[]
    for condition in c["dataset"]["conditions"]:
        for seed in c["training"]["seeds"]:
            run=f"{condition}_s{seed}"; runfolder=attempt_dir()/run
            if (runfolder/"complete.json").exists():
                completed.append(run)
                t=json.loads((runfolder/"timing.json").read_text()); t.pop("training_loss",None)
                t["run"]=run; t["A_monitor_training_ratio"]=t["primary_A_seconds"]/t["training_seconds"]
                t["AB_monitor_training_ratio"]=(t["primary_A_seconds"]+t["validation_B_seconds"])/t["training_seconds"]
                time_rows.append(t)
                rc=json.loads((runfolder/"offline_recolour.json").read_text()); rc["run"]=run; recolor.append(rc)
    timing=pd.DataFrame(time_rows); recol=pd.DataFrame(recolor)
    timing.to_csv(tables/"timing.csv",index=False); recol.to_csv(tables/"offline_recolour.csv",index=False)
    numeric_cols=[col for col in original if col.endswith("_abs") or col.endswith("_share")]
    jitter=[]
    for (run,step),df in original[original.panel=="A"].groupby(["run","step"]):
        if len(df)<2: continue
        jitter.append({"run":run,"step":int(step),"max_difference":float(np.max(np.ptp(df[numeric_cols].to_numpy(dtype=float),axis=0)))})
    pd.DataFrame(jitter).to_csv(tables/"identical_probe_jitter.csv",index=False)
    structural=endpoints[["run","conv1_abs","conv2_abs","conv3_abs","head_abs","bias_norm","bias_sensitivity","total","head_fraction_of_deep"]].copy()
    structural["bias_sensitivity_over_weight_total"]=structural.bias_sensitivity/structural.total
    structural["head_aggregation_flag"]=structural.head_fraction_of_deep>=.5
    structural["bias_flag"]=structural.bias_sensitivity_over_weight_total>=.1
    structural.to_csv(tables/"structural_endpoints.csv",index=False)
    latest=combined[(combined.probe_per_class==largest)&(combined.prefix==.1)]
    level_pass=latest[~latest.feature.str.endswith("_slope")].passes_screen.any()
    slope_pass=latest[latest.feature.str.endswith("_slope")].passes_screen.any()
    check=json.loads((attempt_dir()/"correctness_status.json").read_text())
    all_complete=len(completed)==6
    jitter_max=max((r["max_difference"] for r in jitter),default=None)
    repeat_stable=jitter_max is not None and jitter_max<=1e-5
    flags=bool(structural.head_aggregation_flag.any() or structural.bias_flag.any())
    cost_ratio=float((timing.primary_A_seconds.sum()+timing.validation_B_seconds.sum())/timing.training_seconds.sum()) if len(timing) else None
    cost_flag=cost_ratio is not None and cost_ratio>c["measurement"]["excessive_monitoring_ratio"]
    # Contrast is descriptive; fixed 5-point flip-rate difference is only a transparent diagnostic screen.
    contrasts=recol.groupby(recol.run.str.split("_s").str[0])[["flip_rate","mean_true_class_probability_change"]].mean() if len(recol) else pd.DataFrame()
    contrast=float(contrasts.loc["correlated","flip_rate"]-contrasts.loc["independent","flip_rate"]) if len(contrasts)==2 else None
    if not check["passed"]: verdict="correctness_failure_repair_before_training"
    elif not all_complete: verdict="incomplete_execution_no_feasibility_conclusion"
    elif not repeat_stable: verdict="identical_probe_instability_resolve_numerics"
    elif not level_pass: verdict="sampling_dominated_reconsider_measurement_feasibility" if largest==256 else "larger_probe_followup_required"
    elif not slope_pass: verdict="levels_reliable_temporal_feasibility_unresolved"
    elif flags: verdict="reliable_level_and_slope_variation_targeted_structural_control_required"
    elif cost_flag: verdict="reliable_variation_revise_monitoring_budget_and_revalidate"
    else: verdict="proceed_to_larger_completed_training_outcome_pilot"
    decision={"attempt":c["attempt"],"decision":verdict,"completed_runs":completed,"correctness_passed":check["passed"],"repeat_stable":repeat_stable,
              "numerical_jitter_max":jitter_max,"probe_sizes_measured":measured_sizes,"selected_probe_budget_per_class":largest,
              "at_least_one_level_passes":bool(level_pass),"at_least_one_slope_passes":bool(slope_pass),"head_or_bias_diagnostic_flag":flags,
              "source_health_eligible_count":int(endpoints.eligible.sum()),"AB_monitor_training_ratio":cost_ratio,"excessive_monitoring_flag":cost_flag,
              "correlated_minus_independent_flip_rate":contrast,"background_contrast_absent_or_weak":bool(contrast is None or contrast<=.05),
              "scientific_scope":"Descriptive diagnostics across two fixed dataset conditions with three paired seeds. No population inference, WGA prediction, audit gain, mechanism or novelty claim.",
              "configuration_sha256":f["configuration_sha256"]}
    write_json(folder/"decision.json",decision)
    # All files are local; report references are relative for portability within the attempt.
    data_ver=json.loads((data_dir_path()/"verification.json").read_text())
    provenance=json.loads((attempt_dir()/"provenance.json").read_text())
    checks=json.loads((attempt_dir()/"correctness_metrics.json").read_text())
    health_cols=endpoints[["run","health_accuracy","health_loss","eligible"]]
    total_train=float(timing.training_seconds.sum()); total_A=float(timing.primary_A_seconds.sum()); total_B=float(timing.validation_B_seconds.sum())
    additional=0.
    for size in measured_sizes:
        tt=measurement_table(size)
        if size==64: tt=tt[tt["repeat"]>0]
        additional+=float(tt.seconds.sum())
    summary_bytes=sum(p.stat().st_size for p in tables.glob("*.csv"))
    decision["summary_storage_bytes"]=summary_bytes
    decision["report_generation_seconds"]=time.perf_counter()-start
    write_json(folder/"decision.json",decision)
    relview=combined[["probe_per_class","prefix","feature","noise","spread","numerical_floor","screen_ratio","passes_screen"]]
    text=f"""# GradientGuard first feasibility test - attempt {c['attempt']}

## Result and scope

**Decision: `{verdict}`.** Completed {len(completed)}/6 real prefixes. Correctness: {check['passed']}. Identical-probe numerical stability: {repeat_stable} (maximum summary difference {jitter_max}). At 10%, at least one retained level passes the frozen ratio >=3 screen: {bool(level_pass)}; at least one slope: {bool(slope_pass)}. Probe budgets measured: {measured_sizes}; final diagnostic budget: {largest} examples/class/panel.

The question is whether early layer-wise summaries vary reproducibly between these runs beyond numerical and probe sampling variation. These are **two fixed controlled settings with three paired seeds**, sharing benchmark identities, not six independent setting families. A passing screen is engineering evidence on these settings. It establishes no final WGA prediction, audit discovery gain, generalisation significance, methodological novelty or causal learning mechanism. No forecasting model was trained and no terminal validation/test evaluation was performed.

## Frozen design and data verification

Empty project inspected before modification; no applicable AGENTS.md or existing intended mapping/horizon. The unchanged proposal supplied CNN A's structure and left mapping/horizon open. User defaults: ten separate digits, fixed HSV palette, MAGNITUDE_LARGE, no label/core noise, Adam lr=.001, betas=(.9,.999), weight_decay=0, batch=128, no augmentation, 20 planned epochs, no dropped final batch. Configuration hash: `{f['configuration_sha256']}`. The resolved configuration and rationale were written before pilot training in `../../runs/{c['attempt']}/frozen_config.json`.

Training examples: {f['train_n']}; updates/epoch: {f['updates_per_epoch']}; planned horizon: {f['full_planned_updates']} updates; actual endpoint: {f['endpoint']}. Checkpoint updates: {f['checkpoint_updates']}; fractions: {f['actual_fractions']}. Initialization is diagnostic and excluded from summaries. Full horizon was never adjusted from observed curves.

SpuCo official commit `{provenance['spuco_commit']}`, installed version `{provenance['packages'].get('spuco')}`. The inspected generator assigns its class-associated background to the configured fraction; independence is 1/K=.1. Official source-training allocation excludes its fixed validation IDs. Reserve, by ordinary class only, 256/class in each of A and B and 128/class for health. Nested ranks fix 64,128,256 probes. Training IDs, reservations and digit order are identical across conditions. One initialization per seed is saved and shared across conditions. Data seed=1701, reservation seed=2903; initialization/training seeds=11,22,33; epoch permutation Generator seed=training_seed*100000+epoch. No DataLoader workers or stochastic transforms are used.

Verification passed: {data_ver['passed']}. Source identity disjointness, official validation exclusion, labels, background palette, finite RGB values, and **every nonzero foreground pixel** were checked. Pixel errors are zero. Realised alignments and all role/class contingency tables are in the data directory. The torchvision downloader fetched its standard raw archives; test images/outcomes were not used. Background annotations remain in `data/{c['attempt']}/offline_diagnostics` and are absent from forecasting feature files.

![Matched identities](../../data/{c['attempt']}/matched_images.png)

## Implementation correctness

CNN A: 3→16→32→64 convolutions, 3x3/pad1 with trainable biases and ReLU; max pool after first two stages; global spatial average; 64→10 biased head. No BN/dropout. Primary shallow=conv1 weights, middle=conv2 weights, deep=conv3+head weights. Biases excluded. Per-layer signed/absolute totals, means per weight, convolution-only shares, head fraction and bias norms/sensitivities are separate diagnostics.

At frozen weights, gradients of each microbatch mean CE are accumulated with microbatch_size/total_probe_size; only then compute |weight*gradient| and aggregate. autograd.grad avoids .grad writes. Evaluation mode, module training flags and Python/NumPy/CPU/CUDA RNG states are restored. Optimizer is not passed to measurement. Gradients are discarded after summarization. Mean-loss gradient accumulation uses model dtype (float32 training; float64 analytical checks). epsilon=1e-12; denominator and relative epsilon logged; zero/nonfinite or >=1% stabilizer contribution invalidates shares.

Analytical state A normalized sensitivities: {checks.get('analytical_A',{}).get('normalized_sensitivities')}; state B: {checks.get('analytical_B',{}).get('normalized_sensitivities')}. These are correctness fixtures, not feasibility evidence. Full actual errors, gradcheck settings, isolation checks and neutrality comparisons are in `../../runs/{c['attempt']}/correctness_metrics.json`. pytest log/JUnit histories preserve attempted checks. Analytical tolerance: atol=1e-9,rtol=1e-7; float32 gradients/totals atol=1e-6,rtol=1e-4, with a separate 1e-5 block-share screen. Gradcheck eps=1e-6,atol=1e-5,rtol=.001. Tolerances were not relaxed after failures.

Checks cover embedded analytical states, selected multiplicative gates, the common-gate signed/absolute distinction, unequal-final microbatch invariance, positive channel+bias rescaling with outgoing compensation, repeated frozen probes, unchanged parameters/optimizer/RNG/gradients/modes, and deterministic training with/without probes on CPU and CUDA. A separate 12-update smoke run is labelled SMOKE ONLY and excluded from all feasibility tables.

## All six real prefixes and source health

Keep every run, including ineligible runs. Eligibility is accuracy>=.80, mean CE<=ln(10), finite loss and telemetry at 10%; eligibility never filters the reliability calculation.

{mdtable(health_cols)}

![Source health](figures/source_health.png)

## Reliability of all eight retained features

At 5% and 10%, each A/B panel separately gives shallow/deep last, checkpoint mean, population SD and OLS slope versus actual full-horizon fraction. Initialization excluded; no interpolation; any missing/invalid checkpoint invalidates the complete feature vector. Noise=RMS(A-B)/sqrt(2); spread=sample SD across six runwise (A+B)/2 values (ddof=1). Raw ratios and zero/flat/invalid statuses are retained in CSV.

The screen uses spread/max(noise,numerical_floor). A conservative prospective level/mean/SD floor is 1e-5 share units; slope floor is 2e-5 divided by the prefix's last-minus-first observed fraction. A measured repeated-feature RMS/sqrt(2) larger than this replaces the floor. Exact zero noise gives undefined raw ratio, never an artificial finite/infinite success; screen uses the floor. A feature below the floor is flat. No confidence intervals or significance tests are computed.

{mdtable(relview)}

Follow-up is checkpoint-only. If any initial feature failed the screen at either prefix, the same checkpoints were remeasured with the fixed nested 128 and 256/class subsets. No retraining or changed horizon/seed/definition. Panels share fixed source-distribution realizations and include class-balanced reservations by design; background groups were not used to select them. Independent A/B differences quantify this one split's sampling sensitivity, not all possible probe pools.

![All feature reliability](figures/reliability_all_features.png)
![Eight features at 5%](figures/eight_features_0.05.png)
![Eight features at 10%](figures/eight_features_0.1.png)

## Architectural, classifier and bias diagnostics

{mdtable(structural)}

Head/bias diagnostic flag: {flags}. The prospectively recorded descriptive screens are head>=50% of deep sensitivity or bias sensitivity>=10% of weight total. Such a flag motivates a targeted architectural/bias control; it does not prove dominance or confounding. The deep block includes two learned layers and unequal parameter counts. Six runs cannot rule out these explanations. Convolution-only profiles and means per weight are provided to inspect aggregation; no bias-free retraining/control was performed here.

![Primary blocks, all runs and both panels](figures/primary_shares.png)
![Four separate layers](figures/four_layer_totals.png)
![Signed separate layers](figures/signed_layer_totals.png)
![Convolution-only profiles](figures/convolution_profiles.png)
![Means per weight](figures/per_weight_means.png)
![Bias and head](figures/bias_diagnostics.png)

## Offline background manipulation check

Use the first 32 nested reserved health examples/class (320 total), alternative background=(original background+1) modulo K. Preserve original foreground pixels and labels exactly; only original zero pixels recoloured. Outcomes stay outside feature exports. Correlated-minus-independent mean prediction-flip rate: {contrast}. A difference <=.05 is descriptively flagged weak/absent, not tested statistically; inspect probability changes too.

{mdtable(recol[['run','n','flip_rate','mean_true_class_probability_change','original_accuracy','alternative_accuracy','seconds']])}

## Timing, memory and storage

Dataset preparation: {data_ver['seconds']:.2f}s. Total training: {total_train:.2f}s; primary A monitoring: {total_A:.2f}s; additional B monitoring: {total_B:.2f}s. Additional checkpoint reliability measurements: {additional:.2f}s. Warm-up, source health, checkpoint save time and recolouring inference are separate. Report generation: {decision['report_generation_seconds']:.2f}s. Summary CSV storage: {summary_bytes} bytes. Checkpoint and dataset storage are separate reproducibility costs. GPU operations are synchronized for timed regions. Training time includes batch transfer/forward/backward/update and excludes monitors; CPU index permutation setup and data materialization outside measure() are not comprehensively timed. Probe timers include microbatch transfer/gradients/aggregation but exclude caller's indexing of CPU tensors. Therefore wall-clock recurring overhead may be higher than these component timers.

{mdtable(timing)}

Combined A+B monitoring/training ratio: {cost_ratio}; excessive-cost flag (>1): {cost_flag}. This threshold was frozen before analysis and is a pilot budget signal, not a deployment standard. GPU peak allocated/reserved and incremental allocated probe memory are measured with PyTorch counters; process RSS before/after and Windows process peak working set are logged. Process peak working set is cumulative and cannot isolate transient CPU probe allocations. GPU allocator reserved memory differs from actual device-wide usage, and WDDM does not give reliable per-process device totals. No claim of exact system-wide incremental memory is made. Initialization monitoring is separately identifiable by step=0; warm-up was excluded from feature summaries.

## Reproducibility, failed setup attempts and limitations

Python `{provenance['hardware']['python']}`; device `{provenance['hardware']['device']}` / `{provenance['hardware']['gpu']}`; CUDA runtime `{provenance['hardware']['cuda_runtime']}`. Exact versions, OS, RAM, driver/cuDNN/build, process cache paths and source hashes are in provenance.json and requirements-lock.txt. All project assets/caches/temp files are under the requested root; the selected existing bundled interpreter is external read-only. The requested 3.11 standalone installation was blocked by Windows Application Control (_socket/DLL and build interpreter), preserved under `.venv_python311_blocked`. Automatic approval review rejected the signed python.org installer command as blocked by policy. Existing bundled Python 3.12 was used as the compatible fallback; no security setting was changed and no global package environment modified. Installer failure logs are preserved in root runs/. The official SpuCo checkout contains case-colliding demo notebook paths on Windows; dataset source is unchanged.

Deterministic algorithms strict, cudnn deterministic true/benchmark false, TF32/reduced precision disabled, no AMP, float32 training, CPU float64 analytical checks, four CPU threads, num_workers=0 equivalent explicit batch indexing. Training permutation is independent of monitoring RNG. Checkpoints preserve model, Adam, step, RNG, config and code hashes. Resume checks compatibility and starts from the last committed optimizer update, using deterministic epoch ordering. Bitwise reproducibility across different hardware/software is not promised. Three identical probes assess deterministic numerical behavior on this build, not hardware portability. Official validation/test outcomes untouched.

## Decision rules and next actions

- Correctness failure: repair before further training; preserve failed history.
- Identical-probe instability: resolve numerical/stochastic behavior first.
- Reliability improves at larger probes: adopt and document the measured revised probe budget; remeasure saved checkpoints rather than retrain.
- Sampling dominates even at 256/class: reconsider measurement feasibility.
- Reliable levels but flat/unstable slopes: temporal feasibility unresolved.
- Apparent head/bias dominance: require a targeted control before interpreting redistribution.
- Reliable level and slope variation surviving diagnostics: proceed to a larger completed-training outcome pilot, without treating this as prediction evidence.
- Weak/absent background-dependence contrast: investigate construction/manipulation before interpreting a weak setting effect.
- Excessive monitoring cost: revise checkpoint/probe budget and revalidate reliability.

For this attempt the machine-readable decision above applies, with independent cost, manipulation and structural flags retained in decision.json. Nothing in this experiment measures terminal WGA, outcome diversity or audit gains. A larger outcome pilot needs completed training, additional independent setting families and appropriately separated outcomes. See `../../instructions/STEP_BY_STEP.txt` for exact commands and expected artifacts.

## Primary implementation references

- [Official SpuCo repository](https://github.com/BigML-CS-UCLA/SpuCo) - pinned implementation inspected locally.
- [SpuCo dataset guide](https://spuco.readthedocs.io/en/latest/user-guide/spuco_datasets.html).
- [PyTorch local installation](https://pytorch.org/get-started/locally/) and [documented 2.7.1 wheel pair](https://pytorch.org/get-started/previous-versions/).
- [PyTorch randomness guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
"""
    (folder/"REPORT.md").write_text(text,encoding="utf-8")
    print("REPORT",folder/"REPORT.md",flush=True)
    print(json.dumps(decision,indent=2),flush=True)

def data_dir_path():
    return ROOT/"data"/config()["attempt"]
