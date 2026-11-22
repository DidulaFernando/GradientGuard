"""Readable synthesis of frozen amended-study outputs; no fitting or new selection."""
from pathlib import Path
import csv, json, hashlib

ROOT = Path(__file__).resolve().parent
O = ROOT/'runs/20261122_fresh02'
R = ROOT/'reports/20261122_fresh02'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def records(name):
    with (O/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write(p, obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     +['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def main():
    assert read(O/'execution_completion.json')['finished']
    assert read(O/'final_validation_audit.json')['passed']
    gate=read(O/'analysis_gate.json'); support=read(O/'evaluation_support.json')
    assert not gate['passed'], 'This signed-off synthesis is specific to the failed progression gate.'
    metrics=records('metrics.csv'); contrasts=records('contrasts.csv')
    intervals=records('intervals.csv'); influence=records('leave_one_family_out.csv')
    primary=lambda r:r['cohort']=='eligible' and r.get('prefix','0.1')=='0.1'
    a=[r for r in metrics if primary(r) and r['variant']=='A']
    n=support['N']; k=n//3; k25=n//4; fails=support['outcomes']['1']['models']
    banks=['E','logistic_B','logistic_L2','logistic_L6','logistic_N','logistic_NZ','logistic_M','logistic_MZ',
           'histogram_E','histogram_B','histogram_L6','histogram_N','histogram_NZ','histogram_M','histogram_MZ']
    index={r['model']:r for r in a}
    model_table=table(['Frozen model',f'Discoveries / {k} audits',f'Discoveries / {k25} audits','AP','AUROC','Brier'],
        [[b,index[b]['D33'],index[b]['D25'],f"{float(index[b]['AP']):.4f}",f"{float(index[b]['AUROC']):.4f}",f"{float(index[b]['Brier']):.4f}"] for b in banks])
    target=lambda r:r['cohort']=='eligible' and r['added']=='logistic_MZ' and r['base']=='logistic_M' and r['denominator']=='3'
    panels=[r for r in contrasts if target(r)]
    panel_rows=[]
    for p in panels:
        ci=next(r for r in intervals if target(r) and r['variant']==p['variant'])
        lo=[int(r['difference']) for r in influence if target(r) and r['variant']==p['variant']]
        panel_rows.append([p['variant'],p['D_added'],p['D_base'],p['difference'],f"[{float(ci['low']):.4f}, {float(ci['high']):.4f}]",f'{min(lo)} to {max(lo)}'])
    panel_table=table(['Panel procedure','MZ','M','Difference','95% family interval (Delta20)','LOFO difference range'],panel_rows)
    # Complete-budget summaries use all budgets, never a newly selected primary.
    curves=records('discovery_curves.csv'); pairs=[('logistic_MZ','logistic_M'),('logistic_L6','E'),('logistic_L6','logistic_B'),('logistic_NZ','logistic_N')]
    curve_rows=[]
    for added,base in pairs:
        ar={int(r['k']):int(r['discoveries']) for r in curves if primary(r) and r['variant']=='A' and r['model']==added}
        br={int(r['k']):int(r['discoveries']) for r in curves if primary(r) and r['variant']=='A' and r['model']==base}
        assert set(ar)==set(br)==set(range(n+1))
        ds=[ar[q]-br[q] for q in range(n+1)]
        curve_rows.append([added,base,min(ds),max(ds),sum(d>0 for d in ds),sum(d==0 for d in ds),sum(d<0 for d in ds)])
    curve_table=table(['Added model','Control','Minimum difference','Maximum difference','Budgets ahead','Tied','Behind'],curve_rows)
    pr=records('evaluation_predictions.csv'); labels={r['run']:float(r['wga']) for r in records('terminal_outcomes.csv')}
    family_sets={}
    for p in pr:
        if primary(p) and p['variant']=='A' and p['model']=='logistic_MZ':
            family_sets.setdefault(p['family'],set()).add(labels[p['run']]<.75)
    mixed=sum(len(v)==2 for v in family_sets.values())
    costs=read(O/'costs.json'); completion=read(O/'execution_completion.json')
    mem=[read(p).get('peak_gpu_allocated',0) for p in (O/'classifiers').glob('*/complete.json')]
    f=read(O/'predictor_freeze.json'); choices=f['choices']['eligible_0.1']
    # This supplements, rather than changes, the frozen manuscript decision.
    decision=read(R/'decision.json')
    decision['interpretation']={
        'primary_incremental_discovery_observed':False,
        'sensitivity_specific_value':'not_supported_by_primary_matched_comparison',
        'mixed_eligible_evaluation_families':mixed,
        'recommendation':'Revise or pause the current sensitivity-specific representation; no automatic larger replication or manuscript promotion.',
        'limitations':['conditional on successfully constructed source pool','shared MNIST benchmark and one architecture','bootstrap conditional on frozen predictors','panel dependence','isolated incremental Z cost not measured'],
        'postprocessing_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(R/'decision.json',decision)
    write(R/'interpretation_summary.json',{'panel_rows':panel_rows,'whole_curve_rows':curve_rows,'mixed_eligible_families':mixed,'peak_gpu_allocated_bytes_max':max(mem),'primary_model_choices':choices,'costs':costs})
    text=f'''# GradientGuard amended fresh-family validation

Attempt **20261122_fresh02** — 22 November 2026.

**The experiment completed and was adequately evaluable. It did not demonstrate the predefined sensitivity-specific improvement.** At the prospectively frozen one-third capacity, logistic MZ and logistic M each discovered **26 of 34 failures in 35 inspections**. Adding the six connection-sensitivity summaries did not improve that primary decision. This is evidence about the tested representation, controls and available SpuCoMNIST pool, not a disproof of every possible layer-information approach.

## Execution, amendment and preservation

Of 128 planned development models, 124 completed training; four were unavailable because two fixed 0.99-alignment source constructions failed the official generator's group-presence validation. All completed development models were retained and reused, **not retrained**. The original stopped attempt and its failed complete-development execution requirement remain unchanged.

All 64 evaluation source constructions were preflighted after predictor freezing and before evaluation training. Two fixed constructions were rejected by the same official assertion, leaving 124 available evaluation models. All 124 completed 60 epochs, or 19,560 optimiser updates, without numerical divergence. Overall: **256 planned, 248 complete (124 reused + 124 newly trained), eight data-unavailable, zero diverged and zero incomplete**. Development exclusions are families 61 construction 1 and 62 construction 0; evaluation exclusions are families 56 construction 0 and 59 construction 0. Each rejected construction accounts for two planned initialisations. Seeds were not replaced, assertions were not disabled, and other valid constructions remained in their assigned families.

Missingness is concentrated at 0.99 alignment; it is not assumed random or harmless. Results are conditional on successfully constructed candidates. Exact identities, seeds and official rejection reasons are in `unavailable/`, with configuration and family accounting tables in the attempt.

The complete recorded dependency environment matches: Python 3.12.14, torch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA runtime 12.8, RTX 5060 Ti. No environment repair, package change, global installation or security change was needed. Twenty-three automated checks passed. Completed-output audits verified imported development bytes, checkpoint hashes, full horizons, schedules, source/configuration freezes, matched panels/banks and bounded complete curves. All 7,171 protected inputs were unchanged; an independent repository audit also verified 1,813 original scientific files unchanged.

## Information boundary and support

The amendment explicitly changed only the all-128-development execution requirement. Development outcomes were already known. The original design, family assignments, seeds, 60-epoch horizon, health rule, WGA cutoffs, feature banks, tuning grids and statistical gates remained fixed. The amended freeze preceded verified development support and predictor fitting; fitted predictors preceded all-source evaluation preflight/training; early risks and eligibility were frozen before the one-time test unseal. No tuning followed evaluation outcomes.

Development source-normal support: 107 models in 29 families; 38 failures in 14 families and 69 non-failures in 19 families, with four mixed-outcome families and valid five-fold grouped CV. Evaluation source-normal support: **106 models in 28 families; 34 failures in 12 families and 72 non-failures in 20 families**, including {mixed} mixed-outcome families. All original minimum support requirements pass. The full evaluation cohort contains 124 completed models and is reported separately. Trajectories are nested in families and share MNIST; they are not 248 independent experimental settings.

## Frozen primary and secondary audit results

Panel A is primary; observation is the first 10% of planned updates. E is the development-selected conventional predictor. B adds global sensitivity. L2 adds layer endpoints; L6 adds six shallow/deep endpoint, mean and population-SD shares. N adds norm-product allocation; NZ adds those six sensitivity shares. M is the rich matched structural control; MZ adds exactly the same six shares. Slopes are not primary inputs. E and the strongest non-Z control were selected using development families only; the selected non-Z control was **{gate['selected_nonZ']}**.

{model_table}

For N={n}, failure prevalence is {fails/n:.2%}, the primary budget is {k} and the secondary 25% budget is {k25}. Random inspection expects {k*fails/n:.3f} failures at the primary budget; the oracle ceiling is {min(k,fails)}. M and MZ both have eight further discoveries of oracle headroom. The useful-gain target Delta20 >= 1 was attainable in principle, but the observed primary Delta20 is **0**.

The matched primary 95% whole-family interval is **[{gate['interval']['low']:.4f}, {gate['interval']['high']:.4f}]** in Delta20 units, with 5,000 valid draws and zero invalid draws. It includes zero. This interval is conditional on frozen predictors and realised families, not a population equivalence proof or a powered confirmatory claim.

There are limited supporting gains: logistic NZ finds three more failures than N, and L6 finds three more than L2 at one-third capacity. L6 finds one more than E/B at the secondary 25% capacity. Histogram L6 finds 28 at one-third, compared with histogram E/B's 27. These are smaller-control or secondary findings; they do not establish improvement over rich M or satisfy the sensitivity-specific primary gate. All histogram M/MZ primary results tie. No favourable budget or learner is promoted after evaluation.

## Panels, families and whole curves

{panel_table}

A_on_B applies the frozen A predictor to B measurements. B_refit uses the already-frozen A hyperparameter choices with development B fitting, performed before held-out outcomes. Neither replaces panel A. The primary MZ-M tie becomes a **one-discovery loss** under A_on_B and remains a tie under B_refit. All 32 panel-A family omissions retain a zero primary difference; this is absence of a gain, not robustness of a positive effect. Other panel/family comparisons, including gains and losses, remain in the detailed tables.

Complete panel-A curves use every budget 0 through {n}; the counts below include both endpoints and correlated adjacent budgets. They are descriptive, not multiple independent tests.

{curve_table}

![Complete source-normal and full-cohort discovery curves](discovery_curves.png)

![Layer and matched structural comparisons with both frozen capacities](allocation_control_curves.png)

![Complete matched-control differences across probe procedures](matched_control_panel_curves.png)

The separately labelled full cohort contains 45 failures among 124 models. At its recomputed one-third budget of 41 audits, E/B/L6/M/MZ discover 29/30/31/32/32 respectively; MZ-M still ties. At 31 audits (25%), the same methods discover 27/27/27/25/25. These full-cohort observations do not replace the intended source-normal analysis.

All cohorts, prefixes (10% primary and 5% secondary), learner variants, panel variants, fixed 0.70/0.80 failure-cutoff sensitivities, AP/AUROC/Brier metrics and negative comparisons are archived. See `metrics.csv`, `contrasts.csv`, `intervals.csv`, `secondary_cutoffs.csv`, `discovery_curves.csv` and `leave_one_family_out.csv`. Bootstrap and omissions keep whole families intact, including families without eligible models, and recompute candidate sizes and budgets. Intervals are pointwise and exclude forecaster-selection uncertainty. Group-level counts and marginal Wilson intervals are in `group_accuracies.csv`; they are not simultaneous uncertainty bounds for the minimum group. Cutoff-near cases remain visible.

![Source health versus terminal worst-group accuracy](health_outcomes.png)

## Costs and limits

The combined conservative GPU-stage ledger is **{costs['conservative_gpu_budget']['charged_seconds']/3600:.3f} hours**, including 9,401.188 seconds charged to the original attempt, within the unchanged 24-hour cap. The amended all-stage invocation took {completion['wall_this_invocation']/3600:.3f} hours. Summed component timings over all completed imported/new models are training {costs['classifier_component_sums']['training']/3600:.3f} hours, panel A monitoring {costs['classifier_component_sums']['monitor_A']:.1f} seconds, panel B validation {costs['classifier_component_sums']['monitor_B']:.1f} seconds, diagnostic initialisation {costs['classifier_component_sums']['initialization']:.1f} seconds and early health {costs['classifier_component_sums']['health']:.1f} seconds. These component totals are not an extra charge on top of the combined ledger. Per-split per-model records are in `classifier_costs.csv`.

Compact cached evaluation-source arrays use {costs['dataset_bytes']/2**20:.2f} MiB; local classifier checkpoints use {costs['checkpoint_bytes']/2**20:.2f} MiB. Identity/background manifests are additional storage. The largest recorded allocated GPU peak is {max(mem)/2**20:.1f} MiB; allocator peaks may be cumulative within their original process and are not isolated incremental-probe peaks. CPU peak memory is not available. Panel A monitoring includes conventional telemetry, structural diagnostics and bounded output clustering, so an isolated incremental cost for Z is not established. Source preparation, forecaster selection and research validation overhead are not equivalent to the cost of a single production monitoring panel. No economic audit savings have been demonstrated.

## Scientific decision and manuscript

**Implementation correctness:** passed. **Evaluability:** passed. **Sensitivity-specific primary benefit:** not supported in this study. **Operational benefit:** not established. Some secondary/limited-control improvements are worth preserving, but their panel dependence and failure to beat the rich matched primary control prevent promotion to the central claim.

Recommendation: **revise or pause investment in the current sensitivity-specific representation**, discuss these fresh results with the supervisor, and use the matched-control result to narrow the research claim. Do not automatically expand the grid, retune on these outcomes, launch Waterbirds or claim that a larger replication will make the existing feature bank succeed. Any materially revised representation or experiment needs a separately justified protocol and fresh evaluation. The result does not establish causal shortcut detection, sensitivity-specific novelty, transfer to natural images or within-setting discrimination across all families. Prediction depth and full reproductions of competing systems remain missing comparators.

The manuscript-update gate is **failed**. No v1.3 is created; v1.2 and the proposal remain unchanged. This report and decision record include unfavourable results rather than changing the manuscript gate.

## Reproduction and artifacts

Protocol, environment/lock, source snapshot, seed/run/family manifests, readiness, predictor and prediction freezes, chronological events, source preflight, early features, terminal group outcomes, cost records and all scalar analyses are under `layer_allocation/runs/20261122_fresh02`. Binary classifier weights and raw/cached arrays stay project-local; compact evidence and frozen forecasters are published according to repository policy.

Instructions: `layer_allocation/amended_validation/instructions/STEP_BY_STEP.txt`. Use the documented resume wrapper only if a process was interrupted; never run it concurrently. The completed-output audit is `layer_allocation/complete_amended_validation.py`; readable synthesis is `layer_allocation/interpret_amended_validation.py`. Reports regenerate from existing frozen evidence without classifier training, forecaster refitting or new outcome selection. Run the completion audit again after this synthesis to refresh preservation and final artifact hashes.
'''
    (R/'REPORT.md').write_text(text,encoding='utf8')
    write(R/'MANUSCRIPT_RECOMMENDATION.json',{'create_v1_3':False,'gate_passed':gate['passed'],'preserve_v1_2':True,'recommendation':decision['interpretation']['recommendation'],'disclose_all_results_if_manually_revising':True})
    print(json.dumps({'report':str(R/'REPORT.md'),'mixed_eligible_families':mixed,'gate_passed':gate['passed']}))

if __name__=='__main__':
    main()
