from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md(df,columns=None):
    if columns:df=df[columns]
    def fmt(v):
        if pd.isna(v):return 'undefined'
        if isinstance(v,(float,np.floating)):return f'{v:.5g}'
        return str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def report():
    get=lambda name:pd.read_csv(ATTEMPT/'tables'/f'{name}.csv')
    curves=get('all_budget_curves');contrasts=get('all_budget_contrasts');marked=get('marked_budgets')
    coef=get('slope_coefficients');comparison=get('L6_L8_comparisons');replace=get('reference_effects')
    lofo=get('leave_one_family_out');intervals=get('bootstrap_intervals');diversity=get('family_diversity')
    normal=curves[(curves.cohort=='eligible')&(curves.variant=='A')]
    main=marked[(marked.cohort=='eligible')&(marked.variant=='A')&(marked.mark.isin(['nine','twelve']))]
    main=main.pivot(index='model',columns='k',values='discoveries').reset_index()
    main.columns=['model','nine_audits','twelve_audits']
    primary_diff=contrasts[(contrasts.cohort=='eligible')&(contrasts.variant=='A')]
    positiveE=primary_diff[(primary_diff.control=='E')&(primary_diff.difference>0)]
    positiveL6=primary_diff[(primary_diff.control=='logistic_L6')&(primary_diff.difference>0)]
    if len(positiveE) and not len(positiveL6):
        recommendation='Revise the contribution toward order-free layer summaries; pursue separately frozen independent validation and novelty assessment.'
        conclusion='Encouraging added discovery at some audit capacities, with temporal-slope specificity unsupported.'
    elif len(positiveE) and len(positiveL6):
        recommendation='Continue to a separately frozen independent validation study, with panel/family sensitivity and temporal attribution resolved first.'
        conclusion='Exploratory layer and temporal advantages occur at some budgets; stability determines how cautiously to proceed.'
    else:
        recommendation='Stop investing in this representation for incremental audit discovery under the tested design; revise the hypothesis before further training.'
        conclusion='No positive incremental discovery over E across the inspected primary-cohort budget curve.'
    summary=[]
    for (cohort,variant,control),g in contrasts.groupby(['cohort','variant','control']):
        summary.append({'cohort':cohort,'variant':variant,'control':control,
            'positive_budgets':','.join(map(str,g.loc[g.difference>0,'k'].tolist())) or 'none',
            'negative_budgets':','.join(map(str,g.loc[g.difference<0,'k'].tolist())) or 'none',
            'tied_budget_count':int((g.difference==0).sum()),'max_extra_discoveries':int(g.difference.max()),'minimum_difference':int(g.difference.min())})
    summary=save_table('curve_direction_summary',summary)
    recall=[]
    for (cohort,variant,model),g in curves.groupby(['cohort','variant','model']):
        r=g[g.discoveries==g.F].iloc[0]
        recall.append({'cohort':cohort,'variant':variant,'model':model,'F':int(r.F),'first_k_all_failures_found':int(r.k)})
    recall=save_table('full_recall_budgets',recall)
    outcomes=get('marked_budgets');stability=[]
    for (cohort,variant,control,requested),g in lofo[lofo.view=='fraction'].groupby(['cohort','variant','control','requested']):
        stability.append({'cohort':cohort,'variant':variant,'control':control,'fraction_rule':requested,
                          'positive_omissions':int((g.difference>0).sum()),'tied_omissions':int((g.difference==0).sum()),
                          'negative_omissions':int((g.difference<0).sum()),'min_Delta20':float(g.Delta20.min()),'max_Delta20':float(g.Delta20.max())})
    stability=save_table('influence_summary',stability)
    figures=ATTEMPT/'figures';plt.rcParams.update({'font.size':9,'figure.dpi':145})
    colors={'E':'#64748b','logistic_B':'#b45309','logistic_L6':'#059669','logistic_L8':'#2563eb','histogram_B':'#a855f7'}
    for cohort in ['eligible','full']:
        fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
        for ax,variant in zip(axs,['A','B_frozen_A','B_existing_refit']):
            z=curves[(curves.cohort==cohort)&(curves.variant==variant)]
            for model,g in z.groupby('model'):
                ax.step(g.k,g.discoveries,where='post',label=model,color=colors[model],alpha=.9)
            reference=z[z.model=='E'];ax.plot(reference.k,reference.oracle_ceiling,'k--',alpha=.5,label='Oracle')
            ax.plot(reference.k,reference.random_expected,':',color='gray',label='Random expectation')
            ax.axvline(9,color='gray',ls=':',alpha=.5);ax.axvline(12,color='gray',ls='--',alpha=.5)
            ax.scatter([9,12],reference.set_index('k').loc[[9,12],'discoveries'],c=colors['E'],s=18)
            ax.set(xlabel='Models inspected (k)',title=f'{cohort}: {variant}');ax.set_xticks(sorted(set([0,9,12,len(reference)-1])))
        axs[0].set_ylabel('Failing models discovered');axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'discovery_curves_{cohort}.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,control in zip(axs,['E','logistic_B','logistic_L6']):
        for variant in ['A','B_frozen_A','B_existing_refit']:
            d=contrasts[(contrasts.cohort=='eligible')&(contrasts.variant==variant)&(contrasts.control==control)]
            ax.step(d.k,d.difference,where='post',label=variant)
        ax.axhline(0,color='gray',lw=.8);ax.axvline(9,color='gray',ls=':');ax.axvline(12,color='gray',ls='--')
        ax.set(xlabel='Models inspected',ylabel='Additional discoveries',title='L8 minus '+control)
    axs[-1].legend();fig.tight_layout();fig.savefig(figures/'paired_gain_curves.png');plt.close(fig)
    components=get('per_run_slope_contributions');z=components[(components.cohort=='eligible')&(components.variant=='A')]
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(np.arange(len(z)),z.shallow_slope_component,'o-',label='Shallow slope')
    ax.plot(np.arange(len(z)),z.deep_slope_component,'o-',label='Deep slope');ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Evaluation models in identifier order',ylabel='Contribution to L8 log odds',title='Frozen standardized slope contributions');ax.legend();fig.tight_layout();fig.savefig(figures/'slope_contributions.png');plt.close(fig)
    rc=get('reference_curves');fig,ax=plt.subplots(figsize=(8,4))
    z=normal[normal.model=='logistic_L8'];ax.step(z.k,z.discoveries,where='post',label='Original L8',color=colors['logistic_L8'])
    for mode,g in rc[(rc.cohort=='eligible')&(rc.variant=='A')].groupby('replacement'):ax.step(g.k,g.discoveries,where='post',label='Training reference: '+mode,alpha=.8)
    ax.set(xlabel='Models inspected',ylabel='Failures discovered',title='Frozen L8 input replacement; no refitting');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'slope_replacement_curves.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,f in zip(axs,['1/4','1/3']):
        z=lofo[(lofo.cohort=='eligible')&(lofo.variant=='A')&(lofo.control=='E')&(lofo.view=='fraction')&(lofo.requested==f)]
        ax.bar(z.omitted_family.astype(str),z.Delta20,color='#2563eb');ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel='Omitted setting family',ylabel='L8 minus E: Delta20',title='Recomputed capacity: '+f)
    fig.tight_layout();fig.savefig(figures/'family_influence.png');plt.close(fig)
    selected_intervals=intervals[(intervals.cohort=='eligible')&(intervals.fraction_rule.isin(['1/4','1/3']))]
    primary_replacement=replace[(replace.cohort=='eligible')&(replace.variant=='A')&(replace.k.isin([9,12]))].drop_duplicates()
    coefficient_table=coef[(coef.cohort=='eligible')&(coef.variant=='A')]
    identity_table=comparison[(comparison.cohort=='eligible')&(comparison.variant=='A')&(comparison.k.isin([9,12]))]
    common=get('common_coefficients');c=common[(common.cohort=='eligible')&(common.variant=='A')]
    env=read(ATTEMPT/'environment/verification.json');repair=read(ATTEMPT/'environment/repair.json')
    pipeline_audit=read(ATTEMPT/'reports/pipeline_immutability.json')
    small_changes=main.to_dict(orient='records')
    decision={'post_hoc':True,'recommendation':recommendation,'conclusion':conclusion,'primary_original_unchanged':True,
        'primary_cohort':{'N':36,'families':9,'F':12,'nonfailures':24},'nine_twelve_discoveries':small_changes,
        'primary_curve_direction':summary[(summary.cohort=='eligible')&(summary.variant=='A')].to_dict(orient='records'),
        'slopes_reach_estimators':bool(coef.retained.all()),'no_fitting':pipeline_audit,
        'environment_complete_lock_matches':env['complete_lock_matches'],'new_training_runs':0,
        'limitations':read(CONFIG)['limitations']+['Input replacement is an off-distribution diagnostic, not a retrained control or causal effect.',
            'The original 25% useful-gain target is not transferred to a favourable post-hoc budget.',
            'Eligible held-out binary labels are homogeneous within families; this does not demonstrate within-setting seed discrimination.']}
    write(ATTEMPT/'reports/decision.json',decision)
    report_text=f'''# GradientGuard diagnostic follow-up — 18 November 2026

**{conclusion}**

Recommendation: **{recommendation}**

This analysis is post-hoc. All outcomes were already known before this diagnostic plan was frozen. It changes no original model, forecast, label, family, threshold or scientific decision. Small improvements are highlighted below with their scope and limitations.

## Audit capacity: primary source-normal cohort

There are 36 models, 12 failures, nine eligible setting families and ten original held-out families. Nine audits are 25%; twelve audits are one third. Discovery counts:

{md(main)}

At nine audits every primary method still discovers eight failures. The original result is unchanged. The oracle ceiling is nine, leaving E room for only one extra discovery and a maximum Delta20 headroom of 20/36=0.5556, below the original target of 1. Increasing capacity creates a different decision; it cannot retrospectively validate the original primary hypothesis.

All budget directions, including losses and ties, are reported rather than selecting only a favourable budget:

{md(summary[(summary.cohort=='eligible')&(summary.variant=='A')],['control','positive_budgets','negative_budgets','tied_budget_count','max_extra_discoveries','minimum_difference'])}

Earliest budget recovering every observed failure (descriptive, selected using known outcomes):

{md(recall[(recall.cohort=='eligible')&(recall.variant=='A')])}

The complete 0..N curves, random expectations, oracle ceilings, precision, recall, headroom and paired differences are in all_budget_curves.csv and all_budget_contrasts.csv. At k=0 precision is undefined; at full capacity all observed failures are recovered. Rank tables identify the actual audit selections.

## Slopes and preprocessing

All frozen pipeline predictions were re-evaluated and compared with the original saved predictions before interpreting slope changes. Feature construction was independently adapted read-only and checked against both development/evaluation panel-A exports. No classifier or forecaster fitting occurred.

{md(coefficient_table,['feature','retained','training_mean','training_scale','fitted_variance','coefficient','standardized_min','standardized_max','contribution_mean_absolute','contribution_max_absolute'])}

The lineage table traces all input columns through imputation, scaling and constant filtering to estimator indices. Variation tables report missingness, distinct values and population SD. A retained, varying feature can still make only a small contribution. Coefficients are conditional associations and can share information with correlated conventional/static features.

L6/L8 equality checks (do not infer probability/ranking equality from equal AP/AUROC):

{md(identity_table,['k','probabilities_exact','probabilities_close','max_absolute_probability_difference','complete_order_identical','selected_set_identical','selected_overlap','AP_L6','AP_L8','AUROC_L6','AUROC_L8'])}

Primary L6 C={c.C_L6.iloc[0]}, L8 C={c.C_L8.iloc[0]}. The maximum absolute difference in shared standardized coefficients is {c.coefficient_difference.abs().max():.6g}; maximum common training-mean difference is {c.training_mean_difference.abs().max():.6g}. L6 and L8 are separately fitted historical models; their comparison is not simply deleting two coefficients from the same fit. Full-cohort and panel-specific hyperparameters and coefficients are retained in common_coefficients.csv.

## Frozen L8 slope-reference replacement

Each slope, and then both slopes, was replaced with the fitted development-training scaler mean in imputed input space. All other input values were preserved. Replacement produces zero standardized slope values and its logit difference was verified against the negative sum of removed coefficient contributions.

{md(primary_replacement,['replacement','k','max_absolute_probability_change','max_logit_identity_error','complete_order_identical','selected_set_identical','original_discoveries','replacement_discoveries'])}

This is inference with a frozen pipeline, not a retrained slope-free model. It may create unusual/off-distribution combinations, is not causal evidence and does not replace L6. Coefficients, per-run contributions, replacement probabilities and complete replacement curves are exported.

## Panels, families and uncertainty

All panel results at nine/twelve inspections:

{md(marked[(marked.cohort=='eligible')&(marked.mark.isin(['nine','twelve']))],['variant','model','k','discoveries','precision','recall','oracle_ceiling'])}

Panel A remains primary. B_frozen_A applies original A models to B. B_existing_refit uses already fitted B pipelines with original A-selected choices; no new refit occurred. Both panels use exactly the same candidates and eligibility.

{md(stability[stability.cohort=='eligible'])}

LOFO removes complete families and recomputes N, rankings and floor(N/4) or floor(N/3). The family with no source-normal model is included and its omission leaves that pool unchanged. Fixed-count nine/twelve views are separately labelled. All eligible held-out families have homogeneous binary labels: no demonstration of successful/failing seed discrimination within a setting is possible.

Pointwise family-bootstrap intervals:

{md(selected_intervals,['variant','control','fraction_rule','k','difference','Delta20','interval_low','interval_high','valid_resamples','invalid_resamples'])}

The bootstrap uses 2,000 paired draws of all ten setting families with seed 130005, preserving constructions and seeds and recomputing pool sizes/rankings/budgets. Original duplicate identifier and invalid empty/single-outcome rules are retained for comparability. Invalid counts are explicit; raw single-class audit counts remain defined. The original 25% interval and LOFO results were reproduced within the frozen tolerance. Intervals condition on selected forecasters, exclude selection uncertainty and are not simultaneous confidence bands over searched budgets. Panel/budget points are overlapping diagnostics, not independent replications.

## Environment and implementation

The matching Python 3.12.14 64-bit interpreter was available in the updated Codex runtime. Only home/executable in .venv/pyvenv.cfg were repaired, with the old config backed up and its historical command retained. All {env['resolved_requirement_lines']} dependency records match the original lock; no package installation or upgrade was needed. GPU: {env['gpu']}; CUDA runtime: {env['cuda_runtime']}. The first full pipeline import failed with a Windows Application Control message. A fresh-process component check and unchanged verification command subsequently succeeded; no security setting was changed. Both failure and retry logs are preserved. The cause of that transient loader failure is not established.

Tests cover original-result reproduction, candidate identities, ties, curve bounds/monotonicity/full recall, invalid inputs, exact fractional budgets, family resampling, slope lineage/reference replacement, no-fitting guards and fitted-state preservation. Test logs and a machine-readable validation record distinguish execution checks from scientific evidence. Previous artifacts are hashed before and after the analysis. No old report command was invoked.

## Qualified decision

{recommendation}

A positive discovery difference at a larger capacity is an encouraging exploratory observation about where the representation might help. It is not proof of temporal specificity or novelty. Compare L8 with L6 and the strongest frozen non-temporal control at that same capacity. If L6 explains the gain, the defensible next direction is order-free layer summaries rather than a slope success claim. Do not append favourable seeds, change the original audit target or retune on these held-out outcomes. Any independent validation requires a new frozen design, adequate setting diversity, within-setting outcome variation where feasible, appropriate baseline capacity and sufficient audit headroom.

## Artifacts

Tables: ../tables/. Figures: ../figures/. Frozen plan and source hashes: ../config/frozen_plan.json. Environment backups/lock comparison: ../environment/. Stage and test logs: ../logs/. Decision: decision.json. Reproduction commands: E:\\Final Year Project\\Implementation\\Feasibility\\instructions\\STEP_BY_STEP_DIAGNOSTICS.txt.

![Primary discovery curves](../figures/discovery_curves_eligible.png)

![All-run secondary curves](../figures/discovery_curves_full.png)

![Paired differences](../figures/paired_gain_curves.png)

![Slope contributions](../figures/slope_contributions.png)

![Slope replacement](../figures/slope_replacement_curves.png)

![Family influence](../figures/family_influence.png)
'''
    (ATTEMPT/'reports/REPORT.md').write_text(report_text,encoding='utf8')
    event('report_complete',recommendation=recommendation)
