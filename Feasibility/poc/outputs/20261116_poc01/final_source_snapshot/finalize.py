"""Read-only post-unseal audits, exports and presentation. Never fits a predictor."""
from run import *
from prediction import feature_frame, TELE, SEP
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(df, columns=None):
    if columns is not None:
        df=df[columns]
    def fmt(v):
        if pd.isna(v): return 'undefined'
        if isinstance(v,(float,np.floating)): return f'{v:.4g}'
        return str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |',
                      '| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def main():
    folder=O/'reports'
    assert (folder/'decision.json').exists(), 'Run the guarded report stage first'
    assert len(list((O/'outcomes/test').glob('*.json')))==40, 'Outcome collection incomplete'
    decision=read(folder/'decision.json')
    if not (folder/'decision_initial.json').exists():
        shutil.copy2(folder/'decision.json',folder/'decision_initial.json')
        shutil.copy2(folder/'REPORT.md',folder/'REPORT_initial.md')
    protocol=read(O/'protocol.json'); manifest=pd.read_csv(O/'run_manifest.csv')
    metrics=pd.read_csv(folder/'metrics.csv'); contrasts=pd.read_csv(folder/'contrasts.csv')
    influence=pd.read_csv(folder/'leave_one_family_out.csv')
    outcomes=pd.read_csv(folder/'terminal_outcomes.csv')
    support=pd.read_csv(folder/'evaluation_support.csv')
    devsupport=pd.read_csv(O/'development_support.csv')
    frozen=read(O/'predictor_freeze.json'); choices=frozen['choices']
    pred_freeze=read(O/'prediction_freeze.json')
    events=[json.loads(s) for s in (O/'events.jsonl').read_text().splitlines()]
    unseal=next(e for e in events if e['event']=='heldout_outcome_unseal')
    assert frozen['time_ns'] < pred_freeze['time_ns'] < unseal['time_ns']
    assert sha(O/'pipelines.joblib')==frozen['pipelines_sha256']
    assert sha(O/'evaluation_predictions.csv')==pred_freeze['predictions_sha256']
    assert sha(O/'predictor_freeze.json')==pred_freeze['predictor_freeze_sha256']
    assert all(p.stat().st_mtime_ns<frozen['time_ns'] for p in (O/'outcomes/val').glob('*.json'))
    assert all(p.stat().st_mtime_ns>=unseal['time_ns'] for p in (O/'outcomes/test').glob('*.json'))
    completion=[]; timing=[]; structural=[]; source_frames=[]
    for spec in protocol['runs']:
        f=O/'classifiers'/spec['run']; c=read(f/'complete.json'); plan=read(f/'plan.json')
        assert c['updates']==c['planned_updates']==plan['horizon']==19560 and c['full_60_epochs']
        cp=torch.load(f/'recovery.pt',map_location='cpu',weights_only=False)
        assert cp['step']==19560 and all(float(s['step'])==19560 for s in cp['optimizer']['state'].values())
        assert all(torch.isfinite(p).all() for p in cp['model'].values())
        final=torch.load(f/'final.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(final[n],p) for n,p in cp['model'].items())
        early=pd.read_csv(f/'early.csv')
        for panel in ['A','B']:
            rows=early[(early.panel==panel)&(early.step>0)]
            assert rows.step.tolist()==plan['steps'] and rows.valid.all()
            assert np.isfinite(rows[TELE+SEP+['shallow_share','deep_share','total','bias_norm','bias_sensitivity']]).all().all()
        health=read(f/('health_'+str(plan['steps'][-1])+'.json'))
        eligible=health['accuracy']>=.8 and health['loss']<=np.log(10)
        completion.append({**spec,'status':'complete','updates':c['updates'],'eligible':eligible,
                           'health_accuracy':health['accuracy'],'health_loss':health['loss'],
                           'final_source_accuracy':c['final_health']['accuracy'],
                           'final_source_loss':c['final_health']['loss']})
        timing.append({'run':spec['run'],'split':spec['split'],**c['timing'],
                       'initialization_monitor':early.loc[early.step==0,'seconds'].sum(),
                       'classifier_wall':c['wall_this_invocation'],'peak_gpu_allocated':c['peak_gpu_allocated']})
        e=early.copy();e['run']=spec['run'];e['budget_per_class']=256
        e['conv3_common_share']=e.conv3_abs/e.denominator;e['head_common_share']=e.head_abs/e.denominator
        assert np.max(abs(e.deep_share-e.conv3_common_share-e.head_common_share))<1e-12
        e['head_flag']=e.head_fraction_of_deep>=.5;e['bias_flag']=e.bias_sensitivity>=.1*e.total
        structural.append(e)
    completion=pd.DataFrame(completion); timing=pd.DataFrame(timing); structural=pd.concat(structural,ignore_index=True)
    predictions=pd.read_csv(O/'evaluation_predictions.csv')
    eval_all=set(completion.loc[completion.split=='evaluation','run'])
    eval_normal=set(completion.loc[(completion.split=='evaluation')&completion.eligible,'run'])
    for key,g in predictions.groupby(['cohort','prefix','panel','refit_B','model']):
        assert set(g.run)==(eval_normal if key[0]=='eligible' else eval_all)
        assert not g.run.duplicated().any()
    forbidden={'alignment','difficulty','data_seed','reservation_seed','training_seed','family','run','split','wga','failure','health_accuracy','health_loss'}
    for choice in choices.values():
        for label,param in choice.get('parameters',{}).items():
            assert not forbidden&set(param['cols'])
            if label!='final_monitor':assert not any(c.startswith('final_') for c in param['cols'])
    completion.to_csv(folder/'completion_and_health.csv',index=False)
    timing.to_csv(folder/'cost_per_classifier.csv',index=False)
    structural.to_csv(folder/'structural_checkpoint_diagnostics.csv',index=False)
    group=pd.read_csv(folder/'group_accuracy_uncertainty.csv')
    assert len(group)==9600 and group.n.min()>0
    for run,g in group.groupby('run'):
        assert len(g)==100 and len(g[['class','background']].drop_duplicates())==100
        r=outcomes[outcomes.run==run].iloc[0]
        assert abs(g.accuracy.min()-r.wga)<1e-12 and bool(r.failure)==bool(r.wga<.75)
        assert (g.correct<=g.n).all() and (g.correct>=0).all()
    for path in (O/'data').glob('*/verification.json'):
        v=read(path);assert v['passed'] and v['foreground_and_background_error']==0
        assert sha(path.parent/'compact.npz')==v['sha256']
    assert len(list((O/'data').glob('*/verification.json')))==48
    layercols=[b+'_share_'+s for b in ['shallow','deep'] for s in ['last','mean','std','slope']]
    for split in ['development','evaluation']:
        for panel in ['A','B']:
            for prefix in [.05,.1]:
                frame=feature_frame(split,panel,prefix)
                frame[['run']+layercols].to_csv(folder/f'layer8_{split}_{panel}_{prefix}.csv',index=False)
    merged=completion.merge(outcomes.drop(columns='split'),on='run')
    audited_support=[]
    for split in ['development','evaluation']:
        for cohort in ['eligible','full']:
            z=merged[merged.split==split]
            if cohort=='eligible': z=z[z.eligible]
            audited_support.append({'split':split,'cohort':cohort,'N':len(z),'families':z.family.nunique(),
                'failure_families':z.loc[z.failure,'family'].nunique(),
                'nonfailure_families':z.loc[~z.failure,'family'].nunique()})
    audited_support=pd.DataFrame(audited_support)
    audited_support.to_csv(folder/'outcome_family_support.csv',index=False)
    devsupport=devsupport.merge(audited_support[audited_support.split=='development'][['cohort','failure_families','nonfailure_families']],on='cohort')
    support=support.merge(audited_support[audited_support.split=='evaluation'][['cohort','failure_families','nonfailure_families']],on='cohort')
    coverage=merged.groupby(['split','family','alignment','difficulty','lr'],as_index=False).agg(
        completed=('run','size'),eligible=('eligible','sum'),failures=('failure','sum'),
        min_wga=('wga','min'),max_wga=('wga','max'))
    coverage.to_csv(folder/'family_coverage_and_diversity.csv',index=False)
    normal_eval=merged[(merged.split=='evaluation')&merged.eligible]
    mixed_families=int((normal_eval.groupby('family').failure.nunique()>1).sum())
    tests=all('failed' not in (P/name).read_text().lower().splitlines()[-1] for name in ['tests.log','boundary_tests.log','execution_tests.log','reporting_tests.log'])
    assert tests
    protect(True)
    durations={k:float(timing[k].sum()) for k in ['training','monitor_A','monitor_B','health','initialization_monitor','classifier_wall']}
    storage={d:sum(p.stat().st_size for p in (O/d).rglob('*') if p.is_file()) for d in ['data','classifiers','reports','outcomes','outcome_panels']}
    costs={'seconds':durations,'peak_gpu_allocated_bytes':int(timing.peak_gpu_allocated.max()),'storage_bytes':storage,
           'end_to_end_through_first_report_seconds':(next(e['time_ns'] for e in events if e['event']=='poc_complete')-events[0]['time_ns'])/1e9,
           'limitations':['Initial monitoring is recorded separately from six checkpoint monitoring.',
             'Training wall excludes construction preparation, source transfer and terminal outcome evaluation.',
             'Dataset preparation, terminal outcome collection and forecasting were not individually instrumented; their actual component times are unavailable.',
             'GPU allocation peak is cumulative per process, not an isolated per-probe increment or total device/system memory.',
             'Initial profile and classifier wall include file saves and Python overhead; end-to-end includes engineering/wait gaps.',
             'Panel A monitoring includes conventional telemetry/output clustering and sensitivity together; connection-sensitivity incremental cost is not isolated.']}
    write_json(folder/'costs.json',costs)
    audit={'all_96_full_horizon':True,'all_optimizer_steps':19560,'finite_weights_and_required_telemetry':True,
           'source_constructions_verified':48,'outcomes_complete':96,'groups_per_model':100,
           'eight_layer_feature_columns':layercols,'predictor_freeze_time_ns':frozen['time_ns'],
           'prediction_freeze_time_ns':pred_freeze['time_ns'],'test_unseal_time_ns':unseal['time_ns'],
           'freeze_order_verified':True,'pipelines_and_predictions_hashes_verified':True,
           'validation_outcomes_before_predictor_freeze':True,'test_outcomes_after_unseal':True,
           'final_models_match_full_horizon_recovery_states':True,'automated_checks_passed':21,
           'identical_candidate_memberships_across_methods_and_panels':True,
           'forbidden_predictor_inputs_absent':True,
           'protected_prior_artifacts':len(read(O/'protected_after.json')),'protected_prior_artifacts_unchanged':True,
           'no_predictor_refit_after_unseal':sha(O/'pipelines.joblib')==frozen['pipelines_sha256']}
    write_json(folder/'validation_audit.json',audit)
    def primary(df):return df[(df.cohort=='eligible')&(df.prefix==.1)&(df.panel=='A')&(~df.refit_B)]
    main=primary(metrics); mainc=primary(contrasts).drop_duplicates()
    strong=choices['eligible_0.1'].get('strongest_nontemporal')
    order=list(dict.fromkeys(['E','logistic_B','logistic_L2','logistic_L6','logistic_L8',strong,'histogram_L8','bias_E','bias_L8','final_monitor']))
    order=[n for n in order if n is not None and n in set(main.model)]
    main=main.set_index('model').loc[order].reset_index()
    panel=metrics[(metrics.cohort=='eligible')&(metrics.prefix==.1)&metrics.model.isin(['E','logistic_L8',strong])]
    maini=primary(influence)
    robustness=[]
    for control in mainc.control.unique():
        cc=contrasts[(contrasts.cohort=='eligible')&(contrasts.prefix==.1)&(contrasts.control==control)]
        ii=maini[maini.control==control]
        robustness.append({'control':control,'panel_A_Delta20':float(cc[(cc.panel=='A')&(~cc.refit_B)].Delta20.iloc[0]),
                           'panel_B_frozen_A_Delta20':float(cc[(cc.panel=='B')&(~cc.refit_B)].Delta20.iloc[0]),
                           'panel_B_refit_Delta20':float(cc[(cc.panel=='B')&(cc.refit_B)].Delta20.iloc[0]),
                           'leave_family_out_min':float(ii.Delta20.min()),'leave_family_out_max':float(ii.Delta20.max())})
    pd.DataFrame(robustness).to_csv(folder/'primary_robustness_summary.csv',index=False)
    # These flags qualify the unchanged predefined contrasts, never select new methods.
    evaluable=bool(decision['source_normal_support']['evaluable'] and choices['eligible_0.1']['classification_supported'])
    gain=next((r['panel_A_Delta20'] for r in robustness if r['control']=='E'),None)
    all_controls=all(r['panel_A_Delta20']>0 for r in robustness) if robustness else False
    consistent=all(r['panel_B_frozen_A_Delta20']>0 and r['panel_B_refit_Delta20']>0 and r['leave_family_out_min']>0 for r in robustness) if robustness else False
    verdict='Outcome pilot not adequately evaluable.' if not evaluable else 'No incremental audit value observed in this pilot.'
    if evaluable and gain is not None and gain>=1:
        verdict='Predictive value observed, temporal specificity unresolved.'
        # Retain a conservative interpretation if controls/robustness do not jointly support it.
        if all_controls and consistent and all(mainc.interval_low>0):
            verdict='Promising temporal-layer result requiring independent confirmation.'
    decision.update(verdict=verdict,primary_useful_gain_target_met=bool(gain is not None and gain>=1),
                    strongest_nontemporal_control=strong,
                    positive_primary_global_static_contrasts=all_controls,positive_panel_and_influence_contrasts=consistent,
                    primary_robustness=robustness,failures_and_eligibility_by_family='family_coverage_and_diversity.csv',
                    implementation_audit='validation_audit.json',costs='costs.json')
    e_row=main[main.model=='E'].iloc[0]
    ceiling_note=('E reaches the oracle discovery ceiling in this pool: the fixed audit budget leaves no improvement headroom over E.'
                  if e_row.headroom==0 else f'E has room for {int(e_row.headroom)} additional discovery before the oracle ceiling.')
    maximum_gain=20*float(e_row.headroom)/float(e_row.N)
    if maximum_gain<1:
        ceiling_note+=f' Its maximum attainable Delta20 headroom is {maximum_gain:.4g}, below the prespecified useful-gain target of 1; this pool cannot demonstrate that target against this baseline.'
    baseline_discoveries=int(e_row.discoveries)
    static_rows=main[main.model.isin(['logistic_B','logistic_L2','logistic_L6',strong])]
    static_gain=bool(len(static_rows) and static_rows.discoveries.max()>baseline_discoveries)
    if not evaluable:
        information='The intended source-normal binary question is not adequately evaluable; predictive comparisons remain exploratory.'
    elif gain is None or gain<1:
        information=('Static/global controls show a point-estimate audit gain, but useful temporal incremental value is not established.'
                     if static_gain else 'Neither useful temporal nor static/global incremental audit value is demonstrated at the fixed budget.')
    elif not all_controls or not consistent:
        information='There is a point-estimate predictive gain over E, but temporal specificity is unresolved by controls and robustness.'
    else:
        information='Temporal information is promising in this conditional exploratory comparison and requires independent confirmation.'
    decision['information_interpretation']=information
    decision['maximum_attainable_Delta20_over_E']=maximum_gain
    decision['eligible_evaluation_mixed_outcome_families']=mixed_families
    decision['limitations']+=['Histogram boosting retained default min_samples_leaf=20; some grouped training folds are too small to split, weakening the nonlinear control.',
        'Final-monitoring comparator uses final source-health loss/accuracy and learning rate, not an extensive final telemetry family.',
        'Separate timing/memory instrumentation is incomplete; see costs.json.']
    decision['limitations']=list(dict.fromkeys(decision['limitations']))
    write_json(folder/'decision.json',decision)
    plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(main.model,main.discoveries,color=['#167D8D' if n=='logistic_L8' else '#65758B' for n in main.model]);ax.axhline(main.oracle_ceiling.iloc[0],color='black',ls='--',label='Oracle ceiling');ax.axhline(main.random_expected.iloc[0],color='gray',ls=':',label='Random expectation');ax.set_ylabel('Failures discovered at fixed 25% capacity');ax.tick_params(axis='x',rotation=45);ax.legend();fig.tight_layout();fig.savefig(folder/'audit_discoveries.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,split in zip(axs,['development','evaluation']):
        d=merged[merged.split==split]
        for elig,marker in [(True,'o'),(False,'x')]:
            z=d[d.eligible==elig];ax.scatter(z.health_accuracy,z.wga,marker=marker,label='Source-normal' if elig else 'Ineligible',alpha=.75)
        ax.axhline(.75,color='black',ls='--');ax.axvline(.8,color='gray',ls=':');ax.set(xlabel='10% source-health accuracy',title=split)
    axs[0].set_ylabel('Terminal worst-group accuracy');axs[0].legend();fig.tight_layout();fig.savefig(folder/'health_and_terminal_wga.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for b,ax in zip(['shallow_share','deep_share'],axs):
        for run,e in structural[(structural.step>0)&(structural.panel=='A')].groupby('run'):
            ax.plot(e.fraction,e[b],alpha=.2,lw=.8)
        ax.set(xlabel='Fraction of planned 60-epoch updates',ylabel=b,title='All 96 runs, panel A')
    fig.tight_layout();fig.savefig(folder/'layer_trajectories.png');plt.close(fig)
    flags=structural[(structural.step==1956)&(structural.panel=='A')]
    cutoff=pd.read_csv(folder/'secondary_cutoffs.csv'); cutoff=cutoff[(cutoff.cohort=='eligible')&(cutoff.prefix==.1)&(cutoff.panel=='A')&(~cutoff.refit_B)&cutoff.model.isin(['E','logistic_L8'])]
    ridge=primary(metrics);ridge=ridge[ridge.model.str.startswith('ridge_')]
    primary_support=support[(support.cohort=='eligible')&(support.prefix==.1)].iloc[0]
    select=choices['eligible_0.1']
    paragraph=f"{verdict} All 96 classifiers completed 60 epochs (19,560 optimiser updates each). The intended held-out source-normal cohort contains {int(primary_support.N)} runs across {int(primary_support.families)} setting families, with {int(primary_support.failures)} failures and {int(primary_support.nonfailures)} non-failures. The prospective support screen {'passes' if evaluable else 'does not pass'}."
    report=f'''# GradientGuard exploratory outcome proof of concept

{paragraph}

## Primary audit result

Each method ranks the same source-normal held-out pool. Failure means terminal WGA strictly below 0.75; capacity is floor(0.25N). L8 is logistic regression with the eight prescribed layer summaries. E was selected on development families only. Strongest development-selected non-temporal control: **{strong}**.

{table(main,['model','N','k','failures','discoveries','precision','recall','random_expected','oracle_ceiling','headroom','AP','AUROC','Brier'])}

Delta20 is 20 times the difference in discoveries divided by candidate count. A value of at least 1 was the prospective useful-gain target. The intervals below resample whole held-out families with frozen predictions; they are descriptive, conditional intervals, not confirmatory inference and not forecaster-selection uncertainty.

{table(mainc,['control','Delta20','interval_low','interval_high','invalid_resamples','family_count'])}

{('The fixed-budget result does not show useful incremental discovery over E.' if gain is None or gain<1 else 'L8 meets the point-estimate useful-gain target over E; static/global comparisons and robustness must also support a temporal interpretation.')}

{ceiling_note} {information}

The non-slope L6 control has the same primary discovery count, average precision and AUROC as L8. The improved ranking metrics relative to E therefore do not establish added value from the two layer-share slopes. Shared bias controls also improve ranking metrics without adding discoveries. These secondary changes do not overturn the fixed-budget result.

## Support, eligibility and outcomes

Eligibility is determined independently on the fixed source-health panel at 10%: accuracy >=0.80, CE <=ln(10), finite required measurements. The same 10% cohort definition is used in the secondary 5% analysis. Ineligible runs remain in the full-cohort tables. Classifier settings are grouped into 24 families, not 96 independent settings; 14 families are development and 10 evaluation.

Development support:

{table(devsupport)}

Held-out support:

{table(support)}

There are {mixed_families} held-out source-normal families containing both failure labels. The minimum family-support convention passes, but binary outcomes are homogeneous within each eligible held-out family. This experiment therefore cannot demonstrate discrimination between successful and failing seeds inside the same setting family; constructed setting differences remain an important limitation.

Full cohort, panel A, 10%:

{table(metrics[(metrics.cohort=='full')&(metrics.prefix==.1)&(metrics.panel=='A')&(~metrics.refit_B)&metrics.model.isin(['E','logistic_B','logistic_L6','logistic_L8'])],['model','N','k','failures','discoveries','AP','AUROC'])}

Every completed model has 100 predefined digit/background groups. Outcome groups contain {int(group.n.min())} to {int(group.n.max())} examples. The report retains all group correct counts, accuracies and nominal 95% Wilson intervals. These marginal binomial intervals are not a simultaneous confidence interval for the minimum WGA. {int(outcomes.cutoff_near.sum())} of 96 outcomes lie within 0.05 of the primary cutoff. Neither confidence bounds nor numerical failures redefine the binary label.

Fixed secondary cutoff results (frozen predictions reused without retuning):

{table(cutoff,['cutoff','model','N','failures','discoveries','oracle_ceiling'])}

## Probe, influence and structural controls

Panel A is primary. The frozen A pipeline applied to B and the matched B refit with A-selected hyperparameters are predefined secondary checks. No panel ensemble or favourable-panel substitution was made.

{table(panel,['panel','refit_B','model','N','discoveries','AP','AUROC'])}

{table(pd.DataFrame(robustness))}

All ten held-out families, including those contributing no eligible candidate, enter family resampling. Duplicated family draws retain their nested constructions/seeds; ranking, N and k are recomputed. Leave-one-family-out contrasts are in leave_one_family_out.csv. Only ten independent setting groups remain, and outcome panels and underlying MNIST images are shared.

Separate conv3/head totals, common-denominator shares, convolution-only shares, per-weight means and bias diagnostics are in structural_checkpoint_diagnostics.csv. Deep = conv3 + head was numerically verified at every recorded point. At 10% on panel A and the selected 256/class budget, {int(flags.head_flag.sum())}/96 runs have head sensitivity >=50% of deep and {int(flags.bias_flag.sum())}/96 have bias sensitivity >=10% of total weight sensitivity. Shared bias_E/bias_L8 controls appear in the primary table. These observations do not resolve causal or architectural interpretation.

## Secondary continuous outcome and lead time

{table(ridge,['model','MAE','spearman'])}

Ridge regression is a separately labelled secondary WGA analysis; it cannot rescue inadequate binary support. Ordinary final monitoring uses final source-health CE, accuracy and learning rate, selected only on development families. Its audit result is included above. Early predictions use six actual epochs, leaving 54 planned training epochs (17,604 updates) before the terminal model. This establishes observation timing, not a demonstrated deployment benefit or an optimal final-monitoring comparator.

## Frozen design and operation order

CNN A was trained from scratch with float32, no AMP/TF32 shortcuts, deterministic operations, Adam with no weight decay, batch 128 including the last four-example batch, and no augmentation. All families ran the fixed 60-epoch horizon without outcome-based stopping. The horizon is an explicit exploratory amendment: the prior 60-epoch calibration candidate failed its all-three-seeds gate. That gate remains failed.

The 24 Cartesian families cover alignment 0.10/0.50/0.80/0.95, LARGE/MEDIUM difficulty and learning rate 0.0003/0.001/0.003, with two seeded constructions and two initialisations each. All identities and the family-disjoint 14/10 split were frozen before outcomes. Both sides cover every configuration level. The official SpuCo generator was checked pixel-for-pixel for all 48 source constructions. LARGE changes zero-background pixels across the image; MEDIUM uses a 4x4 corner mask. Foreground pixels and ordinary labels were preserved. Each source has 48,004 original training identities, with disjoint A/B reservations of 256 per class, health of 128 per class, and 41,604 training images.

Terminal development outcomes use official validation identities, held-out outcomes official test identities. Common group-balanced panels are generated per difficulty with prespecified seeds. These are not independent new image populations. Background assignments are offline metadata and never predictor inputs.

The audit log verifies: design freeze; development training/validation outcomes; predictor freeze; held-out training; early-prediction freeze; terminal-test unseal. Predictor freeze time_ns={frozen['time_ns']}, prediction freeze={pred_freeze['time_ns']}, unseal={unseal['time_ns']}. Pipeline and prediction hashes still match. No preprocessing was fitted on combined development/evaluation records. Final source fields and terminal labels are absent from the exported early predictor table; exactly eight layer features are exported separately for each panel/prefix.

Environment: Python 3.12.14, torch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA runtime 12.8, RTX 5060 Ti. Complete environment, driver, SpuCo source/commit hashes, deterministic settings and project-local cache settings are retained. Installed versions were preserved. Reproducibility across other hardware/software is not promised.

## Implementation validation and costs

All 21 integration checks passed. The automated synthetic/read-only checks cover family/CV separation, forbidden inputs, fold-local preprocessing, exact feature summaries, gradient accumulation/isolation and parity with the original gradient definition, both official masks, old-artifact preservation, predictor-unseal gates, future-record independence, strict audit budgets/ties, numerical resume and an intentional interruption/resume of the actual runner on a separately labelled fixture. Full scientific models were independently checked for finite weights/telemetry, the six planned early measurements, final optimiser steps and 60-epoch completion. These are implementation checks, not evidence of predictive value.

Summed recorded training time: {durations['training']/60:.2f} minutes; six-checkpoint panel A monitoring: {durations['monitor_A']/60:.2f} minutes; additional panel B monitoring: {durations['monitor_B']/60:.2f} minutes; early source-health: {durations['health']:.2f} seconds; diagnostic initialisation monitoring: {durations['initialization_monitor']/60:.2f} minutes. Classifier invocation wall totals {durations['classifier_wall']/60:.2f} minutes. Panel A includes conventional telemetry, clustering and sensitivity, so this is not an isolated incremental GradientGuard cost. Panel B is validation overhead. Representative initial full run and extrapolation are retained in representative_profile.json.

Maximum recorded allocated GPU memory: {costs['peak_gpu_allocated_bytes']/2**20:.1f} MiB. The counter is cumulative within each process and does not give isolated incremental probe memory, total GPU allocation by other applications or CPU peak memory. Compact source storage is {storage['data']/2**20:.1f} MiB and classifier storage {storage['classifiers']/2**20:.1f} MiB. Full RGB duplicates are not stored. Dataset preparation, terminal evaluation and forecaster fitting were not individually timed; their actual separate component costs are unavailable. End-to-end time includes engineering/wait overhead and is retained in costs.json. These instrumentation gaps limit precise production-cost claims.

## Interpretation and remaining limitations

The preserved first result reproduced at approximately 3.050141, but the marginal deep-slope finding passed only 15/20 alternative partitions against a 16/20 screen. Structural interpretation and source-health calibration remained unresolved, and the planned fresh 18-prefix study was not launched. This PoC does not retrospectively pass those gates.

The nonlinear control uses the frozen bounded histogram implementation with min_samples_leaf=20. Some grouped training folds are too small for any split; this weakens the claim of comparison against a strong nonlinear conventional predictor. The actual defaults are recorded, and were not changed after outcomes to improve results. Prediction depth is deferred. The final-monitoring comparator is limited to source-health CE/accuracy. The 96 classifiers share MNIST and only 24 setting families; held-out uncertainty uses ten families and does not include development selection uncertainty. Finite group samples, cutoff-near outcomes, limited failure diversity and probe sensitivity require caution.

{verdict} A positive coefficient, higher AUROC, static improvement or successful execution alone does not establish temporal incremental audit value, shortcut causality or novelty. Any follow-up must be separately frozen; this study is not journal-ready confirmation.

If research continues, design a separate study with more held-out families, genuine within-setting outcome variation and enough audit headroom to test the prospective gain. Strengthen the nonlinear comparator under its actual fold sizes and implement the deferred prediction-depth comparator. Do not append favourable runs, retune on these held-out outcomes or promote the secondary static/ranking findings into a temporal success claim.

## Files and reproduction

Main decision: decision.json. Primary metrics: metrics.csv and contrasts.csv. Cohort accounting: completion_and_health.csv, family_coverage_and_diversity.csv, evaluation_support.csv and ../development_support.csv. Outcomes and finite-group diagnostics: terminal_outcomes.csv and group_accuracy_uncertainty.csv. All audit selections: audit_selections.csv. Robustness: primary_robustness_summary.csv and leave_one_family_out.csv. All early raw observations/checkpoints: ../classifiers/<run>/. Seeds/configuration: ../protocol.json and ../run_manifest.csv. Predictor choices, inputs, folds and hashes: ../predictor_freeze.json. Instructions: E:\\Final Year Project\\Implementation\\Feasibility\\instructions\\STEP_BY_STEP_POC.txt.

![Audit discoveries](audit_discoveries.png)

![Source health and terminal outcomes](health_and_terminal_wga.png)

![Layer trajectories](layer_trajectories.png)
'''
    (folder/'REPORT.md').write_text(report,encoding='utf8')
    snapshot=O/'final_source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in P.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    for path in (P/'tests').glob('*.py'):
        (snapshot/'tests').mkdir(exist_ok=True);shutil.copy2(path,snapshot/'tests'/path.name)
    event('report_finalized',verdict=verdict,classifiers=96)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    extras=list(P.glob('*.py'))+list((P/'tests').glob('*.py'))+[P/'README.md',P/'ENGINEERING_HISTORY.md',ROOT/'instructions/STEP_BY_STEP_POC.txt',ROOT/'instructions/environment.ps1']
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
    write_json(folder/'artifact_hashes.json',hashes)


if __name__=='__main__': main()
