"""Descriptive post-hoc evaluation. Never claims a fresh holdout."""
from core import *
from predictors import discovery,ranking,outcomes
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
def curve(y,risk,ids):
    y=np.asarray(y,dtype=int);n=len(y);f=int(y.sum())
    if not n:raise ValueError('Empty cohort')
    d=np.r_[0,np.cumsum(y[ranking(risk,ids)])]
    return [{'k':k,'N':n,'F':f,'fraction':k/n,'discoveries':int(d[k]),'precision':float(d[k]/k) if k else None,'recall':float(d[k]/f) if f else None,'random_expectation':k*f/n,'oracle':min(k,f),'headroom':int(min(k,f)-d[k])} for k in range(n+1)]
def analyse():
    require();pf=read(out()/'prediction_freeze.json');assert sha(out()/'predictions.csv')==pf['sha256']
    assert pf['predictor_freeze_sha256']==sha(out()/'predictor_freeze.json')
    event('post_hoc_evaluation_analysis_started')
    labels=outcomes('evaluation');p=pd.read_csv(out()/'predictions.csv');p['failure']=p.run.map(labels).astype(int)
    metrics=[];curves=[];coverage=[]
    for keys,sub in p.groupby(['cohort','prefix','variant','model']):
        cohort,prefix,variant,model=keys;y=sub.failure.to_numpy();n=len(y)
        assert sub.run.nunique()==n and np.isfinite(sub.risk).all()
        metrics.append(dict(zip(['cohort','prefix','variant','model'],keys))|{'N':n,'families':sub.family.nunique(),'F':int(y.sum()),'nonfailures':int(n-y.sum()),'k25':n//4,'D25':discovery(y,sub.risk,sub.run,n//4),'k33':n//3,'D33':discovery(y,sub.risk,sub.run,n//3),'D9':discovery(y,sub.risk,sub.run,9),'D12':discovery(y,sub.risk,sub.run,12),'AP':float(average_precision_score(y,sub.risk)) if y.sum() else None,'AUROC':float(roc_auc_score(y,sub.risk)) if len(set(y))==2 else None,'Brier':float(brier_score_loss(y,sub.risk))})
        curves.extend(dict(zip(['cohort','prefix','variant','model'],keys))|r for r in curve(y,sub.risk,sub.run))
    met=pd.DataFrame(metrics);met.to_csv(out()/'metrics.csv',index=False);pd.DataFrame(curves).to_csv(out()/'discovery_curves.csv',index=False)
    first=p[(p.prefix==.1)&(p.variant=='A')&(p.model=='logistic_E')]
    for (cohort,family),sub in first.groupby(['cohort','family']):coverage.append({'cohort':cohort,'family':family,'N':len(sub),'F':int(sub.failure.sum()),'nonfailures':int((1-sub.failure).sum())})
    pd.DataFrame(coverage).to_csv(out()/'family_support.csv',index=False)
    allfamilies=sorted(r['family'] for r in read(OLD/'protocol.json')['families'] if r['split']=='evaluation')
    contrasts=[];influence=[];intervals=[]
    for (cohort,variant),sub in p[p.prefix==.1].groupby(['cohort','variant']):
        pivot=sub.pivot(index=['run','family','failure'],columns='model',values='risk').reset_index()
        for kind in cfg()['learners']:
            for added,base in cfg()['comparisons']:
                a=kind+'_'+added;b=kind+'_'+base
                for fraction in cfg()['capacity_fractions']:
                    n=len(pivot);k=int(fraction*n);da=discovery(pivot.failure,pivot[a],pivot.run,k);db=discovery(pivot.failure,pivot[b],pivot.run,k)
                    row={'cohort':cohort,'variant':variant,'learner':kind,'added':added,'base':base,'fraction':fraction,'N':n,'k':k,'D_added':da,'D_base':db,'difference':da-db,'Delta20':20*(da-db)/n};contrasts.append(row)
                    for f in allfamilies:
                        q=pivot[pivot.family!=f];kk=int(len(q)*fraction)
                        influence.append(row|{'omitted_family':f,'N':len(q),'k':kk,'difference':discovery(q.failure,q[a],q.run,kk)-discovery(q.failure,q[b],q.run,kk)})
        # Same whole-family draws across all methods and capacities.
        rng=np.random.default_rng(cfg()['bootstrap_seed']);results={};invalid=0
        for iteration in range(cfg()['bootstrap_replicates']):
            draw=rng.choice(allfamilies,len(allfamilies),replace=True)
            indices=[];ids=[]
            for occurrence,f in enumerate(draw):
                ix=np.flatnonzero(pivot.family.to_numpy()==f);indices.extend(ix);ids.extend([f'{r}__draw{occurrence:02d}' for r in pivot.run.iloc[ix]])
            y=pivot.failure.to_numpy()[indices]
            if len(y)==0 or len(set(y))<2:invalid+=1;continue
            for kind in cfg()['learners']:
                for added,base in cfg()['comparisons']:
                    a=pivot[kind+'_'+added].to_numpy()[indices];b=pivot[kind+'_'+base].to_numpy()[indices]
                    for fraction in cfg()['capacity_fractions']:
                        k=int(len(y)*fraction);delta=20*(discovery(y,a,ids,k)-discovery(y,b,ids,k))/len(y)
                        results.setdefault((kind,added,base,fraction),[]).append(delta)
        for (kind,added,base,fraction),values in results.items():
            lo,hi=np.quantile(values,[.025,.975]);intervals.append({'cohort':cohort,'variant':variant,'learner':kind,'added':added,'base':base,'fraction':fraction,'low':float(lo),'high':float(hi),'valid':len(values),'invalid':invalid,'seed':cfg()['bootstrap_seed']})
    pd.DataFrame(contrasts).to_csv(out()/'contrasts.csv',index=False);pd.DataFrame(influence).to_csv(out()/'leave_one_family_out.csv',index=False);pd.DataFrame(intervals).to_csv(out()/'intervals.csv',index=False)
    render_report()
def render_report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'reports'/cfg()['attempt'];folder.mkdir(parents=True,exist_ok=True)
    met=pd.read_csv(out()/'metrics.csv');curves=pd.read_csv(out()/'discovery_curves.csv');contr=pd.read_csv(out()/'contrasts.csv');interval=pd.read_csv(out()/'intervals.csv');inf=pd.read_csv(out()/'leave_one_family_out.csv')
    f,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,cohort in zip(axes,['eligible','full']):
        df=curves[(curves.cohort==cohort)&(curves.prefix==.1)&(curves.variant=='A')]
        for name in ['logistic_E','logistic_B','logistic_L6','logistic_M','logistic_MZ']:
            d=df[df.model==name];ax.step(d.k,d.discoveries,where='post',label=name.replace('logistic_',''))
        d=df[df.model=='logistic_E'];ax.plot(d.k,d.random_expectation,'--',color='gray',label='Random expectation');ax.plot(d.k,d.oracle,':',color='gray',label='Oracle');ax.axvline(len(d)//4,color='gray',alpha=.25);ax.axvline((len(d)-1)//3,color='gray',alpha=.25)
        ax.set(xlabel='Models inspected',ylabel='Failures discovered',title=cohort+'; consumed PoC outcomes');ax.legend(fontsize=8)
    f.tight_layout();f.savefig(folder/'discovery_curves.png',dpi=160);plt.close(f)
    primary=contr[(contr.cohort=='eligible')&(contr.variant=='A')&(contr.learner=='logistic')&(contr.added=='MZ')&(contr.fraction>.3)].iloc[0]
    pi=interval[(interval.cohort=='eligible')&(interval.variant=='A')&(interval.learner=='logistic')&(interval.added=='MZ')&(interval.fraction>.3)].iloc[0]
    recommendation='Proceed to separately frozen validation only if the narrower question remains worthwhile; do not claim sensitivity-specific novelty from this reused evaluation.'
    verdict='Exploratory conditional added value observed; independent confirmation required' if primary.difference>0 else 'No exploratory discovery increment beyond the matched structural bank at one-third capacity'
    decision={'attempt':cfg()['attempt'],'status':'complete','post_hoc':True,'classifiers_trained':0,'saved_classifiers_measured':96,'checkpoint_panel_measurements':1152,'original_reproduced':read(out()/'original_reproduction.json')['passed'],'measurement_parity':read(out()/'measurement_audit.json'),'primary_exploratory_comparison':primary.to_dict(),'conditional_pointwise_interval':pi.to_dict(),'verdict':verdict,'recommendation':recommendation,'limitations':['All evaluation outcomes previously inspected; post-hoc protocol freeze cannot restore untouched status','Single benchmark and CNN; only three failure-bearing eligible evaluation families and no mixed-outcome eligible family','Large correlated structural bank relative to development sample; negative result cannot show absence of all useful sensitivity information','Bounded learners and finite control bank do not exhaust existing approaches','No Waterbirds, new outcome families, direct transfer, causal shortcut claim or economic savings established']}
    write(folder/'decision.json',decision)
    tables=[]
    primarymet=met[(met.cohort=='eligible')&(met.prefix==.1)&(met.variant=='A')]
    lines=['| Predictor | D9 | D12 | AP | AUROC | Brier |','|---|---:|---:|---:|---:|---:|']
    for r in primarymet.itertuples():lines.append(f'| {r.model} | {r.D9} | {r.D12} | {r.AP:.4f} | {r.AUROC:.4f} | {r.Brier:.4f} |')
    structural=contr[(contr.added=='MZ')&(contr.fraction>.3)]
    paneltable=['| Cohort | Panel variant | Learner | MZ discovered | M discovered | Difference |','|---|---|---|---:|---:|---:|']
    for r in structural.itertuples():paneltable.append(f'| {r.cohort} | {r.variant} | {r.learner} | {r.D_added} | {r.D_base} | {r.difference} |')
    param=read(out()/'predictor_freeze.json');audit=read(out()/'measurement_audit.json');env=read(ROOT/'environment/verification.json')
    influence=inf[(inf.cohort=='eligible')&(inf.variant=='A')&(inf.learner=='logistic')&(inf.added=='MZ')&(inf.fraction>.3)]
    text=f'''# GradientGuard: first layer-allocation structural-control implementation

21 November 2026. Attempt `{cfg()['attempt']}`. **Post-hoc exploratory engineering and conditional-addition study.**

## Scope and execution
No new image classifier was trained. All 96 original completed classifiers were remeasured at six frozen early checkpoints on both original 256/class panels (1,152 measurements). Both 5% and 10% prefixes were exported. Original data, outcomes, predictor freezes and reports remain read-only. New forecasters were fitted only on original development families; their evaluation nevertheless reuses already consumed outcomes. This is not prospective confirmation.

## Implementation correctness and environment
The broken venv base-interpreter path was backed up and repaired to the available matching Python 3.12.14. All {env['resolved_lines']} package records match the saved lock; torch {env['torch']}, CUDA {env['cuda']}, GPU {env['gpu']}. No package installation was needed. Isolation and microbatch checks are separate from predictive evidence. Original frozen predictions and original nine-audit counts reproduced. Maximum checkpoint scalar parity error: {audit['max_parity_error']:.9g}. Full test output, source snapshot and preservation hashes accompany this attempt.

## Frozen exploratory comparison
E inputs are fixed to the original development-selected bank. B adds total sensitivity. L2 and L6 reproduce the nested endpoint/six-share representations. N adds a six-summary norm-product proxy. M adds per-layer weight/mean-gradient L2 norms and marginal statistics plus shared bias diagnostics. MZ adds exactly the six L6 shares to M; NZ similarly adds them to N. Last, mean and population SD use the original schedule; no slope enters Z. The structural bank includes head diagnostics on the same footing as other layers.

Every matched pair uses the same data, family folds, preprocessing, learner and grid allowance. Regularisation may differ after development selection. Logistic and bounded histogram controls use C=0.1/1/10 or depth=1/2/3. Histogram min-leaf=5 was prospectively specified for this *new exploratory* comparison; historical boosting defaults/results are untouched. Selection uses development discovery at one-third capacity, AP tie-break, smaller grid on exact ties. One-third is pilot-motivated, not a newly independent primary result. Fold-local preprocessing never uses evaluation data.

Original source-normal evaluation has 36 models, 12 failures and nine eligible families. Full evaluation has 40 models. Eligibility and failure threshold remain fixed. M inputs: {param['choices']['eligible_0.1']['models']['logistic_M']['inputs']}; MZ inputs: {param['choices']['eligible_0.1']['models']['logistic_MZ']['inputs']}. This dimensionality is large relative to 48 eligible development runs and is a limitation even with regularisation.

## Descriptive evaluation results: source-normal, 10%, panel A

{chr(10).join(lines)}

Original published E/B/L6/L8 values are reproduced separately in `original_reproduction.json`; these new E/B fits can differ because the new exploratory selection rule differs. They must not replace the original primary result.

## Panel and cohort sensitivity: matched MZ versus M at one-third

{chr(10).join(paneltable)}

Panel A is primary. A_on_B applies the new frozen A model to B inputs. B_refit uses a development B fit with the A-selected hyperparameters, not B-based tuning. Full curves and both 25%/one-third comparisons are exported, including losses and ties.

## Family uncertainty and influence
Eligible panel-A logistic MZ-M at one-third: {int(primary.D_added)} versus {int(primary.D_base)} discoveries; Delta20={primary.Delta20:.6f}. The paired-family pointwise 95% descriptive interval is [{pi.low:.6f}, {pi.high:.6f}], with {int(pi.valid)} valid and {int(pi.invalid)} invalid draws of 2,000, seed {cfg()['bootstrap_seed']}. Empty and single-label draws are excluded and counted. All ten original evaluation families participate in sampling, including the one with no eligible models. Nested models remain together. These intervals condition on selected forecasters, exclude selection uncertainty and are not simultaneous curve intervals or a significance claim. Leave-one-family-out differences at recomputed one-third capacity range from {int(influence.difference.min())} to {int(influence.difference.max())}.

## Cost and artifacts
The sum of synchronized checkpoint-panel gradient/statistic timings is {pd.read_csv(out()/'measurements.csv').seconds.sum():.3f} seconds. Extraction invocation wall time: {audit['seconds_this_invocation']:.3f} seconds (resume invocation cost is separate from cumulative measurement cost). Development fitting cost: {param['fitting_seconds']:.3f} seconds. GPU allocated peak/incremental counters, data reconstruction overhead, complete stage wall times and output bytes are recorded separately. These are research validation costs, not isolated production L6 monitoring cost. Dataset reconstruction uses the original compact identities/background assignments and cached raw MNIST; no new dataset or label generation occurs.

## Qualified conclusion
**{verdict}.** {recommendation}

Positive differences are candidates for fresh validation, not proof that coordinate-product information is unique or causal. Ties/losses may motivate narrowing the representation, but cannot show that all layer information is useless. Family concentration, panel variation, control-bank dimensionality, baseline headroom and learner limitations bound interpretation. No changes to the manuscript's completed evidence are implied. A future untouched outcome study and separately frozen second benchmark remain necessary before a publication-level novelty claim.

## Reproduction
Use `instructions/STEP_BY_STEP.txt` and `run.ps1`. `all` safely resumes scalar extraction and skips already frozen fits; `report` regenerates only this attempt's report. Never invoke legacy fitting/training/report commands for this stage. Prior artifact verification is mandatory at completion.
'''
    (folder/'REPORT.md').write_text(text,encoding='utf8')
    event('report_generated',verdict=verdict)
