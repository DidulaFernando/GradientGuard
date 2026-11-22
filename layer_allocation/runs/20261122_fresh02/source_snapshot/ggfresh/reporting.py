from .core import *
from .prediction import discovery,ranking,support
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

def accounting():
    rows=[]
    for s in protocol()['runs']:
        folder=O/'classifiers'/s['run'];r={k:s[k] for k in ['run','family','split']};r['status']='not_launched'
        if (folder/'complete.json').exists():
            c=read(folder/'complete.json');assert c['updates']==c['planned_updates']==19560
            r.update(status='complete',updates=c['updates'],**{'seconds_'+k:v for k,v in c['timing'].items()})
        elif (folder/'diverged.json').exists():r.update(status='diverged',reason=read(folder/'diverged.json')['reason'])
        elif folder.exists():r['status']='incomplete'
        rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(O/'completion_accounting.csv',index=False);return df

def diagnostic_tables():
    rows=[];groups=[];timing=[]
    for s in protocol()['runs']:
        folder=O/'classifiers'/s['run']
        if not (folder/'complete.json').exists():continue
        c=read(folder/'complete.json');plan=read(folder/'plan.json');h=read(folder/f"health_{plan['steps'][-1]}.json")
        name='val' if s['split']=='development' else 'test';path=O/'outcomes'/name/(s['run']+'.json')
        label=read(path) if path.exists() else None
        rows.append({'run':s['run'],'family':s['family'],'split':s['split'],'health_accuracy':h['accuracy'],'health_loss':h['loss'],'healthy':h['accuracy']>=.8 and h['loss']<=np.log(10),'wga':label['wga'] if label else None,'failure':label['failure'] if label else None,'cutoff_near':label['cutoff_near'] if label else None})
        if label:groups.extend({'run':s['run'],'family':s['family'],'split':name,**g} for g in label['groups'])
        timing.append({'run':s['run'],'split':s['split'],**c['timing'],'wall':c['wall_this_invocation'],'outcome_seconds':label['seconds'] if label else 0})
    frame=pd.DataFrame(rows);frame.to_csv(O/'health_and_outcomes.csv',index=False);pd.DataFrame(groups).to_csv(O/'all_group_accuracy_uncertainty.csv',index=False);pd.DataFrame(timing).to_csv(O/'classifier_costs.csv',index=False)
    if len(frame):frame.groupby(['split','family']).agg(models=('run','size'),healthy=('healthy','sum'),failures=('failure','sum')).reset_index().to_csv(O/'family_coverage.csv',index=False)
    write_json(O/'costs.json',{'classifier_component_sums':pd.DataFrame(timing).select_dtypes(include='number').sum().to_dict() if timing else {},'conservative_gpu_budget':read(O/'gpu_budget.json') if (O/'gpu_budget.json').exists() else None,'dataset_bytes':sum(p.stat().st_size for p in (O/'data').rglob('*.npz')),'checkpoint_bytes':sum(p.stat().st_size for p in (O/'classifiers').rglob('*.pt')),'limitations':'Training synchronized; primary-panel path includes telemetry/structural/clustering, not isolated sensitivity increment. GPU counters are per-observation allocated peaks; no historical CPU peak or demonstrated monetary savings.'})
    if len(frame) and frame.wga.notna().any():
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder=BASE/'reports'/cfg()['attempt'];folder.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(6,4));d=frame[frame.wga.notna()];ax.scatter(d.health_accuracy,d.wga,c=d.family,cmap='tab20',s=22);ax.axvline(.8,color='gray',linestyle='--');ax.axhline(.75,color='gray',linestyle='--');ax.set(xlabel='Source health accuracy at10%',ylabel='Terminal WGA',title='Observed completed trajectories; shared MNIST');fig.tight_layout();fig.savefig(folder/'health_outcomes.png',dpi=150);plt.close(fig)

def report_stopped(reason):
    require();a=accounting();diagnostic_tables();folder=BASE/'reports'/cfg()['attempt'];folder.mkdir(parents=True,exist_ok=True)
    ready=read(O/'readiness.json') if (O/'readiness.json').exists() else None
    decision={'attempt':cfg()['attempt'],'status':'stopped_predefined_gate' if ready is not None and not ready['passed'] else 'execution_limited','reason':reason,'planned_classifiers':256,'completed':int((a.status=='complete').sum()),'diverged':int((a.status=='diverged').sum()),'incomplete':int((a.status=='incomplete').sum()),'not_launched':int((a.status=='not_launched').sum()),'readiness':ready,'source_normal_core_evaluable':False,'evaluation_terminal_outcomes_accessed':(O/'unseal.json').exists(),'manuscript_v1_3_gate_passed':False,'manuscript_updated':False,'conclusion':'Outcome study not adequately evaluable; development readiness or execution constraint blocks dependent stages. This is not a method superiority result.'}
    write_json(folder/'decision.json',decision)
    if ready:
        table=f"Eligible development: {ready['N']} models in {ready['families']} families. Outcome counts/families: {json.dumps(ready['outcomes'])}. Requirements:20 failures and20 non-failures, each in8families, valid5/3fold CV. Readiness passed={ready['passed']}."
    else:table='Readiness could not yet be assessed.'
    text=f'''# GradientGuard fresh-family validation: {cfg()['attempt']}

## Decision
**{decision['conclusion']}**

Reason: {reason}

## What executed
Planned64 families/256 complete60-epoch trajectories,32 development/32 evaluation families. Completed {decision['completed']}, diverged {decision['diverged']}, incomplete {decision['incomplete']}, not launched {decision['not_launched']}. No completed classifier was shortened to improve an outcome. See completion_accounting.csv for every planned identity.

## Development support and predefined gate
{table}

Readiness uses health/outcome support and grouped-fold validity, not a favourable L6/MZ advantage. When it fails, evaluation generation/training, predictor fitting and terminal test access are stopped as predefined. No cutoff, seed, horizon or family grid was changed. No evaluation audit gain, AUROC or sensitivity-specific result can be claimed from this stopped stage. A full-cohort view cannot rescue the intended source-normal claim.

## Protocol and environment
The new fixed design is8alignment levels ×2official background difficulties ×4new learning rates. Families do not exactly overlap the original24settings. Two constructions and two initialisations per family stay together. New seeds, configuration-only split, source, tests, support rules and manuscript gate were frozen before training. Numerical stack and complete dependency lock match the archived environment. Models share MNIST; this is not external benchmark validation. No Waterbirds or causal shortcut interpretation is supplied.

## Evidence and costs
Read protocol_freeze.json, environment.json, tests.xml, readiness.json, representative_profile.json, gpu_budget.json, source_snapshot, completion_accounting.csv and preservation_audit.json under the attempt. GPU budget is conservative cumulative wall time in GPU stages, including data/monitor/checkpoint overhead, capped at24hours. Scientific timing and health/outcome records are retained per completed classifier. Any failed engineering attempt remains in logs.

## Manuscript
The automatic v1.3 gate did not pass. Version1.2 and all earlier manuscripts remain unchanged. This report preserves the inconclusive result; the user may decide how to incorporate it manually. A new design amendment would require a separately identified protocol, not silently expanding this study.

## Next step
Inspect the specific support shortfall before committing to additional outcomes. Discuss a separately justified design with the supervisor; do not lower the frozen thresholds or interpret software completion as predictive evidence. Exact status/resume/report commands are in fresh_validation/instructions/STEP_BY_STEP.txt. Re-running all cannot bypass a failed readiness gate.
'''
    (folder/'REPORT.md').write_text(text,encoding='utf8');write_json(folder/'MANUSCRIPT_RECOMMENDATION.json',{'create_v1_3':False,'reason':reason,'user_manual_revision_required_if_desired':True});event('fresh_study_reported',decision=decision['status'])

def curves(y,risk,ids):
    y=np.asarray(y,dtype=int);n=len(y);f=int(y.sum())
    if not n:raise ValueError('Empty candidate cohort')
    d=np.r_[0,np.cumsum(y[ranking(risk,ids)])]
    return [{'N':n,'F':f,'k':k,'fraction':k/n,'discoveries':int(d[k]),'precision':float(d[k]/k) if k else None,'recall':float(d[k]/f) if f else None,'random':k*f/n,'oracle':min(k,f),'headroom':int(min(k,f)-d[k])} for k in range(n+1)]

def family_draw(pivot,draw):
    ix=[];ids=[]
    for j,family in enumerate(draw):
        selectedix=np.flatnonzero(pivot.family.to_numpy()==family);ix.extend(selectedix);ids.extend(f'{r}__occ{j:02}' for r in pivot.run.iloc[selectedix])
    return ix,ids

def analyse():
    allow_test_outcomes();assert (O/'unseal.json').exists()
    from .study import preserve
    preserve()
    a=accounting();labels={p.stem:read(p) for p in (O/'outcomes/test').glob('*.json')};pr=pd.read_csv(O/'evaluation_predictions.csv');pr['wga']=pr.run.map({r:v['wga'] for r,v in labels.items()});assert pr.wga.notna().all();pr['failure']=(pr.wga<.75).astype(int)
    primary=pr[(pr.cohort=='eligible')&(pr.prefix==.1)&(pr.variant=='A')&(pr.model=='logistic_MZ')]
    sup=support(primary,evaluation=True);write_json(O/'evaluation_support.json',sup)
    met=[];cv=[];contr=[];influence=[];intervals=[];selected=read(O/'predictor_freeze.json')['choices']['eligible_0.1']['strongest_nonZ']
    comparisons=[('logistic_MZ','logistic_M'),('logistic_MZ','logistic_B'),('logistic_MZ',selected),('logistic_L6','E'),('logistic_L6','logistic_B'),('logistic_NZ','logistic_N'),('logistic_L6','logistic_L2'),('histogram_MZ','histogram_M'),('histogram_NZ','histogram_N')]
    comparisons=list(dict.fromkeys(comparisons));families=[r['family'] for r in protocol()['families'] if r['split']=='evaluation']
    for keys,df in pr.groupby(['cohort','prefix','variant','model']):
        md=dict(zip(['cohort','prefix','variant','model'],keys));n=len(df);y=df.failure.to_numpy()
        met.append(md|{'N':n,'F':int(y.sum()),'D25':discovery(y,df.risk,df.run,n//4),'D33':discovery(y,df.risk,df.run,n//3),'AP':float(average_precision_score(y,df.risk)) if y.sum() else None,'AUROC':float(roc_auc_score(y,df.risk)) if df.failure.nunique()==2 else None,'Brier':float(brier_score_loss(y,df.risk))});cv.extend(md|r for r in curves(y,df.risk,df.run))
    for (cohort,variant),df in pr[pr.prefix==.1].groupby(['cohort','variant']):
        pivot=df.pivot(index=['run','family','failure'],columns='model',values='risk').reset_index();n=len(pivot)
        for added,base in comparisons:
            for denominator in [3,4]:
                k=n//denominator;delta=discovery(pivot.failure,pivot[added],pivot.run,k)-discovery(pivot.failure,pivot[base],pivot.run,k)
                r={'cohort':cohort,'variant':variant,'added':added,'base':base,'denominator':denominator,'N':n,'k':k,'D_added':discovery(pivot.failure,pivot[added],pivot.run,k),'D_base':discovery(pivot.failure,pivot[base],pivot.run,k),'difference':delta,'Delta20':20*delta/n};contr.append(r)
                for family in families:
                    q=pivot[pivot.family!=family];kk=len(q)//denominator
                    influence.append(r|{'omitted_family':family,'N':len(q),'k':kk,'difference':discovery(q.failure,q[added],q.run,kk)-discovery(q.failure,q[base],q.run,kk)})
        rng=np.random.default_rng(cfg()['bootstrap_seed']);values={};invalid=0
        for iteration in range(cfg()['bootstrap_replicates']):
            draw=rng.choice(families,len(families),replace=True);ix,ids=family_draw(pivot,draw)
            y=pivot.failure.to_numpy()[ix]
            if not len(y) or len(set(y))<2:invalid+=1;continue
            for added,base in comparisons:
                for denominator in [3,4]:
                    k=len(y)//denominator;d=20*(discovery(y,pivot[added].to_numpy()[ix],ids,k)-discovery(y,pivot[base].to_numpy()[ix],ids,k))/len(y);values.setdefault((added,base,denominator),[]).append(d)
        for (added,base,denominator),v in values.items():
            lo,hi=np.quantile(v,[.025,.975]);intervals.append({'cohort':cohort,'variant':variant,'added':added,'base':base,'denominator':denominator,'low':float(lo),'high':float(hi),'valid':len(v),'invalid':invalid,'seed':cfg()['bootstrap_seed']})
    pd.DataFrame(met).to_csv(O/'metrics.csv',index=False);pd.DataFrame(cv).to_csv(O/'discovery_curves.csv',index=False);pd.DataFrame(contr).to_csv(O/'contrasts.csv',index=False);pd.DataFrame(influence).to_csv(O/'leave_one_family_out.csv',index=False);pd.DataFrame(intervals).to_csv(O/'intervals.csv',index=False)
    pd.DataFrame([{k:v for k,v in label.items() if k!='groups'} for label in labels.values()]).to_csv(O/'terminal_outcomes.csv',index=False)
    pd.DataFrame([{'run':r,**g} for r,v in labels.items() for g in v['groups']]).to_csv(O/'group_accuracies.csv',index=False)
    secondary=[]
    for cutoff in [.7,.8]:
        for keys,df in pr.groupby(['cohort','prefix','variant','model']):
            y=(df.wga<cutoff).astype(int);secondary.append(dict(zip(['cohort','prefix','variant','model'],keys))|{'cutoff':cutoff,'N':len(df),'F':int(y.sum()),'D25':discovery(y,df.risk,df.run,len(df)//4),'D33':discovery(y,df.risk,df.run,len(df)//3)})
    pd.DataFrame(secondary).to_csv(O/'secondary_cutoffs.csv',index=False)
    p=pd.DataFrame(contr);i=pd.DataFrame(intervals);l=pd.DataFrame(influence)
    def row(base,variant='A'):return p[(p.cohort=='eligible')&(p.variant==variant)&(p.added=='logistic_MZ')&(p.base==base)&(p.denominator==3)].iloc[0]
    main=row('logistic_M');inter=i[(i.cohort=='eligible')&(i.variant=='A')&(i.added=='logistic_MZ')&(i.base=='logistic_M')&(i.denominator==3)].iloc[0]
    lof=l[(l.cohort=='eligible')&(l.variant=='A')&(l.added=='logistic_MZ')&(l.base=='logistic_M')&(l.denominator==3)]
    fitted=read(O/'predictor_freeze.json');predicted=read(O/'prediction_freeze.json');unsealed=read(O/'unseal.json')
    boundary=bool(require()['time_ns']<fitted['time_ns']<predicted['time_ns']<unsealed['time_ns'])
    flags={'correctness_preservation_boundary':boundary,'evaluation_support':sup['passed'],'useful_gain':main.Delta20>=1,'interval_above_zero':inter.low>0,'beats_B':row('logistic_B').difference>0,'beats_selected_nonZ':row(selected).difference>0,'both_B_variants_positive':row('logistic_M','A_on_B').difference>0 and row('logistic_M','B_refit').difference>0,'no_negative_family_omission':bool((lof.difference>=0).all())}
    write_json(O/'analysis_gate.json',{'flags':{k:bool(v) for k,v in flags.items()},'passed':gate(flags),'primary':main.to_dict(),'interval':inter.to_dict(),'selected_nonZ':selected})
    report_evaluated()

def report_evaluated():
    folder=BASE/'reports'/cfg()['attempt'];folder.mkdir(parents=True,exist_ok=True);a=accounting();diagnostic_tables();g=read(O/'analysis_gate.json');support_record=read(O/'evaluation_support.json')
    verdict='Promising sensitivity-specific result requiring independent external confirmation' if g['passed'] else ('Outcome study not adequately evaluable' if not support_record['passed'] else 'Fresh study does not meet the predefined sensitivity-specific progression gate')
    write_json(folder/'decision.json',{'attempt':cfg()['attempt'],'status':'complete','verdict':verdict,'planned_classifiers':256,'completed':int((a.status=='complete').sum()),'diverged':int((a.status=='diverged').sum()),'incomplete':int((a.status=='incomplete').sum()),'not_launched':int((a.status=='not_launched').sum()),'evaluation_support':support_record,'manuscript_v1_3_gate_passed':g['passed'],'gate':g,'manuscript_updated':False})
    met=pd.read_csv(O/'metrics.csv');lines=['| Model | N | Failures | D25 | D33 | AP | AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for r in met[(met.cohort=='eligible')&(met.prefix==.1)&(met.variant=='A')].itertuples():lines.append(f'| {r.model} | {r.N} | {r.F} | {r.D25} | {r.D33} | {r.AP:.4f} | {r.AUROC:.4f} |')
    text=f'''# Fresh-family GradientGuard validation

Attempt {cfg()['attempt']}. **{verdict}.**

The study prospectively froze64 new setting families/256 full60-epoch trajectories. Development families were used for readiness, conventional-bank selection and fitting. Evaluation predictors and risks were frozen before one-time terminal test access. Related models remained grouped by family. This is within-benchmark validation on shared MNIST images, not Waterbirds, direct transfer or causal identification.

## Primary source-normal results at10%
{chr(10).join(lines)}

Primary capacity=floor(N/3), secondary=floor(N/4). The one-third choice was informed by consumed pilot diagnostics before this new freeze. Full curves, all learner/panel variants, source-normal/full cohorts and5% secondary prefixes are retained. Read contrasts.csv, intervals.csv and leave_one_family_out.csv rather than selecting favourable points.

## Support and progression gate
Evaluation support: {json.dumps(support_record)}

Gate flags: {json.dumps(g['flags'])}

Primary matched comparison: {json.dumps(g['primary'])}

Conditional family interval: {json.dumps(g['interval'])}

Bootstrap uses5000 draws of whole32families, nested seeds/constructions together, recomputed budgets and deterministic occurrence identities. Empty/single-label draws are counted invalid. Pointwise intervals exclude forecaster-selection uncertainty and are not simultaneous curve evidence. Few failure-bearing families, if present, remain a limitation despite trajectory counts. Marginal Wilson group intervals are not simultaneous uncertainty for the worst group.

## Interpretation and manuscript
Scientific ranking value, coordinate-product specificity and operational savings are separate claims. A control-bank/learner-specific gain is not universal novelty. Actual monitoring/resource totals and source-health coverage accompany the outcome accounting. The v1.3 update is permitted only if every frozen gate passes; otherwise v1.2 remains unchanged and the user receives a separate recommendation. All unfavourable comparisons remain in the archive even if a manuscript update is permitted. Prediction depth and full reproductions of competing systems remain unimplemented. No monetary savings were demonstrated.

## Reproducibility
See protocol_freeze.json, source_snapshot, environment and dependency locks, manifests, checkpoint plans, tests, freezes, GPU-stage budget ledger and preservation audit. Instructions are under fresh_validation/instructions/STEP_BY_STEP.txt. Complete accounting: {a.status.value_counts().to_dict()}.
'''
    (folder/'REPORT.md').write_text(text,encoding='utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=pd.read_csv(O/'discovery_curves.csv');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,cohort in zip(axes,['eligible','full']):
        for name in ['E','logistic_B','logistic_N','logistic_M','logistic_MZ']:
            d=c[(c.cohort==cohort)&(c.prefix==.1)&(c.variant=='A')&(c.model==name)];ax.step(d.k,d.discoveries,where='post',label=name)
        ax.set(title=cohort,xlabel='Models inspected',ylabel='Failures discovered');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(folder/'discovery_curves.png',dpi=160);plt.close(fig)
    write_json(folder/'MANUSCRIPT_RECOMMENDATION.json',{'create_v1_3':g['passed'],'reason':verdict,'report_all_negative_secondary_results':True})
