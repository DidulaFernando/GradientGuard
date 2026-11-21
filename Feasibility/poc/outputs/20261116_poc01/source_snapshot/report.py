from run import *
from prediction import feature_frame,support,selection_score
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,mean_absolute_error
from scipy.stats import spearmanr
def audit(df):
    n=len(df);k=n//4
    selected=df.sort_values(['risk','run'],ascending=[False,True]).head(k)
    discoveries=int(selected.failure.sum());fail=int(df.failure.sum())
    return {'N':n,'k':k,'failures':fail,'discoveries':discoveries,'precision':discoveries/k if k else None,'recall':discoveries/fail if fail else None,
            'prevalence':fail/n if n else None,'random_expected':k*fail/n if n else None,'oracle_ceiling':min(k,fail),'headroom':min(k,fail)-discoveries,
            'AP':average_precision_score(df.failure,df.risk) if fail else None,'AUROC':roc_auc_score(df.failure,df.risk) if len(set(df.failure))==2 else None,
            'Brier':brier_score_loss(df.failure,df.risk) if df.risk.between(0,1).all() else None}
def report():
    folder=O/'reports';folder.mkdir(exist_ok=True);pred=pd.read_csv(O/'evaluation_predictions.csv');manifest=pd.read_csv(O/'run_manifest.csv');outcome=[];groups=[]
    for split in ['val','test']:
        for path in (O/'outcomes'/split).glob('*.json'):
            r=read(path);outcome.append({k:v for k,v in r.items() if k!='groups'})
            groups.extend({'run':r['run'],'split':split,**g} for g in r['groups'])
    outcomes=pd.DataFrame(outcome);outcomes.to_csv(folder/'terminal_outcomes.csv',index=False);pd.DataFrame(groups).to_csv(folder/'group_accuracy_uncertainty.csv',index=False)
    df=pred.merge(outcomes,on='run').merge(manifest[['run','family']],on='run')
    results=[];contrasts=[];influence=[];selections=[]
    choices=read(O/'predictor_freeze.json')['choices'];support_rows=[]
    for prefix in (.1,.05):
        early=feature_frame('evaluation',prefix=prefix).merge(outcomes[['run','wga']],on='run')
        for cohort in ['eligible','full']:
            s=early[early.eligible] if cohort=='eligible' else early
            y=(s.wga<.75).astype(int);support_rows.append({'cohort':cohort,'prefix':prefix,'N':len(s),'families':s.family.nunique(),'failures':int(y.sum()),'nonfailures':int((y==0).sum()),'evaluable':support(s,y,True)})
    for key,g in df.groupby(['cohort','prefix','panel','refit_B']):
        cohort,prefix,panel,refit=key;models={name:d for name,d in g.groupby('model')};strong=choices[f'{cohort}_{prefix}'].get('strongest_nontemporal')
        for name,d in models.items():
            if name.startswith('ridge_'):
                results.append({'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'model':name,'MAE':mean_absolute_error(d.wga,-d.risk),'spearman':float(spearmanr(d.wga,-d.risk).statistic)})
                continue
            res=audit(d);results.append({'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'model':name,**res})
            selected=d.sort_values(['risk','run'],ascending=[False,True]).head(len(d)//4)
            selections.extend({'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'model':name,'run':r} for r in selected.run)
        if 'logistic_L8' not in models:continue
        primary=models['logistic_L8']
        for control in ['E','logistic_B','logistic_L6',strong]:
            if control not in models:continue
            base=models[control];n=len(primary);gain=20*(audit(primary)['discoveries']-audit(base)['discoveries'])/n
            rng=np.random.default_rng(130005);families=sorted(primary.family.unique());values=[];invalid=0
            for _ in range(2000):
                draw=rng.choice(families,len(families),replace=True)
                a=pd.concat([primary[primary.family==f].assign(run=lambda q:q.run+'_'+str(i)) for i,f in enumerate(draw)],ignore_index=True)
                b=pd.concat([base[base.family==f].assign(run=lambda q:q.run+'_'+str(i)) for i,f in enumerate(draw)],ignore_index=True)
                if len(set(a.failure))<2:invalid+=1;continue
                values.append(20*(audit(a)['discoveries']-audit(b)['discoveries'])/len(a))
            contrasts.append({'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'control':control,'Delta20':gain,'interval_low':np.quantile(values,.025) if values else None,'interval_high':np.quantile(values,.975) if values else None,'invalid_resamples':invalid,'family_count':len(families)})
            for f in families:
                a=primary[primary.family!=f];b=base[base.family!=f]
                influence.append({'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'control':control,'omitted_family':f,'Delta20':20*(audit(a)['discoveries']-audit(b)['discoveries'])/len(a)})
    pd.DataFrame(results).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(folder/'contrasts.csv',index=False);pd.DataFrame(influence).to_csv(folder/'leave_one_family_out.csv',index=False);pd.DataFrame(selections).to_csv(folder/'audit_selections.csv',index=False);pd.DataFrame(support_rows).to_csv(folder/'evaluation_support.csv',index=False)
    # Fixed secondary thresholds reuse frozen predictions, with no refit or cutoff selection.
    secondary=[]
    for cutoff in [.7,.8]:
        for key,g in df[~df.model.str.startswith('ridge_')].groupby(['cohort','prefix','panel','refit_B','model']):
            g=g.copy();g['failure']=g.wga<cutoff;secondary.append(dict(zip(['cohort','prefix','panel','refit_B','model'],key))|{'cutoff':cutoff,**audit(g)})
    pd.DataFrame(secondary).to_csv(folder/'secondary_cutoffs.csv',index=False)
    primary=[r for r in support_rows if r['cohort']=='eligible' and r['prefix']==.1][0]
    verdict='Outcome pilot not adequately evaluable.' if not primary['evaluable'] or not choices['eligible_0.1']['classification_supported'] else 'No incremental audit value observed in this pilot.'
    main=[r for r in contrasts if r['cohort']=='eligible' and r['prefix']==.1 and r['panel']=='A' and not r['refit_B']]
    if main and primary['evaluable'] and any(r['control']=='E' and r['Delta20']>=1 for r in main):
        verdict='Predictive value observed, temporal specificity unresolved.'
        if all(r['Delta20']>0 for r in main) and all(r['interval_low'] is not None and r['interval_low']>0 for r in main):verdict='Promising temporal-layer result requiring independent confirmation.'
    protect(True);completed=len(list((O/'classifiers').glob('*/complete.json')))
    decision={'verdict':verdict,'completed_classifiers':completed,'planned':96,'source_normal_support':primary,'primary_contrasts':main,'limitations':['Exploratory; only 10 held-out setting families','Shared MNIST images and outcome panels','Partition-sensitive temporal measurements and unresolved head/denominator interpretation','Prediction-depth comparator deferred','Family bootstrap conditional on selected forecasters; excludes development selection uncertainty'],'legacy_preserved':True}
    write_json(folder/'decision.json',decision)
    text='# GradientGuard exploratory outcome PoC\n\n'+verdict+'\n\n'+json.dumps(decision,indent=2)+'\n\nSee metrics.csv, contrasts.csv, development_support.csv (attempt root), evaluation_support.csv, secondary_cutoffs.csv, group_accuracy_uncertainty.csv and audit_selections.csv. Models were selected only on development validation outcomes; held-out test outcomes were unsealed after predictor and prediction freezes. The 60-epoch horizon is an explicitly user-authorized exploratory amendment, not a passed calibration gate.\n'
    (folder/'REPORT.md').write_text(text,encoding='utf8');event('poc_complete',completed=completed,verdict=verdict)
