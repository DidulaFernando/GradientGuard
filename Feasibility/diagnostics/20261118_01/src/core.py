"""Frozen prediction diagnostics. No legacy imports or write operations on old attempts."""
from environment import ATTEMPT,ROOT,POC,read,write,sha,verify_prior
from pathlib import Path
from fractions import Fraction
from contextlib import contextmanager
import json,time,sys,hashlib
import numpy as np
import pandas as pd

CONFIG=ATTEMPT/'config/analysis.json'
SLOPES=['shallow_share_slope','deep_share_slope']
TELE=['loss','accuracy','worst_class','confidence','correct_confidence','entropy','gradient_norm','growth','gns']
SEP=['separation','smallest_cluster','cluster_gap']

def event(name,**kwargs):
    row={'time_ns':time.time_ns(),'event':name,**kwargs}
    with (ATTEMPT/'logs/events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    write(ATTEMPT/'status.json',row)
    print(json.dumps(row),flush=True)

def code_hashes():
    return {str(p.relative_to(ATTEMPT)):sha(p) for d in ['src','tests'] for p in (ATTEMPT/d).rglob('*.py')}

def freeze():
    assert read(ATTEMPT/'environment/verification.json')['passed']
    verify_prior('freeze')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in POC.rglob('*') if p.is_file()}
    obj={'post_hoc':True,'outcomes_already_unsealed':True,'analysis':read(CONFIG),
         'inputs':inputs,'config_sha256':sha(CONFIG),'source_hashes':code_hashes(),
         'environment_verification_sha256':sha(ATTEMPT/'environment/verification.json'),
         'time_ns':time.time_ns()}
    path=ATTEMPT/'config/frozen_plan.json'
    if path.exists():
        old=read(path);assert all(old[k]==obj[k] for k in obj if k!='time_ns'),'Diagnostic plan/source drift'
    else:write(path,obj)
    event('plan_frozen',inputs=len(inputs),post_hoc=True)

def guard():
    obj=read(ATTEMPT/'config/frozen_plan.json')
    assert obj['config_sha256']==sha(CONFIG)
    assert obj['source_hashes']==code_hashes(),'Source changed since diagnostic freeze'
    assert obj['environment_verification_sha256']==sha(ATTEMPT/'environment/verification.json')
    for n,h in obj['inputs'].items():assert sha(ROOT/n)==h,f'Input changed: {n}'
    return obj

def save_table(name,records):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(ATTEMPT/'tables'/f'{name}.csv',index=False)
    return frame

def ranked(frame):
    if not set(['run','risk','failure'])<=set(frame):raise ValueError('Missing audit fields')
    if frame.run.isna().any() or frame.run.duplicated().any():raise ValueError('Missing/duplicate identities')
    if not np.isfinite(frame.risk.to_numpy(float)).all():raise ValueError('Invalid risk')
    if frame.failure.isna().any() or not frame.failure.isin([True,False,0,1]).all():raise ValueError('Invalid outcome')
    return frame.sort_values(['risk','run'],ascending=[False,True])

def budget(n,fraction):
    f=Fraction(str(fraction))
    if not 0<=f<=1:raise ValueError('Invalid capacity fraction')
    return n*f.numerator//f.denominator

def curve(frame):
    r=ranked(frame);n=len(r);fail=int(r.failure.sum());cum=np.r_[0,np.cumsum(r.failure.to_numpy(int))]
    return pd.DataFrame([{'N':n,'F':fail,'k':k,'fraction':k/n if n else np.nan,
        'discoveries':int(cum[k]),'precision':float(cum[k]/k) if k else np.nan,
        'recall':float(cum[k]/fail) if fail else np.nan,'random_expected':k*fail/n if n else np.nan,
        'oracle_ceiling':min(k,fail),'headroom':min(k,fail)-int(cum[k])} for k in range(n+1)])

def compare_sets(a,b,k):
    aa=ranked(a).run.tolist();bb=ranked(b).run.tolist()
    assert set(aa)==set(bb),'Different candidates'
    return {'complete_order_identical':aa==bb,'selected_set_identical':set(aa[:k])==set(bb[:k]),
            'selected_overlap':len(set(aa[:k])&set(bb[:k]))}

def resample_indices(fam,ids,draw):
    by={f:np.flatnonzero(fam==f) for f in set(draw)}
    ix=np.concatenate([by[f] for f in draw])
    copies=np.concatenate([np.array([str(ids[i])+'_'+str(j) for i in by[f]],dtype=str) for j,f in enumerate(draw)])
    return ix,copies

def load_inputs():
    manifest=pd.read_csv(POC/'run_manifest.csv');completion=pd.read_csv(POC/'reports/completion_and_health.csv')
    outcomes=pd.read_csv(POC/'reports/terminal_outcomes.csv');pred=pd.read_csv(POC/'evaluation_predictions.csv')
    evaluation=completion[completion.split=='evaluation'].merge(outcomes.drop(columns='split'),on='run',validate='one_to_one')
    assert np.array_equal(evaluation.failure.to_numpy(),(evaluation.wga<.75).to_numpy())
    assert len(completion)==96 and (completion.updates==19560).all()
    for path in (POC/'classifiers').glob('*/complete.json'):
        c=read(path);assert c['full_60_epochs'] and c['updates']==c['planned_updates']==19560
    pred=pred[pred.prefix==.1].merge(evaluation[['run','family','failure','eligible']],on='run',validate='many_to_one')
    cfg=read(CONFIG);groups={};families=sorted(manifest.loc[manifest.split=='evaluation','family'].unique().tolist())
    for cohort in cfg['cohorts']:
        expected=set(evaluation.loc[evaluation.eligible,'run']) if cohort=='eligible' else set(evaluation.run)
        models=list(dict.fromkeys(cfg['models']+[read(POC/'predictor_freeze.json')['choices'][cohort+'_0.1']['strongest_nontemporal']]))
        for variant in cfg['variants']:
            for model in models:
                d=pred[(pred.cohort==cohort)&(pred.panel==variant['panel'])&(pred.refit_B==variant['refit_B'])&(pred.model==model)].copy()
                assert set(d.run)==expected and not d.run.duplicated().any()
                ranked(d)
                groups[(cohort,variant['name'],model)]=d.sort_values('run').reset_index(drop=True)
    normal=evaluation[evaluation.eligible]
    assert len(normal)==36 and normal.family.nunique()==9 and int(normal.failure.sum())==12
    original=[]
    old=pd.read_csv(POC/'reports/metrics.csv')
    for model in cfg['models']:
        d=groups[('eligible','A',model)];r=curve(d).iloc[9]
        assert int(r.discoveries)==8 and int(r.oracle_ceiling)==9
        original.append({'model':model,'N':len(d),'F':int(d.failure.sum()),'k':9,'discoveries':int(r.discoveries),'oracle':int(r.oracle_ceiling)})
    l6=old[(old.cohort=='eligible')&(old.prefix==.1)&(old.panel=='A')&(~old.refit_B)&(old.model=='logistic_L6')].iloc[0]
    l8=old[(old.cohort=='eligible')&(old.prefix==.1)&(old.panel=='A')&(~old.refit_B)&(old.model=='logistic_L8')].iloc[0]
    assert l6.AP==l8.AP and l6.AUROC==l8.AUROC
    assert read(POC/'predictor_freeze.json')['choices']['eligible_0.1']['strongest_nontemporal']=='logistic_B'
    return groups,evaluation,families,original

def stats(v,x):
    signs=np.sign(np.diff(v));signs=signs[signs!=0]
    return {'first':v[0],'last':v[-1],'change':v[-1]-v[0],'mean':v.mean(),'std':v.std(ddof=0),'min':v.min(),'max':v.max(),
            'area':np.trapezoid(v,x),'slope':np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean()),'reversals':int(np.sum(signs[1:]!=signs[:-1]))}

def features(split,panel):
    """Read-only adaptation of original feature_frame; no terminal input fields."""
    rows=[]
    for spec in read(POC/'protocol.json')['runs']:
        if spec['split']!=split:continue
        f=POC/'classifiers'/spec['run'];t=pd.read_csv(f/'early.csv');d=t[(t.panel==panel)&(t.fraction<=.1+1e-12)&(t.step>0)].sort_values('step')
        plan=read(f/'plan.json');assert d.step.tolist()==plan['steps']
        h=read(f/('health_'+str(plan['steps'][-1])+'.json'))
        row={'run':spec['run'],'family':spec['family'],'split':split,'lr':spec['lr'],
             'eligible':bool(h['accuracy']>=.8 and h['loss']<=np.log(10) and d.valid.all()),'health_accuracy':h['accuracy'],'health_loss':h['loss']}
        for col in TELE+SEP+['shallow_share','deep_share','total','bias_norm','bias_sensitivity']:
            ss=stats(d[col].to_numpy(float),d.fraction.to_numpy())
            for s in ['last','mean','std','slope']:row[col+'_'+s]=ss[s]
            if col in ['loss','accuracy','growth','gns']:
                for s,val in ss.items():row['rich_'+col+'_'+s]=val
        row['saturation']=d.saturation.dropna().iloc[0] if d.saturation.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

@contextmanager
def no_fitting():
    """Reject fitting calls in this process; no synthetic estimator fitting either."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.ensemble import HistGradientBoostingClassifier
    originals=[];calls=[]
    def blocked(*args,**kwargs):
        calls.append('blocked_fit');raise RuntimeError('Fitting is prohibited in this diagnostic')
    for cls in [Pipeline,SimpleImputer,StandardScaler,VarianceThreshold,LogisticRegression,Ridge,HistGradientBoostingClassifier]:
        for name in ['fit','fit_transform','partial_fit']:
            if hasattr(cls,name):originals.append((cls,name,getattr(cls,name)));setattr(cls,name,blocked)
    try:yield calls
    finally:
        for cls,name,fn in reversed(originals):setattr(cls,name,fn)
