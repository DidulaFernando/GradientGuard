"""Bounded exploratory refits; original forecasters remain read-only."""
from core import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
import joblib
from structural_measurement import LAYERS
MODELS=['E','B','L2','L6','N','NZ','M','MZ']
def allstats(v,x):
    r=summaries(v,x);v=np.asarray(v,dtype=float);x=np.asarray(x,dtype=float)
    signs=np.sign(np.diff(v));signs=signs[signs!=0]
    r.update(first=float(v[0]),change=float(v[-1]-v[0]),min=float(v.min()),max=float(v.max()),area=float(np.trapezoid(v,x)),slope=float(np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean())),reversals=int(np.sum(signs[1:]!=signs[:-1])))
    return r
def features():
    require();measure=pd.read_csv(out()/'measurements.csv');rows=[]
    assert len(measure)==1152 and not measure.duplicated(['run','step','panel']).any()
    specs=read(OLD/'protocol.json')['runs']
    audit=[]
    for spec in specs:
        original=pd.read_csv(OLD/'classifiers'/spec['run']/'early.csv')
        plan=read(OLD/'classifiers'/spec['run']/'plan.json')
        for prefix in cfg()['prefixes']:
            saved=pd.read_csv(OLD/f"{spec['split']}_features_{prefix}.csv").set_index('run').loc[spec['run']]
            for panel in cfg()['panels']:
                df=original[(original.panel==panel)&(original.step>0)&(original.fraction<=prefix+1e-12)].sort_values('step')
                new=measure[(measure.run==spec['run'])&(measure.panel==panel)&(measure.fraction<=prefix+1e-12)].sort_values('step')
                expected=[s for s in plan['steps'] if s<=round(prefix*plan['horizon'])]
                assert df.step.tolist()==new.step.tolist()==expected
                x=df.fraction.to_numpy();row={'run':spec['run'],'family':spec['family'],'split':spec['split'],'lr':spec['lr'],'eligible':bool(saved.eligible),'prefix':prefix,'panel':panel}
                tele=['loss','accuracy','worst_class','confidence','correct_confidence','entropy','gradient_norm','growth','gns','separation','smallest_cluster','cluster_gap','total','bias_norm','bias_sensitivity']
                for col in tele:
                    ss=allstats(df[col].to_numpy(),x)
                    for stat in ['last','mean','std','slope']:row[col+'_'+stat]=ss[stat]
                    if col in ['loss','accuracy','growth','gns']:
                        for stat,value in ss.items():row['rich_'+col+'_'+stat]=value
                row['saturation']=float(original[(original.panel==panel)&(original.step==plan['steps'][0])].saturation.iloc[0])
                # Z from newly recomputed mean gradients, preserving original denominator.
                for col in ['shallow_share','deep_share']:
                    ss=allstats(new[col].to_numpy(),x)
                    for stat in ['last','mean','std']:row[col+'_'+stat]=ss[stat]
                structural=[f'{l}_{kind}_{stat}' for l in LAYERS for kind in ['w','g'] for stat in ['l2']+cfg()['marginals']]+['proxy_shallow','proxy_deep']
                for col in structural:
                    for stat,value in summaries(new[col].to_numpy(),x).items():row[col+'_'+stat]=value
                if panel=='A':
                    cols=[c for c in row if c in saved.index and c not in ['run','split','eligible']]
                    errors=[abs(float(row[c])-float(saved[c])) for c in cols]
                    assert all(np.isclose(float(row[c]),float(saved[c]),atol=cfg()['atol'],rtol=cfg()['rtol']) for c in cols),spec['run']
                    audit.append({'run':spec['run'],'prefix':prefix,'max_error':max(errors),'columns':len(cols)})
                rows.append(row)
    frame=pd.DataFrame(rows);assert len(frame)==384
    frame.to_csv(out()/'early_features.csv',index=False)
    write(out()/'feature_parity.json',{'checked':len(audit),'max_error':max(r['max_error'] for r in audit),'all_passed':True,'records':audit})
def inputs(base,name):
    b=list(base)
    assert not set(b)&{'run','family','split','eligible','alignment','difficulty','wga','failure','final_accuracy','final_loss','data_seed','reservation_seed'}
    if name=='E':return b
    b += ['total_'+s for s in ['last','mean','std','slope']]
    if name=='B':return b
    z=six()
    if name=='L2':return b+[c for c in z if c.endswith('_last')]
    if name=='L6':return b+z
    n=b+[f'proxy_{l}_{s}' for l in ['shallow','deep'] for s in cfg()['summary_stats']]
    if name=='N':return n
    if name=='NZ':return n+z
    m=n+[f'{l}_{kind}_{stat}_{s}' for l in LAYERS for kind in ['w','g'] for stat in ['l2']+cfg()['marginals'] for s in cfg()['summary_stats']]+[f'{b}_{s}' for b in ['bias_norm','bias_sensitivity'] for s in cfg()['summary_stats']]
    assert len(m)==len(set(m))
    return m if name=='M' else m+z
def pipe(kind,value):
    if kind=='logistic':est=LogisticRegression(C=value,max_iter=3000,random_state=cfg()['seed'])
    else:est=HistGradientBoostingClassifier(max_depth=value,random_state=cfg()['seed'],**cfg()['histogram'])
    return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),VarianceThreshold(),est)
def ranking(risks,ids):
    r=np.asarray(risks,dtype=float);ids=np.asarray(ids,dtype=str)
    if len(r)!=len(ids) or not np.isfinite(r).all():raise ValueError('Invalid risk vector')
    return np.lexsort((ids,-r))
def discovery(y,risk,ids,k):
    y=np.asarray(y,dtype=int)
    if not 0<=k<=len(y) or not np.isin(y,[0,1]).all():raise ValueError('Invalid candidates/budget')
    return int(y[ranking(risk,ids)[:k]].sum())
def score(y,risk,ids):return discovery(y,risk,ids,int(len(y)*cfg()['selection_fraction'])),float(average_precision_score(y,risk))
def folds(frame,original):
    fs=[]
    for f in original:
        a=np.flatnonzero(frame.family.isin(f['train_families']));b=np.flatnonzero(frame.family.isin(f['validation_families']))
        assert not set(frame.family.iloc[a])&set(frame.family.iloc[b]) and set(a)|set(b)==set(range(len(frame)))
        assert len(set(frame.failure.iloc[a]))==len(set(frame.failure.iloc[b]))==2
        fs.append((a,b))
    assert sorted(np.concatenate([b for a,b in fs]).tolist())==list(range(len(frame)))
    return fs
def outcomes(split):
    # Development labels only during fitting; evaluation labels accessed by analysis after prediction freeze.
    folder=OLD/'outcomes'/('val' if split=='development' else 'test')
    return {p.stem:read(p)['wga']<.75 for p in folder.glob('*.json')}
def fit():
    require();assert not (out()/'predictor_freeze.json').exists(),'Already fitted: use predict/analyse; never overwrite freeze'
    frame=pd.read_csv(out()/'early_features.csv');original=read(OLD/'predictor_freeze.json')['choices'];models={};params={};oof=[]
    labels=outcomes('development');start=time.perf_counter()
    with threadpool_limits(limits=4):
        for cohort in cfg()['cohorts']:
            for prefix in cfg()['prefixes']:
                key=f'{cohort}_{prefix}';choice=original[key]
                sub=frame[(frame.split=='development')&(frame.panel=='A')&(frame.prefix==prefix)]
                if cohort=='eligible':sub=sub[sub.eligible]
                sub=sub.sort_values('run').reset_index(drop=True);sub['failure']=sub.run.map(labels).astype(int)
                assert len(sub)==choice['n'] and sub.failure.sum()==choice['failures']
                fs=folds(sub,choice['folds']);base=choice['parameters']['E']['cols'];params[key]={'folds':choice['folds'],'models':{}};models[key]={}
                b=frame[(frame.split=='development')&(frame.panel=='B')&(frame.prefix==prefix)].set_index('run').loc[sub.run].reset_index()
                for kind in cfg()['learners']:
                    for name in MODELS:
                        cols=inputs(base,name);best=None
                        assert np.isfinite(sub[cols].to_numpy()).all()
                        for value in cfg()['C'] if kind=='logistic' else cfg()['depth']:
                            pr=np.zeros(len(sub))
                            for a,v in fs:
                                model=pipe(kind,value);model.fit(sub.iloc[a][cols],sub.failure.iloc[a]);pr[v]=model.predict_proba(sub.iloc[v][cols])[:,1]
                            sc=score(sub.failure,pr,sub.run)
                            if best is None or sc>best[0]:best=(sc,value,pr)
                        model=pipe(kind,best[1]);model.fit(sub[cols],sub.failure)
                        bm=pipe(kind,best[1]);bm.fit(b[cols],sub.failure)
                        label=kind+'_'+name;models[key][label]=(model,bm,cols)
                        params[key]['models'][label]={'value':best[1],'cols':cols,'oof_discoveries':best[0][0],'oof_AP':best[0][1],'inputs':len(cols),'retained':int(model[-2].get_support().sum())}
                        oof.extend({'cohort':cohort,'prefix':prefix,'model':label,'run':r,'family':f,'failure':int(y),'risk':float(v)} for r,f,y,v in zip(sub.run,sub.family,sub.failure,best[2]))
                controls=[(p['oof_discoveries'],p['oof_AP'],label) for label,p in params[key]['models'].items() if label.split('_',1)[1] in ['B','L2','N','M']]
                params[key]['strongest_structural_control']=max(controls,key=lambda t:t[:2])[2]
                event('development_fit_complete',cohort=cohort,prefix=prefix,n=len(sub),models=16,seconds=round(time.perf_counter()-start,2))
    joblib.dump(models,out()/'pipelines.joblib');pd.DataFrame(oof).to_csv(out()/'development_oof.csv',index=False)
    write(out()/'predictor_freeze.json',{'time_ns':time.time_ns(),'post_hoc':True,'choices':params,'pipelines_sha256':sha(out()/'pipelines.joblib'),'features_sha256':sha(out()/'early_features.csv'),'protocol_sha256':sha(out()/'protocol_freeze.json'),'fitting_seconds':time.perf_counter()-start})
    event('new_exploratory_predictors_frozen')
def predict():
    require();freeze=read(out()/'predictor_freeze.json');assert sha(out()/'pipelines.joblib')==freeze['pipelines_sha256'] and sha(out()/'early_features.csv')==freeze['features_sha256']
    dest=out()/'predictions.csv'
    if (out()/'prediction_freeze.json').exists():assert sha(dest)==read(out()/'prediction_freeze.json')['sha256'];return
    models=joblib.load(out()/'pipelines.joblib');frame=pd.read_csv(out()/'early_features.csv');rows=[]
    for key,collection in models.items():
        cohort,prefix=key.split('_');prefix=float(prefix)
        a=frame[(frame.split=='evaluation')&(frame.prefix==prefix)&(frame.panel=='A')]
        if cohort=='eligible':a=a[a.eligible]
        for variant,panel,use_B in [('A','A',False),('A_on_B','B',False),('B_refit','B',True)]:
            sub=frame[(frame.split=='evaluation')&(frame.prefix==prefix)&(frame.panel==panel)].set_index('run').loc[a.run].reset_index()
            assert sub.run.tolist()==a.run.tolist()
            for label,(m,bm,cols) in collection.items():
                risks=(bm if use_B else m).predict_proba(sub[cols])[:,1]
                rows.extend({'run':r,'family':int(f),'cohort':cohort,'prefix':prefix,'variant':variant,'model':label,'risk':float(v)} for r,f,v in zip(sub.run,sub.family,risks))
    pd.DataFrame(rows).to_csv(dest,index=False)
    write(out()/'prediction_freeze.json',{'time_ns':time.time_ns(),'sha256':sha(dest),'predictor_freeze_sha256':sha(out()/'predictor_freeze.json'),'evaluation_previously_consumed':True})
    event('exploratory_predictions_saved',rows=len(rows))
def reproduce():
    require();frame=pd.read_csv(out()/'early_features.csv');old_models=joblib.load(OLD/'pipelines.joblib');saved=pd.read_csv(OLD/'evaluation_predictions.csv');audit=[]
    for cohort in cfg()['cohorts']:
        for prefix in cfg()['prefixes']:
            sub=frame[(frame.split=='evaluation')&(frame.prefix==prefix)&(frame.panel=='A')]
            if cohort=='eligible':sub=sub[sub.eligible]
            for label in ['E','logistic_B','logistic_L2','logistic_L6']:
                m,cols=old_models[f'{cohort}_{prefix}'][label]
                pr=m.predict_proba(sub[cols])[:,1]
                ref=saved[(saved.cohort==cohort)&(saved.prefix==prefix)&(saved.panel=='A')&(~saved.refit_B)&(saved.model==label)].set_index('run').loc[sub.run]
                error=float(np.max(np.abs(pr-ref.risk.to_numpy())));assert error<1e-5,(cohort,prefix,label,error)
                audit.append({'cohort':cohort,'prefix':prefix,'model':label,'maximum_probability_error':error})
    primary=saved[(saved.cohort=='eligible')&(saved.prefix==.1)&(saved.panel=='A')&(~saved.refit_B)]
    ys=outcomes('evaluation')
    for label in ['E','logistic_B','logistic_L6','logistic_L8']:
        sub=primary[primary.model==label];y=sub.run.map(ys).astype(int)
        assert len(sub)==36 and y.sum()==12 and discovery(y,sub.risk,sub.run,9)==8
    write(out()/'original_reproduction.json',{'passed':True,'comparisons':audit,'original_nine_audit_counts':[8,8,8,8],'original_pipeline_unchanged':True})
