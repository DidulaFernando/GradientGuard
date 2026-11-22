from .core import *
from .measure_data import LAYERS
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
import joblib
TELE=['loss','accuracy','worst_class','confidence','correct_confidence','entropy','gradient_norm','growth','gns']
SEP=['separation','smallest_cluster','cluster_gap']
MODELS=['E','B','L2','L6','N','NZ','M','MZ']
def six():return [b+'_share_'+s for b in ['shallow','deep'] for s in ['last','mean','std']]
def summaries(v,x):
    v=np.asarray(v,dtype=float)
    return {'last':float(v[-1]),'mean':float(v.mean()),'std':float(v.std(ddof=0))}
def allstats(v,x):
    r=summaries(v,x);v=np.asarray(v,dtype=float);x=np.asarray(x,dtype=float)
    signs=np.sign(np.diff(v));signs=signs[signs!=0]
    r.update(first=float(v[0]),change=float(v[-1]-v[0]),min=float(v.min()),max=float(v.max()),area=float(np.trapezoid(v,x)),slope=float(np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean())),reversals=int(np.sum(signs[1:]!=signs[:-1])))
    return r

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

def conventional(rich=False,separation=False):
    cols=['lr','saturation']+[c+'_'+s for c in TELE for s in ['last','mean','std','slope']]
    if rich:
        cols=[c for c in cols if not any(c.startswith(n+'_') for n in ['loss','accuracy','growth','gns'])]+['rich_'+c+'_'+s for c in ['loss','accuracy','growth','gns'] for s in ['first','last','change','mean','std','min','max','area','slope','reversals']]
    if separation:cols += [c+'_'+s for c in SEP for s in ['last','mean','std','slope']]
    return cols

def feature_frame(split):
    require();rows=[]
    for spec in protocol()['runs']:
        folder=O/'classifiers'/spec['run']
        if spec['split']!=split or not (folder/'complete.json').exists():continue
        plan=read(folder/'plan.json');early=pd.read_csv(folder/'early.csv');h=read(folder/f"health_{plan['steps'][-1]}.json")
        full=early[(early.panel=='A')&(early.step>0)]
        full_required=['loss','accuracy','gradient_norm','growth','gns','total','shallow_share','deep_share','bias_norm','bias_sensitivity']
        full_valid=bool(full.valid.all() and np.isfinite(full[full_required].to_numpy()).all())
        for prefix in cfg()['prefixes']:
            for panel in ['A','B']:
                df=early[(early.panel==panel)&(early.step>0)&(early.fraction<=prefix+1e-12)].sort_values('step')
                expected=[s for s in plan['steps'] if s<=round(prefix*plan['horizon'])]
                assert df.step.tolist()==expected and len(expected)>1
                x=df.fraction.to_numpy()
                required=['loss','accuracy','gradient_norm','growth','gns','total','shallow_share','deep_share','bias_norm','bias_sensitivity']
                valid=bool(df.valid.all() and np.isfinite(df[required].to_numpy()).all())
                eligible=bool(h['accuracy']>=.8 and h['loss']<=np.log(10) and np.isfinite([h['accuracy'],h['loss']]).all() and full_valid)
                row={'run':spec['run'],'family':spec['family'],'split':split,'prefix':prefix,'panel':panel,'eligible':eligible,'feature_valid':valid,'health_accuracy':h['accuracy'],'health_loss':h['loss'],'lr':spec['lr']}
                for col in TELE+SEP+['total','shallow_share','deep_share','bias_norm','bias_sensitivity']:
                    ss=allstats(df[col].to_numpy(),x)
                    for stat in ['last','mean','std','slope']:
                        if col in ['shallow_share','deep_share'] and stat=='slope':continue
                        row[col+'_'+stat]=ss[stat]
                    if col in ['loss','accuracy','growth','gns']:
                        for stat,v in ss.items():row['rich_'+col+'_'+stat]=v
                row['saturation']=float(early[(early.panel==panel)&(early.step==plan['steps'][0])].saturation.iloc[0])
                for col in [f'{l}_{kind}_{stat}' for l in LAYERS for kind in ['w','g'] for stat in ['l2']+cfg()['marginals']]+['proxy_shallow','proxy_deep']:
                    for stat,v in summaries(df[col].to_numpy(),x).items():row[col+'_'+stat]=v
                rows.append(row)
    frame=pd.DataFrame(rows)
    assert not {'failure','wga','final_accuracy','final_loss','alignment','difficulty'}&set(frame.columns)
    if len(frame):
        # Panel membership cannot redefine eligibility; A determines the frozen cohort.
        a=frame[frame.panel=='A'].set_index(['run','prefix']).eligible
        frame['eligible']=[bool(a.loc[(r,p)]) for r,p in zip(frame.run,frame.prefix)]
    frame.to_csv(O/f'{split}_features.csv',index=False)
    return frame

def valid_folds(frame):
    y=frame.failure.astype(int)
    for n in cfg()['cv_folds']:
        if frame.family.nunique()<n:continue
        fs=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=cfg()['cv_seed']).split(frame,y,frame.family))
        if all(len(set(y.iloc[a]))==len(set(y.iloc[b]))==2 and not set(frame.family.iloc[a])&set(frame.family.iloc[b]) for a,b in fs):return fs
    return None

def support(frame,evaluation=False):
    rule=cfg()['evaluation_support' if evaluation else 'development_support'];counts={}
    for value in [0,1]:
        sub=frame[frame.failure==value];counts[str(value)]={'models':len(sub),'families':sub.family.nunique()}
    ok=all(v['models']>=rule['minimum_each'] and v['families']>=rule['families_each'] for v in counts.values())
    if evaluation:ok=ok and len(frame)>=rule['minimum_n'] and frame.family.nunique()>=rule['minimum_families']
    return {'passed':bool(ok),'N':len(frame),'families':frame.family.nunique(),'outcomes':counts,'requirements':rule}

def dev_labels():
    return {p.stem:float(read(p)['wga']) for p in (O/'outcomes/val').glob('*.json')}

def readiness():
    require();frame=feature_frame('development');labels=dev_labels()
    frame=frame[(frame.panel=='A')&(frame.prefix==.1)&frame.eligible&frame.feature_valid].copy();frame['wga']=frame.run.map(labels);assert frame.wga.notna().all();frame['failure']=(frame.wga<.75).astype(int)
    audit=support(frame);fs=valid_folds(frame) if len(frame) and frame.failure.nunique()==2 else None
    specs=[s for s in protocol()['runs'] if s['split']=='development'];complete=sum((O/'classifiers'/s['run']/'complete.json').exists() for s in specs);diverged=sum((O/'classifiers'/s['run']/'diverged.json').exists() for s in specs)
    audit.update(valid_grouped_folds=fs is not None,completed=complete,diverged=diverged,incomplete=len(specs)-complete-diverged,planned=len(specs),selection_uses_predictive_advantage=False)
    audit['unavailable']=len([r for r in specs if (O/'unavailable'/(r['run']+'.json')).exists()])
    audit['unaccounted']=audit['incomplete']-audit['unavailable']
    audit['passed']=bool(audit['passed'] and fs is not None and audit['unaccounted']==0)
    if fs is not None:audit['folds']=[{'train_families':sorted(frame.family.iloc[a].unique().tolist()),'validation_families':sorted(frame.family.iloc[b].unique().tolist())} for a,b in fs]
    frame.to_csv(O/'development_readiness_cohort.csv',index=False);write_json(O/'readiness.json',audit);event('development_readiness',**{k:audit[k] for k in ['passed','N','families','completed','diverged','incomplete']});return audit['passed']

def selection(y,p,ids):return discovery(y,p,ids,len(y)//3),float(average_precision_score(y,p))
def tune(frame,cols,kind,fs):
    best=None;y=frame.failure.astype(int)
    for value in cfg()['C'] if kind=='logistic' else cfg()['depth']:
        pred=np.zeros(len(frame))
        for a,b in fs:
            m=pipe(kind,value);m.fit(frame.iloc[a][cols],y.iloc[a]);pred[b]=m.predict_proba(frame.iloc[b][cols])[:,1]
        sc=selection(y,pred,frame.run)
        if best is None or sc>best[0]:best=(sc,value,pred)
    m=pipe(kind,best[1]);m.fit(frame[cols],y)
    return m,best[1],best[0],best[2]

def fit():
    require();assert read(O/'readiness.json')['passed']
    assert not (O/'outcomes/test').exists(),'Evaluation already accessed; refitting prohibited'
    if (O/'predictor_freeze.json').exists():assert sha(O/'pipelines.joblib')==read(O/'predictor_freeze.json')['pipelines_sha256'];return
    frame=pd.read_csv(O/'development_features.csv');labels=dev_labels();allmodels={};choices={};oof=[];start=time.perf_counter()
    with threadpool_limits(limits=4):
        for cohort in ['eligible','full']:
            for prefix in cfg()['prefixes']:
                sub=frame[(frame.panel=='A')&(frame.prefix==prefix)&frame.feature_valid]
                if cohort=='eligible':sub=sub[sub.eligible]
                sub=sub.sort_values('run').reset_index(drop=True);sub['wga']=sub.run.map(labels);sub['failure']=(sub.wga<.75).astype(int);fs=valid_folds(sub)
                if fs is None:raise ValueError('No valid grouped folds for predefined cohort/prefix')
                key=f'{cohort}_{prefix}';candidates=[]
                for rich,sep,kind in itertools.product([False,True],[False,True],['logistic','histogram']):
                    cols=conventional(rich,sep);m,value,sc,pred=tune(sub,cols,kind,fs);candidates.append((sc,rich,sep,kind,cols,m,value,pred))
                win=max(candidates,key=lambda v:v[0]);base=win[4];collection={};params={};control=[]
                b=frame[(frame.panel=='B')&(frame.prefix==prefix)].set_index('run').loc[sub.run].reset_index()
                for kind,name in itertools.product(['logistic','histogram'],MODELS):
                    cols=inputs(base,name);m,value,sc,pred=tune(sub,cols,kind,fs);bm=pipe(kind,value);bm.fit(b[cols],sub.failure)
                    label=kind+'_'+name;collection[label]=(m,bm,cols);params[label]={'kind':kind,'value':value,'cols':cols,'oof_discoveries':sc[0],'oof_AP':sc[1]}
                    if name in ['E','B','N','M']:control.append((sc,label))
                    oof.extend({'cohort':cohort,'prefix':prefix,'model':label,'run':r,'family':int(f),'failure':int(y),'risk':float(v)} for r,f,y,v in zip(sub.run,sub.family,sub.failure,pred))
                # E denotes the actual conventional candidate selected across learner classes.
                bm=pipe(win[3],win[6]);bm.fit(b[base],sub.failure);collection['E']=(win[5],bm,base);params['E']={'kind':win[3],'value':win[6],'cols':base,'oof_discoveries':win[0][0],'oof_AP':win[0][1],'rich':win[1],'separation':win[2]}
                control.append((win[0],'E'))
                choices[key]={'parameters':params,'strongest_nonZ':max(control,key=lambda v:v[0])[1],'folds':[{'train_families':sorted(sub.family.iloc[a].unique().tolist()),'validation_families':sorted(sub.family.iloc[b].unique().tolist())} for a,b in fs],'N':len(sub)};allmodels[key]=collection
                event('fresh_predictor_cohort_fit',cohort=cohort,prefix=prefix,models=len(collection),N=len(sub))
    joblib.dump(allmodels,O/'pipelines.joblib');pd.DataFrame(oof).to_csv(O/'development_oof.csv',index=False)
    write_json(O/'predictor_freeze.json',{'time_ns':time.time_ns(),'choices':choices,'pipelines_sha256':sha(O/'pipelines.joblib'),'development_features_sha256':sha(O/'development_features.csv'),'protocol_sha256':sha(O/'protocol.json'),'source_hashes':require()['source_hashes'],'seconds':time.perf_counter()-start})
    event('fresh_predictors_frozen')

def predict():
    allow_evaluation_training()
    if (O/'prediction_freeze.json').exists():allow_test_outcomes();return
    assert not (O/'outcomes/test').exists()
    frame=feature_frame('evaluation');models=joblib.load(O/'pipelines.joblib');records=[]
    for key,collection in models.items():
        cohort,prefix=key.split('_');prefix=float(prefix);a=frame[(frame.panel=='A')&(frame.prefix==prefix)&frame.feature_valid]
        if cohort=='eligible':a=a[a.eligible]
        if len(a)==0:continue
        for variant,panel,refit in [('A','A',False),('A_on_B','B',False),('B_refit','B',True)]:
            sub=frame[(frame.panel==panel)&(frame.prefix==prefix)].set_index('run').loc[a.run].reset_index();assert sub.run.tolist()==a.run.tolist()
            for label,(m,bm,cols) in collection.items():
                risks=(bm if refit else m).predict_proba(sub[cols])[:,1]
                records.extend({'cohort':cohort,'prefix':prefix,'variant':variant,'model':label,'run':r,'family':int(f),'risk':float(v)} for r,f,v in zip(sub.run,sub.family,risks))
    pd.DataFrame(records).to_csv(O/'evaluation_predictions.csv',index=False)
    write_json(O/'prediction_freeze.json',{'time_ns':time.time_ns(),'predictor_freeze_sha256':sha(O/'predictor_freeze.json'),'predictions_sha256':sha(O/'evaluation_predictions.csv'),'evaluation_features_sha256':sha(O/'evaluation_features.csv')});event('fresh_predictions_frozen',rows=len(records))
