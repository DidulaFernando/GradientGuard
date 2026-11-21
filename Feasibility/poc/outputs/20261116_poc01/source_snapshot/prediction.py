from run import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,mean_absolute_error
from scipy.stats import spearmanr
import joblib

TELE=['loss','accuracy','worst_class','confidence','correct_confidence','entropy','gradient_norm','growth','gns']
SEP=['separation','smallest_cluster','cluster_gap']
def stats(v,x):
    signs=np.sign(np.diff(v));signs=signs[signs!=0]
    return {'first':v[0],'last':v[-1],'change':v[-1]-v[0],'mean':v.mean(),'std':v.std(ddof=0),'min':v.min(),'max':v.max(),'area':np.trapezoid(v,x),'slope':np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean()),'reversals':int(np.sum(signs[1:]!=signs[:-1]))}
def feature_frame(split,panel='A',prefix=.1):
    rows=[]
    for spec in read(O/'protocol.json')['runs']:
        if spec['split']!=split:continue
        folder=O/'classifiers'/spec['run']
        if not (folder/'complete.json').exists():continue
        table=pd.read_csv(folder/'early.csv');df=table[(table.panel==panel)&(table.fraction<=prefix+1e-12)&(table.step>0)].sort_values('step')
        plan=read(folder/'plan.json');expected=[s for s in plan['steps'] if s<=round(prefix*plan['horizon'])]
        assert df.step.tolist()==expected
        h=read(folder/f"health_{plan['steps'][-1]}.json")
        row={'run':spec['run'],'family':spec['family'],'split':split,'lr':spec['lr'],'eligible':bool(h['accuracy']>=.8 and h['loss']<=np.log(10) and df.valid.all()),'health_accuracy':h['accuracy'],'health_loss':h['loss']}
        x=df.fraction.to_numpy()
        for col in TELE+SEP+['shallow_share','deep_share','total','bias_norm','bias_sensitivity']:
            values=df[col].to_numpy(dtype=float);ss=stats(values,x)
            for s in ['last','mean','std','slope']:row[col+'_'+s]=ss[s]
            if col in ['loss','accuracy','growth','gns']:
                for s,val in ss.items():row['rich_'+col+'_'+s]=val
        row['saturation']=df.saturation.dropna().iloc[0] if df.saturation.notna().any() else np.nan
        final=read(folder/'complete.json')['final_health'];row['final_loss']=final['loss'];row['final_accuracy']=final['accuracy']
        rows.append(row)
    return pd.DataFrame(rows)
def inputs(rich=False,separation=False):
    cols=['lr','saturation']+[c+'_'+s for c in TELE for s in ['last','mean','std','slope']]
    if rich:cols=[c for c in cols if not any(c.startswith(n+'_') for n in ['loss','accuracy','growth','gns'])]+['rich_'+c+'_'+s for c in ['loss','accuracy','growth','gns'] for s in ['first','last','change','mean','std','min','max','area','slope','reversals']]
    if separation:cols += [c+'_'+s for c in SEP for s in ['last','mean','std','slope']]
    assert not any(c in cols for c in ['alignment','difficulty','data_seed','reservation_seed','family','run','split','wga','failure'])
    return cols
def ladder(base,name):
    cols=list(base)
    if name=='E':return cols
    cols += ['total_'+s for s in ['last','mean','std','slope']]
    if name=='B':return cols
    cols += [b+'_share_last' for b in ['shallow','deep']]
    if name=='L2':return cols
    cols += [b+'_share_'+s for b in ['shallow','deep'] for s in ['mean','std']]
    if name=='L6':return cols
    return cols+[b+'_share_slope' for b in ['shallow','deep']]
def pipeline(kind,value):
    est=LogisticRegression(C=value,max_iter=3000,random_state=130002) if kind=='logistic' else HistGradientBoostingClassifier(max_depth=value,max_iter=100,learning_rate=.05,early_stopping=False,random_state=130002)
    return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),VarianceThreshold(),est)
def selection_score(y,p,ids):
    n=len(y);k=n//4;order=np.lexsort((np.asarray(ids,dtype=str),-np.asarray(p)))[:k]
    return int(np.asarray(y)[order].sum()),float(average_precision_score(y,p))
def folds(df,y):
    for n in (3,2):
        try:fs=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=130003).split(df,y,df.family))
        except ValueError:continue
        if all(len(set(y.iloc[a]))==2 and len(set(y.iloc[b]))==2 and not set(df.family.iloc[a])&set(df.family.iloc[b]) for a,b in fs):return fs
    return None
def support(df,y,evaluation=False):
    minimum=5 if evaluation else 8
    return bool(len(df)>=(20 if evaluation else 16) and df.family.nunique()>=(6 if evaluation else 3) and all(int((y==v).sum())>=minimum and df.loc[y==v,'family'].nunique()>=3 for v in (0,1)))
def tune(df,y,cols,kind,fs):
    best=None
    for value in ([.1,1,10] if kind=='logistic' else [1,2,3]):
        pred=np.zeros(len(df))
        for a,b in fs:
            model=pipeline(kind,value);model.fit(df.iloc[a][cols],y.iloc[a]);pred[b]=model.predict_proba(df.iloc[b][cols])[:,1]
        score=selection_score(y,pred,df.run)
        if best is None or score>best[0]:best=(score,value,pred)
    model=pipeline(kind,best[1]);model.fit(df[cols],y)
    return model,best[1],best[0],best[2]
def fit():
    assert not list((O/'outcomes/test').glob('*.json')),'Held-out outcomes already accessed; predictor refit prohibited'
    assert all((O/'classifiers'/r['run']/'complete.json').exists() or (O/'classifiers'/r['run']/'diverged.json').exists() for r in read(O/'protocol.json')['runs'] if r['split']=='development')
    fitted={};choices={};support_rows=[]
    for prefix in (.1,.05):
        df=feature_frame('development',prefix=prefix);df['wga']=[read(O/'outcomes/val'/(r+'.json'))['wga'] for r in df.run];df.to_csv(O/f'development_features_{prefix}.csv',index=False)
        for cohort in ['eligible','full']:
            sub=df[df.eligible].reset_index(drop=True) if cohort=='eligible' else df.reset_index(drop=True)
            y=(sub.wga<.75).astype(int); fs=folds(sub,y) if len(set(y))==2 else None;ok=support(sub,y) and fs is not None
            key=f'{cohort}_{prefix}';choices[key]={'classification_supported':ok,'n':len(sub),'failures':int(y.sum()),'nonfailures':int((y==0).sum()),'families':sub.family.nunique()};support_rows.append({'cohort':cohort,'prefix':prefix,**choices[key]})
            base=inputs(False,True)
            if ok:
                candidates=[]
                for rich,sep,kind in itertools.product([False,True],[False,True],['logistic','histogram']):
                    cols=inputs(rich,sep);model,value,score,pred=tune(sub,y,cols,kind,fs);candidates.append((score,rich,sep,kind,cols,model,value))
                # max preserves first candidate on exact ties, matching frozen simplicity ordering.
                win=max(candidates,key=lambda z:z[0]);base=win[4];models={'E':(win[5],base)};params={'E':{'kind':win[3],'value':win[6],'cols':base}}
                control=[]
                for kind,name in itertools.product(['logistic','histogram'],['B','L2','L6','L8']):
                    cols=ladder(base,name);m,v,s,pr=tune(sub,y,cols,kind,fs);label=kind+'_'+name;models[label]=(m,cols);params[label]={'kind':kind,'value':v,'cols':cols};
                    if name!='L8':control.append((s,label))
                for name in ['E','L8']:
                    cols=ladder(base,name)+[b+'_'+s for b in ['bias_norm','bias_sensitivity'] for s in ['last','slope']]
                    m,v,s,pr=tune(sub,y,cols,'logistic',fs);label='bias_'+name;models[label]=(m,cols);params[label]={'kind':'logistic','value':v,'cols':cols}
                m,v,s,pr=tune(sub,y,['lr','final_loss','final_accuracy'],'logistic',fs);models['final_monitor']=(m,['lr','final_loss','final_accuracy']);params['final_monitor']={'kind':'logistic','value':v,'cols':['lr','final_loss','final_accuracy']}
                choices[key].update(strongest_nontemporal=max(control,key=lambda z:z[0])[1],parameters=params,folds=[{'train_families':sorted(sub.family.iloc[a].unique().tolist()),'validation_families':sorted(sub.family.iloc[b].unique().tolist())} for a,b in fs]);fitted[key]=models
                # Matched panel-B refit uses fixed panel-A hyperparameters and memberships.
                bdf=feature_frame('development','B',prefix).set_index('run').loc[sub.run].reset_index()
                fitted[key+'_B']={}
                for label,(model,cols) in models.items():
                    pp=params[label];bm=pipeline(pp['kind'],pp['value']);bm.fit(bdf[cols],y);fitted[key+'_B'][label]=(bm,cols)
            # Continuous secondary ladder, always separated from binary support.
            if len(sub)>=6 and sub.family.nunique()>=3:
                rg={}
                for name in ['E','B','L2','L6','L8']:
                    cols=ladder(base,name);best=None
                    for alpha in [.1,1,10]:
                        scores=[]
                        for a,b in GroupKFold(n_splits=min(3,sub.family.nunique())).split(sub,groups=sub.family):
                            m=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha));m.fit(sub.iloc[a][cols],sub.wga.iloc[a]);scores.append(mean_absolute_error(sub.wga.iloc[b],m.predict(sub.iloc[b][cols])))
                        if best is None or np.mean(scores)<best[0]:best=(np.mean(scores),alpha)
                    m=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=best[1]));m.fit(sub[cols],sub.wga);rg[name]=(m,cols);choices[key]['ridge_'+name]={'alpha':best[1],'cols':cols}
                fitted[key+'_ridge']=rg
    joblib.dump(fitted,O/'pipelines.joblib');pd.DataFrame(support_rows).to_csv(O/'development_support.csv',index=False)
    write_json(O/'predictor_freeze.json',{'time_ns':time.time_ns(),'choices':choices,'pipelines_sha256':sha(O/'pipelines.joblib'),'protocol_sha256':sha(O/'protocol.json'),'source_sha256':read(O/'source_freeze.json')})
    event('predictors_frozen',models=len(fitted))
def predict():
    frozen=read(O/'predictor_freeze.json');assert sha(O/'pipelines.joblib')==frozen['pipelines_sha256'];models=joblib.load(O/'pipelines.joblib');records=[]
    for prefix in (.1,.05):
        a=feature_frame('evaluation','A',prefix);b=feature_frame('evaluation','B',prefix);a.to_csv(O/f'evaluation_features_{prefix}.csv',index=False)
        for cohort in ['eligible','full']:
            key=f'{cohort}_{prefix}';aa=a[a.eligible] if cohort=='eligible' else a
            for panel,refit,df,mkey in [('A',False,aa,key),('B',False,b[b.run.isin(aa.run)],key),('B',True,b[b.run.isin(aa.run)],key+'_B')]:
                for label,(m,cols) in models.get(mkey,{}).items():
                    pr=m.predict_proba(df[cols])[:,1]
                    records.extend({'run':r,'cohort':cohort,'prefix':prefix,'panel':panel,'refit_B':refit,'model':label,'risk':float(v)} for r,v in zip(df.run,pr))
            for label,(m,cols) in models.get(key+'_ridge',{}).items():
                records.extend({'run':r,'cohort':cohort,'prefix':prefix,'panel':'A','refit_B':False,'model':'ridge_'+label,'risk':float(-v)} for r,v in zip(aa.run,m.predict(aa[cols])))
    pd.DataFrame(records,columns=['run','cohort','prefix','panel','refit_B','model','risk']).to_csv(O/'evaluation_predictions.csv',index=False)
    write_json(O/'prediction_freeze.json',{'time_ns':time.time_ns(),'predictions_sha256':sha(O/'evaluation_predictions.csv'),'predictor_freeze_sha256':sha(O/'predictor_freeze.json')});event('evaluation_predictions_frozen')

