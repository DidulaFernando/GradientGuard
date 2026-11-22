"""Bounded development-only analysis. New fits reconstruct Z, never failure labels."""
from pathlib import Path
import os, sys, json, hashlib, time, subprocess, platform, itertools, math, argparse
import xml.etree.ElementTree as ET
HOME=Path(__file__).resolve().parent
BASE=HOME.parent
REPO=BASE.parent
for key,rel in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/pycache','PIP_CACHE_DIR':'cache/pip','MPLCONFIGDIR':'cache/matplotlib','TORCH_HOME':'cache/torch','XDG_CACHE_HOME':'cache/xdg'}.items():
    (BASE/rel).mkdir(parents=True,exist_ok=True);os.environ[key]=str(BASE/rel)
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
CFG=read(HOME/'config.json');SRC=BASE/'runs'/CFG['source_attempt'];O=BASE/'runs'/CFG['attempt'];R=BASE/'reports'/CFG['attempt'];Z=CFG['features']
READS=[]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf8')
def source_hashes():return {str(p.relative_to(HOME)):sha(p) for p in sorted(HOME.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.ps1','.md','.txt'] and '__pycache__' not in p.parts}
def allowed_paths():
    fixed=['development_features.csv','development_oof.csv','run_manifest.csv','predictor_freeze.json','readiness.json','development_import.json','requirements-lock.txt','environment.json']
    ids=[r['run'] for r in read(SRC/'development_import.json')['records']]
    return {str((SRC/p).resolve()) for p in fixed}|{str((SRC/'outcomes/val'/f'{r}.json').resolve()) for r in ids}
def check_input(p):
    p=Path(p).resolve()
    if str(p) not in allowed_paths():raise ValueError('Outside development semantic-input allowlist: '+str(p))
    READS.append(str(p));return p
def csv_input(p):return pd.read_csv(check_input(p))
def json_input(p):return read(check_input(p))
def protect():
    inherited=read(SRC/'protocol_freeze.json')['protected_inputs']
    for path,h in inherited.items():assert sha(path)==h,'Inherited protected input changed: '+path
    for folder in [SRC,BASE/'reports'/CFG['source_attempt'],BASE/'amended_validation',REPO/'papers']:
        for p in folder.rglob('*'):
            if p.is_file() and not {'cache','tmp','__pycache__'}&set(p.parts):inherited[str(p)]=sha(p)
    return inherited
def verify():
    f=read(O/'plan_freeze.json');assert f['source_hashes']==source_hashes(),'Diagnostic implementation changed after freeze'
    assert sha(O/'plan.json')==f['plan_sha256']
    changed=[p for p,h in f['protected_inputs'].items() if not Path(p).is_file() or sha(p)!=h]
    result={'passed':not changed,'protected_files':len(f['protected_inputs']),'changed_paths':changed}
    write(O/'preservation_after.json',result);assert not changed,changed
    return result
def freeze():
    O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (O/'plan_freeze.json').exists():verify();return
    suites=list(ET.parse(O/'tests.xml').getroot().iter('testsuite'))
    assert suites and all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 and int(s.get('tests','0'))>=16 for s in suites),'Tests must pass before freeze'
    (O/'tests_before_freeze.xml').write_bytes((O/'tests.xml').read_bytes())
    expected=(SRC/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines()
    actual=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True).splitlines()
    env={'python':sys.version,'executable':sys.executable,'os':platform.platform(),'lock_matches':set(actual)==set(expected),'missing':sorted(set(expected)-set(actual)),'extra':sorted(set(actual)-set(expected)),'precision':'CPU float64; fixed single-thread linear algebra; no GPU training','settings':{k:os.environ.get(k) for k in ['TEMP','TMP','PYTHONPYCACHEPREFIX','PIP_CACHE_DIR','MPLCONFIGDIR']}}
    write(O/'environment.json',env);assert env['lock_matches'] and env['python']==read(SRC/'environment.json')['python']
    (O/'requirements-lock.txt').write_text('\n'.join(actual)+'\n',encoding='utf8')
    inputs={p:sha(p) for p in sorted(allowed_paths())};protected=protect()
    write(O/'plan.json',CFG);write(O/'plan_freeze.json',{'time_ns':time.time_ns(),'plan_sha256':sha(O/'plan.json'),'source_hashes':source_hashes(),'inputs':inputs,'protected_inputs':protected,'tests_sha256':sha(O/'tests_before_freeze.xml'),'post_hoc':True,'heldout_semantic_access_authorised':False})
    for rel in source_hashes():
        p=O/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HOME/rel).read_bytes())
    print(json.dumps({'stage':'frozen','protected_files':len(protected),'inputs':len(inputs),'lock_matches':True}),flush=True)
def variance_parts(values,families):
    v=np.asarray(values,dtype=np.float64);g=np.asarray(families);assert np.isfinite(v).all()
    center=np.array([v[g==f].mean() for f in g]);within=float(np.sum((v-center)**2));between=float(np.sum((center-v.mean())**2));total=float(np.sum((v-v.mean())**2))
    assert np.isclose(total,within+between,atol=1e-12,rtol=1e-10)
    return {'total_ss':total,'within_ss':within,'between_ss':between,'between_fraction':between/total if total>CFG['flat_floor']**2 else None,'within_fraction':within/total if total>CFG['flat_floor']**2 else None,'sd':float(v.std()),'unique':int(len(np.unique(v)))}
def auc_within(y,x):
    y=np.asarray(y,dtype=int);x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    if len(x)!=len(y) or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid pair labels/features')
    a=x[y==1];b=x[y==0]
    if not len(a) or not len(b):raise ValueError('A mixed family is required')
    d=a[:,None,:]-b[None,:,:];return ((d>0)+.5*(d==0)).mean(axis=(0,1))
def fixed_reconstruction(x,y,folds):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);pred=np.full(y.shape,np.nan);base=np.full(y.shape,np.nan);units=np.full(y.shape,np.nan);audits=[]
    for fold,(train,test) in enumerate(folds):
        assert not set(train)&set(test)
        mu=y[train].mean(0);sd=y[train].std(0);scale=np.where(sd>CFG['flat_floor'],sd,1.)
        pipeline=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=CFG['ridge_alpha'],solver='svd'))
        pipeline.fit(x[train],(y[train]-mu)/scale)
        result=np.asarray(pipeline.predict(x[test])).reshape(len(test),y.shape[1])
        pred[test]=result*scale+mu;base[test]=mu;units[test]=scale
        audits.append({'fold':fold,'train_rows':len(train),'test_rows':len(test),'imputation_mean':pipeline.steps[0][1].statistics_.tolist(),'input_scaling_mean':pipeline.steps[1][1].mean_.tolist(),'target_mean':mu.tolist(),'target_sd':sd.tolist()})
    assert np.isfinite(pred).all() and np.isfinite(y).all()
    return pred,base,units,audits
def reconstruction_scores(y,p,b,u):
    out=[]
    for j in range(y.shape[1]):
        sse=float(np.sum((y[:,j]-p[:,j])**2));sst=float(np.sum((y[:,j]-y[:,j].mean())**2));null=float(np.sum((y[:,j]-b[:,j])**2))
        out.append({'R2':1-sse/sst if sst>CFG['flat_floor']**2 else None,'RMSE':float(np.sqrt(sse/len(y))),'skill_vs_fold_training_mean':1-sse/null if null>CFG['flat_floor']**2 else None,'standardised_RMSE':float(np.sqrt(np.mean(((y[:,j]-p[:,j])/u[:,j])**2)))})
    return out
def fold_indices(frame,saved):
    folds=[];seen=[]
    for f in saved:
        tf=set(f['train_families']);vf=set(f['validation_families']);assert not tf&vf
        train=np.flatnonzero(frame.family.isin(tf));test=np.flatnonzero(frame.family.isin(vf))
        assert len(train)+len(test)==len(frame) and len(test)>0
        seen.extend(test.tolist());folds.append((train,test))
    assert sorted(seen)==list(range(len(frame)));return folds
def conditional_null(y,x,families):
    """24-column joint raw/residual/panel statistic; no outcome-trained risk scores."""
    y=np.asarray(y,int);x=np.asarray(x,float);families=np.asarray(families);options=[];observed=[];families_used=[]
    for fam in sorted(set(families)):
        ix=np.flatnonzero(families==fam);yy=y[ix]
        if len(set(yy))<2:continue
        family_options=[]
        for chosen in itertools.combinations(range(len(ix)),int(yy.sum())):
            labels=np.zeros(len(ix),int);labels[list(chosen)]=1;family_options.append(auc_within(labels,x[ix]))
        options.append(np.asarray(family_options));observed.append(auc_within(yy,x[ix]));families_used.append(int(fam))
    if not options:return None,[]
    obs=np.mean(observed,axis=0);combinations=math.prod(len(a) for a in options)
    if combinations<=CFG['permutation_exact_limit']:
        draws=np.array([sum((options[j][a] for j,a in enumerate(combo)))/len(options) for combo in itertools.product(*(range(len(a)) for a in options))]);mode='exact';offset=0
    else:
        rng=np.random.default_rng(CFG['permutation_seed']);draws=sum(a[rng.integers(len(a),size=CFG['permutation_draws'])] for a in options)/len(options);mode='monte_carlo';offset=1
    dev=np.abs(draws-.5);target=np.abs(obs-.5);maxdev=dev.max(1)
    details=[{'concordance':float(obs[j]),'two_sided_conditional_p':float((offset+np.sum(dev[:,j]>=target[j]-1e-12))/(offset+len(draws))),'max24_adjusted_p':float((offset+np.sum(maxdev>=target[j]-1e-12))/(offset+len(draws)))} for j in range(x.shape[1])]
    return {'families':families_used,'mixed_families':len(options),'assignments':combinations,'draws':len(draws),'mode':mode,'seed':CFG['permutation_seed']},details
def datasets():
    frame=csv_input(SRC/'development_features.csv');manifest=csv_input(SRC/'run_manifest.csv');frozen=json_input(SRC/'predictor_freeze.json');imported=json_input(SRC/'development_import.json')
    dev=manifest[manifest.split=='development'];ids={r['run'] for r in imported['records']}
    assert len(ids)==124 and len(dev)==128 and not ids-set(dev.run)
    assert set(frame.run)==ids and set(frame.split)=={'development'}
    family_map=manifest.set_index('run').family
    assert frame.family.tolist()==frame.run.map(family_map).tolist()
    outcomes={r:json_input(SRC/'outcomes/val'/f'{r}.json') for r in sorted(ids)}
    assert all(v['split']=='val' and v['run']==r and bool(v['failure'])==(v['wga']<.75) for r,v in outcomes.items())
    frame=frame[frame.prefix==CFG['prefix']].copy();frame['wga']=frame.run.map({r:v['wga'] for r,v in outcomes.items()});frame['failure']=(frame.wga<.75).astype(int)
    assert not frame.duplicated(['run','panel']).any()
    return frame,manifest,frozen
def analyse():
    verify();start=time.perf_counter();frame,manifest,frozen=datasets();oof=csv_input(SRC/'development_oof.csv')
    all_summary=[];cohorts=[];vrows=[];recon=[];predrows=[];audits=[];membership=[];noise=[];correlations=[];within=[];pairs=[];perm=[];riskrows=[];familystats=[]
    for cohort in CFG['cohorts']:
        af=frame[(frame.panel=='A')&frame.feature_valid].copy()
        if cohort=='eligible':af=af[af.eligible]
        af=af.sort_values('run').reset_index(drop=True);ids=af.run.tolist();key=cohort+'_0.1';choice=frozen['choices'][key];folds=fold_indices(af,choice['folds'])
        assert len(af)==choice['N']
        for i,(tr,te) in enumerate(folds):membership.extend({'cohort':cohort,'fold':i,'role':role,'run':af.run.iloc[j],'family':int(af.family.iloc[j])} for role,ix in [('train',tr),('validation',te)] for j in ix)
        mixed=[]
        for fam,g in af.groupby('family'):
            nf=int(g.failure.sum());nn=len(g)-nf
            familystats.append({'cohort':cohort,'family':int(fam),'models':len(g),'failures':nf,'nonfailures':nn,'mixed':nf>0 and nn>0,'failure_nonfailure_pairs':nf*nn})
            if nf and nn:mixed.append(int(fam))
        cohorts.append({'cohort':cohort,'N':len(af),'families':af.family.nunique(),'failures':int(af.failure.sum()),'nonfailures':int((1-af.failure).sum()),'mixed_families':mixed,'mixed_family_models':int(af[af.family.isin(mixed)].shape[0]),'folds':len(folds)})
        paneldata={};residuals={}
        for panel in CFG['panels']:
            f=frame[frame.panel==panel].set_index('run').loc[ids].reset_index();assert f.feature_valid.all() and f.family.tolist()==af.family.tolist();paneldata[panel]=f
            y=f[Z].to_numpy(np.float64);assert np.isfinite(y).all()
            for z in Z:vrows.append({'cohort':cohort,'panel':panel,'feature':z,**variance_parts(f[z],f.family)})
            for bank in CFG['reconstruction_banks']:
                columns=choice['parameters']['logistic_'+bank]['cols']
                assert not set(columns)&set(Z) and not set(columns)&{'failure','wga','family','run','eligible','health_accuracy','health_loss','alignment','difficulty','split','data_seed','reservation_seed'}
                x=f[columns].to_numpy(np.float64);assert not np.isinf(x).any()
                with threadpool_limits(limits=1):p,b,u,a=fixed_reconstruction(x,y,folds)
                audits.extend({'cohort':cohort,'panel':panel,'bank':bank,'columns':columns,**v} for v in a)
                recon.extend({'cohort':cohort,'panel':panel,'bank':bank,'feature':z,**s} for z,s in zip(Z,reconstruction_scores(y,p,b,u)))
                predrows.extend({'cohort':cohort,'panel':panel,'bank':bank,'run':r,'family':int(fam),'feature':z,'observed':float(y[i,j]),'predicted':float(p[i,j]),'residual':float(y[i,j]-p[i,j]),'train_reference':float(b[i,j]),'train_sd':float(u[i,j])} for i,(r,fam) in enumerate(zip(f.run,f.family)) for j,z in enumerate(Z))
                if bank=='M':residuals[panel]=y-p
            for z in Z:
                proxy=z.replace('shallow_share','proxy_shallow').replace('deep_share','proxy_deep')
                raw=f[[z,proxy]].astype(float);centered=raw-raw.groupby(f.family).transform('mean')
                correlations.append({'cohort':cohort,'panel':panel,'feature':z,'proxy':proxy,'pearson':raw.corr().iloc[0,1],'spearman':raw.corr(method='spearman').iloc[0,1],'within_pearson':centered.corr().iloc[0,1],'within_spearman':centered.corr(method='spearman').iloc[0,1]})
        names=[];columns=[]
        for panel in CFG['panels']:
            for representation,values in [('raw',paneldata[panel][Z].to_numpy(float)),('M_residual',residuals[panel])]:
                columns.append(values);names.extend((panel,representation,z) for z in Z)
        combined=np.column_stack(columns);meta,stats=conditional_null(af.failure,combined,af.family)
        if meta:perm.append({'cohort':cohort,**meta})
        for col,(panel,representation,z) in enumerate(names):
            family_values=[]
            for fam in mixed:
                ix=np.flatnonzero(af.family.to_numpy()==fam);score=float(auc_within(af.failure.iloc[ix],combined[ix,col])[0]);family_values.append(score)
                within.append({'cohort':cohort,'panel':panel,'representation':representation,'feature':z,'family':fam,'concordance':score})
                for fi in ix[af.failure.iloc[ix].to_numpy()==1]:
                    for ni in ix[af.failure.iloc[ix].to_numpy()==0]:
                        fr=manifest.set_index('run').loc[af.run.iloc[fi]];nr=manifest.set_index('run').loc[af.run.iloc[ni]]
                        pairs.append({'cohort':cohort,'panel':panel,'representation':representation,'feature':z,'family':fam,'failure_run':af.run.iloc[fi],'nonfailure_run':af.run.iloc[ni],'same_construction':bool(fr.construction==nr.construction),'difference':combined[fi,col]-combined[ni,col]})
            if meta:
                omission=[np.mean([v for j,v in enumerate(family_values) if j!=i]) for i in range(len(family_values))] if len(family_values)>1 else []
                all_summary.append({'cohort':cohort,'panel':panel,'representation':representation,'feature':z,'mixed_families':len(mixed),'leave_one_family_min':min(omission) if omission else None,'leave_one_family_max':max(omission) if omission else None,**stats[col]})
        for z in Z:
            a=paneldata['A'][z];b=paneldata['B'][z];ac=a-a.groupby(af.family).transform('mean');bc=b-b.groupby(af.family).transform('mean')
            for label,x,y in [('raw',a,b),('family_centred',ac,bc)]:
                spread=float(((x+y)/2).std(ddof=0));sampling=float(np.sqrt(np.mean((x-y)**2)/2))
                noise.append({'cohort':cohort,'feature':z,'representation':label,'panel_difference_RMS_over_sqrt2':sampling,'spread':spread,'spread_over_panel_difference':spread/sampling if sampling>CFG['flat_floor'] else None,'panel_correlation':float(np.corrcoef(x,y)[0,1]) if x.std()>CFG['flat_floor'] and y.std()>CFG['flat_floor'] else None})
        rr=oof[(oof.cohort==cohort)&(oof.prefix==.1)]
        for model in ['logistic_B','logistic_L6','logistic_M','logistic_MZ']:
            q=rr[rr.model==model].set_index('run').loc[ids];assert q.failure.tolist()==af.failure.tolist()
            scores=[]
            for fam in mixed:
                ix=np.flatnonzero(af.family.to_numpy()==fam);s=float(auc_within(af.failure.iloc[ix],q.risk.iloc[ix])[0]);scores.append(s)
                riskrows.append({'cohort':cohort,'model':model,'family':fam,'concordance':s,'note':'existing development-selection OOF; no new fit'})
            riskrows.append({'cohort':cohort,'model':model,'family':'equal_family_mean','concordance':float(np.mean(scores)) if scores else None,'note':'descriptive development data, not independent validation'})
        print(json.dumps({'stage':'cohort_complete','cohort':cohort,'N':len(af),'mixed_families':len(mixed),'auxiliary_fits':len(CFG['panels'])*len(CFG['reconstruction_banks'])*len(folds)}),flush=True)
    for name,data in [('cohort_support',cohorts),('family_support',familystats),('variance_decomposition',vrows),('reconstruction_scores',recon),('reconstruction_oof',predrows),('fold_membership',membership),('matched_proxy_correlations',correlations),('panel_sampling_diagnostics',noise),('within_family_feature_concordance',within),('within_family_summary',all_summary),('within_family_pairs',pairs),('existing_oof_within_family',riskrows)]:pd.DataFrame(data).to_csv(O/(name+'.csv'),index=False)
    write(O/'preprocessing_audit.json',audits);write(O/'conditional_permutation.json',perm);write(O/'semantic_read_audit.json',{'unique_paths':sorted(set(READS)),'development_only':all(p in allowed_paths() for p in READS),'evaluation_content_loaded':False,'binary_pipelines_loaded':False,'new_risk_models_fit':0,'new_classifiers_fit':0,'fixed_auxiliary_reconstruction_fits':len(audits)})
    verify();write(O/'analysis_complete.json',{'finished':True,'time_ns':time.time_ns(),'seconds':time.perf_counter()-start,'source_hashes':source_hashes()});print(json.dumps({'stage':'analysis_complete','seconds':time.perf_counter()-start}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','analyse','verify','status']);args=ap.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='analyse':analyse()
    elif args.command=='verify':print(json.dumps(verify()))
    else:print(json.dumps({'frozen':(O/'plan_freeze.json').exists(),'complete':read(O/'analysis_complete.json') if (O/'analysis_complete.json').exists() else None}))
