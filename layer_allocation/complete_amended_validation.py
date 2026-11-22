"""Completed amended-study audit and explicit imported/excluded accounting."""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent/'amended_validation'))
from ggfresh.core import *
from ggfresh.study import preserve
import joblib

def complete():
    require();allow_test_outcomes();preservation=preserve()
    assert read(O/'execution_completion.json')['finished']
    f=read(O/'predictor_freeze.json');p=read(O/'prediction_freeze.json');u=read(O/'unseal.json');preflight=read(O/'evaluation_preflight.json')
    assert require()['time_ns']<f['time_ns']<preflight['time_ns']<p['time_ns']<u['time_ns']
    assert sha(O/'pipelines.joblib')==f['pipelines_sha256']
    assert sha(O/'evaluation_predictions.csv')==p['predictions_sha256']
    assert preflight['constructions']==64
    rows=[];hashes_out={}
    for s in protocol()['runs']:
        folder=O/'classifiers'/s['run'];r={k:s[k] for k in ['run','family','construction','split','alignment','difficulty','lr']}
        if (folder/'complete.json').exists():
            c=read(folder/'complete.json');assert c['updates']==c['planned_updates']==19560 and sha(folder/'final.pt')==c['final_sha256'];r['status']='complete'
            if s['split']=='evaluation':assert (folder/'plan.json').stat().st_mtime_ns>=preflight['time_ns']
            plan=read(folder/'plan.json');early=pd.read_csv(folder/'early.csv')
            for panel in ['A','B']:assert early[early.panel==panel].step.tolist()==[0]+plan['steps']
            hashes_out[str(folder.relative_to(O)/'final.pt')]=c['final_sha256']
        elif (O/'unavailable'/(s['run']+'.json')).exists():
            r['status']='data_preparation_failed';r['reason']=read(O/'unavailable'/(s['run']+'.json'))['reason'];assert not folder.exists()
        elif (folder/'diverged.json').exists():r['status']='numerical_divergence'
        else:raise AssertionError('Unaccounted planned run: '+s['run'])
        r['reused_development']=s['split']=='development' and r['status']=='complete';rows.append(r)
    accounting=pd.DataFrame(rows);accounting.to_csv(O/'run_status_expanded.csv',index=False)
    accounting.to_csv(O/'completion_accounting.csv',index=False)
    accounting.groupby(['split','alignment','difficulty','lr','status']).size().rename('models').reset_index().to_csv(O/'availability_by_configuration.csv',index=False)
    accounting.groupby(['split','family','status']).size().rename('models').reset_index().to_csv(O/'availability_by_family.csv',index=False)
    assert accounting[accounting.reused_development].shape[0]==124
    old=BASE/'runs/20261121_fresh01'
    for record in read(O/'development_import.json')['records']:
        a=O/'classifiers'/record['run'];b=old/'classifiers'/record['run']
        for name in ['initial.pt','early.csv','final.pt','complete.json','plan.json']:assert sha(a/name)==sha(b/name)
    pred=pd.read_csv(O/'evaluation_predictions.csv')
    for keys,g in pred.groupby(['cohort','prefix','model']):
        sets=[set(d.run) for _,d in g.groupby('variant')];assert len(sets)==3 and sets[0]==sets[1]==sets[2]
    models=joblib.load(O/'pipelines.joblib')
    lineage=[]
    for key,collection in models.items():
        for kind in ['logistic','histogram']:
            assert collection[kind+'_MZ'][2]==collection[kind+'_M'][2]+[b+'_share_'+s for b in ['shallow','deep'] for s in ['last','mean','std']]
            pa,pb,columns=collection[kind+'_MZ']
            for variant,pipeline in [('A',pa),('B_refit',pb)]:
                retained=pipeline.named_steps['variancethreshold'].get_support()
                assert len(retained)==len(columns)
                final_columns=[name for name,keep in zip(columns,retained) if keep]
                assert pipeline.steps[-1][1].n_features_in_==len(final_columns)
                for name in columns[-6:]:
                    lineage.append({'cohort_prefix':key,'learner':kind,'variant':variant,'feature':name,'retained':bool(retained[columns.index(name)]),'estimator_position':final_columns.index(name) if name in final_columns else None})
    pd.DataFrame(lineage).to_csv(O/'sensitivity_feature_lineage.csv',index=False)
    assert sha(O/'pipelines.joblib')==f['pipelines_sha256'],'Read-only inspection must not mutate frozen pipelines'
    curves=pd.read_csv(O/'discovery_curves.csv')
    for keys,g in curves.groupby(['cohort','prefix','variant','model']):
        g=g.sort_values('k');assert g.k.tolist()==list(range(int(g.N.iloc[0])+1))
        assert g.discoveries.iloc[-1]==g.F.iloc[0] and (g.discoveries.diff().dropna()>=0).all() and (g.discoveries<=g.oracle).all()
    tests=list(ET.parse(O/'tests.xml').getroot().iter('testsuite'));assert all(int(t.get('failures','0'))==int(t.get('errors','0'))==0 for t in tests)
    counts=accounting.groupby(['split','status']).size().to_dict();counts={str(k):int(v) for k,v in counts.items()}
    reportdir=BASE/'reports'/cfg()['attempt'];decision=read(reportdir/'decision.json');gate_record=read(O/'analysis_gate.json')
    decision.update(amended=True,development_reused=124,development_newly_trained=0,development_unavailable=4,construction_policy='Fixed exact official empty-source-subgroup rejections only; no seed replacement or assertion override',status_counts=counts,source_normal_core_evaluable=read(O/'evaluation_support.json')['passed'],conditional_on_available_pool=True)
    decision.update(completed=int((accounting.status=='complete').sum()),unavailable=int((accounting.status=='data_preparation_failed').sum()),diverged=int((accounting.status=='numerical_divergence').sum()),incomplete=0,not_launched=0,evaluation_newly_trained=int(((accounting.split=='evaluation')&(accounting.status=='complete')).sum()))
    write_json(reportdir/'decision.json',decision)
    report=reportdir/'REPORT.md';content=report.read_text(encoding='utf8').split('\n## Amendment and completed-output audit')[0]
    disclosure='Of 128 planned development models, 124 completed training; four were unavailable because two fixed 0.99-alignment source constructions failed the official generator group-presence validation. All completed development models were retained and reused, not retrained.'
    content+='\n## Amendment and completed-output audit\n'+disclosure+'\n\nStatus accounting: '+json.dumps(counts)+'. All64evaluation source constructions were preflighted before evaluation training. Unavailable evaluation identities were recorded without inspecting model performance. Missingness is concentrated in extreme-alignment settings and is not assumed random or harmless. Comparative results are conditional on successfully constructed candidates.\n\nAll imported development bytes checked unchanged; frozen protocol/source and '+str(preservation['protected_files'])+' prior-input hashes pass. Predictor freeze precedes evaluation preflight/training; risk freeze precedes one-time test unseal. Matched columns, panels, candidate pools, complete curves and full19,560-update horizons passed completed-output checks. Original source-normal support, grid, seeds, numerical environment, learner grids, capacities and manuscript criteria remain unchanged.\n\nThe24-hour conservative GPU-stage cap includes9401.188seconds from the original attempt. Current combined ledger: '+str(round(read(O/'gpu_budget.json')['charged_seconds']/3600,3))+'hours. No global packages or security settings changed. Source preparation failures are not subgroup-failure labels or numerical divergence.\n'
    report.write_text(content,encoding='utf8')
    write_json(O/'final_validation_audit.json',{'passed':True,'preservation':preservation,'status_counts':counts,'source_sha256':sha(Path(__file__)),'predictor_boundary':True,'candidate_panel_and_bank_checks':True,'development_import_unchanged':True,'manuscript_gate':gate_record})
    write_json(O/'checkpoint_hashes.json',hashes_out)
    scalar_manifest={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.suffix not in ['.log','.pt'] and p.name not in ['artifact_hashes.json','final_artifact_hashes.json']}
    write_json(O/'artifact_hashes.json',scalar_manifest)
    manifest={str(p.relative_to(BASE)):sha(p) for folder in [O,reportdir] for p in folder.rglob('*') if p.is_file() and p.suffix not in ['.log','.pt','.npz'] and p.name!='final_artifact_hashes.json'}
    write_json(O/'final_artifact_hashes.json',manifest)
    print(json.dumps({'passed':True,'status_counts':counts,'gate_passed':gate_record['passed'],'primary':gate_record['primary'],'interval':gate_record['interval']},indent=2))
if __name__=='__main__':complete()
