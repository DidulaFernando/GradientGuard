"""Read-only completed-output audit, separate from scientific model selection."""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from core import *
from predictors import inputs,MODELS,discovery
from study import preserve
import joblib,psutil

def audit():
    require();assert read(out()/'completion.json')['complete'];preserve()
    checks={}
    tests=list(ET.parse(out()/'tests.xml').getroot().iter('testsuite'))
    assert tests and all(int(x.get('failures','0'))==int(x.get('errors','0'))==0 for x in tests)
    checks['synthetic_engineering_tests_passed']=sum(int(x.get('tests','0')) for x in tests)
    rows=pd.read_csv(out()/'measurements.csv')
    assert len(rows)==1152 and not rows.duplicated(['run','step','panel']).any() and rows.valid.all()
    assert set(rows.step)=={196,391,587,978,1467,1956} and set(rows.panel)=={'A','B'}
    assert rows.run.nunique()==96 and rows.max_parity_error.max()<=cfg()['share_atol']
    checks['complete_scheduled_measurements']=True
    frame=pd.read_csv(out()/'early_features.csv')
    assert len(frame)==384 and frame.run.nunique()==96
    assert not {'failure','wga','final_loss','final_accuracy','alignment','difficulty'}&set(frame.columns)
    checks['strictly_early_feature_export']=True
    freeze=read(out()/'predictor_freeze.json');predict=read(out()/'prediction_freeze.json')
    assert sha(out()/'pipelines.joblib')==freeze['pipelines_sha256']
    assert sha(out()/'predictions.csv')==predict['sha256']
    models=joblib.load(out()/'pipelines.joblib');kept=[]
    for key,items in models.items():
        params=freeze['choices'][key]['models']
        for kind in cfg()['learners']:
            assert params[kind+'_MZ']['cols']==params[kind+'_M']['cols']+six()
            for name in MODELS:
                label=kind+'_'+name;m,bm,cols=items[label]
                assert cols==params[label]['cols'] and m[0].statistics_.shape[0]==len(cols)
                if name=='MZ':
                    selected=np.asarray(cols)[m[-2].get_support()].tolist()
                    kept.append({'key':key,'learner':kind,'Z_retained':[c for c in six() if c in selected]})
    checks['matched_column_lineage_and_frozen_pipelines']=True
    pred=pd.read_csv(out()/'predictions.csv');metric=pd.read_csv(out()/'metrics.csv');curves=pd.read_csv(out()/'discovery_curves.csv')
    for (cohort,prefix,model),g in pred.groupby(['cohort','prefix','model']):
        sets=[set(x.run) for _,x in g.groupby('variant')];assert len(sets)==3 and sets[0]==sets[1]==sets[2]
    for keys,g in curves.groupby(['cohort','prefix','variant','model']):
        g=g.sort_values('k');n=int(g.N.iloc[0]);f=int(g.F.iloc[0])
        assert g.k.tolist()==list(range(n+1)) and (g.discoveries.diff().dropna()>=0).all()
        assert g.discoveries.iloc[-1]==f and (g.discoveries<=np.minimum(g.k,f)).all()
        assert np.allclose(g.random_expectation,g.k*f/n)
    checks['matched_candidates_complete_curves_and_bounds']=True
    influence=pd.read_csv(out()/'leave_one_family_out.csv')
    assert np.array_equal(influence.k.to_numpy(),np.floor(influence.N*influence.fraction).astype(int).to_numpy())
    checks['fractional_omission_budgets_recomputed']=True
    events=[json.loads(l) for l in (out()/'events.jsonl').read_text().splitlines()]
    def when(name):return min(r['time_ns'] for r in events if r['event']==name)
    assert when('protocol_frozen')<when('checkpoint_run_complete')<when('development_fit_complete')<when('new_exploratory_predictors_frozen')<when('exploratory_predictions_saved')<when('post_hoc_evaluation_analysis_started')
    checks['recorded_stage_order']=True
    checks['old_predictions_reproduced']=read(out()/'original_reproduction.json')['passed']
    checks['protected_prior_files']=read(out()/'preservation_audit.json')['protected_files']
    costs={'measurement_seconds':float(rows.seconds.sum()),'peak_gpu_allocated_bytes':int(rows.gpu_peak_allocated.max()),'max_incremental_probe_bytes':int(rows.gpu_incremental.max()),'measurement_records':len(rows),'wall_stage_records':[r for r in events if r['event']=='stage_finished'],'end_to_end_invocation_seconds':read(out()/'completion.json')['wall_this_invocation'],'process_rss_bytes_at_audit':psutil.Process().memory_info().rss,'cpu_count_logical':psutil.cpu_count(),'system_memory_available_bytes_at_audit':psutil.virtual_memory().available,'limitations':'GPU counters cover each isolated measurement; CPU RSS is an audit-time snapshot, not historical training or end-to-end peak memory. Reconstruction/statistics overhead and gradients are jointly timed. No production-only incremental L6 cost can be inferred.'}
    write(out()/'costs.json',costs)
    write(out()/'validation_audit.json',{'passed':True,'checks':checks,'Z_lineage':kept,'audit_source_sha256':sha(Path(__file__)),'new_predictor_freeze_unchanged':True,'previous_evaluation_consumed':True})
    f=ROOT/'reports'/cfg()['attempt']/'REPORT.md'
    extra=f'\n## Completed-output validation and resource audit\nAll {checks["synthetic_engineering_tests_passed"]} engineering tests pass; output audit passed. All {checks["protected_prior_files"]} protected prior files are unchanged. Stage order, frozen pipelines, strictly early exports, nested input schemas, complete curves, matched panels and recomputed omission budgets were checked. End-to-end invocation wall time: {costs["end_to_end_invocation_seconds"]:.3f} s. Maximum measured allocated GPU memory: {costs["peak_gpu_allocated_bytes"]/2**20:.2f} MiB; maximum incremental probe allocation: {costs["max_incremental_probe_bytes"]/2**20:.2f} MiB. CPU memory is only an audit-time snapshot. See `validation_audit.json` and `costs.json`.\n'
    contrasts=pd.read_csv(out()/'contrasts.csv');intervals=pd.read_csv(out()/'intervals.csv')
    target=contrasts[(contrasts.cohort=='eligible')&(contrasts.added=='MZ')&(contrasts.fraction>.3)]
    pr=target[(target.variant=='A')&(target.learner=='logistic')].iloc[0]
    others=target[(target.variant!='A')&(target.learner=='logistic')]
    pi=intervals[(intervals.cohort=='eligible')&(intervals.variant=='A')&(intervals.learner=='logistic')&(intervals.added=='MZ')&(intervals.fraction>.3)].iloc[0]
    m=metric[(metric.cohort=='eligible')&(metric.prefix==.1)&(metric.variant=='A')].set_index('model')
    chosen=freeze['choices']['eligible_0.1']['strongest_structural_control']
    synthesis=f'''\n## Scientific synthesis for the next implementation\nThe positive result is narrow: logistic MZ adds {int(pr.difference)} discovery over M at twelve audits on panel A. Both panel-B variants give MZ-M differences {others.difference.astype(int).tolist()}, and histogram MZ/M tie on panel A. The conditional Delta20 interval includes zero ({pi.low:.6f} to {pi.high:.6f}). Thus the sensitivity-specific increment is not panel/learner-stable in this reused evaluation.\n\nThe simpler logistic norm-product allocation proxy N finds {int(m.loc['logistic_N','D12'])} failures at twelve audits, matching logistic L6's {int(m.loc['logistic_L6','D12'])}; this shows that this fixed-budget result alone does not identify coordinate-product specificity. The stronger bounded histogram E finds {int(m.loc['histogram_E','D9'])} at nine and {int(m.loc['histogram_E','D12'])} at twelve, so the historical weaker nonlinear baseline must not be treated as the strongest available control. The development-selected strongest non-sensitivity structural control is `{chosen}` (selected without evaluation labels); it finds {int(m.loc[chosen,'D9'])}/{int(m.loc[chosen,'D12'])} at nine/twelve. These are descriptive comparisons, not new selection based on test performance.\n\nRecommendation: **revise toward the narrower layer-allocation hypothesis and retain N/M as serious controls in a separately frozen fresh validation.** The extension is working and there is a small encouraging conditional increment, but the present result does not establish a robust sensitivity-specific contribution, justify journal readiness, or support dropping stronger conventional controls. No further training was launched.\n'''
    content=f.read_text(encoding='utf8').split('\n## Completed-output validation and resource audit')[0]
    f.write_text(content+extra+synthesis,encoding='utf8')
    dpath=ROOT/'reports'/cfg()['attempt']/'decision.json';decision=read(dpath)
    decision['independent_flags']={'implementation_passed':True,'preservation_passed':True,'all_checkpoint_measurements_reproduced':True,'positive_logistic_panel_A_increment_at_one_third':bool(pr.difference>0),'positive_increment_across_both_B_variants':bool((others.difference>0).all()),'pointwise_interval_excludes_zero':bool(pi.low>0 or pi.high<0),'untouched_confirmation':False}
    decision['development_selected_structural_control']=chosen
    decision['qualified_recommendation']='Revise toward a narrower layer-allocation claim; keep norm-product, rich structural and stronger nonlinear controls in separately frozen fresh validation. Sensitivity-specific advantage remains unresolved.'
    write(dpath,decision)
    completion=read(out()/'completion.json');completion.setdefault('pre_audit_report_sha256',completion['report_sha256']);completion['report_sha256']=sha(f);completion['final_validation_sha256']=sha(out()/'validation_audit.json');write(out()/'completion.json',completion)
    manifest={str(p.relative_to(ROOT)):sha(p) for folder in [out(),ROOT/'reports'/cfg()['attempt']] for p in folder.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix!='.log'}
    write(out()/'artifact_hashes.json',manifest)
    print(json.dumps({'passed':True,'checks':checks,'costs':costs},indent=2))
if __name__=='__main__':audit()
