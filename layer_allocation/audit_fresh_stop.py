"""Final read-only/data-accounting audit of the stopped fresh-family attempt."""
from pathlib import Path
import sys,json,time,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent/'fresh_validation'))
from ggfresh.core import *
from ggfresh.study import preserve
import psutil

def finalise():
    require();assert sha(O/'protocol.json')==require()['protocol_sha256']
    assert read(O/'execution_completion.json')['finished']
    assert not read(O/'readiness.json')['passed']
    preservation=preserve();specs=protocol()['runs'];failures=[read(p) for p in (O/'preparation_failures').glob('*.json')]
    unavailable={run:f for f in failures for run in f['affected_runs']}
    rows=[];checkpoint_hashes={};early=[]
    for s in specs:
        folder=O/'classifiers'/s['run'];row={k:s[k] for k in ['run','family','construction','split','data_seed','reservation_seed','training_seed']}
        if (folder/'complete.json').exists():
            c=read(folder/'complete.json');plan=read(folder/'plan.json')
            assert c['updates']==c['planned_updates']==19560 and c['full_60_epochs'] and sha(folder/'final.pt')==c['final_sha256']
            assert plan['steps']==[196,391,587,978,1467,1956]
            table=pd.read_csv(folder/'early.csv')
            for panel in ['A','B']:assert table[table.panel==panel].step.tolist()==[0]+plan['steps']
            early.append(table.assign(run=s['run']))
            row.update(status='complete',updates=c['updates'])
            for p in folder.glob('*.pt'):checkpoint_hashes[str(p.relative_to(O))]=sha(p)
            outcome=read(O/'outcomes/val'/(s['run']+'.json'));assert len(outcome['groups'])==100 and all(g['n']>0 for g in outcome['groups'])
        elif (folder/'diverged.json').exists():row['status']='numerical_divergence'
        elif s['run'] in unavailable:
            f=unavailable[s['run']];assert f['data_seed']==s['data_seed'] and f['reservation_seed']==s['reservation_seed'];assert not folder.exists()
            row.update(status='data_preparation_failed',reason=f['reason'])
        elif folder.exists():row['status']='interrupted_training'
        else:row['status']='not_launched'
        rows.append(row)
    frame=pd.DataFrame(rows);assert len(frame)==256
    assert not (O/'outcomes/test').exists() and not (O/'unseal.json').exists() and not (O/'predictor_freeze.json').exists() and not (O/'prediction_freeze.json').exists() and not (O/'pipelines.joblib').exists()
    assert (frame[frame.split=='evaluation'].status=='not_launched').all()
    assert not (frame[frame.split=='development'].status.isin(['interrupted_training','not_launched'])).any(),'Other feasible development work remains'
    for p in (O/'data').glob('*/verification.json'):
        v=read(p);compact=p.parent/'compact.npz';assert sha(compact)==v['sha256'] and v['foreground_and_background_error']==0
        with np.load(compact) as d:
            assert len(set(d['ids']))==48004
            sets=[set(d['ids'][d['roles']==r]) for r in range(4)]
            assert all(not sets[a]&sets[b] for a in range(4) for b in range(a))
            for role,n in [(1,256),(2,256),(3,128)]:assert np.bincount(d['y'][d['roles']==role],minlength=10).tolist()==[n]*10
    xml=list(ET.parse(O/'tests.xml').getroot().iter('testsuite'));assert xml and all(int(x.get('failures','0'))==int(x.get('errors','0'))==0 for x in xml)
    tests=sum(int(x.get('tests','0')) for x in xml);parity=read(O/'real_checkpoint_parity.json');assert parity['passed']
    ready=read(O/'readiness.json');counts_support=all(v['models']>=20 and v['families']>=8 for v in ready['outcomes'].values());counts=frame.status.value_counts().to_dict()
    devcohort=pd.read_csv(O/'development_readiness_cohort.csv');mixed=int((devcohort.groupby('family').failure.nunique()>1).sum())
    frame.to_csv(O/'run_status_expanded.csv',index=False);pd.DataFrame(failures).drop(columns=['traceback'],errors='ignore').to_csv(O/'preparation_failure_summary.csv',index=False);write_json(O/'checkpoint_hashes.json',checkpoint_hashes)
    telemetry=pd.concat(early,ignore_index=True);monitor=telemetry[telemetry.step>0]
    costs=read(O/'costs.json');costs.update({'freeze_to_completion_wall_seconds':(read(O/'execution_completion.json')['time_ns']-require()['time_ns'])/1e9,'auxiliary_real_checkpoint_audit_seconds':parity['seconds'],'peak_observation_gpu_allocated_bytes':int(telemetry.gpu_peak_allocated.max()),'max_probe_incremental_bytes':int(telemetry.gpu_incremental.max()),'summary_and_metadata_bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file() and p.suffix not in ['.pt','.npz']),'audit_process_rss_bytes':psutil.Process().memory_info().rss,'initial_failed_generation_cost_note':'Original rejected construction overhead before exception was not separately charged; exact total preparation/exception overhead is not isolated. GPU-stage ledger remains far below24h; auxiliary parity cost recorded separately. CPU RSS is a snapshot, not a historical peak.'})
    write_json(O/'costs.json',costs)
    audit={'passed':True,'implementation_tests_passed':tests,'frozen_protocol_hash_matches':True,'frozen_source_hashes_match':True,'preservation':preservation,'run_status_counts':counts,'all_other_feasible_development_completed':True,'unavailable_constructions':len(failures),'scheduled_noninitial_panel_measurements':len(monitor),'invalid_measurement_records':int((~monitor.valid).sum()),'real_checkpoint_parity':parity,'evaluation_boundary_intact':True,'forecasters_fitted':False,'audit_source_sha256':sha(Path(__file__))}
    write_json(O/'final_validation_audit.json',audit)
    reportfolder=BASE/'reports'/cfg()['attempt'];decision=read(reportfolder/'decision.json')
    decision.update({'detailed_status_counts':counts,'data_preparation_failed_models':int(counts.get('data_preparation_failed',0)),'unavailable_constructions':len(failures),'evaluation_models_not_launched':128,'interrupted_training':int(counts.get('interrupted_training',0)),'development_mixed_outcome_families':mixed,'source_normal_core_evaluable':False,'comparison_results_available':False,'comparisons':{k:'Not evaluated; development execution/readiness gate failed' for k in ['L6_vs_E','L6_vs_B','NZ_vs_N','MZ_vs_M','MZ_vs_selected_nonZ']},'independent_flags':{'implementation_checks_passed':True,'preservation_passed':True,'all_source_constructions_available':False,'development_count_family_minima_passed':bool(counts_support),'valid_grouped_folds':bool(ready['valid_grouped_folds']),'all128development_terminal':False,'evaluation_boundary_intact':True,'fresh_outcome_comparison_available':False,'manuscript_update_gate_passed':False},'qualified_conclusion':'Prospective outcome question not evaluated. Fixed-seed official source construction failures blocked complete development execution. The count/family support minima and grouped folds passed. This is not evidence that GradientGuard succeeds or fails.','recommendation':'Before a separately frozen amendment, preflight every planned source construction and agree a justified policy for naturally empty training subgroups at0.99. Do not replace seeds or bypass generator checks within this attempt. Preserve these development trajectories; no automatic new study or v1.3.'})
    write_json(reportfolder/'decision.json',decision)
    report=reportfolder/'REPORT.md';content=report.read_text(encoding='utf8').split('\n## Final execution and construction-failure audit')[0]
    failure_lines=['| Family | Construction | Data seed | Missing source group / reason | Affected models |','|---|---|---:|---|---:|']
    for f in failures:failure_lines.append(f"| {f['family']} | {f['construction']} | {f['data_seed']} | {f['reason']} | {len(f['affected_runs'])} |")
    extra=f'''\n## Final execution and construction-failure audit
All {tests} pre-training checks and the actual-checkpoint parity audit passed. Frozen protocol/source hashes match; {preservation['protected_files']} prior artifacts remain unchanged. Complete status accounting: {json.dumps(counts)}. No interrupted training remains. The {counts.get('data_preparation_failed',0)} unavailable model identities are **data-preparation failures**, not numerical divergences or terminal subgroup-failure labels. Every other feasible development trajectory completed60epochs/19,560updates. All128evaluation trajectories remain unlaunched.

{chr(10).join(failure_lines)}

The pinned official generator validates nonempty class/background source groups and rejected these fixed0.99constructions. Its dataset initialization stopped before a verified training allocation could be accepted. No alignment, seed, palette, sample count, label, noise level or validation assertion was changed. Successful construction verifications retain zero foreground/background reconstruction error and disjoint original-ID reservations. The continuation used the unchanged frozen training/measurement functions and recorded its own scope/hash before running the other feasible development constructions. The initial failed invocation and traces remain preserved.

Development source-normal subset: {ready['N']} models in {ready['families']} families; outcomes {json.dumps(ready['outcomes'])}. There are {mixed} mixed-outcome source-normal development families. Count/family minima alone passed={counts_support}; grouped-fold validity={ready['valid_grouped_folds']}; all128terminal=False. Overall frozen readiness=False. The development support result is encouraging for evaluability, but it is not a predictive improvement. These independent flags must not be reduced to a method-performance verdict. No forecaster fitting, held-out audit discoveries, AP, AUROC, panel comparisons or family-gain intervals were executed or invented.

There are {len(monitor)} retained scheduled noninitial panel records. Invalid records: {audit['invalid_measurement_records']}. Real-checkpoint recorded-scalar parity error: {parity['recorded_measurement_max_error']:.9g}; unequal-microbatch maximum gradient error: {parity['unequal_microbatch_max_gradient_error']:.9g}, within frozen tolerances. Maximum recorded per-observation GPU allocation: {costs['peak_observation_gpu_allocated_bytes']/2**20:.2f}MiB; maximum incremental probe allocation: {costs['max_probe_incremental_bytes']/2**20:.2f}MiB. Conservative GPU-stage ledger: {costs['conservative_gpu_budget']['charged_seconds']/3600:.3f}h of24h; freeze-to-completion wall: {costs['freeze_to_completion_wall_seconds']/3600:.3f}h. Failed-preparation overhead was not separately isolated, and CPU memory is only an audit snapshot. See classifier_costs.csv/costs.json for recorded component sums.

**Conclusion:** the intended fresh outcome comparison is not adequately evaluable under this attempt. The stop is not a negative predictive result for L6/MZ, nor positive evidence for them. A lesson is to preflight every fixed source construction before expensive classifier training; mask/identity tests alone did not reveal this seed-dependent group-coverage constraint. A separately identified, prospectively justified source-construction amendment is required before further outcome work. No seed replacement, assertion bypass, new benchmark, extra families or manuscript v1.3 was launched. Version1.2 remains unchanged.
'''
    report.write_text(content+extra,encoding='utf8');write_json(reportfolder/'MANUSCRIPT_RECOMMENDATION.json',{'create_v1_3':False,'reason':'No held-out comparison; fixed construction/readiness gate blocked evaluation','user_manual_revision_required_if_desired':True})
    record=read(O/'execution_completion.json');record['final_report_sha256']=sha(report);record['final_decision_sha256']=sha(reportfolder/'decision.json');record['final_audit_sha256']=sha(O/'final_validation_audit.json');write_json(O/'execution_completion.json',record)
    manifest={str(p.relative_to(BASE)):sha(p) for folder in [O,reportfolder] for p in folder.rglob('*') if p.is_file() and p.name!='final_artifact_hashes.json' and p.suffix not in ['.log','.pt']}
    write_json(O/'final_artifact_hashes.json',manifest)
    print(json.dumps({'passed':True,'status_counts':counts,'readiness':ready,'protected_files_unchanged':preservation['protected_files'],'manuscript_updated':False},indent=2))
if __name__=='__main__':finalise()
