from pathlib import Path
import sys,time,traceback
from ggfresh.core import *
from ggfresh.study import preflight,freeze,preserve,status
from ggfresh.training import execute,unseal,preflight_evaluation
from ggfresh.prediction import readiness,fit,predict
from ggfresh.reporting import report_stopped,analyse,report_evaluated

def audit():
    require();p=preserve();rows=[]
    from ggfresh.reporting import accounting
    a=accounting()
    for spec in protocol()['runs']:
        f=O/'classifiers'/spec['run']
        if (f/'complete.json').exists():
            c=read(f/'complete.json');assert c['updates']==c['planned_updates']==19560 and sha(f/'final.pt')==c['final_sha256']
            df=pd.read_csv(f/'early.csv');plan=read(f/'plan.json')
            for panel in ['A','B']:assert df[df.panel==panel].step.tolist()==[0]+plan['steps']
            rows.append({'run':spec['run'],'valid_schedule':True,'complete_horizon':True})
    if (O/'unseal.json').exists():
        allow_test_outcomes();u=read(O/'unseal.json');pred=read(O/'prediction_freeze.json');fitted=read(O/'predictor_freeze.json')
        assert require()['time_ns']<fitted['time_ns']<pred['time_ns']<u['time_ns']
    if (O/'readiness.json').exists() and not read(O/'readiness.json')['passed']:
        assert not (O/'outcomes/test').exists() and not (O/'unseal.json').exists()
        assert not any(r['split']=='evaluation' and (O/'classifiers'/r['run']).exists() for r in protocol()['runs'])
    write_json(O/'validation_audit.json',{'passed':True,'completed_classifier_checks':len(rows),'preservation':p,'readiness_gate_enforced':True,'source_hashes_match':True,'checkpoint_records':rows})
    event('fresh_validation_audit_complete',completed=len(rows))
    manifest={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix not in ['.log','.pt']}
    write_json(O/'artifact_hashes.json',manifest)

def all_stages():
    freeze()
    if not readiness():report_stopped('Development source-normal support/CV gate did not qualify');audit();return
    fit();preflight_evaluation();execute('evaluation');predict();unseal();analyse();audit()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','freeze','development','readiness','fit','evaluation','predict','unseal','analyse','report','verify','audit','status','all']);args=ap.parse_args();deterministic();O.mkdir(parents=True,exist_ok=True)
    commands={'preflight':preflight,'freeze':freeze,'development':lambda:execute('development'),'readiness':readiness,'fit':fit,'evaluation':lambda:(preflight_evaluation(),execute('evaluation')),'predict':predict,'unseal':unseal,'analyse':analyse,'verify':preserve,'audit':audit,'status':status,'all':all_stages,'report':lambda:report_evaluated() if (O/'analysis_gate.json').exists() else report_stopped('Readiness or execution limitation; inspect frozen attempt records')}
    start=time.perf_counter()
    try:
        commands[args.command]()
        if args.command=='all':write_json(O/'execution_completion.json',{'finished':True,'time_ns':time.time_ns(),'wall_this_invocation':time.perf_counter()-start,'decision_path':str(BASE/'reports'/cfg()['attempt']/'decision.json')})
    except BudgetExceeded as error:
        report_stopped(str(error));preserve();write_json(O/'execution_limitation.json',{'reason':str(error),'type':'resource_cap','time_ns':time.time_ns()});raise
    except Exception:
        with (O/'failures.log').open('a',encoding='utf8') as f:f.write(traceback.format_exc()+'\n')
        raise
