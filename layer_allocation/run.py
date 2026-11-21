"""Explicit stage entry point, with no legacy command imports."""
from pathlib import Path
import sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from core import *
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','freeze','extract','features','reproduce','fit','predict','analyse','report','verify','status','all']);a=parser.parse_args()
    from study import environment,freeze,preserve
    from structural_measurement import extract
    from predictors import features,reproduce,fit,predict
    from analysis_report import analyse,render_report
    deterministic();out().mkdir(parents=True,exist_ok=True)
    actions={'preflight':environment,'freeze':freeze,'extract':extract,'features':features,'reproduce':reproduce,'fit':fit,'predict':predict,'analyse':analyse,'report':render_report,'verify':preserve,'status':lambda:print(json.dumps({'status':read(out()/'status.json') if (out()/'status.json').exists() else None,'measurements':len(list((out()/'measurements').glob('*.json'))),'predictors_frozen':(out()/'predictor_freeze.json').exists(),'report':(ROOT/'reports'/cfg()['attempt']/'REPORT.md').exists()},indent=2))}
    start=time.perf_counter()
    try:
        stages=['freeze','extract','features','reproduce','fit','predict','analyse','verify'] if a.command=='all' else [a.command]
        for stage in stages:
            if a.command=='all' and stage=='fit' and (out()/'predictor_freeze.json').exists():continue
            before=time.perf_counter();actions[stage]()
            if stage not in ['status','preflight']:event('stage_finished',stage=stage,seconds=time.perf_counter()-before)
        if a.command=='all':
            write(out()/'completion.json',{'complete':True,'end_time_ns':time.time_ns(),'wall_this_invocation':time.perf_counter()-start,'classifiers_trained':0,'protocol_sha256':sha(out()/'protocol_freeze.json'),'predictor_sha256':sha(out()/'predictor_freeze.json'),'prediction_sha256':sha(out()/'prediction_freeze.json'),'report_sha256':sha(ROOT/'reports'/cfg()['attempt']/'REPORT.md'),'new_artifact_bytes':sum(p.stat().st_size for p in out().rglob('*') if p.is_file())})
            event('study_complete',wall_seconds=time.perf_counter()-start)
    except Exception:
        with (out()/'failures.log').open('a',encoding='utf8') as f:f.write(traceback.format_exc()+'\n')
        raise
