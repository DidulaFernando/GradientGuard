"""Guarded diagnostic CLI. Never calls a legacy command or fits any estimator."""
from core import *
import traceback,shutil,xml.etree.ElementTree as ET


def passed_tests():
    path=ATTEMPT/'logs/tests.xml'
    assert path.exists(),'Run diagnostic tests first'
    root=ET.parse(path).getroot();suites=list(root.iter('testsuite'))
    count=sum(int(s.get('tests','0')) for s in suites)
    assert count>0 and all(int(s.get('failures','0'))==int(s.get('errors','0'))==int(s.get('skipped','0'))==0 for s in suites)
    return count


def stage(name):
    guard();passed_tests()
    marker=ATTEMPT/'config'/f'completed_{name}.json'
    if marker.exists():
        m=read(marker);assert m['plan_sha256']==sha(ATTEMPT/'config/frozen_plan.json')
        assert all(sha(ATTEMPT/n)==h for n,h in m['outputs'].items()),'Completed diagnostic output changed'
        event('stage_verified',stage=name);return
    before={str(p.relative_to(ATTEMPT)):sha(p) for folder in ['tables','figures','reports'] for p in (ATTEMPT/folder).rglob('*') if p.is_file()}
    try:
        if name=='curves':
            from capacity import run_curves
            groups,evaluation,families,original=load_inputs();save_table('original_result_audit',original)
            run_curves(groups,evaluation,families)
        elif name=='slopes':
            from slopes import run_slopes
            groups,_,_,_=load_inputs();run_slopes(groups)
        elif name=='stability':
            from capacity import run_stability
            groups,_,families,_=load_inputs();run_stability(groups,families)
        elif name=='report':
            for n in ['curves','slopes','stability']:assert (ATTEMPT/'config'/f'completed_{n}.json').exists()
            from reporting import report
            report()
        else:raise ValueError(name)
    except Exception:
        stamp=str(time.time_ns());failure=ATTEMPT/'failures'/stamp;failure.mkdir(parents=True)
        (failure/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        for folder in ['tables','figures','reports']:
            for p in (ATTEMPT/folder).rglob('*'):
                if p.is_file() and before.get(str(p.relative_to(ATTEMPT)))!=sha(p):
                    dest=failure/p.relative_to(ATTEMPT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        event('stage_failed',stage=name,failure=str(failure));raise
    outputs={str(p.relative_to(ATTEMPT)):sha(p) for folder in ['tables','figures','reports'] for p in (ATTEMPT/folder).rglob('*') if p.is_file() and before.get(str(p.relative_to(ATTEMPT)))!=sha(p)}
    write(marker,{'plan_sha256':sha(ATTEMPT/'config/frozen_plan.json'),'outputs':outputs,'time_ns':time.time_ns()})


def verify():
    guard();count=passed_tests()
    for name in ['curves','slopes','stability','report']:
        m=read(ATTEMPT/'config'/f'completed_{name}.json')
        assert all(sha(ATTEMPT/n)==h for n,h in m['outputs'].items()),f'Output drift: {name}'
    protected=verify_prior('after')
    write(ATTEMPT/'reports/validation.json',{'passed':True,'tests':count,'protected':protected,
        'all_stages_complete':True,'original_predictions_and_pipelines_unchanged':True,
        'no_classifier_training':True,'no_forecaster_fitting':True,'source_frozen':True})
    event('diagnostics_complete',tests=count)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='freeze':freeze()
    elif cmd=='preflight':
        from environment import verify_environment
        verify_prior('preflight');verify_environment();print('Preflight passed')
    elif cmd=='run':
        for n in ['curves','slopes','stability','report']:stage(n)
        verify()
    elif cmd=='verify':verify()
    elif cmd=='status':
        print(json.dumps({'completed_stages':[n for n in ['curves','slopes','stability','report'] if (ATTEMPT/'config'/f'completed_{n}.json').exists()],
                          'latest':read(ATTEMPT/'status.json') if (ATTEMPT/'status.json').exists() else None},indent=2))
    else:stage(cmd)
