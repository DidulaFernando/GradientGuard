"""Recorded engineering continuation after official fixed-seed construction rejection.

No frozen numeric source, settings, seeds or assertions are modified. Unavailable
constructions remain unavailable and block readiness; other development work runs.
"""
from pathlib import Path
import sys,time,itertools,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent/'fresh_validation'))
from ggfresh.core import *
from ggfresh.training import train,evaluate
from ggfresh.measure_data import source,outcome_panel
from ggfresh.prediction import readiness
from ggfresh.reporting import report_stopped
from ggfresh.study import preserve

def finish():
    require();deterministic();assert not (O/'predictor_freeze.json').exists() and not (O/'outcomes/test').exists()
    plan=O/'continuation_plan.json';record={'time_ns':time.time_ns(),'source_sha256':sha(Path(__file__)),'original_protocol_sha256':sha(O/'protocol.json'),'scope':'Finish remaining feasible development groups only. No evaluation, seed substitution, generator patch or horizon/threshold changes. Record official No examples subgroup assertions separately and stop readiness with unlaunched models visible.'}
    if not plan.exists():write_json(plan,record)
    else:assert read(plan)['source_sha256']==record['source_sha256']
    failed=O/'preparation_failures';failed.mkdir(exist_ok=True)
    # Preserve the already observed failed construction without regenerating it.
    prior=next(s for s in protocol()['runs'] if s['family']==61 and s['construction']==1)
    marker=failed/'f61_c1.json'
    if not marker.exists():write_json(marker,{'family':61,'construction':1,'data_seed':prior['data_seed'],'reservation_seed':prior['reservation_seed'],'reason':'Official SpuCo assertion: No examples in (9, 1), considering reducing spurious correlation strength','source':'Original execution.log/failures.log; not retried or regenerated','affected_runs':[s['run'] for s in protocol()['runs'] if s['family']==61 and s['construction']==1]})
    selected=[s for s in protocol()['runs'] if s['split']=='development'];budget=GpuBudget();panels={}
    for key,group in itertools.groupby(selected,key=lambda s:(s['family'],s['construction'])):
        group=list(group);tag=f'f{key[0]:02d}_c{key[1]}'
        if (failed/(tag+'.json')).exists():continue
        todo=[s for s in group if not (O/'classifiers'/s['run']/'complete.json').exists() and not (O/'classifiers'/s['run']/'diverged.json').exists()]
        if not todo:continue
        try:d=source(group[0]);budget.tick()
        except AssertionError as error:
            if not str(error).startswith('No examples in '):raise
            write_json(failed/(tag+'.json'),{'family':key[0],'construction':key[1],'data_seed':group[0]['data_seed'],'reservation_seed':group[0]['reservation_seed'],'reason':str(error),'affected_runs':[s['run'] for s in group],'traceback':traceback.format_exc(),'configuration_unchanged':True})
            event('source_construction_unavailable',family=key[0],construction=key[1],reason=str(error));budget.tick();continue
        for spec in todo:
            train(spec,d,budget)
            if (O/'classifiers'/spec['run']/'complete.json').exists():
                if spec['difficulty'] not in panels:panels[spec['difficulty']]=outcome_panel(spec['difficulty'],'val')
                evaluate(spec,'val',panels[spec['difficulty']]);budget.tick()
        del d;torch.cuda.empty_cache()
    budget.tick();passed=readiness();assert not passed,'Unavailable planned constructions must block frozen all128terminal readiness'
    report_stopped('Official pinned SpuCo rejected fixed0.99-alignment construction(s) with an empty required source subgroup; all remaining feasible development work completed. No seeds or generator settings changed.')
    preserve()
    write_json(O/'execution_completion.json',{'finished':True,'status':'stopped_unavailable_construction_and_readiness_gate','time_ns':time.time_ns(),'continuation_source_sha256':sha(Path(__file__)),'fixed_protocol_sha256':sha(O/'protocol.json'),'evaluation_launched':False})
    event('fresh_continuation_complete',readiness_passed=False,preparation_failures=len(list(failed.glob('*.json'))))
if __name__=='__main__':finish()
