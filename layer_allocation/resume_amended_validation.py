"""Resume frozen amended evidence without rewriting its preflight chronology.

No numerical code, model settings, seeds or predictor choices are changed.
Do not run concurrently with another training/evaluation process.
"""
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'amended_validation'))
from ggfresh.core import *
from ggfresh.study import preserve
from ggfresh.training import execute, unseal, preflight_evaluation
from ggfresh.prediction import predict
from ggfresh.reporting import analyse


def resume():
    start = time.perf_counter()
    deterministic()
    require()
    preserve()
    allow_evaluation_training()
    if (O / 'execution_completion.json').exists():
        assert read(O / 'execution_completion.json')['finished']
        print('Execution already completed. Use the final audit; no stage rerun.')
        return
    preflight = O / 'evaluation_preflight.json'
    if not preflight.exists():
        assert not any((O/'classifiers'/r['run']).exists()
                       for r in protocol()['runs'] if r['split']=='evaluation')
        preflight_evaluation()
    record = read(preflight)
    assert record['constructions'] == len(record['records']) == 64
    assert record['predictor_sha256'] == sha(O/'predictor_freeze.json')
    # Preserve the completed preflight's timestamp/hash; per-model recovery is
    # checked by the frozen training code before an optimiser step is taken.
    execute('evaluation')
    predict()
    unseal()
    analyse()
    preserve()
    write_json(O/'execution_completion.json', {
        'finished': True, 'time_ns': time.time_ns(),
        'wall_this_invocation': time.perf_counter()-start,
        'resumed_without_preflight_rewrite': True,
        'decision_path': str(BASE/'reports'/cfg()['attempt']/'decision.json')})
    print('Resumed execution completed. Run complete_amended_validation.py.')


if __name__ == '__main__':
    resume()
