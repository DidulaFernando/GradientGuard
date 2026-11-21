"""Fail-closed stage launcher; classifier code and protocol remain immutable."""
from run import *
if __name__=='__main__':
    command=sys.argv[1]
    if command=='freeze' and (O/'protocol.json').exists():
        raise SystemExit('This attempt is already frozen. Use poc/verify.py; do not rewrite frozen records.')
    if command=='predict' and list((O/'outcomes/test').glob('*.json')):
        raise SystemExit('Held-out outcomes are already unsealed. Verify the original prediction freeze; do not regenerate it.')
    if command!='freeze':
        frozen=read(O/'source_freeze.json')
        for filename in ['run.py','study.py']:assert sha(P/filename)==frozen[filename],f'Classifier source drift: {filename}'
        if command=='fit':assert not list((O/'outcomes/test').glob('*.json')),'Outcome access prevents refitting'
        if command in ['evaluation','predict','unseal']:
            f=read(O/'predictor_freeze.json');assert f['protocol_sha256']==sha(O/'protocol.json') and f['pipelines_sha256']==sha(O/'pipelines.joblib')
        if command in ['predict','unseal']:
            for spec in read(O/'protocol.json')['runs']:
                if spec['split']=='evaluation':assert (O/'classifiers'/spec['run']/'complete.json').exists() or (O/'classifiers'/spec['run']/'diverged.json').exists(),'Evaluation classifier unfinished'
        if command=='unseal':
            f=read(O/'prediction_freeze.json');assert f['predictor_freeze_sha256']==sha(O/'predictor_freeze.json') and f['predictions_sha256']==sha(O/'evaluation_predictions.csv')
    raise SystemExit(subprocess.call([sys.executable,str(P/'run.py'),command]))
