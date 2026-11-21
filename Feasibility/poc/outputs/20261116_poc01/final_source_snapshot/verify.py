"""Read-only scientific-artifact preflight; never rewrites any frozen record."""
from run import *


def main():
    protect()
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    assert set(lock.splitlines())==set((O/'requirements-lock.txt').read_text().splitlines())
    frozen=read(O/'source_freeze.json')
    assert all(sha(P/name)==frozen[name] for name in ['run.py','study.py'])
    manifest=read(O/'protocol.json')['runs']
    assert len(manifest)==96 and len({r['run'] for r in manifest})==96
    dev={r['family'] for r in manifest if r['split']=='development'}
    test={r['family'] for r in manifest if r['split']=='evaluation'}
    assert len(dev)==14 and len(test)==10 and not dev&test
    if (O/'predictor_freeze.json').exists():
        f=read(O/'predictor_freeze.json')
        assert sha(O/'protocol.json')==f['protocol_sha256']
        assert sha(O/'pipelines.joblib')==f['pipelines_sha256']
    if (O/'prediction_freeze.json').exists():
        f=read(O/'prediction_freeze.json')
        assert sha(O/'predictor_freeze.json')==f['predictor_freeze_sha256']
        assert sha(O/'evaluation_predictions.csv')==f['predictions_sha256']
    if (O/'reports/artifact_hashes.json').exists():
        for name,expected in read(O/'reports/artifact_hashes.json').items():
            assert sha(ROOT/name)==expected, f'Artifact drift: {name}'
    result={'passed':True,'environment_lock_matches':True,'legacy_and_proposal_preserved':True,
            'frozen_classifier_source_matches':True,'families':24,'planned_classifiers':96,
            'completed':len(list((O/'classifiers').glob('*/complete.json'))),
            'diverged':len(list((O/'classifiers').glob('*/diverged.json'))),
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
