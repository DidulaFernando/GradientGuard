"""Independent scalar/output audit; standard library only, no model loading/fitting."""
from pathlib import Path
import csv,json,math,hashlib,itertools,statistics,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
HOME=BASE/'development_diagnostics'
O=BASE/'runs/20261122_devdiag01'
R=BASE/'reports/20261122_devdiag01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(name):
    with (O/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def key(r,fields):return tuple(r[f] for f in fields)
def audit():
    freeze=read(O/'plan_freeze.json');complete=read(O/'analysis_complete.json')
    assert freeze['time_ns']<complete['time_ns'] and complete['finished']
    for p,h in freeze['protected_inputs'].items():assert sha(p)==h,p
    for p,h in freeze['inputs'].items():assert sha(p)==h,p
    for p,h in freeze['source_hashes'].items():assert sha(HOME/p)==h,p
    assert sha(O/'plan.json')==freeze['plan_sha256']
    suites=list(ET.parse(O/'tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)==17
    assert all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 for s in suites)
    access=read(O/'semantic_read_audit.json')
    assert access['development_only'] and not access['evaluation_content_loaded'] and not access['binary_pipelines_loaded']
    assert not access['new_classifiers_fit'] and not access['new_risk_models_fit'] and access['fixed_auxiliary_reconstruction_fits']==60
    assert set(access['unique_paths'])<=set(freeze['inputs'])
    folds=rows('fold_membership.csv')
    for k,g in itertools.groupby(sorted(folds,key=lambda r:key(r,['cohort','fold'])),key=lambda r:key(r,['cohort','fold'])):
        g=list(g);train={r['family'] for r in g if r['role']=='train'};test={r['family'] for r in g if r['role']=='validation'};assert not train&test
    predicted=rows('reconstruction_oof.csv');scores=rows('reconstruction_scores.csv')
    assert len(predicted)==(107+124)*2*3*6 and len(scores)==72
    fields=['cohort','panel','bank','feature'];indexed={key(r,fields):r for r in scores}
    for k,g in itertools.groupby(sorted(predicted,key=lambda r:key(r,fields)),key=lambda r:key(r,fields)):
        g=list(g);y=[float(r['observed']) for r in g];mu=statistics.mean(y)
        sse=math.fsum((float(r['observed'])-float(r['predicted']))**2 for r in g)
        total=math.fsum((v-mu)**2 for v in y)
        null=math.fsum((float(r['observed'])-float(r['train_reference']))**2 for r in g)
        close(indexed[k]['R2'],1-sse/total);close(indexed[k]['RMSE'],math.sqrt(sse/len(g)))
        close(indexed[k]['skill_vs_fold_training_mean'],1-sse/null)
    pairs=rows('within_family_pairs.csv');within=rows('within_family_feature_concordance.csv')
    fields=['cohort','panel','representation','feature','family'];indexed={key(r,fields):r for r in within}
    for k,g in itertools.groupby(sorted(pairs,key=lambda r:key(r,fields)),key=lambda r:key(r,fields)):
        g=list(g);value=statistics.mean(1 if float(r['difference'])>0 else .5 if float(r['difference'])==0 else 0 for r in g);close(value,indexed[k]['concordance'])
    stats=rows('within_family_summary.csv');assert len(stats)==48
    for s in stats:
        group=[float(r['concordance']) for r in within if key(r,fields[:4])==key(s,fields[:4])]
        close(s['concordance'],statistics.mean(group))
        assert 0<=float(s['two_sided_conditional_p'])<=float(s['max24_adjusted_p'])<=1
    supports=rows('family_support.csv')
    for record in read(O/'conditional_permutation.json'):
        fs=[r for r in supports if r['cohort']==record['cohort'] and r['mixed']=='True']
        count=math.prod(math.comb(int(r['models']),int(r['failures'])) for r in fs)
        assert count==record['assignments']==record['draws'] and record['mode']=='exact'
    result={'passed':True,'tests':17,'prior_files_unchanged':len(freeze['protected_inputs']),'inputs_unchanged':len(freeze['inputs']),'reconstruction_scores_recomputed':72,'paired_summary_rows_recomputed':len(stats),'heldout_semantic_access':False,'new_classifiers':0,'new_failure_forecasters':0,'fixed_auxiliary_fits':60,'source_sha256':sha(Path(__file__)),'numerical_tolerance':{'atol':1e-10,'rtol':1e-10}}
    (O/'final_output_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    manifest={str(p.relative_to(BASE)):sha(p) for folder in [O,R] for p in folder.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix!='.log'}
    (O/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    assert all(sha(BASE/p)==h for p,h in manifest.items())
    print(json.dumps(result,indent=2))
if __name__=='__main__':audit()
