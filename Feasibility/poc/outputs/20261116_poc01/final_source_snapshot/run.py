import sys,os,json,time,hashlib,itertools,random,platform,subprocess,inspect
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parent
sys.path.insert(0,str(ROOT/'src'))
from common import torch,np,sha,write_json,save_torch,seed_all,deterministic,isolated,rng_state,restore_rng
from model import CNNA
from measurement import summarize,health
import pandas as pd
from torch.nn.functional import cross_entropy
O=P/'outputs/20261116_poc01'; O.mkdir(parents=True,exist_ok=True)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def event(name,**kw):
    r={'time_ns':time.time_ns(),'event':name,**kw}
    with (O/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
    write_json(O/'status.json',r); print(json.dumps(r),flush=True)
def protect(final=False):
    dirs=['src','tests','configs','runs','reports','checkpoints','data','followup']
    files={str(p.relative_to(ROOT)):sha(p) for d in dirs for p in (ROOT/d).rglob('*') if p.is_file()}
    proposal=Path('C:/Users/rangi/Documents/ChatGPT/Explanations/GradientGuard_Research_Proposal_Didula_Fernando_v5.pdf')
    files['proposal']=sha(proposal)
    if final:
        assert read(O/'protected_before.json')==files,'Prior artifacts changed'
        write_json(O/'protected_after.json',files)
    elif not (O/'protected_before.json').exists():write_json(O/'protected_before.json',files)
    else: assert read(O/'protected_before.json')==files
def freeze():
    protect()
    import importlib.metadata as md
    old=read(ROOT/'runs/20261116_01/provenance.json')
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    assert set(lock.splitlines())==set((ROOT/'runs/20261116_01/requirements-lock.txt').read_text().splitlines())
    assert torch.__version__=='2.7.1+cu128' and torch.cuda.get_device_name(0)==old['hardware']['gpu']
    source=read(ROOT/'followup/outputs/20261116_03/reports/decision.json')
    assert source['saved_partition_target']['target_summary']['passing_partitions']==15
    assert source['calibration_selected_full_epochs'] is None and source['fresh_stage_status']['completed_prefixes']==0
    families=[{'family':i,'alignment':a,'difficulty':d,'lr':lr} for i,(a,d,lr) in enumerate(itertools.product([.1,.5,.8,.95],['MAGNITUDE_LARGE','MAGNITUDE_MEDIUM'],[.0003,.001,.003]))]
    for attempt in range(1000):
        chosen=set(np.random.default_rng(81001+attempt).choice(24,14,replace=False).tolist())
        if all(set(f[k] for f in families if (f['family'] in chosen)==side)==set(f[k] for f in families) for k in ['alignment','difficulty','lr'] for side in [True,False]):break
    runs=[]
    for f in families:
        f['split']='development' if f['family'] in chosen else 'evaluation'
        for construction in range(2):
            for init in range(2):
                runs.append({**f,'construction':construction,'run':f"f{f['family']:02d}_c{construction}_s{init}",'data_seed':90000+f['family']*10+construction,'reservation_seed':100000+f['family']*10+construction,'training_seed':110000+f['family']*10+construction*2+init})
    protocol={'attempt':'20261116_poc01','horizon_epochs':60,'batch':128,'fractions':[.01,.02,.03,.05,.075,.1],
       'amendment':'User-authorised exploratory outcome study despite unresolved prior partition stability, aggregation and health calibration. 60 epochs is NOT a validated health solution. Prior gates remain failed.',
       'split_seed':81001+attempt,'outcome_seeds':{'val_large':120001,'val_medium':120002,'test_large':120003,'test_medium':120004},
       'failure_cutoffs':[.75,.70,.80],'health':{'accuracy':.8,'ce':'ln(10)'},'probe':256,'epsilon':1e-12,
       'families':families,'runs':runs,'learner_grid':{'C':[.1,1,10],'depth':[1,2,3],'iterations':100,'learning_rate':.05},
       'telemetry':'Panel mean CE, accuracy, minimum class accuracy, mean max probability, mean max probability on correct examples (missing if none), mean entropy; L2 norm of mean probe gradient; weight L2 norm/initial norm-1; mean coordinatewise abs(microbatch gradient mean)/(population SD+1e-12), 4096 layer-stratified fixed coordinates; saturation snapshot at1%: fraction of pre-ReLU activations <=0 over all three layers and probe images.',
       'separation':'Per class kmeans k=2,3 random_state=130001 n_init=10 max_iter=100; silhouette on all256 vectors, >=5 per cluster and score>0; max silhouette then smaller k. Class mean score, mean smallest cluster fraction, maximum cluster accuracy gap; unclustered score=0,fraction=1,gap=0.',
       'summaries':'last,mean,population SD,OLS slope versus actual full horizon fraction, excluding init. Rich loss/accuracy/GNS/growth: first,last,change,mean,SD,min,max,trapezoid area vs fraction,slope,nonzero increment sign reversals.',
       'selection':'Grouped OOF discovery at floor(.25N), AP tie-break, deterministic candidate order: compact before rich, no separation before separation, logistic before histogram, smaller grid value. 3 stratified-group folds if all train/validation folds contain both classes else2; no label support -> no classification fit.',
       'support':{'dev_min_each':8,'dev_families_each':3,'eval_n':20,'eval_families':6,'eval_min_each':5,'eval_outcome_families':3},
       'ridge_alpha':[.1,1,10],'bootstrap':{'replicates':2000,'seed':130005,'unit':'whole setting families; nested runs stay together; fixed frozen predictors; invalid class resamples counted'},
       'bias_control':'Shared last and slope of total bias norm and sensitivity added to both logistic E and L8 with same C grid.',
       'panel_B':'Frozen A pipelines on B; separately refit B using A-selected inputs/hyperparameters only, no B-based tuning.',
       'deferred':'Prediction-depth comparator absent; no claim beyond prediction depth or novelty.',
       'primary_ladder':'Logistic L8 vs selected E; B adds total4; L2 adds shares endpoints; L6 adds means/SD; L8 adds slopes; boosting ladder diagnostic; development strongest B/L2/L6 across both learners frozen.',
       'stage_order':['design freeze','development training and validation outcomes','development predictor freeze','evaluation training','early risk predictions freeze','one-time test unseal','predefined analyses']}
    if (O/'protocol.json').exists(): assert read(O/'protocol.json')==protocol
    else:write_json(O/'protocol.json',protocol)
    pd.DataFrame(runs).to_csv(O/'run_manifest.csv',index=False)
    write_json(O/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'os':platform.platform(),'versions':{n:md.version(n) for n in ['numpy','pandas','scikit-learn','torchvision','spuco']},'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'legacy_evidence':source})
    (O/'requirements-lock.txt').write_text(lock,encoding='utf8')
    source_hash={p.name:sha(p) for p in P.glob('*.py')}
    if not (O/'source_freeze.json').exists():write_json(O/'source_freeze.json',source_hash)
    else: assert read(O/'source_freeze.json')==source_hash,'PoC source changed'
    event('protocol_frozen',families=24,runs=96)

if __name__=='__main__':
    deterministic()
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1] in ['development','evaluation']:
        from study import execute
        execute(sys.argv[1])
    elif sys.argv[1]=='fit':
        from prediction import fit
        fit()
    elif sys.argv[1]=='predict':
        from prediction import predict
        predict()
    elif sys.argv[1]=='unseal':
        from study import outcomes
        f=read(O/'prediction_freeze.json');assert f['predictions_sha256']==sha(O/'evaluation_predictions.csv')
        event('heldout_outcome_unseal',predictor_hash=sha(O/'predictor_freeze.json'))
        for spec in read(O/'protocol.json')['runs']:
            if spec['split']=='evaluation' and (O/'classifiers'/spec['run']/'complete.json').exists():outcomes(spec,'test')
    elif sys.argv[1]=='report':
        from report import report
        report()
    elif sys.argv[1]=='status':
        print(json.dumps({'completed':len(list((O/'classifiers').glob('*/complete.json'))),'diverged':len(list((O/'classifiers').glob('*/diverged.json'))),'latest':read(O/'status.json')},indent=2))

