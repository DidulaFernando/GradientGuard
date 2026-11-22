from pathlib import Path
import os,sys,json,time,hashlib,random,platform,subprocess,itertools,contextlib
HOME=Path(__file__).resolve().parents[1]
BASE=HOME.parent
REPO=BASE.parent
LEGACY=REPO/'Feasibility'
OLD=LEGACY/'poc/outputs/20261116_poc01'
CFG=HOME/'configs/fresh.json'
ROOT=LEGACY  # Read-only cached raw MNIST for adapted official-generator helper.
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cfg():return read(CFG)
O=BASE/'runs'/cfg()['attempt']
def settings():
    for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'cache/pip','UV_CACHE_DIR':'cache/uv','MPLCONFIGDIR':'cache/matplotlib','PYTHONPYCACHEPREFIX':'cache/pycache','TORCH_HOME':'cache/torch','XDG_CACHE_HOME':'cache/xdg','CUDA_CACHE_PATH':'cache/cuda'}.items():
        p=BASE/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
settings();sys.path.insert(0,str(LEGACY/'src'))
from common import torch,np,seed_all,deterministic,isolated,rng_state,restore_rng,sync,save_torch
from model import CNNA
import importlib.util
sp=importlib.util.spec_from_file_location('_fresh_legacy_measurement',LEGACY/'src/measurement.py')
legacy_measurement=importlib.util.module_from_spec(sp);sp.loader.exec_module(legacy_measurement)
summarize=legacy_measurement.summarize
mean_gradients=legacy_measurement.mean_gradients
health=legacy_measurement.health
settings()
import pandas as pd
from torch.nn.functional import cross_entropy
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf8');t.replace(p)
def event(name,**kw):
    O.mkdir(parents=True,exist_ok=True);r={'time_ns':time.time_ns(),'event':name,**kw}
    with (O/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
    write_json(O/'status.json',r);print(json.dumps(r),flush=True)
def hashes():
    return {str(p.relative_to(HOME)):sha(p) for p in sorted(HOME.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.ps1'] and not {'__pycache__','cache','tmp'}&set(p.parts)}
def require():
    f=read(O/'protocol_freeze.json');assert f['source_hashes']==hashes(),'Fresh source changed after freeze; preserve attempt and create a new one'
    assert sha(O/'protocol.json')==f['protocol_sha256'],'Protocol content changed'
    return f
def protocol():require();return read(O/'protocol.json')
def allow_evaluation_training():
    require();assert read(O/'readiness.json')['passed']
    f=read(O/'predictor_freeze.json');assert sha(O/'pipelines.joblib')==f['pipelines_sha256']
def allow_test_outcomes():
    allow_evaluation_training();f=read(O/'prediction_freeze.json')
    assert f['predictor_freeze_sha256']==sha(O/'predictor_freeze.json') and f['predictions_sha256']==sha(O/'evaluation_predictions.csv')
def family_design():
    c=cfg();families=[{'family':i,'alignment':a,'difficulty':d,'lr':lr} for i,(a,d,lr) in enumerate(itertools.product(c['alignments'],c['difficulties'],c['learning_rates']))]
    for seed in range(c['split_seed'],c['split_seed']+10000):
        chosen=set(np.random.default_rng(seed).choice(len(families),c['development_families'],replace=False).tolist())
        if all({f[k] for f in families if (f['family'] in chosen)==side}=={f[k] for f in families} for k in ['alignment','difficulty','lr'] for side in [True,False]):break
    else:raise ValueError('Could not construct balanced factor-coverage split')
    runs=[]
    for f in families:
        f['split']='development' if f['family'] in chosen else 'evaluation'
        for construction in range(c['construction_count']):
            for init in range(c['initialisations']):
                runs.append(f|{'construction':construction,'init':init,'run':f"fresh_f{f['family']:02d}_c{construction}_s{init}",'data_seed':c['data_seed_base']+f['family']*10+construction,'reservation_seed':c['reservation_seed_base']+f['family']*10+construction,'training_seed':c['training_seed_base']+f['family']*10+construction*2+init})
    return families,runs,seed
def gate(flags):return bool(flags) and all(bool(v) for v in flags.values())

class BudgetExceeded(RuntimeError):pass
class GpuBudget:
    """Conservative GPU-stage wall time, including data/monitor overhead, not kernel-only time."""
    def __init__(self):self.last=time.perf_counter();self.total=read(O/'gpu_budget.json')['charged_seconds'] if (O/'gpu_budget.json').exists() else 0.
    def tick(self):
        now=time.perf_counter();self.total+=now-self.last;self.last=now
        write_json(O/'gpu_budget.json',{'charged_seconds':self.total,'limit_seconds':cfg()['gpu_budget_seconds'],'definition':'Cumulative wall time in GPU stages including source preparation, checkpoints, health and outcome evaluation; conservative, not hardware occupancy'})
        if self.total>cfg()['gpu_budget_seconds']:raise BudgetExceeded('24 GPU-stage-hour budget exhausted; no design reduction')
