"""Namespaced read-only adapters. Never import legacy command modules."""
from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
LEGACY=REPO/'Feasibility'
OLD=LEGACY/'poc/outputs/20261116_poc01'
for name,rel in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'cache/pip','MPLCONFIGDIR':'cache/matplotlib','PYTHONPYCACHEPREFIX':'cache/pycache','TORCH_HOME':'cache/torch','XDG_CACHE_HOME':'cache/xdg','CUDA_CACHE_PATH':'cache/cuda'}.items():
    p=ROOT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(LEGACY/'src'))
from common import torch,np,deterministic,isolated,rng_state,restore_rng
from model import CNNA
import importlib.util
_spec=importlib.util.spec_from_file_location('_gg_original_measurement',LEGACY/'src/measurement.py')
_legacy_measurement=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_legacy_measurement)
mean_gradients=_legacy_measurement.mean_gradients
summarize=_legacy_measurement.summarize
import pandas as pd
# common configures caches at import; restore all new-work settings to this namespace.
for name,rel in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'cache/pip','MPLCONFIGDIR':'cache/matplotlib','PYTHONPYCACHEPREFIX':'cache/pycache','TORCH_HOME':'cache/torch','XDG_CACHE_HOME':'cache/xdg','CUDA_CACHE_PATH':'cache/cuda'}.items():
    p=ROOT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
CFG=ROOT/'configs/structural.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf8');tmp.replace(p)
def cfg():return read(CFG)
def out():return ROOT/'runs'/cfg()['attempt']
def source_hashes():return {str(p.relative_to(ROOT)):sha(p) for d in ['src','tests','configs'] for p in sorted((ROOT/d).rglob('*')) if p.suffix in ['.py','.json']}
def require():
    freeze=read(out()/'protocol_freeze.json')
    assert freeze['source_hashes']==source_hashes(),'New source/config changed after freeze; create a new attempt'
    return freeze
def event(name,**kw):
    row={'time_ns':time.time_ns(),'event':name,**kw}
    with (out()/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    write(out()/'status.json',row);print(json.dumps(row),flush=True)
def summaries(v,x):
    v=np.asarray(v,dtype=float);x=np.asarray(x,dtype=float)
    if len(v)==0 or not np.isfinite(v).all() or len(x)!=len(v):raise ValueError('Invalid or incomplete summary')
    return {'last':float(v[-1]),'mean':float(v.mean()),'std':float(v.std(ddof=0))}
def six():return [b+'_share_'+s for b in ['shallow','deep'] for s in ['last','mean','std']]
