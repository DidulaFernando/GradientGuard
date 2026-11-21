import sys
import os
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
ROOT = EXT.parent
sys.path.insert(0,str(ROOT/"src"))
from common import torch,np,sha,write_json,save_torch,seed_all,deterministic,device,rng_state,restore_rng,sync,isolated
import yaml
import json
import hashlib
import time
from contextlib import contextmanager

PROTOCOL_PATH=EXT/"configs/protocol.yaml"
def cfg(): return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
def out():
    p=EXT/"outputs"/cfg()["attempt"]; p.mkdir(parents=True,exist_ok=True); return p
def legacy(kind): return ROOT/kind/cfg()["legacy_attempt"]
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def hashes():
    return {str(p.relative_to(EXT)):sha(p) for folder in ("ggfollow","tests","configs") for p in sorted((EXT/folder).rglob("*")) if p.is_file() and p.suffix in (".py",".yaml")} | {"run.py":sha(EXT/"run.py")}
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def write_once(path,obj):
    path=Path(path)
    if path.exists(): assert read_json(path)==json.loads(json.dumps(obj,default=str)),f"Frozen artifact mismatch: {path}"
    else: write_json(path,obj)
def log(stage,**values):
    row={"utc_ns":time.time_ns(),"stage":stage,**values}
    with (out()/"progress.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row,default=str)+"\n")
    write_json(out()/"status.json",row)
    print(json.dumps(row,default=str),flush=True)
@contextmanager
def cost(stage):
    start=time.perf_counter(); failed=None
    try: yield
    except BaseException as e:
        failed=repr(e); raise
    finally:
        import psutil
        mi=psutil.Process().memory_info()
        row={"stage":stage,"seconds":time.perf_counter()-start,"failed":failed,"utc_ns":time.time_ns(),
             "process_rss":mi.rss,"peak_working_set":getattr(mi,"peak_wset",None)}
        with (out()/"costs.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
def require_protocol():
    frozen=read_json(out()/"protocol_frozen.json")
    assert frozen["protocol_sha256"]==sha(PROTOCOL_PATH),"Protocol changed"
    # Only numerical source files govern measurement/training resume; reporting code may evolve with a separate hash log.
    for rel in ("ggfollow/core.py","ggfollow/metrics.py","ggfollow/partitions.py","ggfollow/datasets.py","ggfollow/training.py","ggfollow/measuring.py"):
        assert frozen["extension_source_hashes"][rel]==sha(EXT/rel),f"Numerical source changed: {rel}"
    return frozen
def require_reproduction():
    require_protocol()
    assert read_json(out()/"reproduction/audit.json")["passed"],"Original reproduction unresolved"
