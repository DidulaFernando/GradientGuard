"""Commands never activate a shell environment; use .venv/Scripts/python.exe directly."""
from common import ROOT, attempt_dir, config, write_json, source_hashes, deterministic
import argparse
import subprocess
import sys
import time
import json

def checks():
    folder=attempt_dir()/"check_history"/str(time.time_ns()); folder.mkdir(parents=True)
    metrics=attempt_dir()/"correctness_metrics.json"
    if metrics.exists(): (folder/"previous_metrics.json").write_bytes(metrics.read_bytes())
    cmd=[sys.executable,"-m","pytest","tests","-v","--basetemp",str(ROOT/"tmp"/f"pytest_{time.time_ns()}"),
         "-o","cache_dir="+str(ROOT/"cache/pytest"),"--junitxml",str(folder/"junit.xml")]
    result=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8")
    (folder/"pytest.log").write_text(result.stdout,encoding="utf-8")
    print(result.stdout,flush=True)
    write_json(attempt_dir()/"correctness_status.json",{"passed":result.returncode==0,"source_hashes":source_hashes(),"log":str(folder/"pytest.log"),"timestamp_ns":time.time_ns()})
    if result.returncode: raise SystemExit(result.returncode)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["environment","prepare","verify-data","freeze","check","smoke","pilot","remeasure","report","all"])
    p.add_argument("--run")
    p.add_argument("--sizes",type=int,nargs="+",default=[64])
    args=p.parse_args(); deterministic()
    if args.command=="environment":
        from common import device,torch
        print(sys.version,sys.executable,torch.__version__,"device",device())
        if torch.cuda.is_available():
            x=torch.randn(5,5,device="cuda"); print((x@x).sum().item(),torch.cuda.get_device_name(0))
    elif args.command in ("prepare","verify-data"):
        from data import prepare,verify
        (prepare if args.command=="prepare" else verify)()
    elif args.command=="freeze":
        from provenance import freeze
        freeze()
    elif args.command=="check": checks()
    elif args.command=="smoke":
        from runner import smoke
        smoke()
    elif args.command=="pilot":
        from runner import pilot
        pilot(args.run)
    elif args.command=="remeasure":
        from runner import remeasure
        remeasure(tuple(args.sizes))
    elif args.command=="report":
        from analysis import report
        report()
    elif args.command=="all":
        from data import prepare
        from provenance import freeze
        from runner import smoke,pilot,remeasure
        from analysis import report,needs_larger_probes
        prepare(); freeze(); checks(); smoke(); pilot(); remeasure((64,))
        if needs_larger_probes(): remeasure((128,256))
        report()

if __name__=="__main__": main()
