import argparse
import subprocess
import sys
import time
from .core import *

def tests():
    p=out()/"checks"/str(time.time_ns()); p.mkdir(parents=True)
    command=[sys.executable,"-m","pytest",str(EXT/"tests"),"-v","--basetemp",str(ROOT/"tmp"/f"followup_pytest_{time.time_ns()}"),
             "-o","cache_dir="+str(EXT/"cache/pytest"),"--junitxml",str(p/"junit.xml")]
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    (p/"pytest.log").write_text(r.stdout+r.stderr,encoding="utf-8")
    write_json(out()/"checks/status.json",{"passed":r.returncode==0,"source_hashes":hashes(),"log":str(p/"pytest.log")})
    print(r.stdout+r.stderr,flush=True)
    if r.returncode: raise SystemExit(r.returncode)

def main():
    p=argparse.ArgumentParser(description="Namespaced GradientGuard follow-up; legacy artifacts read-only")
    p.add_argument("command",choices=["preflight","freeze","tests","reproduce","partitions","saved","structural","calibrate","freeze-fresh","fresh","fresh-measure","report","verify-protection","status","all"])
    args=p.parse_args(); deterministic()
    with cost("command_"+args.command):
        if args.command=="preflight":
            from .artifacts import preflight
            preflight()
        elif args.command=="freeze":
            from .artifacts import freeze_protocol
            freeze_protocol()
        elif args.command=="tests": tests()
        elif args.command=="reproduce":
            from .measuring import reproduce
            reproduce()
        elif args.command in ("partitions","saved"):
            from .measuring import saved_partitions
            saved_partitions(manifests_only=args.command=="partitions")
        elif args.command=="structural":
            from .reporting import analyze_saved
            analyze_saved()
        elif args.command=="calibrate":
            from .training import calibrate
            calibrate()
        elif args.command=="freeze-fresh":
            from .training import freeze_fresh
            freeze_fresh()
        elif args.command=="fresh":
            from .training import fresh
            fresh()
        elif args.command=="fresh-measure":
            from .measuring import fresh_partitions
            fresh_partitions()
        elif args.command=="report":
            from .reporting import report
            report()
        elif args.command=="verify-protection":
            from .artifacts import protect
            print("PROTECTED LEGACY FILES UNCHANGED",protect(after=True),flush=True)
        elif args.command=="status":
            from .reporting import status
            status()
        elif args.command=="all":
            from .artifacts import preflight,freeze_protocol,protect
            from .measuring import reproduce,saved_partitions,fresh_partitions
            from .training import calibrate,freeze_fresh,fresh
            from .reporting import analyze_saved,report
            preflight(); freeze_protocol(); tests(); reproduce(); saved_partitions(); analyze_saved(); calibrate()
            if freeze_fresh(): fresh(); fresh_partitions()
            protect(after=True); report()
            log("all_completed")
