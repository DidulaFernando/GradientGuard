import sys
import platform
import importlib.metadata as md
import subprocess
import time
from .core import *

def protected_paths():
    roots=[ROOT/"src",ROOT/"tests",ROOT/"configs",legacy("runs"),legacy("data"),legacy("checkpoints"),legacy("reports"),ROOT/"data/mnist"]
    paths=[p for folder in roots for p in folder.rglob("*") if p.is_file()]
    paths.extend([ROOT/"README.md"]+[ROOT/"instructions"/n for n in ("environment.ps1","ENGINEERING_HISTORY.md","STEP_BY_STEP.txt")])
    proposal=Path("C:/Users/rangi/Documents/ChatGPT/Explanations/GradientGuard_Research_Proposal_Didula_Fernando_v5.pdf")
    return sorted(set(paths)),proposal

def protect(after=False):
    paths,proposal=protected_paths()
    records={str(p.relative_to(ROOT)): {"sha256":sha(p),"bytes":p.stat().st_size} for p in paths}
    record={"files":records,"proposal_sha256":sha(proposal)}
    path=out()/"preflight/protected_before.json"
    if after:
        before=read_json(path)
        changed=[p for p in set(before["files"])|set(records) if before["files"].get(p)!=records.get(p)]
        unchanged=not changed and before["proposal_sha256"]==record["proposal_sha256"]
        write_json(out()/"preflight/protected_after.json",record)
        write_json(out()/"preflight/protection_audit.json",{"unchanged":unchanged,"changed":changed,"file_count":len(records),"bytes":sum(v["bytes"] for v in records.values())})
        assert unchanged,f"Legacy artifacts changed: {changed}"
    elif path.exists():
        assert read_json(path)==record,"Protected baseline differs; do not replace it"
    else: write_json(path,record)
    return len(records)

def preflight():
    deterministic()
    with cost("preflight"):
        n=protect()
        old=read_json(legacy("runs")/"provenance.json")
        packages={d.metadata["Name"]:d.version for d in md.distributions()}
        current={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"cpu":platform.processor(),
                 "device":str(device()),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
                 "gpu_properties":str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None}
        drift={key:{"saved":old["hardware"].get(key),"current":value} for key,value in current.items() if old["hardware"].get(key)!=value}
        package_drift={n:{"saved":v,"current":packages.get(n)} for n,v in old["packages"].items() if packages.get(n)!=v}
        freeze=subprocess.check_output([sys.executable,"-m","pip","freeze","--all"],text=True)
        oldlock=(legacy("runs")/"requirements-lock.txt").read_text().splitlines()
        lock_same=set(freeze.splitlines())==set(oldlock)
        for rel,h in old["project_source_hashes"].items():
            assert sha(ROOT/rel)==h and sha(legacy("runs")/"source_snapshot"/rel)==h,rel
        for condition,item in read_json(legacy("data")/"verification.json")["conditions"].items():
            assert sha(legacy("data")/(condition+".pt"))==item["tensor_sha256"]
        assert sha(ROOT/"configs/pilot.yaml")==old["config_hash"]
        import inspect
        from spuco.datasets import SpuCoMNIST
        assert sha(inspect.getfile(SpuCoMNIST))==old["spuco_source_hash"]
        cp_audits=[]
        frozen=read_json(legacy("runs")/"frozen_config.json")
        for condition in ("independent","correlated"):
            for seed in (11,22,33):
                run=f"{condition}_s{seed}"
                for step in [0]+frozen["checkpoint_updates"]:
                    p=legacy("checkpoints")/run/f"step_{step:04d}.pt"
                    cp=torch.load(p,map_location="cpu",weights_only=False)
                    assert cp["step"]==step and cp["config_hash"]==old["config_hash"] and cp["source_hashes"]==old["project_source_hashes"]
                    cp_audits.append({"run":run,"step":step,"sha256":sha(p)})
        write_once(out()/"preflight/checkpoint_hashes.json",cp_audits)
        numerical_names=("torch","torchvision","numpy","pandas","matplotlib","spuco","PyYAML","pytest","psutil")
        numerical_drift={n:{"saved":old["packages"].get(n),"current":md.version(n)} for n in numerical_names if old["packages"].get(n)!=md.version(n)}
        result={"hardware":current,"hardware_drift":drift,"packages":packages,"provenance_enumeration_discrepancies":package_drift,
                "numerical_package_drift":numerical_drift,
                "enumeration_adjudication":"Complete pip freeze --all exactly matches saved lock, numerical versions and installed generator hashes checked explicitly. Old metadata enumeration includes setuptools-vendored distributions, unlike current import ordering. This is a provenance-enumeration discrepancy, not evidence of an installed-version change. No packages installed or changed.",
                "lock_identical":lock_same,"comparable":not drift and not numerical_drift and lock_same,
                "protected_files":n,"environment":{k:os.environ.get(k) for k in old["environment"]},
                "nvidia_smi":subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout,
                "precision":"Legacy deterministic() unchanged; float32 training, CPU float64 analytical checks; no AMP, TF32 or reduced precision; workers=0"}
        prior=out()/"preflight/environment.json"
        if prior.exists(): write_json(out()/f"preflight/environment_history/{time.time_ns()}.json",read_json(prior))
        write_json(prior,result)
        (out()/"preflight/requirements-lock.txt").write_text(freeze,encoding="utf-8")
        assert result["comparable"],f"Environment drift requires documentation before comparative work: {drift}, {numerical_drift}, lock={lock_same}"
        log("preflight_passed",protected_files=n,environment_identical=True)

def freeze_protocol():
    env=read_json(out()/"preflight/environment.json"); assert env["comparable"]
    c=cfg(); old=read_json(legacy("runs")/"frozen_config.json")
    obj={"protocol":c,"protocol_sha256":sha(PROTOCOL_PATH),"extension_source_hashes":hashes(),
         "legacy_configuration_sha256":old["configuration_sha256"],"legacy_measurement_steps":old["checkpoint_updates"],
         "planned_analyses":["reproduce_original_3.0501409990602664","20_partitions_all_8_features_both_prefixes_all_budgets",
            "common_denominator_layers_conv_only_per_weight_bias","additivity_and_uncentred_MSE_cross_term",
            "calibrate_only_health_20_40_60","if_qualified_18_fresh_prefixes_20_partitions_256_with_all_eligible_condition_realisation_subsets"],
         "structural_decision":"No automatic artifact label. Require descriptive review of individual-layer reliability, error cancellation, budget/checkpoint flags. Original main target never replaced.",
         "numerical_semantics":"Unchanged mean gradients and legacy summarize/measure functions. Features use exact legacy formulas; expected cohort explicitly supplied. Missing/invalid runs invalidate planned cohort, eligibility does not silently filter.",
         "partition_interpretation":"20 complementary overlapping partitions of one finite pool; descriptive quantiles/pass fractions, never confidence intervals or independent replication counts.",
         "calibration_selection":"Only health, finite required telemetry and cost; never gradient reliability or recolouring. Smallest qualifying horizon; no grid expansion.",
         "fresh_readiness":"Correctness, stable target for full/eligible cohorts, source coverage and reviewed structure jointly required. Within-condition absence qualifies a pooled result; no prediction claim."}
    write_once(out()/"protocol_frozen.json",obj)
    log("protocol_frozen",attempt=c["attempt"],source_files=len(obj["extension_source_hashes"]))
