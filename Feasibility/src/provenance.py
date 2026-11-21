import sys
import platform
import subprocess
import importlib.metadata as md
import inspect
from common import ROOT, ENV, os, torch, config, deterministic, device, sha, source_hashes, attempt_dir, write_json

def freeze():
    deterministic()
    from data import load
    from spuco.datasets import SpuCoMNIST
    c = config()
    d = load("independent")
    n = d["roles"].count("train")
    updates = (n + c["training"]["batch_size"]-1)//c["training"]["batch_size"]
    horizon = updates*c["training"]["epochs"]
    checkpoints = [round(f*horizon) for f in c["training"]["checkpoint_fractions"]]
    assert len(checkpoints) == len(set(checkpoints)) and min(checkpoints)>0
    out = {"configuration": c, "configuration_sha256": sha(ROOT/"configs/pilot.yaml"),
           "train_n": n, "updates_per_epoch": updates, "full_planned_updates": horizon,
           "checkpoint_updates": checkpoints, "actual_fractions": [s/horizon for s in checkpoints],
           "endpoint": checkpoints[-1], "rounding": "Python round (ties to even), checkpoints distinct",
           "rationale": "Empty existing project; proposal fixes CNN widths but no specific digit mapping or horizon. User defaults adopted prospectively. No empirical tuning.",
           "feature_definition": "8 features of shallow/deep primary block shares: last, mean, population SD, OLS slope vs actual full-horizon fraction; exclude initialization.",
           "numerical_floor": "Level/mean/SD floor 1e-5 share units; slope floor 2e-5/(last-first fraction). Use larger of this and measured repeat RMS/sqrt(2). Ratios use max(sampling noise, numerical floor).",
           "reliability_followup": "If any initial retained feature at either prefix has spread/noise below 3, remeasure all saved checkpoints at 128 and 256 per class. Never retrain.",
           "structural_rule": "Head/bias diagnostics are descriptive; no causal attribution. Head sensitivity >=50% of deep or bias sensitivity >=10% of weight total flags targeted control.",
           "cost_rule": "Combined A+B monitor time/training time >1 flags excessive monitoring; primary A cost also reported."}
    folder = attempt_dir()
    path = folder / "frozen_config.json"
    if path.exists():
        import json
        old = json.loads(path.read_text())
        assert old == out, "Frozen configuration mismatch; make a new attempt"
    else:
        write_json(path, out)
    packages = {dist.metadata["Name"]: dist.version for dist in md.distributions()}
    spuco_root = ROOT/"tools/SpuCo"
    commit = subprocess.check_output(["git", "-C", str(spuco_root), "rev-parse", "HEAD"], text=True).strip()
    source = Path(inspect.getfile(SpuCoMNIST))
    env = {k: os.environ.get(k) for k in (*ENV, "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "PYTHONUTF8")}
    hardware = {"platform": platform.platform(), "cpu": platform.processor(), "python": sys.version,
                "executable": sys.executable, "device": str(device()),
                "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_properties": str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None,
                "torch_build": torch.__config__.show()}
    import psutil
    hardware["memory"] = dict(psutil.virtual_memory()._asdict())
    hardware["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    provenance = {"packages": packages, "hardware": hardware, "environment": env,
                  "spuco_commit": commit, "spuco_source": str(source), "spuco_source_hash": sha(source),
                  "spuco_relevant_hashes": {str(p.relative_to(spuco_root)): sha(p) for p in (spuco_root/"src/spuco").rglob("*.py")},
                  "project_source_hashes": source_hashes(), "config_hash": sha(ROOT/"configs/pilot.yaml"),
                  "precision": "Training float32; analytical CPU float64; no AMP/TF32; deterministic algorithms strict; cudnn benchmark false; Adam foreach=False fused=False.",
                  "limitation": "No cross-hardware or cross-version bitwise guarantee. DataLoader workers=0; training permutation uses isolated CPU Generator per epoch."}
    write_json(folder/"provenance.json", provenance)
    (folder/"requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True), encoding="utf-8")
    print("FROZEN", out, flush=True)

from pathlib import Path
