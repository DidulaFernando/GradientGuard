"""Project-local process configuration. Import before numerical libraries."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ENV = {"TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp", "PIP_CACHE_DIR": "cache/pip",
       "UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "tools/python",
       "UV_PYTHON_BIN_DIR": "tools/bin", "MPLCONFIGDIR": "cache/matplotlib",
       "TORCH_HOME": "cache/torch", "HF_HOME": "cache/huggingface",
       "XDG_CACHE_HOME": "cache/xdg", "NUMBA_CACHE_DIR": "cache/numba",
       "CUDA_CACHE_PATH": "cache/cuda", "PYTHONPYCACHEPREFIX": "cache/pycache"}
for key, rel in ENV.items():
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_MODE"] = "disabled"
import json
import hashlib
import random
import contextlib
import numpy as np
import torch
import yaml

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def config():
    return yaml.safe_load((ROOT / "configs/pilot.yaml").read_text(encoding="utf-8"))

def attempt_dir(kind="runs"):
    p = ROOT / kind / config()["attempt"]
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def deterministic():
    torch.set_num_threads(config()["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])

@contextlib.contextmanager
def isolated(model):
    """autograd.grad leaves .grad alone; restore all module modes and RNG even on error."""
    state = rng_state()
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, mode in modes:
            m.training = mode
        restore_rng(state)

def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

def device():
    requested = config()["training"]["device"]
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)

def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ("src", "tests", "configs")
            for p in sorted((ROOT / folder).rglob("*")) if p.is_file() and p.suffix in (".py", ".yaml")}
