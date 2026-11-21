"""Official generator; identities recovered through its unmodified MNIST Subset."""
import time
import gc
import inspect
from common import ROOT, config, attempt_dir, torch, np, seed_all, sha, write_json, save_torch

def data_dir():
    p = ROOT / "data" / config()["attempt"]
    p.mkdir(parents=True, exist_ok=True)
    return p

def load(condition):
    return torch.load(data_dir() / (condition + ".pt"), map_location="cpu", weights_only=False)

def panel_indices(data, role, n=None):
    idx = np.flatnonzero(np.array(data["roles"]) == role)
    if n is not None:
        idx = np.array([i for i in idx if data["ranks"][i] < n])
    return torch.tensor(idx, dtype=torch.long)

def prepare():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
    from spuco.utils import set_seed
    c = config()
    dc = c["dataset"]
    folder = data_dir()
    if (folder / "verification.json").exists():
        return verify()
    started = time.perf_counter()
    report = {"generator_signature": str(inspect.signature(SpuCoMNIST)), "conditions": {},
              "data_seed": dc["data_seed"], "reservation_seed": dc["reservation_seed"],
              "note": "Official validation/test are never instantiated/evaluated. torchvision download may fetch raw test files."}
    ids_reference = y_reference = roles_reference = ranks_reference = palette_reference = None
    grids = []
    for condition, strength in dc["conditions"].items():
        path = folder / (condition + ".pt")
        if path.exists():
            raise RuntimeError("Partial data preparation exists; preserve it and create a new attempt before regenerating")
        set_seed(dc["data_seed"])
        seed_all(dc["data_seed"])
        ds = SpuCoMNIST(root=str(ROOT / "data/mnist"),
                        spurious_feature_difficulty=getattr(SpuriousFeatureDifficulty, dc["difficulty"]),
                        classes=dc["classes"], spurious_correlation_strength=strength,
                        label_noise=dc["label_noise"], core_feature_noise=dc["core_feature_noise"],
                        split="train", verbose=True, download=True)
        ds.initialize()
        ids = np.array(ds.mnist.indices, dtype=np.int64)
        raw = ds.mnist.dataset.data[torch.from_numpy(ids)].clone()
        original_labels = ds.mnist.dataset.targets[torch.from_numpy(ids)]
        y = torch.tensor(ds.labels, dtype=torch.long)
        bg = torch.tensor(ds.spurious, dtype=torch.long)
        x = torch.stack(ds.data.X)
        palette = np.array(ds.colors[:len(dc["classes"])])
        assert len(ids) == len(x) == len(set(ids.tolist()))
        assert torch.equal(y, original_labels), "Unexpected label corruption"
        assert not ds.is_noisy_label.any() and not ds.data.core_feature_noise.any()
        # Chunk verification avoids a second complete float32 RGB copy.
        max_foreground_error = max_background_error = 0.0
        for start in range(0, len(x), 512):
            rr = raw[start:start+512].float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
            xx = x[start:start+512]
            mask = rr != 0
            max_foreground_error = max(max_foreground_error, (xx[mask]-rr[mask]).abs().max().item())
            expected_bg = torch.tensor(palette[bg[start:start+512]], dtype=x.dtype)[:, :, None, None].expand_as(xx)
            max_background_error = max(max_background_error, (xx[~mask]-expected_bg[~mask]).abs().max().item())
        assert max_foreground_error == 0 and max_background_error == 0
        assert x.shape[1:] == (3, 28, 28) and torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        if ids_reference is None:
            ids_reference, y_reference, palette_reference = ids.copy(), y.clone(), palette.copy()
            roles_reference = np.full(len(ids), "train", dtype="U8")
            ranks_reference = np.full(len(ids), -1, dtype=np.int64)
            rng = np.random.default_rng(dc["reservation_seed"])
            for label in range(len(dc["classes"])):
                shuffled = rng.permutation(np.flatnonzero(y.numpy() == label))
                offset = 0
                for role, n in (("probe_A", dc["probe_pool_per_class"]), ("probe_B", dc["probe_pool_per_class"]), ("health", dc["health_per_class"])):
                    chosen = shuffled[offset:offset+n]
                    assert len(chosen) == n
                    roles_reference[chosen] = role
                    ranks_reference[chosen] = np.arange(n)
                    offset += n
        assert np.array_equal(ids, ids_reference) and torch.equal(y, y_reference)
        assert np.array_equal(palette, palette_reference)
        manifest = pd.DataFrame({"original_mnist_train_id": ids, "digit": original_labels.numpy(),
                                 "class": y.numpy(), "role": roles_reference, "nested_rank": ranks_reference})
        # Offline background metadata is explicitly separate from manifests/features.
        manifest.to_csv(folder / (condition+"_manifest.csv"), index=False)
        offline = folder / "offline_diagnostics"
        offline.mkdir(exist_ok=True)
        pd.DataFrame({"original_mnist_train_id": ids, "class": y.numpy(), "background": bg.numpy()}).to_csv(offline / (condition+"_backgrounds.csv"), index=False)
        counts = manifest.groupby(["role", "class"]).size().unstack(fill_value=0)
        counts.to_csv(folder / (condition+"_class_counts.csv"))
        alignments = {}
        for role in ("all", "train", "probe_A", "probe_B", "health"):
            mask = np.ones(len(ids), dtype=bool) if role == "all" else roles_reference == role
            table = pd.crosstab(y.numpy()[mask], bg.numpy()[mask]).reindex(index=range(10), columns=range(10), fill_value=0)
            table.to_csv(offline / (condition+"_"+role+"_contingency.csv"))
            alignments[role] = float((y.numpy()[mask] == bg.numpy()[mask]).mean())
            if role == "all":
                per_class = np.diag(table) / table.sum(axis=1)
                assert np.max(np.abs(per_class-strength)) < 0.001
                if condition == "independent":
                    assert np.max(np.abs(table.div(table.sum(axis=1), axis=0).to_numpy()-0.1)) < 0.02
        role_ids = {role: set(ids[roles_reference == role].tolist()) for role in ("train", "probe_A", "probe_B", "health")}
        overlaps = {a+"/"+b: len(role_ids[a]&role_ids[b]) for a in role_ids for b in role_ids if a < b}
        assert not any(overlaps.values())
        val_ids = set(ds.mnist.dataset.targets.new_tensor(__import__("spuco.datasets.spuco_mnist_config", fromlist=["config"]).config["val"]).tolist())
        assert not set(ids.tolist()) & val_ids
        selected = [int(torch.nonzero(y == label)[0]) for label in range(10)]
        grids.append(x[selected].clone())
        save_torch(path, {"x": x, "y": y, "raw": raw, "ids": ids,
                          "roles": roles_reference.tolist(), "ranks": ranks_reference.tolist(),
                          "background": bg, "palette": palette, "config_hash": sha(ROOT/"configs/pilot.yaml")})
        report["conditions"][condition] = {"n": len(ids), "train_n": int((roles_reference == "train").sum()),
                 "alignment": alignments, "foreground_error": max_foreground_error,
                 "background_error": max_background_error, "overlaps": overlaps,
                 "official_validation_overlap": 0, "tensor_sha256": sha(path)}
        del x, ds, raw
        gc.collect()
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.8))
    for row, condition in enumerate(dc["conditions"]):
        for label in range(10):
            axes[row, label].imshow(grids[row][label].permute(1,2,0))
            axes[row, label].axis("off")
            axes[row, label].set_title(f"{condition[:4]} digit {label}", fontsize=8)
    fig.suptitle("Matched original MNIST identities; only zero-valued background pixels changed")
    fig.tight_layout()
    fig.savefig(folder / "matched_images.png", dpi=160)
    plt.close(fig)
    report["seconds"] = time.perf_counter()-started
    report["passed"] = True
    write_json(folder / "verification.json", report)
    print("DATA PREPARATION VERIFIED", report, flush=True)

def verify():
    import json
    folder = data_dir()
    report = json.loads((folder / "verification.json").read_text())
    assert report["passed"]
    for condition, item in report["conditions"].items():
        assert sha(folder / (condition+".pt")) == item["tensor_sha256"], "Data hash mismatch"
        d = load(condition)
        assert d["config_hash"] == sha(ROOT/"configs/pilot.yaml"), "Config changed"
    print("DATA HASHES AND CONFIG VERIFIED", flush=True)
