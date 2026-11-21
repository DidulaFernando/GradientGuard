import time
from common import torch, config, isolated, sync
import psutil
from torch.nn.functional import cross_entropy

def mean_gradients(model, x, y, microbatch):
    """Accumulate signed gradients of mean data loss BEFORE taking any absolute values."""
    params = dict(model.named_parameters())
    accum = {n: torch.zeros_like(p) for n, p in params.items()}
    loss_sum = 0.0
    correct = 0
    n = len(y)
    if n == 0:
        raise ValueError("Empty probe")
    with isolated(model), torch.enable_grad():
        for start in range(0, n, microbatch):
            xb = x[start:start+microbatch].to(next(model.parameters()).device)
            yb = y[start:start+microbatch].to(xb.device)
            logits = model(xb)
            loss = cross_entropy(logits, yb, reduction="mean")
            gs = torch.autograd.grad(loss, tuple(params.values()))
            weight = len(yb) / n
            for (name, _), g in zip(params.items(), gs):
                accum[name].add_(g.detach(), alpha=weight)
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
    return accum, loss_sum / n, correct / n

def summarize(model, gradients, epsilon=None):
    c = config()["measurement"]
    epsilon = c["epsilon"] if epsilon is None else epsilon
    out = {}
    totals = []
    for layer in ("conv1", "conv2", "conv3", "head"):
        module = getattr(model, layer)
        z = module.weight.detach() * gradients[layer + ".weight"]
        s = z.abs().sum().item()
        totals.append(s)
        out.update({layer+"_abs": s, layer+"_signed": z.sum().item(),
                    layer+"_per_weight": s / z.numel(),
                    layer+"_weight_norm": module.weight.detach().norm().item(),
                    layer+"_bias_norm": module.bias.detach().norm().item(),
                    layer+"_bias_grad_norm": gradients[layer+".bias"].norm().item(),
                    layer+"_bias_sensitivity": (module.bias.detach()*gradients[layer+".bias"]).abs().sum().item()})
    total = sum(totals)
    denominator = total + epsilon
    out.update(total=total, denominator=denominator, epsilon=epsilon,
               epsilon_fraction=epsilon/denominator if denominator > 0 else 1.0)
    out["valid"] = bool(torch.isfinite(torch.tensor(list(out.values()), dtype=torch.float64)).all()
                        and total > 0 and out["epsilon_fraction"] < c["stabilizer_dominated_fraction"])
    out["invalid_reason"] = "" if out["valid"] else "zero, nonfinite or stabilizer-dominated total"
    for name, s in zip(("shallow", "middle", "deep"), (totals[0], totals[1], totals[2]+totals[3])):
        out[name+"_total"] = s
        out[name+"_share"] = s/denominator if out["valid"] else None
    conv_total = sum(totals[:3])
    for layer, s in zip(("conv1", "conv2", "conv3"), totals[:3]):
        out[layer+"_conv_share"] = s/(conv_total+epsilon)
    out["bias_norm"] = sum(out[l+"_bias_norm"]**2 for l in ("conv1", "conv2", "conv3", "head"))**0.5
    out["bias_sensitivity"] = sum(out[l+"_bias_sensitivity"] for l in ("conv1", "conv2", "conv3", "head"))
    out["head_fraction_of_deep"] = totals[3]/(totals[2]+totals[3]+epsilon)
    return out

def measure(model, x, y, microbatch=None):
    dev = next(model.parameters()).device
    sync(dev)
    baseline = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rss_before = psutil.Process().memory_info().rss
    t = time.perf_counter()
    gradients, loss, accuracy = mean_gradients(model, x, y, microbatch or config()["measurement"]["microbatch"])
    out = summarize(model, gradients)
    del gradients
    sync(dev)
    out.update(probe_loss=loss, probe_accuracy=accuracy, seconds=time.perf_counter()-t,
               process_rss_before=rss_before, process_rss_after=psutil.Process().memory_info().rss,
               process_peak_working_set=psutil.Process().memory_info().peak_wset if hasattr(psutil.Process().memory_info(), "peak_wset") else None,
               gpu_peak_allocated=torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
               gpu_incremental_allocated=torch.cuda.max_memory_allocated(dev)-baseline if dev.type == "cuda" else None,
               gpu_peak_reserved=torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None)
    return out

def health(model, x, y, microbatch=128):
    dev = next(model.parameters()).device
    sync(dev)
    start = time.perf_counter()
    loss = correct = 0
    probs = []
    with isolated(model), torch.no_grad():
        for i in range(0, len(y), microbatch):
            xb, yb = x[i:i+microbatch].to(dev), y[i:i+microbatch].to(dev)
            z = model(xb)
            loss += cross_entropy(z, yb, reduction="sum").item()
            correct += (z.argmax(1)==yb).sum().item()
            probs.append(z.softmax(1).cpu())
    sync(dev)
    return {"loss": loss/len(y), "accuracy": correct/len(y), "seconds": time.perf_counter()-start}, torch.cat(probs)
