"""Read-only paired-family inference on archived frozen risk scores.

No training, tuning, outcome relabelling or writes to scientific attempts.
Integer multiplicity weights are exactly equivalent to replicating whole families.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
out = P / 'evidence'
out.mkdir(exist_ok=True)
inputs = {}

def read(path):
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def scores(y, risk, weights):
    order = np.argsort(-risk, kind='stable')
    r, yy, w = risk[order], y[order], weights[:, order]
    starts = np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1]
    pos = np.add.reduceat(w * yy, starts, axis=1)
    neg = np.add.reduceat(w * (1 - yy), starts, axis=1)
    cp, cn = pos.cumsum(axis=1), neg.cumsum(axis=1)
    tp, tn = pos.sum(axis=1), neg.sum(axis=1)
    ap = (np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp+cn)>0) * pos).sum(axis=1) / tp
    auc = (pos * (tn[:, None] - cn + 0.5 * neg)).sum(axis=1) / (tp * tn)
    brier = (weights * (y - risk)**2).sum(axis=1) / weights.sum(axis=1)
    return np.stack([ap, auc, brier], axis=1)

def analyze(name, pred_path, label_path, manifest_path, reps, seed, pairs, original=False):
    pr, lab, manifest = read(pred_path), read(label_path), read(manifest_path)
    pr = pr[(pr.cohort=='eligible') & (pr.prefix==.1)]
    if original:
        pr = pr[(pr.panel=='A') & (~pr.refit_B)]
        pr['family'] = pr.run.str.extract(r'f(\d+)').astype(int)
    else:
        pr = pr[pr.variant=='A']
    lab = lab[lab.split=='test']
    pivot = pr.pivot(index=['run','family'],columns='model',values='risk').reset_index().merge(lab[['run','failure']],on='run',validate='one_to_one')
    pivot = pivot.sort_values('run').reset_index(drop=True)
    y = pivot.failure.astype(int).to_numpy()
    families = sorted(manifest.loc[manifest.split=='evaluation','family'].unique())
    rng = np.random.default_rng(seed)
    draws = rng.choice(families, size=(reps,len(families)),replace=True)
    weights = np.stack([(draws==f).sum(axis=1) for f in pivot.family],axis=1).astype(float)
    valid = ((weights*y).sum(axis=1)>0) & ((weights*(1-y)).sum(axis=1)>0)
    weights = weights[valid]
    models = sorted({m for pair in pairs for m in pair})
    ms, point = {}, {}
    for m in models:
        risk = pivot[m].to_numpy()
        assert np.isfinite(risk).all()
        ms[m] = scores(y,risk,weights)
        point[m] = scores(y,risk,np.ones((1,len(y))))[0]
        expected = [average_precision_score(y,risk),roc_auc_score(y,risk),brier_score_loss(y,risk)]
        np.testing.assert_allclose(point[m],expected,atol=1e-14,rtol=1e-14)
        # Validate weighted calculation against literal repeated rows.
        for j in range(min(3,len(weights))):
            ix = np.repeat(np.arange(len(y)),weights[j].astype(int))
            expected = [average_precision_score(y[ix],risk[ix]),roc_auc_score(y[ix],risk[ix]),brier_score_loss(y[ix],risk[ix])]
            np.testing.assert_allclose(ms[m][j],expected,atol=1e-14,rtol=1e-14)
    rows=[]
    for added,base in pairs:
        delta=ms[added]-ms[base]
        for j,metric in enumerate(['AP','AUROC','Brier']):
            lo,hi=np.quantile(delta[:,j],[.025,.975])
            rows.append(dict(study=name,added=added,base=base,metric=metric,added_value=point[added][j],base_value=point[base][j],difference=point[added][j]-point[base][j],low=lo,high=hi,N=len(y),F=int(y.sum()),families=len(families),eligible_families=pivot.family.nunique(),replicates=reps,valid=int(valid.sum()),invalid=int((~valid).sum()),seed=seed))
    print(name,len(y),int(valid.sum()),int((~valid).sum()))
    return rows

legacy=ROOT/'Feasibility/poc/outputs/20261116_poc01'
struct=ROOT/'layer_allocation/runs/20261121_structural01'
fresh=ROOT/'layer_allocation/runs/20261122_fresh02'
rows=analyze('original',legacy/'evaluation_predictions.csv',legacy/'reports/terminal_outcomes.csv',legacy/'run_manifest.csv',2000,130005,[(m,'E') for m in ['logistic_B','logistic_L2','logistic_L6','logistic_L8','histogram_L8','bias_E','bias_L8','final_monitor']]+[('logistic_L8','logistic_L6')],True)
rows+=analyze('structural',struct/'predictions.csv',legacy/'reports/terminal_outcomes.csv',legacy/'run_manifest.csv',2000,210212,[('logistic_MZ','logistic_M'),('histogram_MZ','histogram_M'),('logistic_NZ','logistic_N'),('logistic_L6','logistic_B')])
rows+=analyze('fresh',fresh/'evaluation_predictions.csv',fresh/'terminal_outcomes.csv',fresh/'family_manifest.csv',5000,710005,[('logistic_MZ','logistic_M'),('histogram_MZ','histogram_M'),('logistic_L6','E'),('logistic_L6','logistic_B'),('logistic_L6','logistic_L2'),('logistic_NZ','logistic_N'),('histogram_L6','histogram_E')])
pd.DataFrame(rows).to_csv(out/'paired_family_ranking_intervals.csv',index=False,lineterminator='\n')
(out/'ranking_interval_provenance.json').write_text(json.dumps({'input_sha256':inputs,'method':'Paired whole-family percentile bootstrap, identical draw for both frozen models; AP and AUROC recomputed with tie groups and family multiplicities; Brier added minus control, negative favours added; no selection uncertainty; two-sided 95% pointwise intervals; empty/single-label draws excluded for a common metric set. Weighted calculations validated against sklearn and literal duplicated rows.','scope':'Original and structural analyses reuse consumed outcomes; fresh predictors predate fresh evaluation outcomes; intervals added retrospectively without refitting.'},indent=2)+'\n',encoding='utf-8',newline='\n')
