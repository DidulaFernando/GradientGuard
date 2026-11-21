"""Legacy mathematical formulas, generalized to explicit expected run memberships."""
from .core import *
import pandas as pd
import math

SUMMARIES=("last","mean","std","slope")
PRIMARY={"shallow":"shallow_share","deep":"deep_share"}
DIAGNOSTICS={l:l+"_share" for l in ("conv1","conv2","conv3","head")}|{
    l+"_only":l+"_conv_share" for l in ("conv1","conv2","conv3")}|{
    l+"_per_weight":l+"_per_weight" for l in ("conv1","conv2","conv3","head")}|{
    "bias_norm":"bias_norm","bias_sensitivity":"bias_sensitivity","head_fraction":"head_fraction_of_deep"}

def enrich(row):
    row=dict(row)
    den=row["denominator"]
    for l in ("conv1","conv2","conv3","head"):
        row[l+"_share"]=row[l+"_abs"]/den if row["valid"] else None
    c=cfg()["measurement"]
    row["head_flag"]=bool(row["head_fraction_of_deep"]>=c["head_flag"])
    row["bias_ratio"]=row["bias_sensitivity"]/row["total"] if row["total"]>0 else None
    row["bias_flag"]=bool(row["bias_ratio"] is not None and row["bias_ratio"]>=c["bias_flag"])
    row["additivity_error"]=abs(row["deep_share"]-row["conv3_share"]-row["head_share"]) if row["valid"] else None
    if row["valid"]: assert row["additivity_error"]<1e-9
    return row

def summaries(values,x):
    return [values[-1],values.mean(),values.std(ddof=0),np.dot(x-x.mean(),values-values.mean())/np.dot(x-x.mean(),x-x.mean())]

def features(table,schedule,columns=PRIMARY):
    if table.empty: return pd.DataFrame()
    metadata=[n for n in ("realisation","condition","seed","run","partition","budget","panel","repeat") if n in table]
    records=[]
    for key,df in table.groupby(metadata,dropna=False,sort=True):
        key=(key,) if not isinstance(key,tuple) else key
        identity=dict(zip(metadata,key))
        for prefix in cfg()["measurement"]["prefixes"]:
            points=[s for s in schedule["steps"] if s<=round(prefix*schedule["horizon"])]
            observed=df[(df.step>0)&(df.step<=round(prefix*schedule["horizon"]))].sort_values("step")
            values=observed[list(columns.values())].to_numpy(dtype=float)
            valid=bool(observed.step.tolist()==points and observed.valid.astype(bool).all() and np.isfinite(values).all())
            row={**identity,"prefix":prefix,"observations":len(observed),"valid":valid}
            x=observed.fraction.to_numpy()
            for name,column in columns.items():
                vals=summaries(observed[column].to_numpy(),x) if valid else [None]*4
                row.update(dict(zip([name+"_"+s for s in SUMMARIES],vals)))
            records.append(row)
    return pd.DataFrame(records)

def base_floor(feature):
    c=cfg()["measurement"]
    if "_per_weight_" in feature:
        counts={"conv1":432,"conv2":4608,"conv3":18432,"head":640}
        return c["diagnostic_total_floor"]/counts[feature.split("_")[0]]
    if feature.startswith("bias_"): return c["diagnostic_total_floor"]
    return c["numerical_share_floor"]

def reliability(ft,schedule,expected_runs,feature_names=None):
    """An explicit set prevents missing/ineligible runs from silently shrinking the cohort."""
    if feature_names is None: feature_names=[b+"_"+s for b in PRIMARY for s in SUMMARIES]
    records=[]; expected=set(expected_runs)
    for prefix in cfg()["measurement"]["prefixes"]:
        df=ft[ft.prefix==prefix] if len(ft) else ft
        a=df[(df.panel=="A")&(df["repeat"]==0)].set_index("run") if len(df) else pd.DataFrame(columns=feature_names+["valid"])
        b=df[(df.panel=="B")&(df["repeat"]==0)].set_index("run") if len(df) else a.copy()
        steps=[s/schedule["horizon"] for s in schedule["steps"] if s<=round(prefix*schedule["horizon"])]
        for name in feature_names:
            floor=base_floor(name)*(2/(steps[-1]-steps[0]) if name.endswith("_slope") else 1)
            repeated=[]
            if len(df):
                for rep in sorted(df["repeat"].unique()):
                    if rep==0: continue
                    ar=df[(df.panel=="A")&(df["repeat"]==rep)].set_index("run")
                    common=a.index.intersection(ar.index)
                    repeated.extend((a.loc[common,name]-ar.loc[common,name]).tolist())
            repdiff=np.asarray(repeated,dtype=float)
            repeat_invalid=bool(len(repdiff) and not np.isfinite(repdiff).all())
            repeat_noise=float(np.sqrt(np.mean(repdiff**2))/math.sqrt(2)) if len(repdiff) and not repeat_invalid else 0.
            floor=max(floor,repeat_noise)
            actual=set(a.index)|set(b.index)
            valid=bool(len(expected)>=2 and set(a.index)==expected and set(b.index)==expected and not a.index.has_duplicates and not b.index.has_duplicates
                       and a.valid.all() and b.valid.all() and not repeat_invalid)
            av=a.reindex(sorted(expected))[name].to_numpy(dtype=float)
            bv=b.reindex(sorted(expected))[name].to_numpy(dtype=float)
            valid=bool(valid and np.isfinite(av).all() and np.isfinite(bv).all())
            noise=spread=ratio=screen=None; status="invalid/incomplete planned cohort" if len(expected)>=2 else "undefined: fewer than two eligible runs"
            if valid:
                noise=float(np.sqrt(np.mean((av-bv)**2))/math.sqrt(2))
                spread=float(np.std((av+bv)/2,ddof=1))
                ratio=spread/noise if noise>0 else None
                screen=spread/max(noise,floor)
                status="flat / below numerical floor" if spread<=floor else ("zero sampling noise; floor-limited" if noise==0 else "finite")
            records.append({"prefix":prefix,"feature":name,"expected_n":len(expected),"actual_n":len(actual),"expected_runs":";".join(sorted(expected)),
                            "missing_runs":";".join(sorted(expected-actual)),"unexpected_runs":";".join(sorted(actual-expected)),"valid":valid,
                            "noise":noise,"spread":spread,"raw_ratio":ratio,"numerical_repeat_noise":repeat_noise,"numerical_floor":floor,
                            "screen_ratio":screen,"status":status,"passes_screen":bool(valid and spread>floor and screen>=cfg()["measurement"]["ratio_screen"])})
    return pd.DataFrame(records)

def error_decomposition(primary,diagnostic):
    ids=[n for n in ("realisation","condition","seed","run","partition","budget","panel","repeat","prefix") if n in primary]
    together=primary.merge(diagnostic,on=ids,suffixes=("","_diag"),validate="one_to_one")
    rows=[]
    groups=[n for n in ("partition","budget","prefix") if n in together]
    for key,df in together[together["repeat"]==0].groupby(groups):
        key=(key,) if not isinstance(key,tuple) else key
        for summary in SUMMARIES:
            a=df[df.panel=="A"].set_index("run"); b=df[df.panel=="B"].set_index("run")
            runs=sorted(set(a.index)|set(b.index)); a=a.reindex(runs); b=b.reindex(runs)
            ed=(a["deep_"+summary]-b["deep_"+summary]).to_numpy(dtype=float)
            ec=(a["conv3_"+summary]-b["conv3_"+summary]).to_numpy(dtype=float)
            eh=(a["head_"+summary]-b["head_"+summary]).to_numpy(dtype=float)
            additive=summary!="std"
            identity_error=float(np.max(abs(ed-ec-eh))) if np.isfinite(ed).all() else None
            mse_d=float(np.mean(ed**2)); mse_c=float(np.mean(ec**2)); mse_h=float(np.mean(eh**2)); cross=float(2*np.mean(ec*eh))
            rows.append({**dict(zip(groups,key)),"summary":summary,"n":len(runs),"additivity_expected":additive,
                         "max_error_identity_discrepancy":identity_error,"mse_deep":mse_d,"mse_conv3":mse_c,"mse_head":mse_h,
                         "uncentred_cross_term_2mean_error_product":cross,"mse_decomposition_discrepancy":mse_d-mse_c-mse_h-cross,
                         "opposite_sign_error_fraction":float(np.mean(ec*eh<0)),"cancellation_fraction_of_component_mse":-cross/(mse_c+mse_h) if mse_c+mse_h>0 else None,
                         "note":"SD is independently computed; identities and decomposition are not asserted" if not additive else "uncentred MSE cross term, not covariance"})
            if additive and np.isfinite(ed).all():
                assert identity_error<1e-9
                assert abs(mse_d-mse_c-mse_h-cross)<1e-9
    # Additivity of actual A/B additive summaries, not only their errors.
    errors={s:float((together["deep_"+s]-together["conv3_"+s]-together["head_"+s]).abs().max()) for s in ("last","mean","slope")}
    assert max(errors.values())<1e-9
    return pd.DataFrame(rows),errors
