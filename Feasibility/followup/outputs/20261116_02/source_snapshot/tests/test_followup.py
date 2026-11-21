import copy
import importlib.util
import pytest
import pandas as pd
from ggfollow.core import *
from ggfollow import metrics
from ggfollow.partitions import partition,validate_matched,indices

@pytest.fixture(scope="module")
def old_data():
    a=torch.load(legacy("data")/"independent.pt",map_location="cpu",weights_only=False)
    b=torch.load(legacy("data")/"correlated.pt",map_location="cpu",weights_only=False)
    return a,b

@pytest.fixture
def legacy_checks(monkeypatch):
    spec=importlib.util.spec_from_file_location("read_only_legacy_tests",ROOT/"tests/test_correctness.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def redirected(name,value):
        module.RESULTS[name]=value
        write_json(out()/"checks/legacy_reused_metrics.json",module.RESULTS)
    monkeypatch.setattr(module,"record",redirected)
    return module

@pytest.mark.parametrize("state,expected,shares",[("A",[1,1,1,1],[.25,.25,.5]),("B",[1,1.5,.75,.75],[.25,.375,.375])])
def test_legacy_analytical_readonly(legacy_checks,state,expected,shares): legacy_checks.test_analytical(state,expected,shares)
def test_legacy_gradcheck_readonly(legacy_checks): legacy_checks.test_gates_and_signed_distinction()
@pytest.mark.parametrize("dtype",[torch.float64,torch.float32])
def test_legacy_microbatch_readonly(legacy_checks,dtype): legacy_checks.test_microbatch(dtype)
def test_legacy_rescale_readonly(legacy_checks): legacy_checks.test_restricted_rescaling()
@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_legacy_isolation_readonly(legacy_checks,dev): legacy_checks.test_repeated_isolation(dev)
@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_legacy_training_neutrality_readonly(legacy_checks,dev): legacy_checks.test_training_neutrality(dev)

def test_original_arithmetic_exact():
    from ggfollow.measuring import reproduce_arithmetic
    result=reproduce_arithmetic()
    assert abs(result["ratio"]-3.0501409990602664)<1e-12
    assert result["maximum_feature_formula_error"]<1e-12

def test_protected_artifacts_unchanged():
    from ggfollow.artifacts import protect
    assert protect(after=True)>=294

def test_partitions_disjoint_nested_reproducible_and_matched(old_data):
    a,b=old_data; validate_matched(a,b)
    frame=partition(a,41000); frame2=partition(a,41000); other=partition(a,41001)
    pd.testing.assert_frame_equal(frame,frame2)
    assert not frame.equals(other)
    pd.testing.assert_frame_equal(frame,partition(b,41000))
    for panel in ("A","B"):
        ix=[indices(a,panel,n,frame) for n in (64,128,256)]
        assert [len(i) for i in ix]==[640,1280,2560]
        assert set(ix[0].tolist())<set(ix[1].tolist())<set(ix[2].tolist())
        assert all(a["roles"][i] in ("probe_A","probe_B") for i in ix[2])
    assert not set(frame.loc[frame.panel=="A","original_id"])&set(frame.loc[frame.panel=="B","original_id"])
    changed=dict(a); changed["background"]=torch.zeros_like(a["background"])
    pd.testing.assert_frame_equal(frame,partition(changed,41000))

def synthetic_table():
    from ggfollow.measuring import original_schedule
    schedule=original_schedule(); rows=[]
    for run in ("a","b"):
        for panel in ("A","B"):
            for i,step in enumerate(schedule["steps"]):
                conv3=.1+.015*i+(.03 if panel=="A" else -.03)*(1 if run=="a" else -1)
                head=.7-conv3
                rows.append({"run":run,"panel":panel,"repeat":0,"partition":1,"budget":256,"step":step,
                             "fraction":step/schedule["horizon"],"valid":True,"shallow_share":.1,"deep_share":conv3+head,
                             "conv3_share":conv3,"head_share":head})
    return pd.DataFrame(rows),schedule

def test_additive_summaries_sd_independent_and_uncentred_cross_term():
    table,s=synthetic_table()
    # Vary head nonlinearly so SD is demonstrably independent of summation.
    table.loc[table.step==s["steps"][2],"head_share"]+=.07
    table["deep_share"]=table.conv3_share+table.head_share
    primary=metrics.features(table,s)
    diag=metrics.features(table,s,{"conv3":"conv3_share","head":"head_share"})
    error,summary_errors=metrics.error_decomposition(primary,diag)
    assert max(summary_errors.values())<1e-9
    merged=primary.merge(diag,on=["run","panel","repeat","partition","budget","prefix"],suffixes=("","_diag"))
    assert ((merged.deep_std-merged.conv3_std-merged.head_std).abs()>1e-4).any()
    additive=error[error.additivity_expected]
    assert additive.mse_decomposition_discrepancy.abs().max()<1e-9
    assert (additive.uncentred_cross_term_2mean_error_product<0).any()
    assert not error[error.summary=="std"].additivity_expected.any()

def test_common_denominator_and_budget_specific_flags():
    from measurement import summarize
    from model import CNNA
    seed_all(61); m=CNNA(); g={n:torch.ones_like(p) for n,p in m.named_parameters()}
    r=metrics.enrich(summarize(m,g))
    assert abs(r["deep_share"]-r["conv3_share"]-r["head_share"])<1e-9
    assert abs(r["conv1_share"]-r["conv1_conv_share"])>1e-6
    first=dict(r); first["head_fraction_of_deep"]=.51
    second=dict(r); second["head_fraction_of_deep"]=.49
    assert metrics.enrich(first)["head_flag"] and not metrics.enrich(second)["head_flag"]
    first["bias_sensitivity"]=r["total"]*.11; second["bias_sensitivity"]=r["total"]*.09
    assert metrics.enrich(first)["bias_flag"] and not metrics.enrich(second)["bias_flag"]

def test_zero_floor_invalid_incomplete_variable_cohorts():
    from ggfollow.measuring import original_schedule
    s=original_schedule(); feature_names=["deep_last","deep_slope"]
    rows=[]
    for run,v in (("a",.1),("b",.5),("c",.9)):
        for panel in ("A","B"):
            for prefix in (.05,.1): rows.append({"run":run,"panel":panel,"repeat":0,"prefix":prefix,"valid":True,"deep_last":v,"deep_slope":v})
    df=pd.DataFrame(rows)
    r=metrics.reliability(df,s,["a","b","c"],feature_names)
    assert r.valid.all() and r.passes_screen.all() and r.noise.eq(0).all() and r.raw_ratio.isna().all()
    assert metrics.reliability(df[df.run!="c"],s,["a","b","c"],feature_names).valid.eq(False).all()
    assert metrics.reliability(df[df.run!="c"],s,["a","b"],feature_names).valid.all()
    invalid=df.copy(); invalid.loc[0,"valid"]=False
    assert not metrics.reliability(invalid,s,["a","b","c"],feature_names).valid.all()
    flat=df.copy(); flat[feature_names]=.1
    assert not metrics.reliability(flat,s,["a","b","c"],feature_names).passes_screen.any()
    assert not metrics.reliability(df[df.run=="a"],s,["a"],feature_names).valid.any()
    assert not metrics.reliability(df.iloc[:0],s,[],feature_names).valid.any()
    repeated=df[df.panel=="A"].copy(); repeated["repeat"]=1; repeated["deep_last"]+=.2
    rn=metrics.reliability(pd.concat([df,repeated]),s,["a","b","c"],feature_names)
    assert (rn[rn.feature=="deep_last"].numerical_floor>.1).all()

def test_gradient_recomputation_and_absolute_order(tmp_path,legacy_checks,monkeypatch):
    from ggfollow.measuring import ScalarCache,measured
    import ggfollow.measuring as mm
    import measurement as lm
    m,x,_=legacy_checks.fixture_model("A"); x=x.repeat(2,1,1,1); y=torch.tensor([0,1])
    whole,_,_=lm.mean_gradients(m,x,y,1)
    one,_,_=lm.mean_gradients(m,x[:1],y[:1],1); two,_,_=lm.mean_gradients(m,x[1:],y[1:],1)
    expected=lm.summarize(m,whole)["head_abs"]
    wrong=(lm.summarize(m,one)["head_abs"]+lm.summarize(m,two)["head_abs"])/2
    assert abs(expected-wrong)>.01
    calls=[]; original=lm.mean_gradients
    def counted(*a,**kw): calls.append(1); return original(*a,**kw)
    monkeypatch.setattr(lm,"mean_gradients",counted)
    monkeypatch.setattr(mm,"out",lambda:tmp_path)
    cache=ScalarCache("unit_cache"); data={"x":x,"y":y}
    measured(cache,m,data,torch.tensor([0]),{"panel":"A","identity":"first"})
    measured(cache,m,data,torch.tensor([1]),{"panel":"B","identity":"second"})
    measured(cache,m,data,torch.tensor([0]),{"panel":"A","identity":"first"})
    assert len(calls)==2 and len(cache.frame())==2
    cache.close()

def test_calibration_selection_health_only():
    from ggfollow.training import select_horizon
    records=[{"seed":seed,"actual_epochs":epoch,"health_accuracy":.79 if epoch==2 else .85,"health_loss":.6,
              "required_telemetry_finite":True,"finite_weights":True,"feature_ratio":999 if epoch==2 else 0}
             for seed in (101,102,103) for epoch in (2,4,6)]
    selected,_=select_horizon(records); assert selected==40
    for r in records: r["health_accuracy"]=.79
    assert select_horizon(records)[0] is None
    for r in records: r["health_accuracy"]=.99; r["required_telemetry_finite"]=False
    assert select_horizon(records)[0] is None

@pytest.mark.parametrize("dev",["cpu","cuda"])
def test_new_runner_checkpoint_resume(tmp_path,monkeypatch,dev):
    import ggfollow.training as tr
    from model import CNNA
    if dev=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    monkeypatch.setattr(tr,"device",lambda:torch.device(dev))
    monkeypatch.setattr(tr,"out",lambda:tmp_path)
    seed_all(81)
    d={"x":torch.rand(48,3,28,28),"y":torch.arange(48)%10,"roles":["train"]*32+["health"]*16}
    plan={"updates_per_epoch":1,"endpoint":5,"steps":[1,2,3,4,5],"horizon":50}
    full=tr.train_run(d,"checks_training","full",9981,plan,[3,5],False,"plan","synthetic")
    original=tr.checkpoint
    def interrupt(path,*args,**kw):
        original(path,*args,**kw)
        if str(path).endswith("step_0003.pt"): raise RuntimeError("simulated interruption")
    monkeypatch.setattr(tr,"checkpoint",interrupt)
    with pytest.raises(RuntimeError,match="simulated interruption"): tr.train_run(d,"checks_training","resumed",9981,plan,[3,5],False,"plan","synthetic")
    monkeypatch.setattr(tr,"checkpoint",original)
    resumed=tr.train_run(d,"checks_training","resumed",9981,plan,[3,5],False,"plan","synthetic")
    a=torch.load(Path(full["checkpoint_dir"])/"step_0005.pt",weights_only=False,map_location="cpu")
    b=torch.load(Path(resumed["checkpoint_dir"])/"step_0005.pt",weights_only=False,map_location="cpu")
    assert all(torch.equal(a["model"][k],b["model"][k]) for k in a["model"])
    for k in a["optimizer"]["state"]:
        assert all(torch.equal(a["optimizer"]["state"][k][n],b["optimizer"]["state"][k][n]) for n in a["optimizer"]["state"][k])

def test_no_official_validation_test_instantiation(tmp_path,monkeypatch):
    import ggfollow.datasets as ds
    import spuco.datasets as official
    seen=[]
    class StopConstructor(Exception): pass
    def guarded(**kwargs): seen.append(kwargs); raise StopConstructor()
    monkeypatch.setattr(official,"SpuCoMNIST",guarded); monkeypatch.setattr(ds,"out",lambda:tmp_path)
    with pytest.raises(StopConstructor): ds.build("unit",3701,4903,{"independent":.1})
    assert len(seen)==1 and seen[0]["split"]=="train" and seen[0]["download"] is False
