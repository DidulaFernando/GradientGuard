import sys,copy,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ggfresh import core
from ggfresh.core import *
from ggfresh.measure_data import structural,observe,coordinate_indices,SpuCoMNIST,SpuriousFeatureDifficulty
from ggfresh.prediction import inputs,six,conventional,valid_folds,support,pipe,discovery,allstats
from ggfresh.reporting import curves,family_draw
from ggfresh.training import step_batch
import pytest

@pytest.fixture(autouse=True)
def deterministic_test():
    deterministic();state=rng_state();seed_all(731101)
    yield
    restore_rng(state)

def test_manifest_and_nonoverlap():
    f,r,s=family_design();ff,rr,ss=family_design();assert (f,r,s)==(ff,rr,ss)
    assert len(f)==64 and len(r)==256 and len({v['training_seed'] for v in r})==256
    old={(v['alignment'],v['difficulty'],v['lr']) for v in read(OLD/'protocol.json')['families']}
    assert not old&{(v['alignment'],v['difficulty'],v['lr']) for v in f}
    for split in ['development','evaluation']:
        sub=[v for v in f if v['split']==split];assert len(sub)==32
        for key in ['alignment','difficulty','lr']:assert {v[key] for v in sub}=={v[key] for v in f}
    for family in f:assert len([v for v in r if v['family']==family['family']])==4

@pytest.mark.parametrize('difficulty,count',[('MAGNITUDE_LARGE',784),('MAGNITUDE_MEDIUM',16)])
def test_official_masks_and_foreground(difficulty,count):
    color=torch.tensor([.2,.4,.6]);mask=SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,difficulty),color)
    gray=torch.zeros(3,28,28);gray[:,1:20,1:20]=.5;x=gray+mask*(gray==0)
    assert int((mask[0]!=0).sum())==count
    assert torch.equal(x[gray!=0],gray[gray!=0]) and torch.isfinite(x).all()

@pytest.mark.parametrize('state,expected',[('A',[1,1,1,1]),('B',[1,1.5,.75,.75])])
def test_analytical_readonly_fixture(state,expected):
    src=(LEGACY/'tests/test_correctness.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='fixture_model')
    ns={'CNNA':CNNA,'torch':torch,'max_pool2d':torch.nn.functional.max_pool2d};exec(ast.get_source_segment(src,node),ns)
    m,x,y=ns['fixture_model'](state);z=m(x);r=torch.autograd.grad(cross_entropy(z,y),z)[0][0,0].item();g,_,_=mean_gradients(m,x,y,1);out=summarize(m,g)
    assert abs(z[0,0].item()-1)<1e-9
    np.testing.assert_allclose([out[l+'_abs']/abs(r) for l in ['conv1','conv2','conv3','head']],expected,atol=1e-9,rtol=1e-7)

def test_microbatch_and_rng_isolation():
    m=CNNA().double();m.train();x=torch.rand(7,3,28,28,dtype=torch.float64);y=torch.arange(7)
    state=rng_state();before={n:p.clone() for n,p in m.state_dict().items()};a,_,_=mean_gradients(m,x,y,7);b,_,_=mean_gradients(m,x,y,3)
    for n in a:torch.testing.assert_close(a[n],b[n],atol=1e-9,rtol=1e-7)
    after=rng_state();assert state['python']==after['python'] and np.array_equal(state['numpy'][1],after['numpy'][1]) and torch.equal(state['torch'],after['torch'])
    assert all(torch.equal(a,b) for a,b in zip(state['cuda'],after['cuda'])) and m.training
    assert all(torch.equal(v,before[n]) for n,v in m.state_dict().items())

def test_single_pass_observe_matches_original_sensitivity():
    m=CNNA();x=torch.rand(40,3,28,28);y=torch.arange(40)%10;coords=coordinate_indices(m)
    norm=sum(p.double().square().sum().item() for n,p in m.named_parameters() if n.endswith('weight'))**.5
    a=observe(m,x,y,norm,coords,True);g,_,_=mean_gradients(m,x,y,128);b=summarize(m,g)
    for k in ['total','shallow_share','deep_share','bias_norm','bias_sensitivity']:assert a[k]==pytest.approx(b[k],abs=1e-6,rel=1e-4)
    assert np.isfinite(a['saturation']) and a['conv1_w_l2']>0

def test_norm_product_and_marginal_pairing():
    m=CNNA().double();g={n:torch.zeros_like(p) for n,p in m.named_parameters()}
    with torch.no_grad():m.conv1.weight.zero_();m.conv1.weight.flatten()[:2]=torch.tensor([2.,1.])
    g['conv1.weight'].flatten()[:2]=torch.tensor([2.,1.]);a=structural(m,g)
    g['conv1.weight'].flatten()[:2]=torch.tensor([1.,2.]);b=structural(m,g)
    assert a['conv1_U']==b['conv1_U'] and a['conv1_g_absq90']==b['conv1_g_absq90']
    assert a['conv1_co_location']>b['conv1_co_location']

def test_columns_and_matched_banks():
    b=conventional();assert len(b)==38
    for rich in [False,True]:
        for sep in [False,True]:
            base=conventional(rich,sep);m=inputs(base,'M');mz=inputs(base,'MZ');assert mz==m+six() and len(set(mz))==len(mz)
            assert inputs(base,'NZ')==inputs(base,'N')+six()
            assert not any(c.endswith('_slope') for c in six())
    with pytest.raises(AssertionError):inputs(['wga'],'M')

def test_summary_population_and_actual_fraction():
    x=np.array([.01,.03,.1]);v=2*x+1;r=allstats(v,x)
    assert r['slope']==pytest.approx(2) and r['std']==pytest.approx(v.std(ddof=0))

def test_fold_local_preprocessing():
    m=pipe('logistic',1);x=pd.DataFrame({'a':[0,1,2,3],'b':[0,1,0,1]});m.fit(x,[0,0,1,1]);before=m[1].mean_.copy();m.predict_proba(pd.DataFrame({'a':[1e8],'b':[1e8]}));assert np.array_equal(before,m[1].mean_)

def test_group_cv_and_support_rules():
    df=pd.DataFrame([{'family':f,'failure':label} for f in range(16) for label in ([0]*3 if f<8 else [1]*3)])
    assert support(df)['passed'];folds=valid_folds(df);assert folds is not None
    for a,b in folds:assert not set(df.family.iloc[a])&set(df.family.iloc[b])
    assert not support(df[df.family!=15])['passed']
    assert not support(df,evaluation=True)['passed']

def test_checkpoint_resume_and_probe_neutrality(tmp_path):
    a=CNNA();b=copy.deepcopy(a);oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False);ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
    x=torch.rand(8,3,28,28);y=torch.arange(8)
    step_batch(a,oa,x[:4],y[:4]);step_batch(b,ob,x[:4],y[:4]);mean_gradients(b,x,y,3)
    p=tmp_path/'synthetic_smoke.pt';torch.save({'model':b.state_dict(),'optimizer':ob.state_dict(),'rng':rng_state()},p)
    c=CNNA();oc=torch.optim.Adam(c.parameters(),lr=.001,foreach=False,fused=False);cp=torch.load(p,weights_only=False);c.load_state_dict(cp['model']);oc.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    step_batch(a,oa,x[4:],y[4:]);step_batch(c,oc,x[4:],y[4:])
    for n,v in a.state_dict().items():assert torch.equal(v,c.state_dict()[n])
    assert oa.state_dict()['param_groups']==oc.state_dict()['param_groups']

def test_evaluation_guards_before_files_accessed(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'O',tmp_path);monkeypatch.setattr(core,'require',lambda:{})
    with pytest.raises(FileNotFoundError):allow_evaluation_training()
    write_json(tmp_path/'readiness.json',{'passed':False})
    with pytest.raises(AssertionError):allow_evaluation_training()
    write_json(tmp_path/'readiness.json',{'passed':True});(tmp_path/'pipelines.joblib').write_bytes(b'synthetic-fake-pipeline-no-unpickling')
    write_json(tmp_path/'predictor_freeze.json',{'pipelines_sha256':sha(tmp_path/'pipelines.joblib')});allow_evaluation_training()
    with pytest.raises(FileNotFoundError):allow_test_outcomes()
    (tmp_path/'evaluation_predictions.csv').write_text('synthetic')
    write_json(tmp_path/'prediction_freeze.json',{'predictor_freeze_sha256':sha(tmp_path/'predictor_freeze.json'),'predictions_sha256':sha(tmp_path/'evaluation_predictions.csv')});allow_test_outcomes()
    (tmp_path/'evaluation_predictions.csv').write_text('modified')
    with pytest.raises(AssertionError):allow_test_outcomes()

@pytest.mark.parametrize('y',[[0,1,0,1],[0,0,0],[1,1,1]])
def test_audit_curves_and_edge_cases(y):
    r=curves(y,np.ones(len(y)),[str(n) for n in range(len(y))]);assert r[0]['precision'] is None and r[-1]['discoveries']==sum(y)
    assert all(0<=v['discoveries']<=v['oracle'] for v in r)
    assert all(a['discoveries']<=b['discoveries'] for a,b in zip(r,r[1:]))
    assert all(v['random']==pytest.approx(v['k']*sum(y)/len(y)) for v in r)
    assert discovery([1,0],[.5,.5],['b','a'],1)==0
    with pytest.raises(ValueError):discovery([1],[float('nan')],['a'],1)

def test_family_draw_keeps_nested_models():
    df=pd.DataFrame({'run':['a','b','c','d'],'family':[1,1,2,2]});ix,ids=family_draw(df,[2,2,1]);assert ix==[2,3,2,3,0,1] and len(set(ids))==6
    assert len(df[df.family!=1])//3==0

def test_gate_requires_every_condition():
    flags={k:True for k in ['correctness','support','gain','interval','B','control','panels','influence']};assert gate(flags)
    for k in flags:assert not gate(flags|{k:False})
    assert not gate({})

def test_training_and_early_export_contracts():
    t=(HOME/'ggfresh/training.py').read_text();p=(HOME/'ggfresh/prediction.py').read_text();m=(HOME/'ggfresh/measure_data.py').read_text()
    assert "assert step==T" in t and "range(step//per,cfg()['horizon_epochs'])" in t
    frame_code=p.split('def feature_frame')[1].split('def valid_folds')[0]
    assert "['final_health']" not in frame_code and "['wga']" not in frame_code
    fit_code=p.split('def fit():')[1].split('def predict():')[0];assert 'outcomes/test' in fit_code and 'dev_labels()' in fit_code
    assert 'optimizer.step' not in m
