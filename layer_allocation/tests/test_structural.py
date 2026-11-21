"""Synthetic engineering fixtures, never scientific observations."""
import sys,pickle,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from core import *
from structural_measurement import structural,observe
from predictors import inputs,ranking,discovery,pipe,folds
from analysis_report import curve
import pytest

@pytest.fixture(autouse=True)
def deterministic_fixture():
    deterministic();state=rng_state();torch.manual_seed(41)
    yield
    restore_rng(state)

def test_norm_bound_and_coordinate_pairing():
    model=CNNA().double();g={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    with torch.no_grad():
        model.conv1.weight.zero_();model.conv1.weight.flatten()[:2]=torch.tensor([2.,1.])
    g['conv1.weight'].flatten()[:2]=torch.tensor([2.,1.]);a=structural(model,g)
    g['conv1.weight'].flatten()[:2]=torch.tensor([1.,2.]);b=structural(model,g)
    assert a['conv1_U']==b['conv1_U']
    assert a['conv1_co_location']>b['conv1_co_location']
    assert a['conv1_U']*a['conv1_co_location']==pytest.approx(5.)
    assert b['conv1_U']*b['conv1_co_location']==pytest.approx(4.)

def test_population_marginal_and_quantile():
    model=CNNA().double();g={n:torch.zeros_like(p) for n,p in model.named_parameters()};r=structural(model,g)
    assert r['conv2_g_std']==0 and r['conv2_co_location'] is None
    assert r['proxy_shallow']==0 and r['proxy_deep']==0
    w=model.head.weight.detach().numpy().ravel()
    assert r['head_w_absq90']==pytest.approx(np.quantile(abs(w),.9))
    assert r['head_w_std']==pytest.approx(w.std(ddof=0))

def test_microbatch_accumulation_and_isolation():
    model=CNNA().double();model.train();x=torch.rand(7,3,28,28,dtype=torch.float64);y=torch.arange(7)
    for p in model.parameters():p.grad=torch.ones_like(p)
    before={n:p.clone() for n,p in model.state_dict().items()};state=rng_state()
    a,_,_=mean_gradients(model,x,y,7);b,_,_=mean_gradients(model,x,y,3)
    for n in a:torch.testing.assert_close(a[n],b[n],atol=1e-9,rtol=1e-7)
    after=rng_state()
    assert model.training and after['python']==state['python']
    assert np.array_equal(after['numpy'][1],state['numpy'][1]) and after['numpy'][2:]==state['numpy'][2:]
    assert torch.equal(after['torch'],state['torch'])
    assert all(torch.equal(a,b) for a,b in zip(after['cuda'],state['cuda']))
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())

def test_repeated_summaries():
    model=CNNA().double();x=torch.rand(5,3,28,28,dtype=torch.float64);y=torch.arange(5)
    a=observe(model,x,y);b=observe(model,x,y)
    for k in ['total','shallow_share','deep_share','conv1_w_l2','head_g_std']:assert a[k]==b[k]

def test_schema_and_nested_specificity_pair():
    base=['lr','loss_last'];m=inputs(base,'M');z=inputs(base,'MZ')
    assert z==m+six() and len(set(z))==len(z)
    assert len(z)-len(m)==6 and not any(c.endswith('_slope') for c in six())
    assert inputs(base,'L2')==inputs(base,'B')+[c for c in six() if c.endswith('_last')]
    assert inputs(base,'NZ')==inputs(base,'N')+six()
    assert not any(c in z for c in ['family','run','wga','alignment','difficulty','eligible'])
    with pytest.raises(AssertionError):inputs(['wga'],'M')

def test_summary_order_free_and_population():
    a=summaries([1,3,2],[.01,.02,.1]);b=summaries([3,1,2],[.01,.02,.1])
    assert a==b and a['std']==pytest.approx(np.std([1,3,2],ddof=0))
    with pytest.raises(ValueError):summaries([float('nan')],[.1])
    with pytest.raises(ValueError):summaries([],[])

@pytest.mark.parametrize('y',[[0,1,0,1],[0,0,0],[1,1,1]])
def test_curve_bounds_ties_and_full_recovery(y):
    ids=[str(i) for i in range(len(y))];risk=np.ones(len(y));rows=curve(y,risk,ids)
    assert rows[0]['precision'] is None and rows[-1]['discoveries']==sum(y)
    assert all(0<=r['discoveries']<=min(r['k'],sum(y)) for r in rows)
    assert all(a['discoveries']<=b['discoveries'] for a,b in zip(rows,rows[1:]))
    for r in rows:assert r['random_expectation']==pytest.approx(r['k']*sum(y)/len(y))
    assert ranking([.5,.5],['b','a']).tolist()==[1,0]

def test_invalid_inputs():
    with pytest.raises(ValueError):ranking([float('nan')],['a'])
    with pytest.raises(ValueError):curve([],[],[])
    with pytest.raises(ValueError):discovery([1],[.5],['a'],2)

def test_fold_local_preprocessing():
    m=pipe('logistic',1);x=pd.DataFrame({'a':[0,1,2,3],'b':[0,1,0,1]});m.fit(x,[0,0,1,1])
    before=m[1].mean_.copy();m.predict_proba(pd.DataFrame({'a':[1e6],'b':[1e6]}));assert np.array_equal(m[1].mean_,before)

def test_group_folds_and_no_family_splitting():
    df=pd.DataFrame({'family':[0,0,1,1,2,2],'failure':[0,1,0,1,0,1]})
    f=[{'train_families':[b for b in [0,1,2] if b!=a],'validation_families':[a]} for a in [0,1,2]]
    for a,b in folds(df,f):assert not set(df.family.iloc[a])&set(df.family.iloc[b])
    assert int((36-4)/3)==10

def test_legacy_checkpoint_schedule_and_manifest():
    protocol=read(OLD/'protocol.json');assert len(protocol['runs'])==96
    assert len({r['family'] for r in protocol['runs']})==24
    for spec in protocol['runs']:
        plan=read(OLD/'classifiers'/spec['run']/'plan.json')
        assert plan['steps']==[196,391,587,978,1467,1956]
        assert read(OLD/'classifiers'/spec['run']/'complete.json')['full_60_epochs']

def test_original_identity_reservations_and_counts():
    for p in (OLD/'data').glob('*/compact.npz'):
        with np.load(p) as d:
            sets=[set(d['ids'][d['roles']==r]) for r in range(4)]
            assert all(not sets[a]&sets[b] for a in range(4) for b in range(a))
            for role,count in [(1,256),(2,256),(3,128)]:assert np.bincount(d['y'][d['roles']==role],minlength=10).tolist()==[count]*10

def test_checkpoint_loading_determinism():
    p=OLD/'classifiers/f00_c0_s0/early_196.pt';h=sha(p);a=CNNA();b=CNNA()
    a.load_state_dict(torch.load(p,weights_only=True));b.load_state_dict(torch.load(p,weights_only=True))
    x=torch.rand(3,3,28,28);assert torch.equal(a(x),b(x)) and sha(p)==h

def test_resume_identity_and_analysis_boundary():
    code=(ROOT/'src/structural_measurement.py').read_text();entry=(ROOT/'src/predictors.py').read_text()
    assert 'checkpoint_sha256' in code and 'protocol_sha256' in code
    assert "labels=outcomes('development')" in entry.split('def fit():')[1].split('def predict():')[0]
    assert 'outcomes(' not in entry.split('def predict():')[1].split('def reproduce():')[0]
    assert 'torch.optim' not in code and 'optimizer.step' not in code
