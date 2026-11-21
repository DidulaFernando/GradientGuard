from run import *
from prediction import stats,inputs,ladder,pipeline,support
from report import audit
import pytest
def test_manifest():
    p=read(O/'protocol.json');r=p['runs'];assert len(r)==96 and len({x['run'] for x in r})==96
    assert len({x['family'] for x in r if x['split']=='development'})==14
    assert not {x['family'] for x in r if x['split']=='development'}&{x['family'] for x in r if x['split']=='evaluation'}
    assert len({x['training_seed'] for x in r})==96
def test_feature_math():
    x=np.array([.01,.02,.03,.05,.075,.1]);v=2+3*x;s=stats(v,x)
    assert abs(s['slope']-3)<1e-12 and abs(s['std']-v.std(ddof=0))<1e-12 and s['reversals']==0
def test_inputs():
    b=inputs();assert set(ladder(b,'L2'))<set(ladder(b,'L6'))<set(ladder(b,'L8'))
    assert len(set(ladder(b,'L8'))-set(ladder(b,'B')))==8
    assert not set(b)&{'family','run','wga','failure','alignment','difficulty','data_seed','reservation_seed','final_loss'}
def test_fold_local():
    m=pipeline('logistic',1);x=pd.DataFrame({'a':[0.,1.,2.,3.]});m.fit(x,[0,0,1,1]);mean=m.named_steps['standardscaler'].mean_.copy();m.predict_proba(pd.DataFrame({'a':[1000.]}));assert np.array_equal(mean,m.named_steps['standardscaler'].mean_)
def test_audit_ties_and_ceiling():
    d=pd.DataFrame({'run':['a','b','c','d'],'risk':[.5]*4,'failure':[True,False,False,True]});r=audit(d);assert r['k']==1 and r['discoveries']==1 and r['oracle_ceiling']==1 and r['random_expected']==.5
def test_support_singleclass():
    d=pd.DataFrame({'family':np.arange(24)});assert not support(d,pd.Series([1]*24))
def test_gradient_and_isolation():
    from model import CNNA
    from measurement import mean_gradients,summarize
    seed_all(123);m=CNNA().double();x=torch.rand(7,3,28,28,dtype=torch.float64);y=torch.arange(7)
    before={n:p.clone() for n,p in m.state_dict().items()};rng=torch.get_rng_state().clone();a,_,_=mean_gradients(m,x,y,7);b,_,_=mean_gradients(m,x,y,3)
    for n in a:torch.testing.assert_close(a[n],b[n],atol=1e-9,rtol=1e-7)
    assert all(torch.equal(before[n],p) for n,p in m.state_dict().items()) and torch.equal(rng,torch.get_rng_state())
def test_mask_and_foreground():
    from spuco.datasets import SpuCoMNIST,SpuriousFeatureDifficulty
    for difficulty,count in [('MAGNITUDE_LARGE',784),('MAGNITUDE_MEDIUM',16)]:
        bg=SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,difficulty),[1.,1.,1.]);assert int((bg[0]!=0).sum())==count
        gray=torch.rand(3,28,28);gray[gray<.5]=0;x=gray+bg*(gray==0);assert torch.equal(x[gray!=0],gray[gray!=0])
def test_legacy_preserved():protect(True)
