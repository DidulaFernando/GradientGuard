import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import *
import pytest,shutil
from prediction import folds,inputs,ladder,feature_frame
from report import audit

def test_group_folds():
    df=pd.DataFrame({'family':np.repeat(np.arange(14),4),'run':[str(i) for i in range(56)]})
    y=pd.Series(np.repeat(np.arange(14)%2,4));fs=folds(df,y);assert fs is not None
    for a,b in fs:assert not set(df.family.iloc[a])&set(df.family.iloc[b]) and len(set(y.iloc[a]))==2

def test_unseal_gate(tmp_path,monkeypatch):
    import study
    monkeypatch.setattr(study,'O',tmp_path)
    with pytest.raises(AssertionError):study.outcomes({},'test')
    assert not (tmp_path/'outcome_panels').exists()

def test_terminal_edits_do_not_change_early_inputs(tmp_path,monkeypatch):
    import prediction
    protocol=read(O/'protocol.json');spec=next(r for r in protocol['runs'] if (O/'classifiers'/r['run']/'complete.json').exists())
    folder=tmp_path/'classifiers'/spec['run'];folder.mkdir(parents=True)
    for path in (O/'classifiers'/spec['run']).glob('*.json'):shutil.copy2(path,folder/path.name)
    shutil.copy2(O/'classifiers'/spec['run']/'early.csv',folder/'early.csv')
    write_json(tmp_path/'protocol.json',{'runs':[spec]});monkeypatch.setattr(prediction,'O',tmp_path)
    a=prediction.feature_frame(spec['split']);r=read(folder/'complete.json');r['final_health']={'loss':999,'accuracy':0};write_json(folder/'complete.json',r)
    b=prediction.feature_frame(spec['split']);cols=ladder(inputs(True,True),'L8');pd.testing.assert_frame_equal(a[cols],b[cols])

def test_final_updates_are_real():
    paths=list((O/'classifiers').glob('*/complete.json'));assert paths
    for path in paths:
        r=read(path);assert r['updates']==r['planned_updates']==19560 and r['full_60_epochs']
        cp=torch.load(path.parent/'recovery.pt',weights_only=False,map_location='cpu')
        assert cp['step']==19560 and all(float(s['step'])==19560 for s in cp['optimizer']['state'].values())

def test_resume_numerical_primitive(tmp_path):
    from model import CNNA
    seed_all(456);a=CNNA();b=CNNA();b.load_state_dict(a.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False);ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
    x=torch.rand(16,3,28,28);y=torch.arange(16)%10
    def step(m,o):o.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(m(x),y).backward();o.step()
    for _ in range(5):step(a,oa)
    for _ in range(3):step(b,ob)
    save_torch(tmp_path/'cp.pt',{'model':b.state_dict(),'optimizer':ob.state_dict(),'rng':rng_state()})
    cp=torch.load(tmp_path/'cp.pt',weights_only=False);c=CNNA();c.load_state_dict(cp['model']);oc=torch.optim.Adam(c.parameters(),lr=.001,foreach=False,fused=False);oc.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    for _ in range(2):step(c,oc)
    assert all(torch.equal(a.state_dict()[n],v) for n,v in c.state_dict().items())

def test_single_class_and_budget():
    d=pd.DataFrame({'run':[str(i) for i in range(7)],'risk':np.linspace(.1,.9,7),'failure':[0]*7});r=audit(d)
    assert r['k']==1 and r['AUROC'] is None and r['recall'] is None and r['headroom']==0

def test_official_identity_separation():
    from spuco.datasets.spuco_mnist_config import config
    val=set(config['val'])
    for path in (O/'data').glob('*/compact.npz'):
        d=np.load(path);ids=d['ids'];assert not set(ids)&val
        for a,b in itertools.combinations(range(4),2):assert not set(ids[d['roles']==a])&set(ids[d['roles']==b])
