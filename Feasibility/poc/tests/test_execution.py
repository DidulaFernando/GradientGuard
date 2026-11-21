"""Separately labelled synthetic execution/isolation fixtures; never study data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import *
import pytest


def test_new_observer_matches_original_gradient_definition():
    from study import observe, coordinate_indices
    from measurement import mean_gradients
    seed_all(73201)
    model = CNNA().double()
    x = torch.rand(30, 3, 28, 28, dtype=torch.float64)
    y = torch.arange(30) % 10
    optimizer = torch.optim.Adam(model.parameters())
    for p in model.parameters():
        p.grad = torch.full_like(p, .125)
    params = {n: p.clone() for n,p in model.state_dict().items()}
    gradients = {n:p.grad.clone() for n,p in model.named_parameters()}
    rng = rng_state()
    before = optimizer.state_dict()
    norm = sum(p.square().sum().item() for n,p in model.named_parameters() if n.endswith('weight'))**.5
    observed = observe(model,x,y,norm,coordinate_indices(model))
    reference,_,_ = mean_gradients(model,x,y,7)
    expected = summarize(model,reference)
    for key in ['shallow_share','middle_share','deep_share','total']:
        assert observed[key] == pytest.approx(expected[key],abs=1e-9,rel=1e-7)
    assert model.training
    assert all(torch.equal(params[n],p) for n,p in model.state_dict().items())
    assert all(torch.equal(gradients[n],p.grad) for n,p in model.named_parameters())
    assert torch.equal(rng['torch'],rng_state()['torch'])
    assert before == optimizer.state_dict()


def test_actual_runner_interrupted_resume(tmp_path, monkeypatch):
    """Run unchanged trainer on a labelled small fixture; interrupt at recovery."""
    import study
    deterministic()  # Match the guarded command's production entry point.
    monkeypatch.setattr(study,'O',tmp_path)
    monkeypatch.setattr(study,'event',lambda *a,**k:None)
    write_json(tmp_path/'protocol.json',{'fractions':[.01,.02,.03,.05,.075,.1]})
    write_json(tmp_path/'source_freeze.json',{'fixture':'synthetic-only'})
    seed_all(73202)
    count=1280+3*100
    d={'x':torch.rand(count,3,28,28),'y':torch.arange(count)%10,
       'roles':np.array([0]*1280+[1]*100+[2]*100+[3]*100)}
    study.train({'run':'uninterrupted','training_seed':73203,'lr':.001},d)
    original=study.save_torch
    def interrupt(path,value):
        original(path,value)
        if path.name=='recovery.pt' and value['step']==20:
            raise RuntimeError('intentional synthetic interruption')
    monkeypatch.setattr(study,'save_torch',interrupt)
    with pytest.raises(RuntimeError,match='intentional synthetic interruption'):
        study.train({'run':'resumed','training_seed':73203,'lr':.001},d)
    monkeypatch.setattr(study,'save_torch',original)
    study.train({'run':'resumed','training_seed':73203,'lr':.001},d)
    a=torch.load(tmp_path/'classifiers/uninterrupted/final.pt',weights_only=True)
    b=torch.load(tmp_path/'classifiers/resumed/final.pt',weights_only=True)
    assert all(torch.equal(a[n],b[n]) for n in a)
    left=pd.read_csv(tmp_path/'classifiers/uninterrupted/early.csv').drop(columns='seconds')
    right=pd.read_csv(tmp_path/'classifiers/resumed/early.csv').drop(columns='seconds')
    pd.testing.assert_frame_equal(left,right)
    assert read(tmp_path/'classifiers/resumed/complete.json')['updates']==600
