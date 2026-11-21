"""Read-only engineering parity check of the first actual development checkpoint."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent/'fresh_validation'))
from ggfresh.core import *
from ggfresh.measure_data import MNIST,SpuCoMNIST,SpuriousFeatureDifficulty,structural

def audit():
    require();deterministic();start=time.perf_counter()
    spec=next(s for s in protocol()['runs'] if s['split']=='development')
    folder=O/'classifiers'/spec['run'];plan=read(folder/'plan.json');step=plan['steps'][-1];checkpoint=folder/f'early_{step}.pt';checkpoint_hash=sha(checkpoint)
    datafolder=O/'data'/f"f{spec['family']:02d}_c{spec['construction']}";path=datafolder/'compact.npz';assert sha(path)==read(datafolder/'verification.json')['sha256']
    with np.load(path) as z:d={k:z[k] for k in z.files}
    ix=np.flatnonzero(d['roles']==1);raw=MNIST(str(LEGACY/'data/mnist'),train=True,download=False)
    gray=raw.data[d['ids'][ix]].float().div(255).unsqueeze(1).expand(-1,3,-1,-1)
    masks=torch.stack([SpuCoMNIST.create_background(getattr(SpuriousFeatureDifficulty,spec['difficulty']),c) for c in d['palette']]);x=(gray+masks[d['bg'][ix]]*(gray==0)).cuda();y=torch.tensor(d['y'][ix],device='cuda')
    model=CNNA().cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'));state={n:p.clone() for n,p in model.state_dict().items()}
    a,_,_=mean_gradients(model,x,y,128);b,_,_=mean_gradients(model,x,y,127)
    maxg=max((a[n]-b[n]).abs().max().item() for n in a)
    for n in a:torch.testing.assert_close(a[n],b[n],atol=cfg()['atol'],rtol=cfg()['rtol'])
    old=summarize(model,a);unequal=summarize(model,b);record=pd.read_csv(folder/'early.csv');row=record[(record.panel=='A')&(record.step==step)].iloc[0]
    keys=['total','shallow_share','middle_share','deep_share','conv1_abs','conv2_abs','conv3_abs','head_abs','bias_norm','bias_sensitivity']
    errors={k:abs(float(old[k])-float(row[k])) for k in keys}
    for k in keys:assert np.isclose(old[k],row[k],atol=cfg()['atol'],rtol=cfg()['rtol'])
    qerr=max(abs(old[k]-unequal[k]) for k in ['shallow_share','middle_share','deep_share']);assert qerr<=cfg()['share_atol']
    structural_errors={k:abs(v-float(row[k])) for k,v in structural(model,a).items() if v is not None}
    assert max(structural_errors.values())<1e-6
    assert sha(checkpoint)==checkpoint_hash and all(torch.equal(v,state[n]) for n,v in model.state_dict().items())
    del a,b
    sync('cuda')
    result={'passed':True,'fixture':'First development run,10%checkpoint,panelA; selected by manifest order only','run':spec['run'],'checkpoint_sha256':checkpoint_hash,'recorded_measurement_max_error':max(errors.values()),'structural_max_error':max(structural_errors.values()),'unequal_microbatch_max_gradient_error':maxg,'unequal_microbatch_max_share_error':qerr,'atol':cfg()['atol'],'rtol':cfg()['rtol'],'share_atol':cfg()['share_atol'],'seconds':time.perf_counter()-start,'optimizer_constructed':False,'source_sha256':sha(Path(__file__))}
    write_json(O/'real_checkpoint_parity.json',result);print(result)
if __name__=='__main__':audit()
