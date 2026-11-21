from .core import *
from .measure_data import source,observe,coordinate_indices,outcome_panel

def step_batch(model,optimizer,x,y):
    model.train();optimizer.zero_grad(set_to_none=True);loss=cross_entropy(model(x),y)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward();optimizer.step();return loss.detach()

def train(spec,d,budget):
    require();folder=O/'classifiers'/spec['run'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        c=read(folder/'complete.json');assert c['protocol_sha256']==sha(O/'protocol.json') and sha(folder/'final.pt')==c['final_sha256'];return
    if (folder/'diverged.json').exists():return
    seed_all(spec['training_seed']);model=CNNA().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=spec['lr'],betas=(.9,.999),weight_decay=0,foreach=False,fused=False)
    initial={n:p.detach().clone() for n,p in model.state_dict().items()};norm=sum(p.double().square().sum().item() for n,p in initial.items() if n.endswith('weight'))**.5
    coords=coordinate_indices(model);write_json(folder/'gradient_coordinates.json',{k:v.tolist() for k,v in coords.items()})
    ix=torch.tensor(np.flatnonzero(d['roles']==0));per=(len(ix)+127)//128;T=cfg()['horizon_epochs']*per;steps=[round(f*T) for f in cfg()['fractions']]
    plan={'horizon':T,'epochs':cfg()['horizon_epochs'],'updates_per_epoch':per,'steps':steps,'actual_fractions':[s/T for s in steps],'absolute_epochs':[s/per for s in steps],'train_n':len(ix),'batch':128,'drop_last':False,'data_hash':d['data_hash'],'protocol_sha256':sha(O/'protocol.json')}
    if (folder/'plan.json').exists():assert read(folder/'plan.json')==plan
    else:write_json(folder/'plan.json',plan)
    recovery=folder/'recovery.pt';step=0;rows=[];timing={'training':0.,'monitor_A':0.,'monitor_B':0.,'initialization':0.,'health':0.};wall=time.perf_counter()
    if recovery.exists():
        cp=torch.load(recovery,weights_only=False,map_location='cpu');assert cp['protocol']==sha(O/'protocol.json') and cp['source']==require()['source_hashes']
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_rng(cp['rng']);step=cp['step'];rows=cp['rows'];timing=cp['timing']
    else:
        save_torch(folder/'initial.pt',initial)
        for panel,role in [('A',1),('B',2)]:
            pi=torch.tensor(np.flatnonzero(d['roles']==role));r=observe(model,d['x'][pi],d['y'][pi],norm,coords,False);timing['initialization']+=r['seconds'];rows.append(r|{'step':0,'fraction':0.,'panel':panel})
        pd.DataFrame(rows).to_csv(folder/'early.csv',index=False)
    del initial
    x=d['x'].cuda();y=d['y'].cuda()
    try:
        for epoch in range(step//per,cfg()['horizon_epochs']):
            order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(spec['training_seed']*100000+epoch))]
            for within in range(step%per if epoch==step//per else 0,per):
                batch=order[within*128:(within+1)*128].cuda();sync('cuda');ts=time.perf_counter();step_batch(model,optimizer,x[batch],y[batch]);sync('cuda');timing['training']+=time.perf_counter()-ts;step=epoch*per+within+1
                if step in steps:
                    for panel,role in [('A',1),('B',2)]:
                        pi=torch.tensor(np.flatnonzero(d['roles']==role),device='cuda');r=observe(model,x[pi],y[pi],norm,coords,step==steps[0]);timing['monitor_'+panel]+=r['seconds'];rows.append(r|{'step':step,'fraction':step/T,'panel':panel})
                    hi=torch.tensor(np.flatnonzero(d['roles']==3),device='cuda');h,_=health(model,x[hi],y[hi]);timing['health']+=h['seconds'];write_json(folder/f'health_{step}.json',h)
                    save_torch(folder/f'early_{step}.pt',model.state_dict());pd.DataFrame(rows).to_csv(folder/'early.csv',index=False)
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite weights')
            save_torch(recovery,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'step':step,'rows':rows,'timing':timing,'protocol':sha(O/'protocol.json'),'source':require()['source_hashes']})
            budget.tick()
            if epoch%10==9:event('classifier_progress',run=spec['run'],epochs=epoch+1,total=60)
    except FloatingPointError as error:
        write_json(folder/'diverged.json',{'run':spec['run'],'step':step,'reason':str(error),'timing':timing});event('classifier_diverged',run=spec['run'],step=step);return
    assert step==T
    hi=torch.tensor(np.flatnonzero(d['roles']==3),device='cuda');final_health,_=health(model,x[hi],y[hi]);save_torch(folder/'final.pt',model.state_dict())
    write_json(folder/'complete.json',{'run':spec['run'],'updates':step,'planned_updates':T,'full_60_epochs':True,'timing':timing,'wall_this_invocation':time.perf_counter()-wall,'final_health':final_health,'protocol_sha256':sha(O/'protocol.json'),'final_sha256':sha(folder/'final.pt'),'peak_gpu_allocated':torch.cuda.max_memory_allocated()})
    if not (O/'representative_profile.json').exists():
        seconds=time.perf_counter()-wall;projected=seconds*256
        profile={'run':spec['run'],'full_60_epochs':True,'seconds':seconds,'timing':timing,'projected_256_classifier_seconds':projected,'storage_per_classifier_bytes':sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),'projection_rule':'First complete development trajectory extrapolated linearly, excludes construction/fitting/outcomes; conservative GPU-stage ledger enforced separately','within_24_hour_projection':projected<=cfg()['gpu_budget_seconds']}
        write_json(O/'representative_profile.json',profile)
        if not profile['within_24_hour_projection']:raise BudgetExceeded('Representative full-run projection exceeds24h; preserve and report, do not reduce design')
    event('classifier_complete',run=spec['run'],updates=step,seconds=round(time.perf_counter()-wall,2))

def evaluate(spec,split,panels=None):
    require()
    if split=='test':allow_test_outcomes()
    else:assert spec['split']=='development'
    folder=O/'outcomes'/split;p=folder/(spec['run']+'.json')
    if p.exists():return
    model_folder=O/'classifiers'/spec['run'];complete=read(model_folder/'complete.json');assert complete['full_60_epochs'] and complete['updates']==complete['planned_updates']
    x,y,bg=panels if panels is not None else outcome_panel(spec['difficulty'],split)
    model=CNNA().cuda();model.load_state_dict(torch.load(model_folder/'final.pt',weights_only=True,map_location='cuda'))
    ts=time.perf_counter();_,probs=health(model,x,y);correct=(probs.argmax(1)==y.cpu()).numpy();groups=[]
    for c,b in itertools.product(range(10),range(10)):
        mask=(y.cpu().numpy()==c)&(bg==b);n=int(mask.sum());k=int(correct[mask].sum());assert n>0
        ph=k/n;z=1.959963984540054;den=1+z*z/n;center=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        groups.append({'class':c,'background':b,'n':n,'correct':k,'accuracy':ph,'wilson_low':center-half,'wilson_high':center+half})
    wga=min(g['accuracy'] for g in groups)
    write_json(p,{'run':spec['run'],'family':spec['family'],'wga':wga,'failure':wga<.75,'aggregate_accuracy':float(correct.mean()),'groups':groups,'cutoff_near':abs(wga-.75)<=.05,'split':split,'seconds':time.perf_counter()-ts,'final_sha256':sha(model_folder/'final.pt')})

def execute(split):
    require();deterministic()
    if split=='evaluation':allow_evaluation_training()
    budget=GpuBudget();selected=[r for r in protocol()['runs'] if r['split']==split];outcome_cache={}
    for key,group in itertools.groupby(selected,key=lambda r:(r['family'],r['construction'])):
        specs=list(group)
        if all((O/'classifiers'/r['run']/'complete.json').exists() or (O/'classifiers'/r['run']/'diverged.json').exists() for r in specs):d=None
        else:d=source(specs[0]);budget.tick()
        for spec in specs:
            train(spec,d,budget)
            if split=='development' and (O/'classifiers'/spec['run']/'complete.json').exists():
                if spec['difficulty'] not in outcome_cache:outcome_cache[spec['difficulty']]=outcome_panel(spec['difficulty'],'val')
                evaluate(spec,'val',outcome_cache[spec['difficulty']]);budget.tick()
        del d;torch.cuda.empty_cache()
        complete=sum((O/'classifiers'/s['run']/'complete.json').exists() for s in selected)
        event('split_progress',split=split,completed=complete,planned=len(selected),gpu_stage_seconds=round(budget.total,1))
    budget.tick();event('training_split_finished',split=split)

def unseal():
    allow_test_outcomes();require();budget=GpuBudget()
    if not (O/'unseal.json').exists():write_json(O/'unseal.json',{'time_ns':time.time_ns(),'prediction_freeze_sha256':sha(O/'prediction_freeze.json'),'predictor_freeze_sha256':sha(O/'predictor_freeze.json')});event('test_outcomes_unsealed')
    panels={}
    for spec in protocol()['runs']:
        if spec['split']!='evaluation' or not (O/'classifiers'/spec['run']/'complete.json').exists():continue
        if spec['difficulty'] not in panels:panels[spec['difficulty']]=outcome_panel(spec['difficulty'],'test')
        evaluate(spec,'test',panels[spec['difficulty']]);budget.tick()
    event('test_outcomes_complete')
