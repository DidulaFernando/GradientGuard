from .core import *
import importlib.metadata as md,shutil,xml.etree.ElementTree as ET

def preflight():
    O.mkdir(parents=True,exist_ok=True)
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True);old=(OLD/'requirements-lock.txt').read_text(encoding='utf-8-sig')
    saved=read(OLD/'environment.json')
    r={'python':sys.version,'executable':sys.executable,'architecture':platform.architecture(),'os':platform.platform(),'lock_matches':set(lock.splitlines())==set(old.splitlines()),'missing':sorted(set(old.splitlines())-set(lock.splitlines())),'extra':sorted(set(lock.splitlines())-set(old.splitlines())),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'versions':{k:md.version(k) for k in ['torchvision','numpy','pandas','scikit-learn','joblib','matplotlib','spuco']},'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'settings':{k:v for k,v in os.environ.items() if k in ['TEMP','TMP','PYTHONPYCACHEPREFIX','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','MPLCONFIGDIR','PIP_CACHE_DIR']},'git_revision':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
    write_json(O/'environment.json',r);(O/'requirements-lock.txt').write_text(lock,encoding='utf8')
    assert r['lock_matches'] and r['python']==saved['python'] and r['torch']==saved['torch'] and r['cuda']==saved['cuda'] and r['gpu']==saved['gpu'],'Environment drift: inspect record before comparable execution'
    print(json.dumps({k:v for k,v in r.items() if k!='nvidia_smi'},indent=2));return r

def prior_facts():
    s=read(BASE/'reports/20261121_structural01/decision.json');assert s['original_reproduced'] and s['primary_exploratory_comparison']['D_added']==10 and s['primary_exploratory_comparison']['D_base']==9 and not s['independent_flags']['positive_increment_across_both_B_variants'] and not s['independent_flags']['pointwise_interval_excludes_zero']
    old=read(OLD/'reports/decision.json');assert old['completed_classifiers']==96 and old['source_normal_support']['N']==36 and old['source_normal_support']['failures']==12
    rows=pd.read_csv(OLD/'reports/metrics.csv');c=pd.read_csv(BASE/'runs/20261121_structural01/metrics.csv');sub=c[(c.cohort=='eligible')&(c.prefix==.1)&(c.variant=='A')].set_index('model')
    assert sub.loc['logistic_N','D12']==sub.loc['logistic_L6','D12']==10
    return {'old_96_complete':True,'old_source_normal_N':36,'old_failures':12,'original_nine_audit_tie':read(BASE/'runs/20261121_structural01/original_reproduction.json')['original_nine_audit_counts']==[8]*4,'structural_increment_panel_dependent':True,'sensitivity_specificity_unresolved':True}

def preserve():
    f=require();changed=[p for p,h in f['protected_inputs'].items() if not Path(p).exists() or sha(p)!=h]
    r={'protected_files':len(f['protected_inputs']),'unchanged':not changed,'changed_paths':changed};write_json(O/'preservation_audit.json',r);assert not changed,changed;return r

def freeze():
    if (O/'protocol_freeze.json').exists():require();preserve();return
    preflight()
    tests=ET.parse(O/'tests.xml').getroot();assert all(int(t.get('failures','0'))==int(t.get('errors','0'))==0 for t in tests.iter('testsuite'))
    previous=BASE/'runs/20261121_fresh01';oldproto=read(previous/'protocol.json')
    assert not (previous/'outcomes/test').exists() and not (previous/'unseal.json').exists()
    protected=dict(read(previous/'protocol_freeze.json')['protected_inputs'])
    for folder in [previous,BASE/'reports/20261121_fresh01',BASE/'fresh_validation',REPO/'papers']:
        for p in folder.rglob('*'):
            if p.is_file() and not {'cache','tmp','__pycache__'}&set(p.parts):protected[str(p)]=sha(p)
    for p,h in protected.items():assert sha(p)==h,p
    proto=oldproto|cfg();assert proto['runs']==oldproto['runs'] and proto['families']==oldproto['families']
    write_json(O/'protocol.json',proto)
    write_json(O/'protocol_freeze.json',{'time_ns':time.time_ns(),'source_hashes':hashes(),'protocol_sha256':sha(O/'protocol.json'),'protected_inputs':protected,'amendment':cfg()['amendment'],'previous_protocol_sha256':sha(previous/'protocol.json')})
    for rel in hashes():
        dest=O/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HOME/rel,dest)
    for name in ['run_manifest.csv','family_manifest.csv','gpu_budget.json']:shutil.copyfile(previous/name,O/name)
    (O/'unavailable').mkdir(exist_ok=True);imported=[]
    for spec in proto['runs']:
        if spec['split']!='development':continue
        old=previous/'classifiers'/spec['run'];dest=O/'classifiers'/spec['run']
        if (old/'complete.json').exists():
            c=read(old/'complete.json');assert c['updates']==c['planned_updates']==19560 and sha(old/'final.pt')==c['final_sha256']
            shutil.copytree(old,dest)
            for p in old.iterdir():
                if p.is_file():assert sha(p)==sha(dest/p.name)
            (O/'outcomes/val').mkdir(parents=True,exist_ok=True);shutil.copyfile(previous/'outcomes/val'/(spec['run']+'.json'),O/'outcomes/val'/(spec['run']+'.json'))
            imported.append({'run':spec['run'],'source':str(old),'final_sha256':c['final_sha256']})
        else:
            failure=read(previous/'preparation_failures'/f"f{spec['family']:02d}_c{spec['construction']}.json")
            assert spec['run'] in failure['affected_runs'];write_json(O/'unavailable'/(spec['run']+'.json'),failure)
    assert len(imported)==124
    write_json(O/'development_import.json',{'newly_trained':0,'reused':124,'unavailable':4,'records':imported,'original_attempt_preserved':True})
    event('amended_protocol_frozen_and_development_imported',reused=124,unavailable=4);preserve()

def status():
    selected=read(O/'protocol.json')['runs'] if (O/'protocol.json').exists() else []
    counts={}
    for split in ['development','evaluation']:
        specs=[s for s in selected if s['split']==split]
        counts[split]={'planned':len(specs),'complete':sum((O/'classifiers'/s['run']/'complete.json').exists() for s in specs),'diverged':sum((O/'classifiers'/s['run']/'diverged.json').exists() for s in specs)}
    print(json.dumps({'counts':counts,'latest':read(O/'status.json') if (O/'status.json').exists() else None,'readiness':read(O/'readiness.json') if (O/'readiness.json').exists() else None,'gpu_budget':read(O/'gpu_budget.json') if (O/'gpu_budget.json').exists() else None},indent=2))
