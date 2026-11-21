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
    preflight();facts=prior_facts()
    root=ET.parse(O/'tests.xml').getroot();assert all(int(x.get('failures','0'))==int(x.get('errors','0'))==0 for x in root.iter('testsuite'))
    families,runs,seed=family_design();old=read(OLD/'protocol.json');oldkeys={(r['alignment'],r['difficulty'],r['lr']) for r in old['families']}
    assert not oldkeys&{(r['alignment'],r['difficulty'],r['lr']) for r in families}
    oldseeds={r[k] for r in old['runs'] for k in ['data_seed','reservation_seed','training_seed']}
    newseeds={r[k] for r in runs for k in ['data_seed','reservation_seed','training_seed']};assert not oldseeds&newseeds
    assert len({r['training_seed'] for r in runs})==256 and len({r['data_seed'] for r in runs})==128
    protected={str(REPO/rel):h for rel,h in read(BASE/'runs/20261121_structural01/protocol_freeze.json')['inputs'].items()}
    for p in BASE.rglob('*'):
        if p.is_file() and not {'cache','tmp','__pycache__','fresh_validation'}&set(p.parts) and not p.is_relative_to(O):protected[str(p)]=sha(p)
    proposal=REPO.parent/'Presentation/GradientGuard_v9_20261121/GradientGuard_Research_Proposal_Didula_Fernando_v9.pdf'
    if proposal.exists():protected[str(proposal)]=sha(proposal)
    for p,h in protected.items():assert sha(p)==h,p
    proto=cfg()|{'families':families,'runs':runs,'resolved_split_seed':seed,'fresh_setting_families':True,'prospective_within_benchmark':True,'capacity_history':'One-third chosen from consumed pilot diagnostics; new outcomes untouched','prior_facts':facts}
    write_json(O/'protocol.json',proto);pd.DataFrame(runs).to_csv(O/'run_manifest.csv',index=False);pd.DataFrame(families).to_csv(O/'family_manifest.csv',index=False)
    write_json(O/'protocol_freeze.json',{'time_ns':time.time_ns(),'source_hashes':hashes(),'protocol_sha256':sha(O/'protocol.json'),'protected_inputs':protected,'prior_facts':facts})
    for rel in hashes():
        p=O/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HOME/rel,p)
    event('fresh_protocol_frozen',families=64,classifiers=256,protected_files=len(protected));preserve()

def status():
    selected=read(O/'protocol.json')['runs'] if (O/'protocol.json').exists() else []
    counts={}
    for split in ['development','evaluation']:
        specs=[s for s in selected if s['split']==split]
        counts[split]={'planned':len(specs),'complete':sum((O/'classifiers'/s['run']/'complete.json').exists() for s in specs),'diverged':sum((O/'classifiers'/s['run']/'diverged.json').exists() for s in specs)}
    print(json.dumps({'counts':counts,'latest':read(O/'status.json') if (O/'status.json').exists() else None,'readiness':read(O/'readiness.json') if (O/'readiness.json').exists() else None,'gpu_budget':read(O/'gpu_budget.json') if (O/'gpu_budget.json').exists() else None},indent=2))
