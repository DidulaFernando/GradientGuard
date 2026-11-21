"""Project-local interpreter repair and read-only evidence verification."""
from pathlib import Path
import sys, json, hashlib, subprocess, platform, shutil, time, os

ATTEMPT=Path(__file__).resolve().parents[1]
ROOT=ATTEMPT.parents[1]
POC=ROOT/'poc/outputs/20261116_poc01'
PROPOSAL=Path('C:/Users/rangi/Documents/ChatGPT/Explanations/GradientGuard_Research_Proposal_Didula_Fernando_v5.pdf')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def verify_prior(stage):
    recorded=read(POC/'protected_before.json')
    recorded.update(read(POC/'reports/artifact_hashes.json'))
    errors=[k for k,v in recorded.items() if sha(PROPOSAL if k=='proposal' else ROOT/k)!=v]
    assert not errors, f'Historical evidence mismatch: {errors}'
    snapshot={str(p.relative_to(ROOT)):sha(p) for folder in ['src','tests','configs','runs','reports','checkpoints','data','followup','poc','instructions'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='STEP_BY_STEP_DIAGNOSTICS.txt'}
    snapshot['proposal']=sha(PROPOSAL)
    baseline=ATTEMPT/'config/preservation_before.json'
    if baseline.exists():assert snapshot==read(baseline),'Previous evidence changed during diagnostics'
    else:write(baseline,snapshot)
    result={'passed':True,'checked_recorded_hashes':len(recorded),'protected_current_files':len(snapshot),'time_ns':time.time_ns()}
    write(ATTEMPT/f'config/preservation_{stage}.json',snapshot)
    write(ATTEMPT/f'logs/preservation_{stage}.json',result)
    return result

def repair():
    verify_prior('before')
    cfg=ROOT/'.venv/pyvenv.cfg';backup=ATTEMPT/'environment/pyvenv_before.cfg'
    if not backup.exists():shutil.copy2(cfg,backup)
    recorded=read(POC/'environment.json')
    assert sys.version==recorded['python'], 'Base interpreter must match recorded complete version'
    assert platform.architecture()[0]=='64bit'
    old=backup.read_text();lines=[]
    for line in old.splitlines():
        key=line.split('=',1)[0].strip()
        if key=='home':line='home = '+str(Path(sys.executable).parent)
        elif key=='executable':line='executable = '+sys.executable
        lines.append(line)
    proposed='\n'.join(lines)+'\n'
    if cfg.read_text()!=proposed:cfg.write_text(proposed,encoding='utf8')
    shutil.copy2(cfg,ATTEMPT/'environment/pyvenv_after.cfg')
    write(ATTEMPT/'environment/repair.json',{'method':'Update only home/executable in pyvenv.cfg to matching available Codex Python; historical command retained',
          'python':sys.version,'architecture':platform.architecture(),'new_base':sys.executable,
          'before_config_sha256':sha(backup),'after_config_sha256':sha(cfg),
          'packages_installed_removed_or_upgraded':False,'time_ns':time.time_ns()})
    print('Environment configuration repaired; package/import verification still required.')

def verify_environment():
    # Lock equality MUST precede unpickling. No scientific package installation here.
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    expected=(POC/'requirements-lock.txt').read_text()
    (ATTEMPT/'environment/requirements-after.txt').write_text(lock,encoding='utf8')
    assert set(lock.splitlines())==set(expected.splitlines()), 'Resolved package lock mismatch'
    assert sys.version==read(POC/'environment.json')['python']
    import numpy,pandas,sklearn,joblib,matplotlib,torch,torchvision
    import importlib.metadata as md
    assert sha(POC/'pipelines.joblib')==read(POC/'predictor_freeze.json')['pipelines_sha256']
    models=joblib.load(POC/'pipelines.joblib')
    cuda=torch.cuda.is_available()
    if cuda:
        assert torch.equal(torch.tensor([1.,2.],device='cuda')+1,torch.tensor([2.,3.],device='cuda'))
        torch.cuda.synchronize()
    result={'passed':True,'python':sys.version,'executable':sys.executable,'architecture':platform.architecture(),
      'complete_lock_matches':True,'resolved_requirement_lines':len(lock.splitlines()),
      'versions':{n:md.version(n) for n in ['numpy','pandas','scikit-learn','joblib','matplotlib','torch','torchvision','spuco']},
      'cuda_available':cuda,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
      'gpu':torch.cuda.get_device_name(0) if cuda else None,'os':platform.platform(),
      'pipeline_collections_loaded':list(models),'pipeline_sha256':sha(POC/'pipelines.joblib'),
      'prediction_sha256':sha(POC/'evaluation_predictions.csv'),
      'process_settings':{k:os.getenv(k) for k in ['TEMP','TMP','TMPDIR','PIP_CACHE_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CACHE_HOME','PYTHONPYCACHEPREFIX','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']}}
    assert result['gpu']==read(POC/'environment.json')['gpu']
    assert result['cuda_runtime']==read(POC/'environment.json')['cuda']
    write(ATTEMPT/'environment/verification.json',result)
    print(json.dumps({k:result[k] for k in ['passed','python','complete_lock_matches','resolved_requirement_lines','gpu','cuda_runtime']},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='repair':repair()
    elif sys.argv[1]=='verify':verify_environment()
    elif sys.argv[1]=='preserve':print(verify_prior('after'))
