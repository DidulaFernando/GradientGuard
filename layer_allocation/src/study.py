"""Protocol freeze and preservation; all new writes remain namespaced."""
from core import *
import platform,subprocess,importlib.metadata as md
def environment():
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    saved=(OLD/'requirements-lock.txt').read_text(encoding='utf-8-sig')
    expected=set(saved.splitlines());actual=set(lock.splitlines())
    record={'python':sys.version,'architecture':platform.architecture(),'os':platform.platform(),'executable':sys.executable,'complete_lock_matches':expected==actual,'missing':sorted(expected-actual),'extra':sorted(actual-expected),'resolved_lines':len(actual),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'versions':{n:md.version(n) for n in ['numpy','pandas','scikit-learn','torchvision','spuco','joblib','matplotlib']},'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'configuration_repair':'Only home/executable runtime path changed to available matching Python 3.12.14; package directories unchanged; before/after cfg saved','settings':{k:v for k,v in os.environ.items() if k in ['TEMP','TMP','MPLCONFIGDIR','PIP_CACHE_DIR','PYTHONPYCACHEPREFIX','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']}}
    write(ROOT/'environment/verification.json',record)
    (ROOT/'environment/requirements-lock.txt').write_text(lock,encoding='utf8')
    assert expected==actual,'Dependency mismatch: see environment/verification.json'
    old=read(OLD/'environment.json')
    assert sys.version==old['python'] and torch.__version__==old['torch'] and record['gpu']==old['gpu'] and torch.version.cuda==old['cuda']
    assert record['versions']==old['versions']|{k:record['versions'][k] for k in ['joblib','matplotlib']}
    print(json.dumps({k:v for k,v in record.items() if k!='nvidia_smi'},indent=2))
def protected_paths():
    paths=set()
    for line in (REPO/'docs/SCIENTIFIC_ARTIFACTS.sha256').read_text(encoding='utf8').splitlines():
        h,relative=line.split('  ',1);p=REPO/relative;assert sha(p)==h,relative;paths.add(p)
    for folder in [LEGACY/'checkpoints',LEGACY/'data',LEGACY/'followup',LEGACY/'poc',LEGACY/'diagnostics',REPO/'papers']:
        for p in folder.rglob('*'):
            if p.is_file() and not {'cache','tmp','__pycache__'}&set(p.parts):paths.add(p)
    return sorted(paths)
def freeze():
    environment();out().mkdir(parents=True,exist_ok=True)
    if (out()/'protocol_freeze.json').exists():require();preserve();return
    import xml.etree.ElementTree as ET
    test=ET.parse(out()/'tests.xml').getroot()
    assert all(int(t.attrib.get('failures','0'))==0 and int(t.attrib.get('errors','0'))==0 for t in test.iter('testsuite')),'Tests must pass before protocol freeze'
    source=source_hashes();inputs={str(p.relative_to(REPO)):sha(p) for p in protected_paths()}
    write(out()/'protocol_freeze.json',{'time_ns':time.time_ns(),'post_hoc':True,'evaluation_already_unsealed':True,'configuration':cfg(),'source_hashes':source,'inputs':inputs,'amendment':'User authorised first implementation under revised L6 direction. New structural forecasters may be fitted using original development data only. All old results are read-only; the evaluation is consumed and remains exploratory.'})
    write(ROOT/'protocols/20261121_structural01.json',{'freeze_sha256':sha(out()/'protocol_freeze.json'),'configuration':cfg(),'source_hashes':source})
    (out()/'source_snapshot').mkdir(exist_ok=True)
    import shutil
    for rel in source:
        dest=out()/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    event('protocol_frozen',protected_files=len(inputs))
def preserve():
    f=require();changed=[rel for rel,h in f['inputs'].items() if not (REPO/rel).exists() or sha(REPO/rel)!=h]
    write(out()/'preservation_audit.json',{'protected_files':len(f['inputs']),'unchanged':not changed,'changed':changed})
    assert not changed,changed
    return True
