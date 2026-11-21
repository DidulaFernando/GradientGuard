"""Read-only import/preservation audit; outputs are ignored local records.

Uses standard-library Python. Never prints a matched credential value.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'.repository-local'
def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def paths():
    raw=git('ls-files','--cached','--others','--exclude-standard','-z')
    return sorted({p.decode('utf8') for p in raw.split(b'\0') if p})
def snapshot():
    entries=[]
    for rel in paths():
        if not rel.startswith('Feasibility/'):continue
        p=ROOT/rel
        entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    target=LOCAL/'scientific_preservation_before.json'
    if target.exists():raise SystemExit('Snapshot already exists; refusing to overwrite it.')
    LOCAL.mkdir(exist_ok=True)
    target.write_text(json.dumps(entries,indent=2),encoding='utf8')
    print(json.dumps({'scientific_files':len(entries),'bytes':sum(x['bytes'] for x in entries)}))
def verify():
    entries=json.loads((LOCAL/'scientific_preservation_before.json').read_text(encoding='utf8'))
    changed=[x['path'] for x in entries if not (ROOT/x['path']).is_file() or sha(ROOT/x['path'])!=x['sha256']]
    result={'scientific_files':len(entries),'unchanged':not changed,'changed_paths':changed}
    (LOCAL/'scientific_preservation_after.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result));return bool(changed)
PATTERNS={
 'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
 'aws_access_key':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'credential_in_url':re.compile(rb'https?://[^\s/<>"\x27]+:[^\s/<>"\x27]+@'),
}
def inspect(index=False):
    indexed={}
    if index:
        for entry in git('ls-files','--stage','-z').split(b'\0'):
            if not entry:continue
            header,path=entry.split(b'\t',1)
            mode,oid,stage=header.split()
            if stage!=b'0':raise SystemExit('Unmerged index entry blocks publication.')
            indexed[path.decode('utf8')]=oid.decode('ascii')
    selected=sorted(indexed) if index else paths()
    findings=[];large=[];total=0;extensions={}
    for rel in selected:
        p=ROOT/rel;size=p.stat().st_size;total+=size
        extensions[p.suffix]=extensions.get(p.suffix,0)+size
        if size>50*1024*1024:large.append({'path':rel,'bytes':size})
        data=p.read_bytes()
        if p.suffix.lower() not in ['.png','.jpg','.jpeg','.gif','.pdf','.joblib']:
            for kind,pattern in PATTERNS.items():
                if pattern.search(data):findings.append({'path':rel,'kind':kind})
        if index:
            blob=b'blob '+str(len(data)).encode('ascii')+b'\0'+data
            actual=hashlib.new('sha256' if len(indexed[rel])==64 else 'sha1',blob).hexdigest()
            if actual!=indexed[rel]:findings.append({'path':rel,'kind':'index_worktree_byte_mismatch'})
    result={'scope':'index' if index else 'importable','files':len(selected),'bytes':total,'extensions':extensions,'oversize_files':large,'credential_findings':findings}
    LOCAL.mkdir(exist_ok=True)
    (LOCAL/('index_audit.json' if index else 'inventory.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));return bool(large or findings)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['snapshot','verify','inventory','index'])
    args=parser.parse_args()
    if args.mode=='snapshot':snapshot()
    elif args.mode=='verify':sys.exit(verify())
    else:sys.exit(inspect(args.mode=='index'))
