"""Check source/result parity, render every manuscript page, preserve evidence."""
from pathlib import Path
import csv,hashlib,json,re,shutil,zipfile
from pypdf import PdfReader
import pypdfium2 as pdfium

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):
    with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
source=(HERE/'GradientGuard_IEEE_Access_v1_0.tex').read_text(encoding='utf8')
refs=(HERE/'references.tex').read_text(encoding='utf8')
before=json.loads((HERE/'qa/input_hashes_before.json').read_text(encoding='utf8'))
assert all(sha(p)==h for p,h in before.items()),'An evidence input changed'
protected=0
for line in (REPO/'docs/SCIENTIFIC_ARTIFACTS.sha256').read_text(encoding='utf8').splitlines():
    h,relative=line.split('  ',1);assert sha(REPO/relative)==h,relative;protected+=1
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S).group(1)
abstract_words=len(abstract.split());assert 150<=abstract_words<=250,abstract_words
assert not re.search(r'\\cite\{|\\footnote\{|\\begin\{equation\}',abstract)
keys=set()
for group in re.findall(r'\\cite\{([^}]+)\}',source):keys.update(group.split(','))
defined=set(re.findall(r'\\bibitem\{([^}]+)\}',refs));assert keys==defined,(keys-defined,defined-keys)
assert 'Maninda Edirisooriya' in source and 'Didula Fernando' in source
assert 'Acknowledgment' in source and 'OpenAI ChatGPT' in source
table=re.search(r'\\label\{tab:primary\}(.*?)\\end\{table\*\}',source,re.S).group(1)
primary=read_csv(HERE/'evidence/primary_metrics.csv')
mapping={'E':'E','Logistic B':'logistic_B','Logistic L2':'logistic_L2','Logistic L6':'logistic_L6','Logistic L8':'logistic_L8','Histogram L8':'histogram_L8','Shared-bias E':'bias_E','Shared-bias L8':'bias_L8','Final monitoring':'final_monitor'}
for label,model in mapping.items():
    line=next(x for x in table.splitlines() if x.startswith(label+' &'))
    values=line.split('\\\\')[0].split('&')[1:]
    row=next(r for r in primary if r['model']==model)
    for printed,key in zip(values,['discoveries','precision','recall','AP','AUROC','Brier']):
        assert abs(float(printed)-float(row[key]))<=.000051,(label,key,printed,row[key])
log=(HERE/'tmp/GradientGuard_IEEE_Access_v1_0.log').read_text(encoding='utf8',errors='replace')
assert 'Overfull \\hbox' not in log,'Layout contains an overfull horizontal box'
assert not re.search(r'(?:Reference|Citation) .*undefined|undefined references|Missing character:',log), 'Unresolved reference or missing glyph'
compiled=HERE/'tmp/GradientGuard_IEEE_Access_v1_0.pdf';assert compiled.exists()
target=HERE/'GradientGuard_IEEE_Access_v1_0.pdf';shutil.copyfile(compiled,target)
reader=PdfReader(target);texts=[p.extract_text() for p in reader.pages]
assert all(len(t)>150 for t in texts)
assert '\ufffd' not in ''.join(texts)
assert 'Digital Object Identifier' not in texts[0]
assert 'VOLUME 4' not in ''.join(texts)
assert not re.search(r'\[\?+\]', ''.join(texts))
doc=pdfium.PdfDocument(str(target));edges=[]
for n in range(len(doc)):
    page=doc[n];width,height=page.get_size();tp=page.get_textpage()
    for i in range(tp.count_chars()):
        if not tp.get_text_range(i,1).strip():continue
        l,b,r,t=tp.get_charbox(i)
        if l<28 or r>width-28 or b<12 or t>height-10:edges.append((n+1,[l,b,r,t]))
    page.render(scale=1.8).to_pil().save(HERE/'qa'/f'page-{n+1:02}.png')
assert not edges,edges[:10]
(HERE/'qa/page_text.json').write_text(json.dumps(texts,indent=2),encoding='utf8')
validation={'date':'2026-11-21','pages':len(reader.pages),'abstract_words':abstract_words,'references':len(defined),'primary_numerical_table_parity':True,'input_hashes_verified':len(before),'protected_scientific_files_unchanged':protected,'overfull_boxes':0,'unresolved_references':False,'missing_glyphs':False,'edge_violations':len(edges),'no_new_classifier_or_forecaster_training':True,'pdf_sha256':sha(target),'pdf_bytes':target.stat().st_size,'visual_review':'Required on rendered pages before delivery'}
(HERE/'qa/validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
print(json.dumps(validation,indent=2))
