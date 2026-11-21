"""Package a visually reviewed manuscript without changing research artifacts."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pdf=HERE/'GradientGuard_IEEE_Access_v1_1.pdf'
visual=json.loads((HERE/'qa/visual_review.json').read_text(encoding='utf8'))
assert visual['pdf_sha256']==sha(pdf) and not visual['remaining_issues']
validation=json.loads((HERE/'qa/validation.json').read_text(encoding='utf8'))
assert validation['pdf_sha256']==sha(pdf)
validation['visual_review']=f"All {validation['pages']} final page images inspected; no remaining visual defects."
(HERE/'qa/validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
resources={
 'official_template_page':'https://ieeeaccess.ieee.org/authors/preparing-your-article/',
 'official_template_download':'https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip',
 'official_download_status':'Blocked by website browser challenge; not bypassed.',
 'mirror':'https://github.com/TMLlaboratory/latex-template-ieee-access',
 'downloaded_files':{name:sha(HERE/'template'/name) for name in ['ieeeaccess.cls','IEEEtran.cls','logo.png','notaglinelogo.png','bullet.png']},
 'publisher_class_files_unchanged':all(sha(HERE/name)==sha(HERE/'template'/name) for name in ['ieeeaccess.cls','IEEEtran.cls']),
 'compatibility':'Local spotcolor.sty maps declared full-tint blue to CMYK for XeTeX; manuscript-local caption/font/appendix/title/footer overrides. No assigned DOI or publication metadata invented.',
 'compiler':'Tectonic 0.17.0+20260731; project-local package/font cache',
 'authorship':'Didula Fernando and Maninda Edirisooriya, explicitly requested; affiliations/contact/contributions and approval pending.'}
(HERE/'qa/template_and_authoring_provenance.json').write_text(json.dumps(resources,indent=2),encoding='utf8')
files=[HERE/n for n in ['GradientGuard_IEEE_Access_v1_1.tex','references.tex','ieeeaccess.cls','IEEEtran.cls','spotcolor.sty','GradientGuard_IEEE_Access_v1_1.pdf','README.md','EDITORIAL_NOTES.md','SOURCE_PACKAGE_README.txt','REVIEW_RESPONSE.md','prepare_evidence.py','validate_manuscript.py','build.ps1','package_release.py']]
files+=[HERE/'template/bullet.png']
files+=sorted((HERE/'figures').glob('*.pdf'))+sorted((HERE/'evidence').glob('*'))
files+=[HERE/'qa/validation.json',HERE/'qa/visual_review.json',HERE/'qa/template_and_authoring_provenance.json']
files=[p for p in files if p.is_file()]
archive=HERE/'GradientGuard_IEEE_Access_v1_1_source.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(HERE).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert z.read(pdf.name)==pdf.read_bytes()
    assert all(z.read(p.relative_to(HERE).as_posix())==p.read_bytes() for p in files)
manifest={'pdf_sha256':sha(pdf),'pdf_bytes':pdf.stat().st_size,'source_zip_sha256':sha(archive),'source_zip_bytes':archive.stat().st_size,'packaged_files':len(files),'file_hashes':{p.relative_to(HERE).as_posix():sha(p) for p in files},'no_new_training_or_fitting':True,'journal_submission':False,'public_push':False}
(HERE/'qa/delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in manifest.items() if k!='file_hashes'},indent=2))
