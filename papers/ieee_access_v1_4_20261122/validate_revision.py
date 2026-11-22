from pathlib import Path
import hashlib
import json
import re
from zipfile import ZipFile
from docx import Document
from pypdf import PdfReader
from lxml import etree
import copy

P = Path(__file__).resolve().parent
OLD = P.parent/'ieee_access_v1_3_20261122'
d = Document(P/'GradientGuard_IEEE_Access_v1_4.docx')
old = Document(OLD/'GradientGuard_IEEE_Access_v1_3.docx')
text = '\n'.join(x.text for x in d.paragraphs)
checks = {}
checks['page_size_matches_official_template'] = all(int(s.page_width/635)==11520 and int(s.page_height/635)==15660 for s in d.sections)
checks['template_margins'] = all(int(s.left_margin/635)==740 and int(s.right_margin/635)==740 and int(s.bottom_margin/635)==1040 and int(s.top_margin/635)==(1280 if i==0 else 1300) for i,s in enumerate(d.sections))
checks['two_column_gutter'] = all(s._sectPr.xpath('./w:cols')[0].get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}space')=='400' for s in d.sections if s._sectPr.xpath('./w:cols')[0].get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num')=='2')
checks['tables_unchanged'] = [[c.text for row in t.rows for c in row.cells] for t in d.tables] == [[c.text for row in t.rows for c in row.cells] for t in old.tables]
def math_without_font_overrides(document):
    def signature(x):
        return (x.tag,tuple(sorted(x.attrib.items())),x.text,tuple(signature(c) for c in x if c.tag!='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr'))
    result=[]
    for x in document.element.xpath('//m:oMathPara'):
        x=copy.deepcopy(x)
        for rpr in x.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr'):
            rpr.getparent().remove(rpr)
        for mr in list(x.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}r')):
            mt=mr.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}t')
            if mt is not None and mt.text=='&': mr.getparent().remove(mr)
        result.append(signature(x))
    return result
checks['nine_equations_unchanged'] = math_without_font_overrides(d) == math_without_font_overrides(old) and len(d.element.xpath('//m:oMathPara'))==9
checks['three_images_unchanged'] = len(d.inline_shapes)==3
with ZipFile(P/'GradientGuard_IEEE_Access_v1_4.docx') as z, ZipFile(OLD/'GradientGuard_IEEE_Access_v1_3.docx') as oz:
    checks['image_bytes_unchanged'] = all(z.read(n)==oz.read(n) for n in z.namelist() if n.startswith('word/media/'))
    xml = '\n'.join(z.read(n).decode('utf-8') for n in z.namelist() if n.endswith(('.xml','.rels')))
    checks['no_usage_mentions_in_package'] = not re.search(r'OpenAI|ChatGPT|Codex|AI.assist|AI assistance',xml,re.I)
checks['no_usage_mentions_in_source'] = not re.search(r'OpenAI|ChatGPT|Codex|AI.assist|AI assistance',(P/'manuscript_source.tex').read_text(encoding='utf-8'),re.I)
checks['footer_draft_label_removed'] = all('Editorial draft' not in p.text and 'GradientGuard' not in p.text for s in d.sections for part in [s.footer,s.first_page_footer,s.even_page_footer] for p in part.paragraphs)
checks['supervisor_acknowledgement_retained'] = 'The author thanks Maninda Edirisooriya' in text
checks['funding_retained'] = 'this project currently receives no funding' in text
refs = [p.text for p in d.paragraphs if p.style.name=='Ref']
checks['eighteen_sequential_references'] = [int(re.match(r'\[(\d+)\]',x)[1]) for x in refs] == list(range(1,19))
checks['citation_order'] = list(dict.fromkeys(int(n) for m in re.finditer(r'\[([\d, ]+)\]',text[:text.index('References')]) for n in re.findall(r'\d+',m[1]))) == list(range(1,19))
provenance = json.loads((P/'qa/provenance.json').read_text())
checks['v1_3_inputs_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in provenance['prior_files_unchanged'].items())
pdf = PdfReader(P/'qa/render/GradientGuard_IEEE_Access_v1_4.pdf')
rendered = '\n'.join(p.extract_text() for p in pdf.pages)
checks['pdf_template_dimensions_within_one_point_export_rounding'] = all(abs(float(p.mediabox.width)-576)<1 and abs(float(p.mediabox.height)-783)<1 for p in pdf.pages)
checks['pdf_no_usage_mentions'] = not re.search(r'OpenAI|ChatGPT|Codex|AI.assist|AI assistance|Editorial draft',rendered,re.I)
checks['pdf_has_all_tables'] = all('TABLE '+str(n)+'.' in rendered for n in range(1,9))
checks['pdf_has_biography'] = 'Author Biography' in rendered
checks['pdf_has_key_results'] = all(x in rendered for x in ['0.7751','0.9336','0.8160','0.9618','0.2036','0.1648'])
report={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'pages':len(pdf.pages),'scope':'Document checks only; no experiments rerun.'}
(P/'qa/validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report,indent=2))
assert all(checks.values())
