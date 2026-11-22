"""Focused v1.4 revision; preserve the v1.3 manuscript and scientific evidence."""
from pathlib import Path
import copy
import hashlib
import json
import re
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

P = Path(__file__).resolve().parent
OLD = P.parent / 'ieee_access_v1_3_20261122'
SOURCE = OLD / 'GradientGuard_IEEE_Access_v1_3.docx'
TEMPLATE = OLD / 'cache/official_template.docx'
P.joinpath('qa').mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p): sha(p) for p in [SOURCE, OLD/'GradientGuard_IEEE_Access_v1_3.pdf', OLD/'manuscript_source.tex', TEMPLATE]}
d = Document(SOURCE)
t = Document(TEMPLATE)

def ptext(p): return ''.join(x.text or '' for x in p.findall('.//' + qn('w:t')))
for p in list(d.element.body.findall('.//' + qn('w:p'))):
    text = ptext(p)
    if text.startswith('Text drafting and editing in the numbered sections') or text.startswith('[1] OpenAI'):
        p.getparent().remove(p)
    elif text.startswith('The author thanks Maninda Edirisooriya'):
        para = next(x for x in d.paragraphs if x._p is p)
        # Retain only the existing supervisor acknowledgement.
        keep = text[:text.index(' OpenAI ChatGPT')]
        para.text = keep

def replace_spanning_runs(p, pattern, callback):
    nodes = p.findall('.//' + qn('w:t'))
    text = ''.join(n.text or '' for n in nodes)
    positions = []
    at = 0
    for n in nodes:
        positions.append((at, at+len(n.text or ''), n))
        at += len(n.text or '')
    for m in reversed(list(re.finditer(pattern, text))):
        value = callback(m)
        touched = [(a,b,n) for a,b,n in positions if a < m.end() and b > m.start()]
        for i,(a,b,n) in enumerate(touched):
            old = n.text or ''
            start, end = max(0,m.start()-a), min(len(old),m.end()-a)
            n.text = old[:start] + (value if i == 0 else '') + old[end:]

def renumber(m):
    return re.sub(r'\d+', lambda n: str(int(n[0])-1), m[0])
for p in d.element.body.findall('.//' + qn('w:p')):
    replace_spanning_runs(p, r'\[(?:[2-9]|1[0-9])(?:,\s*(?:[2-9]|1[0-9]))*\]', renumber)

# Copy exact page dimensions, margins and column gutter from the retained
# publisher template (same URL as the current publisher download).
front, body = t.sections[0]._sectPr, t.sections[1]._sectPr
for i, sec in enumerate(d.sections):
    sp = sec._sectPr
    ref = front if i == 0 else body
    for tag in ['pgSz','pgMar']:
        current = sp.find(qn('w:'+tag))
        if current is not None:
            sp.replace(current,copy.deepcopy(ref.find(qn('w:'+tag))))
        else:
            sp.append(copy.deepcopy(ref.find(qn('w:'+tag))))
    cols = sp.find(qn('w:cols'))
    cols.set(qn('w:space'), '400' if cols.get(qn('w:num')) == '2' else '720')
    # A neutral page field is suitable for the author manuscript. Volume/year
    # and final pagination are publisher metadata, not values to invent here.
    for part in [sec.footer, sec.first_page_footer, sec.even_page_footer]:
        for p in part.paragraphs:
            p.clear()
            p.style = d.styles['Footer']
            field = OxmlElement('w:fldSimple')
            field.set(qn('w:instr'), 'PAGE')
            run = OxmlElement('w:r')
            txt = OxmlElement('w:t'); txt.text = '1'
            run.append(txt); field.append(run); p._p.append(field)
    order = ['headerReference','footerReference','footnotePr','endnotePr','type','pgSz','pgMar','paperSrc','pgBorders','lnNumType','pgNumType','cols','formProt','vAlign','noEndnote','titlePg','textDirection','bidi','rtlGutter','docGrid','printerSettings','sectPrChange']
    for child in sorted(list(sp),key=lambda x: order.index(x.tag.rsplit('}',1)[-1]) if x.tag.rsplit('}',1)[-1] in order else 100): sp.append(child)
    if i > 0:
        # Cloned sections share relationship ids. Remove only the references,
        # retaining the relationship used by the first section.
        for child in list(sp):
            if child.tag in [qn('w:headerReference'),qn('w:footerReference')]: sp.remove(child)
for p in d.paragraphs:
    if p.style.name in ['H1','H3','H4']:
        p.paragraph_format.keep_with_next=True
    if p.style.name == 'Paper Title':
        for r in p.runs:
            r.font.size = t.styles['Paper Title'].font.size
for rid, rel in list(d.part.rels.items()):
    if rel.is_external and 'openai.com' in rel.target_ref:
        d.part.drop_rel(rid)
# Pandoc retained an unused hyperlink in the empty footnote part.
for part in list(d.part.package.parts):
    for rid, rel in list(part.rels.items()):
        if rel.is_external and 'openai.com' in rel.target_ref:
            part.rels.pop(rid)
# Set the equation font explicitly so the renderer cannot substitute a symbol
# font for ordinary parentheses and brackets. No equation operands change.
mathpr = OxmlElement('m:mathPr')
mathfont = OxmlElement('m:mathFont'); mathfont.set(qn('m:val'), 'Cambria Math')
mathpr.append(mathfont); d.settings.element.append(mathpr)
for mr in d.element.xpath('//m:r'):
    # The converter emitted alignment markers as literal glyphs in equation
    # arrays. They are layout markup, not mathematical operands.
    mt=mr.find(qn('m:t'))
    if mt is not None and mt.text=='&':
        mr.getparent().remove(mr)
        continue
    wrpr = mr.find(qn('w:rPr'))
    if wrpr is None: wrpr = OxmlElement('w:rPr'); mr.insert(0, wrpr)
    fonts = OxmlElement('w:rFonts')
    for name in ['ascii','hAnsi','eastAsia','cs']: fonts.set(qn('w:'+name), 'Cambria Math')
    wrpr.append(fonts)
d.core_properties.comments = ''
d.core_properties.version = '1.4'
d.save(P/'GradientGuard_IEEE_Access_v1_4.docx')

# Keep a matching editable text source for subsequent revisions.
s = (OLD/'manuscript_source.tex').read_text(encoding='utf-8')
s = re.sub(r'\\noindent Text drafting and editing in the numbered sections.*?\n', '', s)
s = re.sub(r'(The author thanks Maninda Edirisooriya[^\n]+?) OpenAI ChatGPT[^\n]*', r'\1', s)
s = re.sub(r'^\[1\]  OpenAI[^\n]*\n', '', s, flags=re.M)
s = re.sub(r'\[(?:[2-9]|1[0-9])(?:,\s*(?:[2-9]|1[0-9]))*\]', renumber, s)
s = s.replace('{figures/', '{../ieee_access_v1_3_20261122/figures/')
(P/'manuscript_source.tex').write_text(s,encoding='utf-8',newline='\n')

assert all(sha(Path(p)) == h for p,h in before.items())
report = {
    'prior_files_unchanged': before,
    'official_template_url': 'https://ieeeaccess.ieee.org/wp-content/uploads/2025/08/Access-Template-2024.docx',
    'template_sha256': sha(TEMPLATE),
    'template_page_twips': [11520,15660],
    'template_body_margins_twips': {'top':1300,'right':740,'bottom':1040,'left':740,'header':360,'footer':640},
    'template_two_column_gutter_twips': 400,
    'scope':'Formatting and requested acknowledgement/disclosure removal only. No scientific results changed.',
}
(P/'qa/provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'output':str(P/'GradientGuard_IEEE_Access_v1_4.docx'),'tables':len(d.tables),'figures':len(d.inline_shapes),'equations':len(d.element.xpath('//m:oMathPara'))}))
