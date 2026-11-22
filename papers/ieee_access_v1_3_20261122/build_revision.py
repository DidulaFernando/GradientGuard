from pathlib import Path
import re, subprocess, hashlib, json, copy
import pypdfium2 as pdfium
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION_START

P=Path(__file__).resolve().parent
OLD=P.parent/'ieee_access_v1_2_20261121'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p):digest(p) for p in OLD.rglob('*') if p.is_file()}
(P/'qa'/'preservation_before.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
s=(OLD/'GradientGuard_IEEE_Access_v1_2.tex').read_text(encoding='utf-8')
s=s[s.index('\\begin{abstract}'):]
s=s.replace('\\end{keywords}', '\\end{keywords}')
s=re.sub(r'\\titlepgskip=.*?\\pagestyle\{headings\}', '',s,flags=re.S)
s=s.replace('\\PARstart{A}{ggregate}', 'Aggregate')
s=s.replace('The uncentred error cross term was positive in all partitions, so net error cancellation did not explain combined reliability.', 'For the measured partition errors, the mean squared error of the combined deep summary equals the conv3 and head mean squared errors plus twice their uncentred mean product. This product was positive in all partitions: errors reinforced, rather than cancelled, on average. This diagnostic concerns probe-panel disagreement, not the causal origin of subgroup failures.')
s=s.replace('The largest recorded allocated GPU-memory counter is 567.2 MiB. It is cumulative per process and does not isolate incremental probe memory, other GPU applications or CPU peak memory.', 'The maximum recorded process-level peak allocated GPU memory is 567.2 MiB. This is an allocation high-water mark, not a sum of allocations over time. It does not isolate incremental probe memory, whole-device memory use, other GPU applications or CPU peak memory.')
s=s.replace('A single positive active channel passes through unit centre kernels and a unit selected head weight; unused connecting weights are zero and unused channels are inactive.', 'The fixture embeds one active channel through all three convolutions. Each active convolution uses a unit centre-kernel weight, and a unit classifier weight connects the pooled channel to one logit. Other connecting weights are zero; negative biases keep unused channels inactive.')
s=s.replace('Its result is therefore an increment over the implemented conventional/global controls, with sensitivity-specific originality still to be established.', 'Any observed increment is conditional on the implemented conventional/global controls. Sensitivity-specific originality remains unestablished because the matched structural comparison has not been executed.')
s=s.replace('The secondary continuous worst-group analysis gives', 'The archived analysis does not provide paired-family uncertainty intervals for average precision, receiver-operating-characteristic area, Brier-score differences or regression errors. Their point estimates are descriptive; they must not be read as statistically established improvements. No additional inference is performed in this editorial revision.\n\nThe secondary continuous worst-group analysis gives')
s=s.replace('CNN A has', 'The convolutional neural network (CNN), CNN A, has',1)
s=s.replace('At the original nine-audit capacity,', 'Average precision (AP), area under the receiver operating characteristic curve (AUROC) and Brier score are secondary ranking/probability metrics; they do not replace the primary discovery count. At the original nine-audit capacity,',1)
s=s.replace('Lower MAE is better.', 'Lower mean absolute error (MAE) is better.')
s=s.replace('\\section*{Acknowledgment}', '''\\section*{Data and Code Availability}
Code, compact identity manifests, configurations, archived tables and frozen forecaster artifacts are available in the public GradientGuard repository~\\cite{repository}. The initial archival import is commit \\texttt{ae2e80c}; study identifiers in Appendix A identify the underlying records. Raw MNIST downloads, generated dataset arrays, local virtual environments and classifier recovery/weight files are not distributed with the repository. Repository instructions describe reconstruction from the pinned official SpuCo implementation; the public clone does not contain every original binary. The evidence manifest accompanying this manuscript identifies its archived inputs.

\\section*{Ethics and Data Scope}
The reported experiments use the public MNIST-derived SpuCoMNIST benchmark. They involve no new participant recruitment, patient records or private personal-data collection. No new human-subject study is reported. This statement describes the experiment rather than asserting an institutional ethics exemption.

\\section*{Funding}
The author reports that this project currently receives no funding.

\\section*{Competing Interests}
The competing-interest declaration requires confirmation by the author before submission.

\\section*{Acknowledgment}''')
s=s.replace('OpenAI ChatGPT, accessed through Codex,', 'The author thanks Maninda Edirisooriya, lecturer at the Informatics Institute of Technology (IIT), Sri Lanka, for supervision of this final-year project (contact: \\texttt{maninda.e@iit.ac.lk}). OpenAI ChatGPT, accessed through Codex~\\cite{ai},')
s=s.replace('The named authors retain responsibility', 'The author retains responsibility')
s=s.replace('\\section{Introduction}', '\\section{Introduction}\n\\noindent Text drafting and editing in the numbered sections and Appendix A were assisted by OpenAI ChatGPT through Codex~\\cite{ai}; the author is responsible for verification.\n')
cm=re.search(r'Our contributions are threefold\. First, (.*?) Second, (.*?) Third, (.*?) The contribution is',s,re.S)
if cm:
    listed='Our contributions are:\n\\begin{enumerate}\n'+''.join('\\item '+v[0].upper()+v[1:]+'\n' for v in cm.groups())+'\\end{enumerate}\nThe contribution is'
    s=s[:cm.start()]+listed+s[cm.end():]
refs=(OLD/'references.tex').read_text(encoding='utf-8')
refs=refs.replace('Publication identity verified in the accepted-paper record; accessible associated manuscript: OpenReview 51TWh8tlSy. ', '')
refs=refs.replace('\\url{https://www.jmlr.org/tmlr/papers/}', '\\url{https://openreview.net/forum?id=51TWh8tlSy}')
refs=refs.replace('\\end{thebibliography}', '\\bibitem{ai} OpenAI, ``ChatGPT and Codex,'' AI-assisted text drafting, editing and software assistance, 2026. [Online]. Available: \\url{https://openai.com/chatgpt/overview/}. Accessed: Nov. 22, 2026.\n\\end{thebibliography}')
keys=re.findall(r'\\bibitem\{([^}]+)\}',refs)
ordered=[]
for citing in re.findall(r'\\cite\{([^}]+)\}',s):
    for key in citing.split(','):
        if key not in ordered:ordered.append(key)
ordered += [k for k in keys if k not in ordered]
items={m[1]:m[0] for m in re.finditer(r'\\bibitem\{([^}]+)\}.*?(?=\\bibitem|\\end\{thebibliography\})',refs,re.S)}
keys=ordered
refs='\\begin{thebibliography}{99}\n'+''.join(items[k] for k in keys)+'\\end{thebibliography}\n'
s=s.replace('\\input{references.tex}',refs)
s=re.sub(r'~?\\cite\{([^}]+)\}',lambda m:' ['+', '.join(str(keys.index(k)+1) for k in m[1].split(','))+']',s)
for cmd,val in [('GG','GradientGuard'),('WGA',r'\\operatorname{WGA}'),('SD',r'\\operatorname{SD}'),('mean',r'\\operatorname{mean}'),('Dtwenty',r'\\Delta_{20}')]:
    s=re.sub(r'\\'+cmd+r'\b',lambda m,v=val:v.replace('\\\\','\\'),s)
# Resolve cross-references before converting; recovery records are never read.
labels={}
for env,prefix in [('table','Table'),('figure','Fig.'),('equation','')]:
    for n,m in enumerate(re.finditer(r'\\begin\{'+env+r'\*?\}(.*?)\\end\{'+env+r'\*?\}',s,re.S),1):
        for lab in re.findall(r'\\label\{([^}]+)\}',m[1]): labels[lab]=str(n)
s=re.sub(r'\\ref\{([^}]+)\}',lambda m:labels.get(m[1],m[1]),s)
s=re.sub(r'\\eqref\{([^}]+)\}',lambda m:'('+labels.get(m[1],m[1])+')',s)
s=s.replace('\\begin{keywords}','\\paragraph{Index Terms}').replace('\\end{keywords}','')
s=s.replace('\\appendices','').replace('\\EOD','').replace('\\end{document}','')
s=re.sub(r'\\begin\{thebibliography\}\{99\}',r'\\section*{References}',s).replace('\\end{thebibliography}','')
s=re.sub(r'\\bibitem\{([^}]+)\}',lambda m:'\n['+str(keys.index(m[1])+1)+'] ',s)
s=re.sub(r'\\begin\{tabularx\}\{\\textwidth\}\{[^\n]+\}',r'\\begin{tabular}{lll}',s).replace('\\end{tabularx}','\\end{tabular}')
# Pandoc's LaTeX reader rejects nested custom column declarations. Preserve every
# cell while replacing only the column formatting specification.
pos=0
while True:
    start=s.find('\\begin{tabular}{',pos)
    if start<0:break
    a=start+len('\\begin{tabular}'); depth=0; end=a
    for end in range(a,len(s)):
        if s[end]=='{':depth+=1
        if s[end]=='}':
            depth-=1
            if depth==0:break
    tail=s[end+1:s.index('\\end{tabular}',end)]
    first=tail.split('\\\\')[0]; n=first.count('&')+1
    s=s[:a]+'{'+'l'*n+'}'+s[end+1:];pos=a+n+2
tn=0
def number_caption(m):
    global tn
    tn+=1
    return m[0].replace('\\caption{','\\caption{TABLE '+str(tn)+'. ',1)
s=re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',number_caption,s,flags=re.S)
fn=0
def number_figure(m):
    global fn
    fn+=1
    return m[0].replace('\\caption{','\\caption{FIGURE '+str(fn)+'. ',1)
s=re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',number_figure,s,flags=re.S)
en=0
def number_equation(m):
    global en
    en+=1
    return m[0].replace('\\end{equation}',r'\tag{'+str(en)+'}\\end{equation}')
s=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}',number_equation,s,flags=re.S)
for f in (OLD/'figures').glob('*.pdf'):
    doc=pdfium.PdfDocument(str(f)); doc[0].render(scale=4).to_pil().save(P/'figures'/(f.stem+'.png'))
    s=s.replace('{'+f.name+'}','{figures/'+f.stem+'.png}')
s+='''
\\section*{Author Biography}
Didula Fernando is a final-year B.Sc. Computer Science student at the Informatics Institute of Technology (IIT), Sri Lanka. His final-year research project investigates early neural-network measurements for prioritising subgroup-failure audits.
'''
front='''\\title{Early Layer Sensitivity for Subgroup Audit Prioritisation with GradientGuard}
\\author{Didula Fernando}
\\date{}
\\maketitle
\\noindent Informatics Institute of Technology (IIT), Sri Lanka.\\\\
Corresponding author: Didula Fernando (\\texttt{didula.20232318@iit.ac.lk}).\\\\
Project supervisor: Maninda Edirisooriya, IIT, Sri Lanka (\\texttt{maninda.e@iit.ac.lk}).
'''
(P/'manuscript_source.tex').write_text(front+s,encoding='utf-8')
pandoc=next((P/'cache'/'pandoc').rglob('pandoc.exe'))
subprocess.run([str(pandoc),'manuscript_source.tex','-f','latex','-t','docx','--reference-doc',str(P/'cache'/'official_template.docx'),'-o','converted.docx'],cwd=P,check=True)
d=Document(P/'converted.docx')
# The LaTeX reader drops captions on table*; restore all eight explicitly.
captions=[]
for block in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',s,re.S):
    a=block.index('\\caption{')+len('\\caption{');depth=1;b=a
    while depth:
        if block[b]=='{':depth+=1
        if block[b]=='}':depth-=1
        b+=1
    result=subprocess.run([str(pandoc),'-f','latex','-t','plain','--wrap=none'],input=block[a:b-1],text=True,encoding='utf-8',capture_output=True,check=True)
    captions.append(result.stdout.strip())
for p in list(d.paragraphs):
    if p.text.startswith('TABLE '):p._p.getparent().remove(p._p)
from docx.text.paragraph import Paragraph
for t,caption in zip(d.tables,captions):
    e=OxmlElement('w:p');t._tbl.addprevious(e);p=Paragraph(e,t._parent);p.add_run(caption)
for sty in ['Normal','PARA','PARA_Indent']:
    d.styles[sty].font.name='Times New Roman';d.styles[sty].font.size=Pt(10)
    d.styles[sty].paragraph_format.line_spacing=1
    d.styles[sty].paragraph_format.space_after=Pt(5)
for sty in ['H1','H3','H4','Fig Caption','Table Caption','Ref']:
    pp=d.styles[sty]._element.find(qn('w:pPr'))
    if pp is not None:
        for e in list(pp):
            if e.tag in [qn('w:numPr'),qn('w:framePr')]:pp.remove(e)
for sec in d.sections:
    sec.page_width=Inches(8); sec.page_height=Inches(10.875)
    sec.left_margin=sec.right_margin=Inches(.514)
    sec.top_margin=Inches(.78); sec.bottom_margin=Inches(.65)
    for part in [sec.header,sec.first_page_header,sec.even_page_header]:
        for p in part.paragraphs:
            p.text='Fernando: Early Layer Sensitivity for Subgroup Audit Prioritisation'
            p.style=d.styles['Header']
    for part in [sec.footer,sec.first_page_footer,sec.even_page_footer]:
        for p in part.paragraphs:
            p.text='GradientGuard | Editorial draft v1.3 | '
            fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); p._p.append(fld)
            for r in p.runs:r.font.size=Pt(7)
stylemap={'Title':'Paper Title','Author':'AU','Body Text':'PARA','First Paragraph':'PARA','Abstract':'Abstract','Heading 1':'H1','Heading 2':'H3','Heading 3':'H4','Caption':'Fig Caption'}
for p in d.paragraphs:
    old=p.style.name
    if old in stylemap and stylemap[old] in d.styles: p.style=d.styles[stylemap[old]]
    if p.text.startswith('[') and re.match(r'^\[\d+\]',p.text):p.style=d.styles['Ref']
    if p.text=='Introduction':
        # End a single-column front matter section immediately before the body.
        prev=OxmlElement('w:p'); props=OxmlElement('w:pPr'); sect=copy.deepcopy(d.sections[-1]._sectPr)
        typ=sect.find(qn('w:type'))
        if typ is None:typ=OxmlElement('w:type');sect.insert(0,typ)
        typ.set(qn('w:val'),'continuous')
        cols=sect.find(qn('w:cols'));cols.set(qn('w:num'),'1')
        props.append(sect);prev.append(props);p._p.addprevious(prev)
    for r in p.runs:
        r.font.color.rgb=RGBColor(0,0,0)
    p.paragraph_format.widow_control=True
    if p.text.startswith('TABLE '):
        p.style=d.styles['Normal'];p.paragraph_format.keep_with_next=True
        for r in p.runs:r.font.size=Pt(9);r.bold=True
    if p.text.startswith('FIGURE '):
        p.style=d.styles['Normal']
        for r in p.runs:r.font.size=Pt(8)
title=d.paragraphs[0];title.style=d.styles['Paper Title']
for r in title.runs:r.font.size=Pt(24);r.bold=True
d.paragraphs[1].style=d.styles['AU']
# Keep identity and contact information before the abstract.
aff=next(p for p in d.paragraphs if p.text.startswith('Informatics Institute'))
d.paragraphs[1]._p.addnext(aff._p)
for text,sty in [('Date of publication and date of current version: to be assigned by IEEE.','DOP'),('Digital Object Identifier: to be assigned by IEEE upon publication.','DOI')]:
    p=title.insert_paragraph_before(text);p.style=d.styles[sty]
eqnum=0
for p in d.paragraphs:
    if p.text=='Index Terms':p.style=d.styles['Normal'];p.runs[0].bold=True
    if p._p.xpath('.//m:oMathPara'):
        eqnum+=1;p.add_run('  ('+str(eqnum)+')').font.size=Pt(9)
# Consistent manual section numbering avoids inherited template list examples.
roman=['I','II','III','IV','V','VI','VII','VIII']; si=0; sub=0
for p in d.paragraphs:
    if p.style.name=='H1' and p.text in ['Introduction','Related Work and Information Boundaries','Connection Sensitivity and Audit Prediction','Experimental Design and Evidence Stages','Results','Computational Cost and Reproducibility','Discussion and Scope of the Findings','Conclusion']:
        p.runs[0].text=roman[si]+'. '+p.runs[0].text;si+=1;sub=0
    elif p.style.name=='H3':
        p.runs[0].text=chr(65+sub)+'. '+p.runs[0].text;sub+=1
    elif p.text=='Analytical Fixture and Evidence Access':p.runs[0].text='APPENDIX A. '+p.runs[0].text
    if p.style.name=='Ref':
        for r in p.runs:r.font.size=Pt(8)
# Official page geometry and two-column body, with wide display objects set below.
cols=d.sections[-1]._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
for t in d.tables:
    t.autofit=False
    n=len(t.columns)
    w=t._tbl.tblPr.find(qn('w:tblW'));w.set(qn('w:type'),'dxa');w.set(qn('w:w'),'10000')
    t.style=d.styles['Normal Table']
    for col in t.columns:col.width=Inches(6.9/n)
    for row in t.rows:
        rp=row._tr.get_or_add_trPr();rp.append(OxmlElement('w:cantSplit'))
    # Tables are put in single-column continuous sections to preserve readable comparisons.
    for cell in t._cells:
        cell.width=Inches(6.9/n)
        for p in cell.paragraphs:
            p.style=d.styles['Normal']
            for r in p.runs:r.font.size=Pt(9)
            p.paragraph_format.space_after=Pt(2)
    for before,num in [(True,'2'),(False,'1')]:
        el=OxmlElement('w:p');pp=OxmlElement('w:pPr');sp=copy.deepcopy(d.sections[-1]._sectPr)
        typ=sp.find(qn('w:type'))
        if typ is None:typ=OxmlElement('w:type');sp.insert(0,typ)
        typ.set(qn('w:val'),'continuous');sp.find(qn('w:cols')).set(qn('w:num'),num)
        pp.append(sp);el.append(pp)
        if before:
            anchor=t._tbl.getprevious()
            if anchor is not None and 'TABLE ' in ''.join(anchor.itertext()):anchor.addprevious(el)
            else:t._tbl.addprevious(el)
        else:t._tbl.addnext(el)
for inline in d.inline_shapes:
    ratio=inline.height/inline.width
    inline.width=Inches(6.85); inline.height=int(inline.width*ratio)
    pic=inline._inline.getparent().getparent().getparent()
    picprops=pic.find(qn('w:pPr'))
    if picprops is None:picprops=OxmlElement('w:pPr');pic.insert(0,picprops)
    picprops.append(OxmlElement('w:keepNext'))
    caption=pic.getnext()
    for before,num in [(True,'2'),(False,'1')]:
        el=OxmlElement('w:p');pp=OxmlElement('w:pPr');sp=copy.deepcopy(d.sections[-1]._sectPr)
        typ=sp.find(qn('w:type'))
        if typ is None:typ=OxmlElement('w:type');sp.insert(0,typ)
        typ.set(qn('w:val'),'continuous');sp.find(qn('w:cols')).set(qn('w:num'),num)
        pp.append(sp);el.append(pp)
        if before:pic.addprevious(el)
        elif caption is not None:caption.addnext(el)
d.core_properties.title='Early Layer Sensitivity for Subgroup Audit Prioritisation with GradientGuard'
d.core_properties.author='Didula Fernando'
d.core_properties.comments='Editorial revision only. Competing-interest declaration requires author confirmation. No new experimental results.'
d.save(P/'GradientGuard_IEEE_Access_v1_3.docx')
after={f:digest(Path(f)) for f in protected}
assert after==protected,'Prior manuscript changed'
(P/'qa'/'preservation_after.json').write_text(json.dumps({'unchanged':True,'files':after},indent=2),encoding='utf-8')
print(json.dumps({'paragraphs':len(d.paragraphs),'tables':len(d.tables),'figures':len(d.inline_shapes),'old_files_unchanged':len(protected)}))
