from pathlib import Path
import re,json,hashlib
from docx import Document
from pypdf import PdfReader

p=Path(__file__).resolve().parent
d=Document(p/'GradientGuard_IEEE_Access_v1_3.docx')
text='\n'.join(x.text for x in d.paragraphs)
checks={
 'eight_native_tables':len(d.tables)==8,
 'three_figures':len(d.inline_shapes)==3,
 'nine_native_equations':len(d.element.xpath('//m:oMathPara'))==9,
 'all_table_captions':all(('TABLE '+str(i)+'.') in text for i in range(1,9)),
 'all_figure_captions':all(('FIGURE '+str(i)+'.') in text for i in range(1,4)),
 'abstract_150_to_250_words':150<=len(next(x.text for x in d.paragraphs if x.style.name=='Abstract').split())<=250,
 'author_email': 'didula.20232318@iit.ac.lk' in text,
 'supervisor_email':'maninda.e@iit.ac.lk' in text,
 'formal_availability':'Data and Code Availability' in text,
 'funding_statement':'currently receives no funding' in text,
 'honest_unconfirmed_competing_interests':'requires confirmation by the author' in text,
 'no_unresolved_crossrefs':not re.search(r'\[(?:eq|tab|fig|sec):',text),
 'no_replacement_characters':'\ufffd' not in d.element.xml,
 'nineteen_references':len(re.findall(r'^\[\d+\]',text,re.M))==19,
 'prior_files_unchanged':all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in json.loads((p/'qa/preservation_before.json').read_text()).items()),
}
pdf=PdfReader(p/'qa/GradientGuard_IEEE_Access_v1_3.pdf')
rendered='\n'.join(x.extract_text() for x in pdf.pages)
checks['rendered_pdf_has_all_tables']=all(('TABLE '+str(i)+'.') in rendered for i in range(1,9))
checks['rendered_pdf_has_biography']='Author Biography' in rendered
checks['rendered_no_conversion_mojibake']='â€' not in rendered and '\ufffd' not in rendered
report={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'pages':len(pdf.pages),'scope':'Document assembly and preservation checks only; no scientific experiments or inference rerun.'}
(p/'qa/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
assert all(checks.values())
