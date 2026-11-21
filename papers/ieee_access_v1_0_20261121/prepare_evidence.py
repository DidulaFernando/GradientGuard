"""Prepare manuscript tables/figures from archived results; no model fitting."""
from pathlib import Path
import csv,hashlib,json,math,shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'Feasibility'
POC=ROOT/'poc/outputs/20261116_poc01'
DIAG=ROOT/'diagnostics/20261118_01'
FOLLOW=ROOT/'followup/outputs/20261116_03'
INPUTS={}
for d in ['evidence','figures','qa','tmp','cache']:(HERE/d).mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):INPUTS[str(p)]=digest(p);return p
def js(p):return json.loads(record(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with record(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def select(data,**criteria):
    return [r for r in data if all(str(r.get(k))==str(v) for k,v in criteria.items())]
def write_csv(name,data):
    with (HERE/'evidence'/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

protocol=js(POC/'protocol.json');audit=js(POC/'reports/validation_audit.json')
first=js(ROOT/'reports/20261116_01/decision.json')
follow=js(FOLLOW/'reports/decision.json');selection=js(FOLLOW/'calibration/selection.json')
costs=js(POC/'reports/costs.json');freeze=js(POC/'predictor_freeze.json')
metrics=rows(POC/'reports/metrics.csv')
marked=rows(DIAG/'tables/marked_budgets.csv')
curves=rows(DIAG/'tables/all_budget_curves.csv')
comparisons=rows(DIAG/'tables/L6_L8_comparisons.csv')
influence=rows(DIAG/'tables/leave_one_family_out.csv')
intervals=rows(DIAG/'tables/bootstrap_intervals.csv')
lineage=rows(DIAG/'tables/slope_coefficients.csv')
replacement=rows(DIAG/'tables/reference_effects.csv')
diversity=rows(DIAG/'tables/family_diversity.csv')
completion=rows(POC/'reports/completion_and_health.csv')
outcomes=rows(POC/'reports/terminal_outcomes.csv')
reliability=rows(ROOT/'reports/20261116_01/tables/reliability_all.csv')
for p in [POC/'reports/REPORT.md',DIAG/'reports/REPORT.md',DIAG/'reports/INTERPRETATION.md',FOLLOW/'reports/REPORT.md',ROOT/'reports/20261116_01/REPORT.md',ROOT/'poc/prediction.py',ROOT/'poc/study.py',ROOT/'src/measurement.py']:
    if p.exists():record(p)
record(Path('E:/Final Year Project/Presentation/GradientGuard_v9_20261121/GradientGuard_Research_Proposal_Didula_Fernando_v9.pdf'))

MODELS=['E','logistic_B','logistic_L2','logistic_L6','logistic_L8','histogram_L8','bias_E','bias_L8','final_monitor']
primary=select(metrics,cohort='eligible',prefix='0.1',panel='A',refit_B='False')
primary=[r for r in primary if r['model'] in MODELS]
assert len(primary)==len(MODELS)
assert audit['all_96_full_horizon'] and audit['groups_per_model']==100
for m in ['E','logistic_B','logistic_L6','logistic_L8']:
    a=select(marked,cohort='eligible',variant='A',model=m,mark='nine');assert len(a)==1 and float(a[0]['discoveries'])==8
    b=select(marked,cohort='eligible',variant='A',model=m,mark='twelve');assert len(b)==1 and float(b[0]['discoveries'])==(10 if m in ['logistic_L6','logistic_L8'] else 8)
assert abs(follow['original_reproduction']['ratio']-3.0501409990602664)<1e-12
assert follow['saved_partition_target']['target_summary']['passing_partitions']==15
assert selection['selected_full_epochs'] is None

write_csv('primary_metrics.csv',primary)
write_csv('marked_budgets.csv',[r for r in marked if r['mark'] in ['nine','twelve']])
write_csv('source_normal_slope_coefficients.csv',select(lineage,cohort='eligible',variant='A'))
write_csv('source_normal_reference_effects.csv',select(replacement,cohort='eligible',variant='A'))
write_csv('source_normal_influence.csv',select(influence,cohort='eligible',variant='A',control='E',view='fraction',requested='1/3'))
write_csv('source_normal_intervals.csv',[r for r in intervals if r['cohort']=='eligible' and r['variant']=='A' and r['control'] in ['E','logistic_B','logistic_L6']])
write_csv('measurement_reliability.csv',reliability)

def text(c,x,y,s,size=8,align='left',font='Helvetica'):
    c.setFont(font,size)
    {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,str(s))
COL={'E':'#415A77','logistic_B':'#B46A35','logistic_L6':'#237C66','logistic_L8':'#752F7F'}
LABEL={'E':'E','logistic_B':'B','logistic_L6':'L6','logistic_L8':'L8'}
def axes(c,x,y,w,h,xmax,ymax,xticks,yticks):
    c.setLineWidth(.5)
    for v in yticks:
        yy=y+h*v/ymax;c.setStrokeColor(colors.HexColor('#DDE2E6'));c.line(x,yy,x+w,yy)
        c.setFillColor(colors.black);text(c,x-6,yy-2.5,v,7,align='right')
    c.setStrokeColor(colors.black);c.line(x,y,x+w,y);c.line(x,y,x,y+h)
    for v in xticks:text(c,x+w*v/xmax,y-12,v,7,align='center')
def curve_plot():
    c=canvas.Canvas(str(HERE/'figures/discovery_curves.pdf'),pagesize=(520,216))
    for j,(cohort,N,F,title) in enumerate([('eligible',36,12,'Source-normal cohort'),('full',40,15,'Full cohort')]):
        x=35+260*j;y=35;w=210;h=142
        axes(c,x,y,w,h,N,F,[0,9,12,N],[0,4,8,12] if F==12 else [0,5,10,15])
        text(c,x+w/2,193,title,9,align='center',font='Helvetica-Bold')
        text(c,x+w/2,181,f'N={N}; observed failures={F}',7,align='center')
        c.setStrokeColor(colors.HexColor('#929AA3'));c.setDash(3,3)
        c.line(x,y,x+w,y+h)
        p=c.beginPath();p.moveTo(x,y);p.lineTo(x+w*F/N,y+h);p.lineTo(x+w,y+h);c.drawPath(p)
        c.setDash()
        for m in ['E','logistic_B','logistic_L6','logistic_L8']:
            rr=sorted(select(curves,cohort=cohort,variant='A',model=m),key=lambda r:int(r['k']))
            assert len(rr)==N+1
            c.setStrokeColor(colors.HexColor(COL[m]));c.setLineWidth(1.1)
            c.setDash(3,1.5) if m=='logistic_L6' else c.setDash()
            p=c.beginPath();p.moveTo(x,y);last=0
            for r in rr:
                xx=x+w*int(r['k'])/N;yy=y+h*int(r['discoveries'])/F
                p.lineTo(xx,y+h*last/F);p.lineTo(xx,yy);last=int(r['discoveries'])
            c.drawPath(p);c.setDash()
        c.setStrokeColor(colors.HexColor('#AAB0B8'));c.setLineWidth(.5);c.setDash(1,2)
        for k in [9,12]:c.line(x+w*k/N,y,x+w*k/N,y+h)
        c.setDash();c.setFillColor(colors.black)
        text(c,x+w/2,7,'Number of models inspected',8,align='center')
        text(c,x+2,y+h-11,'Failures discovered',7)
    for i,m in enumerate(['E','logistic_B','logistic_L6','logistic_L8']):
        xx=113+i*68;c.setStrokeColor(colors.HexColor(COL[m]));c.line(xx,208,xx+15,208)
        c.setFillColor(colors.black);text(c,xx+19,205,LABEL[m],7.5)
    c.save()
def influence_plot():
    rr=sorted(select(influence,cohort='eligible',variant='A',control='E',view='fraction',requested='1/3'),key=lambda r:int(r['omitted_family']))
    assert len(rr)==10
    c=canvas.Canvas(str(HERE/'figures/family_influence.pdf'),pagesize=(255,178))
    x=38;y=38;w=205;h=111;low=-1;high=3
    for v in [-1,0,1,2,3]:
        yy=y+h*(v-low)/(high-low);c.setStrokeColor(colors.HexColor('#DDE2E6'));c.line(x,yy,x+w,yy)
        c.setFillColor(colors.black);text(c,x-5,yy-2.5,v,7,align='right')
    c.setStrokeColor(colors.black);c.line(x,y,x+w,y)
    for i,r in enumerate(rr):
        xx=x+w*(i+.5)/10;v=float(r['Delta20']);yy=y+h*(v-low)/(high-low)
        c.setFillColor(colors.HexColor('#237C66' if v>0 else '#A04D35' if v<0 else '#586675'))
        c.circle(xx,yy,2.4,fill=1,stroke=0)
        c.setFillColor(colors.black);text(c,xx,y-12,r['omitted_family'],7,align='center')
    text(c,x+w/2,11,'Omitted setting family',8,align='center')
    text(c,10,163,'Discovery gain per twenty candidates',8)
    c.save()
def pipeline_plot():
    c=canvas.Canvas(str(HERE/'figures/study_pipeline.pdf'),pagesize=(520,115))
    names=[('Frozen classifiers','24 families; 96 runs'),('Early observations','10% of full updates'),('Development selection','14 intact families'),('Frozen early risks','10 held-out families'),('Terminal group audit','Original and post-hoc views')]
    for i,(a,b) in enumerate(names):
        x=4+i*104;c.setStrokeColor(colors.HexColor('#415A77'));c.setLineWidth(.8);c.roundRect(x,42,96,46,3,stroke=1,fill=0)
        c.setFillColor(colors.black);text(c,x+48,70,a,7.4,align='center',font='Helvetica-Bold');text(c,x+48,55,b,6.8,align='center')
        if i<4:c.line(x+96,65,x+103,65);c.line(x+103,65,x+100,68);c.line(x+103,65,x+100,62)
    text(c,260,20,'Setting families remain intact in splitting, validation and resampling.',8,align='center')
    c.save()
curve_plot();influence_plot();pipeline_plot()

summary={'author':'Didula Fernando','coauthor':'Maninda Edirisooriya','date':'2026-11-21','no_new_training_or_fitting':True,'all_models_complete':audit['all_96_full_horizon'],'classifier_runs':96,'primary':primary,'full_primary':[r for r in select(metrics,cohort='full',prefix='0.1',panel='A',refit_B='False') if r['model'] in MODELS], 'selected_primary_predictors':{k:{'kind':v['kind'],'value':v['value'],'input_count':len(v['cols'])} for k,v in freeze['choices']['eligible_0.1']['parameters'].items()},'first_study':first,'followup':follow,'costs':costs,'input_hashes':INPUTS}
(HERE/'evidence/verified_results.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
baseline=HERE/'qa/input_hashes_before.json'
if baseline.exists():assert json.loads(baseline.read_text(encoding='utf8'))==INPUTS,'Input changed during drafting'
else:baseline.write_text(json.dumps(INPUTS,indent=2),encoding='utf8')
for asset in ['ieeeaccess.cls','IEEEtran.cls']:shutil.copyfile(HERE/'template'/asset,HERE/asset)
print(json.dumps({'verified_inputs':len(INPUTS),'primary_rows':len(primary),'figure_count':3,'selected_primary_predictors':summary['selected_primary_predictors']},indent=2))
