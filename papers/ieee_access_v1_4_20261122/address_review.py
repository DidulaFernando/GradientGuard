"""Update existing v1.4, retaining its prior typography, citations and equations."""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
import csv
import hashlib
import json
import re
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Twips, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
TMP=P/'tmp/review_input'
TMP.mkdir(parents=True,exist_ok=True)
for name in ['GradientGuard_IEEE_Access_v1_4.docx','GradientGuard_IEEE_Access_v1_4.pdf','manuscript_source.tex']:
    if not (TMP/name).exists(): (TMP/name).write_bytes((P/name).read_bytes())
d=Document(TMP/'GradientGuard_IEEE_Access_v1_4.docx')
original=list(d.paragraphs)
source=(TMP/'manuscript_source.tex').read_text(encoding='utf-8')
changes=[]

def tex(text):
    return text.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')

def replace(i,text,prefix=None):
    global source
    p=original[i]
    prior=p.text
    # Keep paragraph properties and the first text run's typography.
    rpr=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    p.clear()
    r=p.add_run(text)
    if rpr is not None: r._r.insert(0,rpr)
    if i==6:
        source,n=re.subn(r'(?<=\\begin\{abstract\}\n).*?(?=\n\\end\{abstract\})',tex(text),source,flags=re.S)
        assert n==1
    elif i in [17,18,19]:
        key='\\item '+prior
        assert key in source
        source=source.replace(key,'\\item '+tex(text),1)
    else:
        prefix=prefix or prior[:55]
        pattern=r'^'+re.escape(prefix)+r'[^\n]*'
        source,n=re.subn(pattern,lambda m:tex(text),source,flags=re.M)
        assert n==1,(i,prefix,n)
    changes.append({'paragraph':i,'before':prior,'after':text})

replace(6,'Early model measurements may help prioritise classifiers for later subgroup-failure audits despite healthy source performance. GradientGuard summarises the allocation of absolute weight-mean-gradient products across network blocks during the first ten percent of training. Analytical and numerical checks establish measurement correctness. An exploratory outcome pilot trained 96 classifiers in 24 setting families. Its source-normal held-out cohort has 36 classifiers and 12 failures: six non-slope layer summaries raise average precision from 0.775 to 0.934, but paired-family ranking intervals include zero improvement and the original nine-audit discovery comparison is tied. A subsequent matched structural-control analysis reuses those outcomes and yields only a panel-dependent increment. A separately frozen fresh-family validation planned 256 classifiers in 64 new families; 248 completed and eight were unavailable because fixed source constructions failed generator validation. Among 106 source-normal evaluation classifiers with 34 failures, a rich structural predictor and the same predictor augmented with six sensitivity shares both discover 26 failures in 35 audits. Their average-precision difference is 0.0005 (95% paired-family interval -0.0057 to 0.0096); AUROC and Brier differences also include zero. Smaller-control and secondary-capacity gains remain, while the sensitivity-specific primary benefit is not demonstrated. The contribution is a reproducible, benchmark-specific evaluation with explicit structural controls and a bounded negative validation result.')
replace(14,'This article reports a staged investigation. Measurement feasibility and a targeted probe/calibration follow-up precede an exploratory 96-classifier outcome pilot and post-hoc capacity/slope diagnostics. The original pilot designated L8 and 25% audit capacity as primary. Its L6 and one-third-capacity observations informed a subsequent reused-outcome structural-control study and then a separately frozen fresh-family validation. The latter tests a matched structural pair at one-third capacity; it does not retrospectively replace the pilot endpoint. We report both the original favourable observations and the completed validation that did not establish the sensitivity-specific primary benefit.')
replace(15,'The pilot layer variants find ten failures at twelve inspections versus eight for conventional/global predictors, but the original nine-audit result is tied and family uncertainty is substantial. In the fresh validation, the rich structural predictor M and its sensitivity-augmented counterpart MZ each find 26 of 34 failures at 35 inspections. Thus the pilot motivates a hypothesis that the stronger test does not support at its predefined primary decision. This negative result narrows the claim from a sensitivity-specific forecasting advance to a reproducible evaluation of a candidate representation.')
replace(18,'We report conventional, global-sensitivity, norm-product and rich structural controls under matched early information, family folds, preprocessing, learner classes and tuning allowances.')
replace(19,'We combine frozen-prediction ablations, paired-family ranking intervals and a fresh-family matched validation, retaining ties, losses, unavailable constructions and the unmet sensitivity-specific primary gate.')
replace(31,'Weight-distribution statistics also predict network accuracy [11]. Bohadana, Schneider and Katz [12] investigate performance prediction using weights and gradients, close technical prior art for internal histories. We do not reproduce their full temporal predictor. We instead implement explicit norm-product and parameter/mean-gradient marginal controls and compare each structural bank with and without the same six sensitivity shares. The fresh matched result does not establish sensitivity-specific superiority over these controls.', 'Weight-distribution statistics also predict network accuracy')
replace(37,'Table 1 states the information boundary relative to close prior work; distinct targets alone do not establish superiority. Historical audited models remain necessary for predictor training. Three claims are evaluated separately: useful layer information beyond conventional/global predictors, coordinate-product sensitivity beyond simpler structural statistics, and operational value relative to monitoring cost. The first has some secondary evidence, the fresh primary specificity comparison is tied, and the third remains unestablished.', 'Table~1 makes the comparison boundary explicit.')
replace(56,'Separate weight/gradient norms do not determine coordinate co-location. For example, weights (2,1) and gradients (2,1) or (1,2) have identical gradient marginals and norms but sensitivity totals five and four. This is an algebraic information distinction, not evidence of subgroup prediction. The matched norm-product and marginal-statistic controls below test whether that distinction adds useful forecast information under the implemented learning procedures.', 'Separate weight/gradient norms do not determine coordinate co-location.')
replace(61,'Let X denote conventional/global inputs and Z the six L6 share summaries. A predictive increment from adding Z is conditional on the bounded learner and the input bank; it is not an estimate of conditional mutual information or a proof that no alternative statistic recovers the signal. We therefore repeat the addition after expanding X with norm-product allocation, parameter/mean-gradient marginals and shared bias diagnostics. The fresh MZ-versus-M result tests this conditional claim and does not demonstrate primary added value.', 'Let $X$ denote the available conventional/global inputs')
replace(66,'The conventional predictor E is selected from compact/rich telemetry, with or without output separation, and two bounded learner classes. B adds four total-sensitivity summaries. L2 adds shallow/deep endpoint shares; L6 adds their means and population standard deviations; L8 adds two slopes. L6 therefore adds six shares, denoted Z. Endpoints are time-anchored, whereas means and spreads discard checkpoint order. L8 remains the original pilot primary candidate. The subsequent validation fixes L6 prospectively and includes L2 to distinguish endpoint-only from multi-checkpoint information; that comparison is secondary to the matched structural test.')
replace(76,'Table 3 separates the original engineering and pilot stages. The conditional 18-prefix feasibility follow-up did not launch because calibration failed its gate. The later authorised 60-epoch pilot does not retrospectively pass that gate. Two subsequent studies are reported separately below: a saved-checkpoint structural analysis using consumed pilot outcomes and a frozen fresh-family validation using new settings. The fresh attempt includes an explicit execution amendment and retained source-construction failures.', 'Table~3 separates the engineering studies from outcome evidence.')
replace(146,'Paired-family intervals for AP, AUROC and Brier differences are now calculated from the archived frozen predictions and reported in Table 8 and the matched-validation tables below. The interval analysis neither refits forecasters nor changes outcomes. The separate regression point estimates in Table 7 remain descriptive; no regression-error interval or improvement claim is made.')
replace(151,'Deep-share additivity is verified at every recorded point. At the pilot 10% endpoint on panel A, no run crosses the descriptive head-at-least-50%-of-deep or bias-at-least-10%-of-weight-total flags. These checks bound simple endpoint explanations but do not establish a layer mechanism. Shared-bias inputs improve pilot ranking without adding primary discoveries. The subsequent N/NZ and M/MZ comparisons directly add the missing matched structural controls.')
replace(159,'The pilot yielded encouraging non-slope layer-summary rankings, with two extra discoveries at twelve inspections and consistent probe variants. The paired-family ranking intervals are wide, and one influential-family omission reverses that capacity comparison. The reused-outcome structural study gives only a panel-dependent MZ increment. In the fresh evaluation the primary MZ-M discovery difference is zero, with ranking and probability intervals spanning zero. The more complete evidence therefore does not establish a sensitivity-specific advantage.')
replace(160,'The pilot two-slope extension reaches the fitted L8 estimator but does not improve its discovery curve over L6. This does not imply that all history is unhelpful. In the fresh study L6 finds three more failures than L2 at one-third capacity, but their AP/AUROC difference intervals include zero, and neither observation overcomes the matched primary tie. Endpoint, prefix-distribution and directional information remain distinct questions.')
replace(161,'The strongest structural alternative is now an executed comparison. M includes conventional/global information, norm-product allocation, layer weight/mean-gradient marginals and shared bias diagnostics; MZ adds exactly Z. Separate regularisation selection is part of the matched learning procedure, not a pure coefficient intervention. M has 186 inputs relative to 107 eligible fresh development models, so dimensionality and bounded learners remain limitations. Nevertheless, the tested conditional specificity claim receives no primary support; simpler controls cannot be dismissed on the basis of the pilot.')
replace(162,'Table 12 separates engineering correctness, exploratory ranking signals and the completed fresh matched-control result. The pilot L8 primary endpoint and fresh MZ-M primary endpoint remain distinct.', 'Table~8 distinguishes what the archive establishes')
# Captions occur inside LaTeX environments; update their literal text separately.
oldcap=original[164].text
original[164].text='TABLE 12. Claim-to-evidence map after matched structural controls and fresh-family validation. Negative and secondary findings retain their original roles.'
for r in original[164].runs:
    r.bold=True;r.font.size=Pt(9);r.font.color.rgb=RGBColor(0,0,0)
source=source.replace('TABLE 8. Claim-to-evidence map. Supported observations remain conditional on the studied software, benchmark and family design. Planned comparisons are not completed findings.','TABLE 12. Claim-to-evidence map after matched structural controls and fresh-family validation. Negative and secondary findings retain their original roles.')
replace(166,'The pilot has only three failure-bearing eligible evaluation families and no mixed-outcome eligible family. The fresh study improves support to twelve failure-bearing and twenty non-failure-bearing eligible families, including four mixed-outcome families, but still shares MNIST identities and outcome panels. Whole-family resampling preserves nested constructions and initialisations; it does not quantify arbitrary dataset shift or forecaster-selection uncertainty. The fresh interval spanning zero is not an equivalence proof. Missing extreme-alignment constructions further condition the result on the available pool.')
replace(168,'No Waterbirds, pretrained-model transfer or other external benchmark result is presented. The fresh 256-planned-run study is within SpuCoMNIST, with 248 completed classifiers and eight data-unavailable trajectories. Prediction depth and full reproductions of related temporal/output methods remain omitted comparators. The improved nonlinear controls address the historical minimum-leaf limitation in the new studies, while the original pilot predictions remain unchanged.')
replace(169,'A. Implications of the completed validation', '\\subsection{A bounded validation contract for the revised claim}')
source=source.replace('A. Implications of the completed validation','\\subsection{Implications of the completed validation}',1)
replace(170,'The fresh protocol already performed the matched addition originally proposed for the next study. Implementation and evaluability gates passed, while the useful-gain, positive-interval and panel-consistency requirements failed. We retain that decision rather than promote a favourable learner, smaller control or secondary budget. The present evidence supports narrowing or revising the sensitivity-specific representation before further validation.')
replace(171,'A materially revised representation needs its own rationale, matched structural controls, development selection and untouched evaluation. Reusing the present outcomes for feature design remains exploratory. A future external-context study must distinguish separately fitted replication of a learning procedure from direct transfer of a frozen predictor. Neither is established by this within-benchmark evaluation.')
replace(172,'The one-third capacity was chosen using consumed pilot diagnostics but frozen before fresh evaluation outcomes. That order makes it the fresh study primary budget without making the earlier twelve-audit result prospective. The construction-availability amendment was made after development outcomes were known and before fresh evaluation training; it changed execution eligibility only, retained fixed identities and did not select replacements by performance.')
replace(173,'Counts of training trajectories alone do not establish inferential precision. Readiness must assess outcome-bearing families, eligibility, valid grouped folds and measurement reliability without requiring a favourable layer advantage. The fresh study met those support requirements and still did not demonstrate the predefined benefit. More data or a second benchmark cannot be assumed to make the current representation succeed.')
replace(179,'GradientGuard provides a reproducible framework for measuring early layer-wise connection sensitivity and testing its value for subgroup-audit prioritisation. The exploratory pilot shows favourable non-slope ranking point estimates but an original primary discovery tie and wide paired-family uncertainty. Matched norm-product and rich structural controls weaken a sensitivity-specific interpretation. In the separately frozen fresh-family validation, MZ and M both discover 26 failures in 35 audits among 106 source-normal classifiers; ranking/probability difference intervals include zero. Smaller-control and secondary gains remain bounded observations. The completed evidence establishes measurement correctness and a transparent negative primary specificity result within SpuCoMNIST, without demonstrating causal shortcut detection, cross-benchmark transfer or operational savings.', 'GradientGuard\\ provides a reproducible framework')
replace(182,'The evidence directory identifies source hashes and scalar tables. Original stages are 20261116_01, 20261116_03, 20261116_poc01 and 20261118_01. The matched saved-checkpoint study is 20261121_structural01. The fresh development attempt 20261121_fresh01 and amended completed validation 20261122_fresh02 retain their execution history and fixed identities. These are linked evidence stages, not independent confirmations. The manuscript interval analysis uses frozen risks and terminal outcomes from these archives without retraining or retuning.')
replace(190,'The author declares no competing financial interests or personal relationships that could have influenced the work reported in this article.')

# Correct old claim-map rows in both representations.
claim_rows=[
['The specified measurement is correctly implemented','Supported by analytical, numerical and isolation checks','Revalidate changes to the statistic'],
['L6 improves pilot ranking point estimates','Exploratory; paired-family intervals include zero','Do not promote to established superiority'],
['Layer features improve the original primary endpoint','Not demonstrated: eight discoveries for all principal methods at nine audits','Retain the original primary tie'],
['Sensitivity adds beyond rich structural statistics','Fresh primary MZ-M tie: 26 versus 26 at 35 audits','Current sensitivity-specific benefit not demonstrated'],
['Multi-checkpoint summaries add beyond endpoints','Fresh L6-L2 discovery gain is secondary; ranking intervals include zero','Separate this claim from sensitivity specificity'],
['The procedure generalises to another benchmark','Unresolved; fresh settings remain within SpuCoMNIST','Separately designed external-context evaluation'],
['Prioritisation yields operational savings','Unresolved; early timing and monitoring costs recorded','Measure incremental costs and justified utility']]
claim=d.tables[7]
for row,values in zip(claim.rows[1:],claim_rows):
    for cell,value in zip(row.cells,values):
        cell.text=value
        for p in cell.paragraphs:
            for r in p.runs:r.font.size=Pt(9)
start=source.index('\\toprule Claim & Current evidence status')
end=source.index('\\bottomrule',start)
source=source[:start]+'\\toprule Claim & Current evidence status & Required next evidence\\\\\\midrule\n'+'\\\\\n'.join(' & '.join(tex(c) for c in row) for row in claim_rows)+'\\\\'+source[end:]

# Build insertions before fixed original anchors, retaining section geometry.
table5=d.tables[4]._tbl
b=list(d.element.body); ti=b.index(table5)
two_break=deepcopy(b[ti-2]);one_break=deepcopy(b[ti+1])

def addp(anchor,text,style='Normal'):
    p=anchor.insert_paragraph_before(text,style=style)
    if style=='H3':
        p.paragraph_format.keep_with_next=True
        for r in p.runs:r.font.color.rgb=RGBColor(0,0,0)
    return p

def table(anchor,caption,headers,rows,widths):
    anchor._p.addprevious(deepcopy(two_break))
    cp=addp(anchor,caption)
    cp.paragraph_format.keep_with_next=True
    cp.paragraph_format.space_after=Pt(5)
    for r in cp.runs:r.bold=True;r.font.size=Pt(9)
    t=d.add_table(rows=1,cols=len(headers))
    anchor._p.addprevious(t._tbl)
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    pr=t._tbl.tblPr
    tw=pr.find(qn('w:tblW'));tw.set(qn('w:type'),'dxa');tw.set(qn('w:w'),str(sum(widths)))
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+side)
        for k,v in [('val','single'),('sz','4'),('color','D9D9D9')]:el.set(qn('w:'+k),v)
        borders.append(el)
    pr.append(borders)
    for col,width in zip(t.columns,widths):col.width=Twips(width)
    allrows=[headers]+rows
    for i,values in enumerate(allrows):
        row=t.rows[0] if i==0 else t.add_row()
        trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        if i==0:trpr.append(OxmlElement('w:tblHeader'))
        for j,(cell,value,width) in enumerate(zip(row.cells,values,widths)):
            cell.width=Twips(width);cell.text=str(value);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EDF2' if i==0 else ('F5F7F9' if i%2 else 'FFFFFF'));tcpr.append(sh)
            margins=OxmlElement('w:tcMar')
            for side,amount in [('top',65),('bottom',65),('left',90),('right',90)]:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),str(amount));el.set(qn('w:type'),'dxa');margins.append(el)
            tcpr.append(margins)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:r.font.size=Pt(8.5);r.bold=(i==0)
    anchor._p.addprevious(deepcopy(one_break))
    return '\\begin{table*}[t]\n\\caption{'+tex(caption)+'}\n\\centering\\small\n\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule '+' & '.join(tex(c) for c in headers)+'\\\\\\midrule\n'+'\\\\\n'.join(' & '.join(tex(str(c)).replace('\n',' ') for c in row) for row in rows)+'\\\\\\bottomrule\n\\end{tabular}\n\\end{table*}\n'

method=[
('F. Matched structural controls and fresh validation','H3'),
('The saved-checkpoint study remeasures all 96 pilot classifiers at six checkpoints on both original panels. N adds six summaries of norm-product allocation: for each learned layer, U is its weight L2 norm times its mean-probe-gradient L2 norm; shallow/deep shares use the sum over conv1, conv2, conv3 and head plus the original stabiliser. NZ adds exactly Z. M extends N with weight and mean-gradient L2 norms plus mean absolute magnitude, signed population standard deviation, absolute median and absolute 90th percentile for each of those four layers, each summarised by endpoint, mean and population standard deviation. It also includes shared bias norm and bias sensitivity summaries. MZ adds exactly the same six Z columns; head and bias controls are shared, not exclusive to MZ.','Normal'),
('Each pair uses identical cohort, checkpoints, panels, family folds, fold-local imputation/scaling/filtering, learner class and grid allowance. Logistic C is 0.1, 1 or 10. New histogram controls use depth 1/2/3, 100 iterations, rate 0.05, minimum leaf size five and disabled early stopping. This improves the small-fold comparator without altering historical predictions. Development out-of-fold discovery at one-third capacity selects each fit, with AP and frozen ordering as tie-breaks. Regularisation may differ between matched fits. The structural study is post hoc because pilot outcomes were already consumed; it cannot provide independent confirmation.','Normal'),
('The fresh design has 64 new setting families: alignments 0.10/0.30/0.50/0.65/0.80/0.90/0.95/0.99, two background difficulties and learning rates 0.0002/0.0005/0.0015/0.004. A configuration-only split assigns 32 families each to development and evaluation; two constructions and two initialisations per family give 256 planned classifiers. Source reservations, 60-epoch horizon, measurement definitions, health and 0.75 failure rules follow the pilot. Five-fold grouped development CV is valid; the fallback is three folds. E and the strongest non-Z control are selected on development only. E selects the 50-input conventional bank with output separation; M/MZ have 186/192 inputs and both select logistic C=0.1. The rich M bank is the development-selected strongest non-Z control.','Normal'),
('An initial fresh attempt stopped after 124 of 128 development classifiers completed: two fixed extreme-alignment source constructions failed the official group-presence assertion, making four trajectories unavailable. The amendment permits these documented preparation failures, retains all completed development classifiers and all planned settings/seeds, and applies the same policy to evaluation without replacement. Development outcomes were already known; the amended freeze precedes readiness checks and predictor fitting, and fitted predictors precede evaluation preflight/training. Early risks are frozen before one-time terminal test unsealing. The amendment does not relax support, feature banks, learner grids, outcome cutoffs or scientific benefit gates.','Normal'),
('The fresh primary estimand is the logistic MZ-M difference in failures discovered at floor(N/3), normalised as Delta20 in (7), among source-normal evaluation candidates at 10%. The quarter-capacity, other learners, 5% prefix, full cohort and panel variants are secondary. The benefit gate requires Delta20 at least one, a positive lower 95% family bound, superiority over global and development-selected strongest non-Z controls, positive MZ-M gains on both B variants and no negative family-omission difference. The one-third choice is pilot-informed but precedes fresh evaluation outcomes.','Normal'),
('For ranking/probability evidence, identical whole-family draws are used for each paired AP, AUROC and Brier contrast, recomputing weighted empirical metrics with score ties. All ten pilot evaluation families are sampled in 2,000 draws (seed 130005); the structural study uses 2,000 draws (210212); the fresh study uses all 32 evaluation families in 5,000 draws (710005), including families with no eligible candidates. Multiplicities replicate every nested run together. Empty/single-label pools are excluded for the common metric set: 1,942 valid/58 invalid in each pilot-based analysis and 5,000 valid/zero invalid in the fresh analysis. Two-sided percentile intervals are pointwise and conditional on frozen forecasters, excluding selection uncertainty and shared-image sampling uncertainty. Ranking intervals were added retrospectively without fitting or tuning.','Normal')]
anchor=original[102]
for text,style in method:addp(anchor,text,style)
methodtex='\n\n'.join(('\\subsection{'+text[3:]+'}') if style=='H3' else tex(text) for text,style in method)
source=source.replace('\\section{Results}',methodtex+'\n\n\\section{Results}',1)

intervals=list(csv.DictReader((P/'evidence/paired_family_ranking_intervals.csv').open()))
def interval(study,added,base,metric):
    row=next(r for r in intervals if (r['study'],r['added'],r['base'],r['metric'])==(study,added,base,metric))
    def f(x):
        v=float(x)
        return f'{0.0 if abs(v)<0.00005 else v:.4f}'
    return f(row['difference'])+'\n['+f(row['low'])+', '+f(row['high'])+']'

anchor=original[153]
resultstex=[]
def rp(text,style='Normal'):
    addp(anchor,text,style)
    resultstex.append('\\subsection{'+text[3:]+'}' if style=='H3' else tex(text))
def rt(caption,headers,rows,widths):resultstex.append(table(anchor,caption,headers,rows,widths))

rp('H. Paired-family ranking uncertainty','H3')
rp('Table 8 adds paired-family difference intervals to the original point estimates in Table 5. The L6-E AP difference is 0.1585 with interval [-0.1875, 0.8382], and its AUROC difference is 0.1458 [-0.1000, 0.6042]. These wide intervals include zero; the favourable pilot point estimates are not statistically established population gains. AP/AUROC differences between L8 and L6 are exactly zero for all retained family draws; their Brier difference is -0.0000745 [-0.0004923, 0.0002976]. Zero-width rank intervals here describe these frozen score orderings, not equivalence under retraining or new families.')
names=[('logistic_B','B - E'),('logistic_L2','L2 - E'),('logistic_L6','L6 - E'),('logistic_L8','L8 - E'),('histogram_L8','Histogram L8 - E'),('bias_E','Shared-bias E - E'),('bias_L8','Shared-bias L8 - E'),('final_monitor','Final monitoring - E')]
rt('TABLE 8. Original pilot paired-family metric differences with 95% pointwise intervals. Positive AP/AUROC and negative Brier favour the first model. All comparisons use E as control; 1,942 valid and 58 invalid draws.',['Comparison','AP difference [95% interval]','AUROC difference [95% interval]','Brier difference [95% interval]'],[[label]+[interval('original',m,'E',metric) for metric in ['AP','AUROC','Brier']] for m,label in names],[2200,2600,2600,2600])

rp('I. Completed saved-checkpoint structural controls','H3')
rp('The reused-outcome analysis reproduces original E/B/L6/L8 results before fitting its new controls. Its N proxy matches logistic L6 at ten discoveries in twelve audits. Rich logistic MZ adds one discovery over M (ten versus nine), but both B-panel variants tie; histogram M/MZ also tie. The logistic MZ-M Delta20 interval at one-third is [0.0000, 0.6250], with 1,942 valid and 58 invalid whole-family draws. This is a weak, panel-dependent exploratory increment, not independent sensitivity-specific evidence. The improved histogram E reaches nine at nine audits and ten at twelve, strengthening the conventional comparison without replacing the original pilot primary result.')
metrics=list(csv.DictReader((ROOT/'layer_allocation/runs/20261121_structural01/metrics.csv').open()))
def mrow(rows,model,cols):
    r=next(r for r in rows if r['cohort']=='eligible' and float(r['prefix'])==.1 and r['variant']=='A' and r['model']==model)
    return [r[c] if c.startswith('D') else f'{float(r[c]):.4f}' for c in cols]
rt('TABLE 9. Reused pilot outcomes with matched structural controls, panel A at 10% (N=36, F=12). All names except Histogram E/M/MZ denote logistic models; new selection uses one-third capacity.',['Predictor','D9 / D12','AP','AUROC','Brier'],[[label,' / '.join(mrow(metrics,model,['D9','D12']))]+mrow(metrics,model,['AP','AUROC','Brier']) for model,label in [('logistic_N','N'),('logistic_NZ','NZ'),('logistic_M','M'),('logistic_MZ','MZ'),('histogram_E','Histogram E'),('histogram_M','Histogram M'),('histogram_MZ','Histogram MZ')]],[2400,1600,2000,2000,2000])
rp('For logistic MZ-M, the paired AP difference is 0.0053 [0.0000, 0.0269], AUROC 0.0035 [0.0000, 0.0179], and Brier 0.0006 [-0.0014, 0.0027]. Both ranking intervals touch zero. Histogram MZ-M metrics and intervals are all zero because the frozen predictions coincide. Logistic NZ-N gives AP -0.0033 [-0.0207, 0.0000] and AUROC -0.0035 [-0.0179, 0.0000]. Thus the coordinate-product representation cannot claim specificity merely from its superiority to the original weaker E/B fits. M/MZ have 174/180 inputs against 48 eligible development models, a substantial dimensionality limitation.')

rp('J. Completed fresh-family matched validation','H3')
rp('Of 256 planned classifiers, 248 complete all 19,560 updates: 124 reused development and 124 newly trained evaluation models. Eight are data-unavailable and none diverged or remained incomplete. Development failures are family 61 construction 1 and family 62 construction 0; evaluation failures are family 56 construction 0 and family 59 construction 0, each affecting two initialisations. All are at 0.99 alignment. Valid constructions remain in their assigned families; missingness is not assumed random. Evaluation source preflight precedes training and uses no model-performance replacement rule.')
rp('Source-normal development support is 107 classifiers in 29 families, with 38 failures in 14 families, 69 non-failures in 19 and four mixed-outcome families. Evaluation has 106 eligible classifiers in 28 families: 34 failures in twelve families, 72 non-failures in twenty and four mixed-outcome families. All unchanged support requirements pass. The full evaluation cohort has 124 classifiers and 45 failures. Implementation, preservation and information-boundary audits pass; the scientific benefit gate does not.')
fm=list(csv.DictReader((ROOT/'layer_allocation/runs/20261122_fresh02/metrics.csv').open()))
rt('TABLE 10. Fresh source-normal evaluation, panel A at 10% (N=106, F=34). D35 is the predefined primary one-third budget; D26 is secondary. E is the development-selected logistic conventional predictor.',['Predictor','D35 / D26','AP','AUROC','Brier'],[[label,' / '.join(mrow(fm,model,['D33','D25']))]+mrow(fm,model,['AP','AUROC','Brier']) for model,label in [('E','E'),('logistic_B','B'),('logistic_L2','L2'),('logistic_L6','L6'),('logistic_N','N'),('logistic_NZ','NZ'),('logistic_M','M'),('logistic_MZ','MZ'),('histogram_E','Histogram E'),('histogram_B','Histogram B'),('histogram_L6','Histogram L6'),('histogram_N','Histogram N'),('histogram_NZ','Histogram NZ'),('histogram_M','Histogram M'),('histogram_MZ','Histogram MZ')]],[2400,1600,2000,2000,2000])
rp('At the fresh primary budget, MZ and M each discover 26 failures in 35 inspections (Table 10). Delta20 is zero, with the archived 95% paired-family interval [-0.2083, 0.1888], 5,000 valid and zero invalid draws. Random expectation is 11.226 failures and the oracle ceiling is 34, leaving eight discoveries of headroom for each model. The useful-gain target was attainable in principle but not achieved. All six added Z columns survive preprocessing and reach both MZ learner classes; the null increment is not caused by silent feature removal.')
rp('Table 11 shows that fresh ranking and probability differences also span zero for the primary logistic pair: AP 0.0005 [-0.0057, 0.0096], AUROC -0.0004 [-0.0046, 0.0034] and Brier -0.0002 [-0.0010, 0.0005]. These conditional intervals neither prove population equivalence nor establish superiority. Histogram MZ and M coincide. Secondary L6-L2 and NZ-N discovery gains are three at one-third; L6 improves E/B by one at the quarter budget, and histogram L6 improves histogram E/B by one at one-third. Their ranking intervals remain compatible with zero, and none replaces the primary structural comparison.')
pairs=[('logistic_MZ','logistic_M','MZ - M'),('histogram_MZ','histogram_M','Histogram MZ - M'),('logistic_L6','E','L6 - E'),('logistic_L6','logistic_B','L6 - B'),('logistic_L6','logistic_L2','L6 - L2'),('logistic_NZ','logistic_N','NZ - N'),('histogram_L6','histogram_E','Histogram L6 - E')]
rt('TABLE 11. Fresh validation paired-family metric differences with 95% pointwise intervals. Positive AP/AUROC and negative Brier favour the first model. All 5,000 draws are valid; ordinary names denote logistic fits.',['Comparison','AP difference [95% interval]','AUROC difference [95% interval]','Brier difference [95% interval]'],[[label]+[interval('fresh',a,b,metric) for metric in ['AP','AUROC','Brier']] for a,b,label in pairs],[2200,2600,2600,2600])
rp('Panel and family diagnostics retain unfavourable results. Frozen A predictors on B inputs give MZ 27 versus M 28 discoveries, with Delta20 interval [-0.3571, 0.1818]; existing B-development refits give 27 versus 27, interval [-0.1961, 0.1696]. Across all 32 family omissions the panel-A primary difference stays zero. Over the 107 integer capacities from zero to 106, MZ is ahead of M at three, tied at 100 and behind at four, with differences from -1 to 1. These correlated capacities are descriptive. In the full cohort, M/MZ each discover 32 at 41 audits and 25 at 31 audits. The sensitivity-specific primary benefit and panel-consistency gates therefore fail despite adequate evaluation support.')
source=source.replace('\\section{Computational Cost and Reproducibility}','\n\n'.join(resultstex)+'\n\n\\section{Computational Cost and Reproducibility}',1)

cost='The added structural extraction takes 78.0 seconds of summed checkpoint-panel measurement and 23.7 seconds of development fitting in the recorded reused-outcome attempt. The fresh combined conservative GPU-stage ledger is 5.239 hours, including the original stopped development attempt, within its 24-hour cap. Summed completed-model components include 3.934 hours of training, 811.3 seconds of panel-A monitoring and 825.8 seconds for panel B. The fresh recorded allocated GPU peak is 575.0 MiB; compact evaluation-source arrays occupy 9.93 MiB and local classifier checkpoints 275.72 MiB. Monitoring includes telemetry, structural diagnostics and clustering, so these totals do not isolate the incremental cost of Z or establish economic audit savings.'
addp(original[156],cost)
source=source.replace('The numerical stack is Python',tex(cost)+'\n\nThe numerical stack is Python',1)

# Stage table grows without altering historical rows.
t=d.tables[2]
for values in [['Matched structural','96 existing classifiers','Consumed pilot outcomes','Post-hoc conditional-addition test'],['Fresh validation','256 planned; 248 complete','New held-out families','Matched primary benefit not demonstrated']]:
    row=t.add_row()
    for cell,value in zip(row.cells,values):
        cell.text=value
        for p in cell.paragraphs:
            for r in p.runs:r.font.size=Pt(9)
# Add the same rows to the corresponding source table.
st=source.index('\\label{tab:stages}') if '\\label{tab:stages}' in source else source.index('TABLE 3.')
end=source.index('\\bottomrule',st)
source=source[:end]+'\nMatched structural & 96 existing classifiers & Consumed pilot outcomes & Post-hoc conditional-addition test\\\\\nFresh validation & 256 planned; 248 complete & New held-out families & Matched primary benefit not demonstrated\\\\'+source[end:]

d.core_properties.version='1.4'
d.core_properties.comments=''
# Let discussion continue after the final full-width claim table. The inherited
# final-section next-page break otherwise strands three table rows on a page.
from docx.enum.section import WD_SECTION_START
d.sections[-1].start_type=WD_SECTION_START.CONTINUOUS
d.save(P/'GradientGuard_IEEE_Access_v1_4.docx')
(P/'manuscript_source.tex').write_text(source,encoding='utf-8',newline='\n')
(P/'evidence/editorial_changes.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf-8',newline='\n')
used=[]
for attempt in ['20261121_structural01','20261122_fresh02']:
    for name in ['metrics.csv','contrasts.csv','intervals.csv','leave_one_family_out.csv','predictor_freeze.json','protocol_freeze.json','prediction_freeze.json','validation_audit.json','costs.json']:
        used.append(ROOT/'layer_allocation/runs'/attempt/name)
    used.append(ROOT/'layer_allocation/reports'/attempt/'REPORT.md')
used.extend(ROOT/'layer_allocation/runs/20261122_fresh02'/n for n in ['completion_accounting.csv','evaluation_support.json','sensitivity_feature_lineage.csv','final_validation_audit.json'])
(P/'evidence/added_evidence_manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in used},indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'paragraphs':len(d.paragraphs),'tables':len(d.tables),'equations':len(d.element.xpath('//m:oMathPara')),'figures':len(d.inline_shapes),'version':d.core_properties.version}))
