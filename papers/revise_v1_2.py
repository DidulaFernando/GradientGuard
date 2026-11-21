from pathlib import Path
import shutil,hashlib,json
base=Path(r'E:\Final Year Project\Implementation\papers')
old=base/'ieee_access_v1_1_20261121';new=base/'ieee_access_v1_2_20261121'
assert not new.exists(),'Preserve existing revision'
new.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p):sha(p) for folder in [base/'ieee_access_v1_0_20261121',old] for p in folder.rglob('*') if p.is_file() and 'cache' not in p.parts and 'tmp' not in p.parts}
for d in ['template','figures','evidence','cache']:shutil.copytree(old/d,new/d)
for p in old.iterdir():
    if p.is_file() and p.suffix not in ['.pdf','.zip'] and p.name!='LayoutCheck.tex':
        name=p.name.replace('v1_1','v1_2')
        if p.suffix in ['.py','.tex','.md','.ps1','.txt']:
            text=p.read_text(encoding='utf8').replace('v1_1','v1_2').replace('v1.1','v1.2').replace('Version 1.1','Version 1.2')
            (new/name).write_text(text,encoding='utf8')
        else:shutil.copyfile(p,new/name)
for d in ['qa','tmp']:(new/d).mkdir()
shutil.copyfile(old/'qa/input_hashes_before.json',new/'qa/input_hashes_before.json')
(new/'qa/prior_draft_preservation.json').write_text(json.dumps(protected,indent=2),encoding='utf8')
p=new/'GradientGuard_IEEE_Access_v1_2.tex';s=p.read_text(encoding='utf8')
def replace(a,b):
    global s
    assert a in s,a[:100]
    s=s.replace(a,b,1)
replace('The research question is: \\emph{Can information about which layers a seemingly healthy model depends on help us choose better models to inspect for hidden subgroup failures?} Here, ``depends on\'\' denotes local loss sensitivity to multiplicative changes in learned weights. It does not imply a causal account of the visual features used by the model. ``Hidden\'\' refers to unavailable terminal group outcomes at the early decision point; the audit groups are predefined.',r'''The operational research question is: \emph{Among classifiers that appear healthy early in training, does the allocation of connection sensitivity across layers improve prioritisation for terminal subgroup-failure audits beyond conventional telemetry, global sensitivity and matched layer-wise structural statistics?} In simple terms, we ask whether knowing where a model's loss is sensitive helps choose which models to inspect. This is local sensitivity to multiplicative weight changes on a specified source probe, not a causal account of visual feature reliance. ``Hidden'' refers to unavailable terminal group outcomes at the early decision point; audit groups are predefined.''')
replace('The strongest supported current claim is that this representation merits a matched conditional-addition test, with the existing observations providing a reason to conduct it.', 'The strongest supported current claim is that this representation merits a matched conditional-addition test, with the existing observations providing a reason to conduct it. Three claims must remain separate: layer allocation adds audit value; coordinate-product sensitivity adds value beyond simpler layer statistics; and that increment justifies its monitoring cost. Evidence for the first does not establish the second or third.')
replace('We use ``non-slope\'\' rather than claiming that L6 contains no history information.', 'We use ``early multi-checkpoint layer summaries\'\' for the prospective L6 focus: its means and spreads retain prefix-distribution information while discarding checkpoint order. Demonstrating value from history requires a fresh L6-versus-L2 comparison; an L6 improvement over E alone cannot establish that multiple observations are needed. L8 remains the original primary experimental candidate, not a retroactively replaced endpoint.')
replace('The main statistical constraint is family diversity.',r'''Table~\ref{tab:claims} distinguishes what the archive establishes from the next study's hypotheses. L6 is a prospective focus selected by this exploratory evidence; the table does not reclassify it as the original primary method.

\begin{table*}[t]
\caption{Claim-to-evidence map. Supported observations remain conditional on the studied software, benchmark and family design. Planned comparisons are not completed findings.}
\label{tab:claims}
\centering\small
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.5cm}YY}
\toprule Claim & Current evidence status & Required next evidence\\\midrule
The specified measurement is correctly implemented & Supported by analytical, numerical and isolation checks & Revalidate extensions without changing the statistic\\
L6 shows encouraging ranking performance & Supported in the exploratory source-normal benchmark cohort & Fresh family-level evaluation with adequate outcome support\\
Layer features improve the original primary audit endpoint & Not demonstrated: all principal methods find eight at nine audits & A prospectively frozen capacity and comparison\\
Connection sensitivity adds beyond matched structural statistics & Unresolved; implemented global controls do not answer this & Identical structural predictor with and without the six L6 shares\\
Multi-checkpoint summaries add beyond endpoint profiles & Not established by the current exploratory comparison & Prospective L6 versus L2 with matched selection\\
The procedure generalises to another benchmark & Unresolved & Separately frozen second-context evaluation\\
Prioritisation yields operational savings & Unresolved; early timing is demonstrated & Incremental monitoring/audit costs and justified utility assumptions\\\bottomrule
\end{tabularx}
\end{table*}

The main statistical constraint is family diversity.''')
replace('A future design should seek adequate development outcome support and precision before spending its final evaluation budget, without replacing unfavourable families or tuning cutoffs after outcomes.', 'The intended primary generalisation is ranking models from previously unseen setting families within the studied benchmark, not necessarily distinguishing initialisation seeds within one setting. Homogeneous outcomes within each eligible family prevent assessing the latter, but do not by themselves invalidate the former. A future design should seek adequate development outcome support and precision before spending its final evaluation budget, without replacing unfavourable families or tuning cutoffs after outcomes. Mixed-outcome families are informative diagnostics, not a requirement imposed retrospectively on every setting.')
replace('The structural bank should include layer weight/gradient norms and their products, distribution statistics for parameters and mean-probe gradients, and shared bias diagnostics.', 'The decisive specificity comparison is a strong conventional/global/structural predictor versus that same input bank augmented with exactly the six L6 shares. The structural bank should include layer weight/gradient norms and their products, distribution statistics for parameters and mean-probe gradients, and shared bias diagnostics. Separate competing feature banks are useful ablations, but cannot substitute for this nested pair.')
replace('Results would remain conditional on two studied benchmark/model contexts, not arbitrary natural-image deployment.', 'New SpuCoMNIST families would test within-benchmark generalisation. A Waterbirds procedure with separately fitted development predictors would test replication of the learning procedure in a second context, not direct transfer of a frozen SpuCoMNIST predictor. Such direct transfer would require a separate compatible protocol. Results would remain conditional on two studied benchmark/model contexts, not arbitrary natural-image deployment.')
replace('An early shortlist is useful when audit slots, specialist time or later data collection must be reserved before terminal outcomes are available.', 'Consider an intended workflow in which a team must reserve specialist audit time or commission a labelled evaluation panel before all candidate training runs finish. An early risk ranking could inform which candidates receive those reservations; historical audited runs would supply the training labels. This is a proposed workflow, not a deployment evaluated here. An early shortlist is useful when audit slots, specialist time or later data collection must be reserved before terminal outcomes are available.')
p.write_text(s,encoding='utf8')
(new/'REVIEW_RESPONSE.md').write_text('''# Final pre-implementation editorial revision: v1.2\n\nThis revision responds to the two v1.1 reviews and the agreed final editorial changes. It adds no training, fitting, re-ranking, new statistical results or literature claims. Subjective review scores are not acceptance probabilities or optimisation targets.\n\n- Replaces ambiguous layer-dependence wording with the operational local-sensitivity research question.\n- Centres prospective work on L6 early multi-checkpoint summaries, preserving original L8/25% primacy. Distinguishes order-free history summaries from directional slopes.\n- Adds a claim-to-evidence table, separating implementation, exploratory ranking, original audit yield, structural specificity, history value, generalisation and economics.\n- Identifies the decisive nested structural pair: identical conventional/global/structural bank, with versus without exactly six L6 shares.\n- Separates unseen-family prediction from within-setting seed discrimination. Mixed-family outcomes are informative, not required in every family.\n- Distinguishes new same-benchmark families, separately fitted second-context replication and frozen-model transfer.\n- Makes the proposed early audit-reservation workflow concrete without claiming savings.\n\nAll reported numerical results are retained. The original tie, uncertainty, full-cohort reversal, family influence and slope limitations remain explicit. Fresh matched structural controls and prospective second-context validation are still necessary; wording cannot establish their outcome or guarantee a review threshold. No implementation study has been launched.\n''',encoding='utf8')
print('Created',new)
