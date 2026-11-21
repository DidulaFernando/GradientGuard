from pathlib import Path
import shutil,hashlib,json
base=Path(__file__).resolve().parent
old=base/'ieee_access_v1_0_20261121';new=base/'ieee_access_v1_1_20261121'
assert not new.exists(), 'Preserve an existing revision'
new.mkdir()
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if p.is_file() and 'cache' not in p.parts and 'tmp' not in p.parts}
for d in ['template','figures','evidence','cache']:shutil.copytree(old/d,new/d)
for p in old.iterdir():
    if p.is_file() and p.suffix not in ['.pdf','.zip'] and p.name!='LayoutCheck.tex':
        name=p.name.replace('v1_0','v1_1')
        if p.suffix in ['.py','.tex','.md','.ps1','.txt']:
            t=p.read_text(encoding='utf8').replace('v1_0','v1_1').replace('v1.0','v1.1').replace('Version 1.0','Version 1.1')
            (new/name).write_text(t,encoding='utf8')
        else:shutil.copyfile(p,new/name)
for d in ['qa','tmp']:(new/d).mkdir()
shutil.copyfile(old/'qa/input_hashes_before.json',new/'qa/input_hashes_before.json')
(new/'qa/prior_draft_preservation.json').write_text(json.dumps(protected,indent=2),encoding='utf8')
p=new/'GradientGuard_IEEE_Access_v1_1.tex';s=p.read_text(encoding='utf8')
anchor='\\section{Connection Sensitivity and Audit Prediction}'
addition=r'''
\subsection{The precise gap and candidate contribution}
The distinguishing question is conditional and decision-specific: does a small allocation representation improve the ranking of source-normal completed runs after the information already supplied by ordinary training telemetry and global sensitivity is considered? This differs from proposing another saliency formula. The derivative, normalisation and descriptive summaries are inherited operations; the candidate methodological contribution is their fixed, low-dimensional use in a carefully separated model-audit forecasting task.

Table~\ref{tab:position} makes the comparison boundary explicit. The distinction is an experimentally testable information contract, not a priority claim for all early layer statistics. A method that requires historical audited models is also different from discovering unknown subgroups without such labels. The strongest supported current claim is that this representation merits a matched conditional-addition test, with the existing observations providing a reason to conduct it.

\begin{table*}[t]
\caption{Position relative to close prior work. Distinct prediction targets and information boundaries motivate the experiment; they do not by themselves establish methodological superiority.}
\label{tab:position}
\centering\small
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.5cm}YYY}
\toprule Prior line & Established information & Distinction investigated here & Evidence needed beyond this pilot\\\midrule
Naik and Monin & Early telemetry and source dynamics predict later outcomes & Fixed 10\%-to-terminal subgroup-audit ranking among source-normal runs & Increment over strong source-side predictors on fresh families\\
SNIP and zero-cost proxies & Weight--gradient saliency and model-ranking totals & Normalised within-model allocation, conditional on total magnitude & A matched global-sensitivity comparison\\
SSNR/T-CET and ZiCo & Layer-aware saliency/gradient aggregation & Six endpoint/mean/spread shares for a completed-run audit decision & Comparison with validated known-aggregation adaptations\\
Weight/gradient predictors & Layer statistics and internal histories predict performance & Coordinate-product allocation as a compact added representation & Rich structural controls with and without exactly the same shares\\
SPARE and subgroup discovery & Early output structure reveals groups for mitigation & Whole-run inspection priority; predefined terminal audit groups & Strong output-signal comparator under matched information access\\\bottomrule
\end{tabularx}
\end{table*}

'''
assert anchor in s;s=s.replace(anchor,addition+anchor,1)
anchor='\\subsection{Prefix features and nested learners}'
addition=r'''
\subsection{A falsifiable specificity argument}
For nonzero $U_l$ in \eqref{eq:norm}, write $S_l=U_l c_l$, where
\begin{equation}
c_l=\frac{\langle|w_l|,|g_l|\rangle}{\|w_l\|_2\|g_l\|_2}\in[0,1].
\end{equation}
This elementary factorisation separates ordinary scale from the co-location of absolute coordinate magnitudes. It supplies a reason to measure allocation, but it does not prove that $c_l$ forecasts group failure. Likewise, a normalised profile is compositional: one block's larger share changes the others. The head, biases and shared denominator must therefore be examined before assigning a depth-specific interpretation.

Let $X$ denote the available conventional/global inputs and $Z$ the six L6 share summaries. Predictive improvement of a bounded learner using $(X,Z)$ over one using $X$ supports an increment for that learner and information set. It is not a direct estimate of conditional mutual information or proof that no alternative statistic can recover the same signal. Repeating the addition after expanding $X$ with structural marginals would provide the missing specificity check. This formulation allows a useful layer-information finding even if connection sensitivity ultimately proves replaceable by simpler statistics.

'''
assert anchor in s;s=s.replace(anchor,addition+anchor,1)
anchor='\\section{Conclusion}'
addition=r'''
\subsection{A bounded validation contract for the revised claim}
The existing results select a research hypothesis, not a new primary result on the consumed test set. The next study should designate L6 prospectively and compare three ordered claims: added audit yield beyond conventional telemetry, beyond a strong global-sensitivity predictor, and a conditional increment beyond a rich structural bank. The matched structural pair must use the same development-selected bank, cohort, folds, preprocessing and learner class, adding only $Z$ to the second input set with comparable hyperparameter allowance. Different fitted regularisation is reported as part of the learning procedure, rather than interpreted as an isolated feature intervention.

The structural bank should include layer weight/gradient norms and their products, distribution statistics for parameters and mean-probe gradients, and shared bias diagnostics. These measurements can be validated using existing checkpoints as engineering fixtures; fitting or selecting against their already known held-out labels would remain exploratory. A claimed sensitivity-specific advance requires fresh evaluation of the fixed pair. This comparison directly tests the alternative explanation identified by both reviewers and the literature audit.

Both benchmark protocols, family allocation, eligibility, failure cutoff, audit capacity, estimand and decision rules must precede new evaluation outcomes. The proposed one-third capacity is motivated by consumed diagnostics and must be disclosed as such. The original nine-audit tie remains unchanged. The proposed second-context Waterbirds experiment would use its own development/evaluation separation; it cannot become independent confirmation if its design or predictor choices are changed in response to the new primary test outcomes. Results would remain conditional on two studied benchmark/model contexts, not arbitrary natural-image deployment.

Development checks should assess the number of failure-bearing families, eligibility, mixed-outcome settings, measurement validity, inference calibration and computational cost. Counts of trajectories alone do not establish precision. Readiness must not depend on observing a favourable L6 advantage, and evaluation settings must not be replaced to obtain balance. If support is inadequate or the structural pair shows no credible increment, the contribution is narrowed accordingly. These are prospective requirements; no new control, benchmark result or interval is claimed in this revision.

\subsection{Operational value as an explicit decision test}
An early shortlist is useful when audit slots, specialist time or later data collection must be reserved before terminal outcomes are available. That is an operating scenario to validate, not an assumed property of this inexpensive benchmark. At equal audit count $k$, define a transparent utility scenario
\begin{equation}
\Delta U=v\,[D_{\mathrm{layer}}(k)-D_{\mathrm{base}}(k)]-\Delta C_m,
\end{equation}
where $v$ is an explicitly assumed value per discovered failure and $\Delta C_m$ is incremental monitoring cost. Shared training and equal-count audit costs cancel under this comparison. A positive observed discovery difference supplies a possible benefit region; it does not demonstrate monetary savings without measured cost and a justified value assumption. If exhaustive auditing is cheaper or the baseline reaches the oracle ceiling, the appropriate conclusion is limited operational headroom. This keeps scientific ranking value, early lead time and practical economics separately assessable.

'''
assert anchor in s;s=s.replace(anchor,addition+anchor,1)
p.write_text(s,encoding='utf8')
(new/'REVIEW_RESPONSE.md').write_text('''# Response to the two v1.0 reviews

This v1.1 revision adds no training, fitting, control results or external validation. Reviewer percentages are subjective assessments, not calibrated acceptance probabilities.

| Concern | Revision possible now | Evidence still required |
|---|---|---|
| Incremental novelty | Explicit prior-work boundary table, conditional-addition claim and scale/co-location decomposition | Fresh matched structural comparisons |
| Retrospective advantage | Preserve original primary result and distinguish prospective L6 hypothesis from post-hoc capacity findings | Frozen new evaluation |
| Limited family diversity | Specify support, family concentration and precision checks as readiness gates | Realised diverse outcomes on new families |
| Generalisation | State independent second-context validation contract and its scope | Executed, frozen Waterbirds study |
| Temporal contribution | Centre the prospective claim on L6; retain L8 as the original primary candidate and slope diagnostic | Independent endpoint/history tests |
| Practical motivation | Add explicit early allocation scenario and incremental-utility equation | Measured incremental costs and a justified use case |

The manuscript's original numerical results and uncertainty remain unchanged. Writing cannot supply the unperformed experiments or guarantee novelty >=7 or publishability >=70%. The strongest achievable revision now is a more precise and falsifiable contribution with a clear evidence boundary.
''',encoding='utf8')
print(new)
