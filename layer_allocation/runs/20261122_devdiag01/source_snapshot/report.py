"""Supervisor-style appraisal of the fixed development diagnosis, not a new study."""
from diagnose import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mdtable(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():lines.append('| '+' | '.join(f'{row[c]:.4f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)

def report():
    verify();assert read(O/'analysis_complete.json')['finished']
    scores=pd.read_csv(O/'reconstruction_scores.csv');variance=pd.read_csv(O/'variance_decomposition.csv');support=pd.read_csv(O/'cohort_support.csv');ws=pd.read_csv(O/'within_family_summary.csv');risk=pd.read_csv(O/'existing_oof_within_family.csv');noise=pd.read_csv(O/'panel_sampling_diagnostics.csv');pairs=pd.read_csv(O/'within_family_pairs.csv');family=pd.read_csv(O/'family_support.csv')
    main=scores[(scores.cohort=='eligible')&(scores.panel=='A')&(scores.bank=='M')]
    count=int((main.R2>=.8).sum());median=float(main.R2.median());mixed=int(len(family[(family.cohort=='eligible')&family.mixed]))
    robust=[]
    for (rep,z),g in ws[ws.cohort=='eligible'].groupby(['representation','feature']):
        if len(g)!=2:continue
        positive=(g.concordance>.5)&(g.leave_one_family_min>.5);negative=(g.concordance<.5)&(g.leave_one_family_max<.5)
        if (g.max24_adjusted_p<CFG['descriptive_alpha']).all() and (positive.all() or negative.all()):robust.append({'representation':rep,'feature':z})
    # All such findings remain retrospective development clues, never promotion gates.
    redundancy='Substantial linear reconstructability in at least four of six features' if count>=4 else 'Fixed linear reconstruction does not establish broad redundancy of the six features'
    verdict='Pause the current sensitivity-specific claim; narrow the FYP or justify a materially different representation before more classifier training.'
    write(R/'decision.json',{'attempt':CFG['attempt'],'status':'complete','post_hoc':True,'development_only':True,'primary_cohort':support[support.cohort=='eligible'].to_dict('records')[0],'M_reconstruction_A_median_R2':median,'features_R2_at_least_0_8':count,'reconstruction_assessment':redundancy,'descriptively_stable_within_features':robust,'within_family_evidence_limited':True,'recommendation':verdict,'heldout_gate_overturned':False,'manuscript_updated':False,'no_new_classifiers_or_risk_forecasters':True,'preservation':read(O/'preservation_after.json')})
    # Plot both panels and all six features; never show only a favourable feature.
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,panel in zip(axes,['A','B']):
        for bank in CFG['reconstruction_banks']:
            q=scores[(scores.cohort=='eligible')&(scores.panel==panel)&(scores.bank==bank)].set_index('feature').loc[Z]
            ax.plot(range(6),q.R2,marker='o',label=bank)
        ax.axhline(0,color='gray');ax.axhline(.8,color='gray',linestyle=':');ax.set_xticks(range(6),[z.replace('_share_','\n') for z in Z],fontsize=8);ax.set(title=f'Panel {panel}: family-held-out reconstruction',ylabel='OOF R-squared',xlabel='Fixed sensitivity feature');ax.legend(title='Input bank')
    fig.tight_layout();fig.savefig(R/'reconstruction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,rep in zip(axes,['raw','M_residual']):
        for panel in ['A','B']:
            q=ws[(ws.cohort=='eligible')&(ws.panel==panel)&(ws.representation==rep)].set_index('feature').loc[Z]
            ax.plot(range(6),q.concordance,marker='o',label=panel)
        ax.axhline(.5,color='gray',linestyle='--');ax.set(ylim=(-.03,1.03),title=f'{rep}: equal-family concordance',ylabel='Higher feature ranks failing run first');ax.set_xticks(range(6),[z.replace('_share_','\n') for z in Z],fontsize=8);ax.legend(title='Probe panel')
    fig.tight_layout();fig.savefig(R/'within_family.png',dpi=150);plt.close(fig)
    recon=scores[scores.cohort=='eligible'].pivot(index='feature',columns=['panel','bank'],values='R2').loc[Z];recon.columns=[p+'_'+b+'_R2' for p,b in recon.columns];recon=recon.reset_index()
    v=variance[(variance.cohort=='eligible')&(variance.panel=='A')]
    r=risk[(risk.cohort=='eligible')&(risk.family=='equal_family_mean')]
    sn=support[support.cohort=='eligible'].iloc[0]
    paired=pairs[(pairs.cohort=='eligible')&(pairs.panel=='A')&(pairs.representation=='raw')&(pairs.feature==Z[0])]
    text=f'''# GradientGuard: bounded development-only diagnosis

Attempt **{CFG['attempt']}**. Post-hoc exploratory diagnosis of development evidence from `20261122_fresh02`. The previous held-out result is already known; this work does not restore untouched evaluation or validate a newly selected method.

## Supervisor-style conclusion

**{verdict}** The diagnosis is useful for explaining limitations, not for rescuing the previous primary tie. {redundancy}. Within-setting outcome comparisons involve only **{mixed} source-normal mixed-outcome families**. That is too narrow to establish a transferable model-level failure signal. These conclusions do not mean all layer information is useless or that the original implementation was incorrect.

My assessment as a research supervisor would be: the implementation and controlled negative result are credible FYP achievements. The sensitivity-specific superiority claim should not currently carry the dissertation or journal argument. Preserve the completed investigation; discuss a narrower comparative contribution with the actual supervisor. A new method or benchmark is warranted only after identifying a distinct, testable mechanism and agreeing a realistic scope. This document is advisory and is not the supervisor's institutional approval.

## What ran and what stayed fixed

Only saved development feature tables, validation-based labels, original family folds, bank column definitions and existing development OOF scores were semantically loaded. No evaluation features, predictions or test outcomes were loaded into these analyses. Protected evaluation files were byte-hashed for preservation only. No binary pipelines were loaded, no CNN was trained, and no failure forecaster was fitted or retuned. Fixed auxiliary Ridge regressions predicted the six features themselves, without failure labels, at alpha=1 and CPU float64. Preprocessing and target scaling were fitted inside each training fold. No grid search, bank selection or feature deletion occurred. Both probe panels use identical development identities and family folds.

The plan, seed, exact formulas, source hashes, input hashes and preservation manifest were frozen before new statistical calculations. The full recorded package lock matches. Synthetic tests are separate. No previous artifact, proposal, manuscript, scientific gate or predictor freeze was amended. The four unavailable development models remain unavailable and were not replaced.

## Development support

{mdtable(support,['cohort','N','families','failures','nonfailures','mixed_family_models','folds'])}

Primary source-normal analysis therefore includes {int(sn.N)} models, but only {len(paired)} failure/nonfailure pairs in {mixed} mixed families inform within-family discrimination. Of these pairs, {int(paired.same_construction.sum())} share a source construction and differ by initialisation; the others also differ by construction. Pairs share models and are not independent experimental units. All homogeneous families remain in reconstruction and variance analyses; they cannot contribute a failure/nonfailure comparison within their family. Family-level pair counts and identities are retained.

## Do the six features repeat existing information?

The table reports out-of-fold R-squared when reconstructing each Z feature from B (conventional/global), N (adds norm-product allocation) and M (adds rich structural statistics/bias). Validation families are absent from that fold's regression training. A score near1 means successful reconstruction; zero is no better than the pooled target mean; negative is worse. The separately exported skill score uses the fold-training mean baseline. All six features are retained in this report, including poor or negative reconstruction scores.

{mdtable(recon,list(recon.columns))}

Panel-A M reconstruction has median R-squared **{median:.4f}**, with **{count}/6** features reaching the predeclared descriptive0.8 marker. High reconstructability supports possible information overlap, but does not prove that residual sensitivity has no relationship with outcomes. Low R-squared only limits this fixed linear reconstruction: nonlinear overlap, regularisation and small family coverage may still matter. These regressions are not failure-prediction models and their scores cannot be compared directly to audit discoveries.

![All six reconstruction results on both panels](reconstruction.png)

## Is variation mostly between settings?

{mdtable(v,['feature','between_fraction','within_fraction','sd'])}

Fractions are an exact run-weighted sample sum-of-squares decomposition, not estimates of population variance components. Large between-family fractions mean a feature mainly separates settings in these data; they do not establish which setting attribute causes the difference. Within-family variability can contain training-seed effects, source-construction effects and probe variation. Matched norm-product correlations and paired-panel noise/spread tables report those descriptive limits separately. Panel A/B comparisons do not measure numerical jitter or provide independent dataset replications.

## Can the features separate outcomes within one setting?

The fixed orientation is larger feature means greater failure risk: concordance1 means every failing/non-failing pair is correctly ordered,0 reverses that ordering and0.5 is tied/random ordering. A value below0.5 may describe an inverse association; no direction was selected to build a new predictor. Each mixed family receives equal weight. M_residual subtracts family-held-out M reconstruction; it does not residualise outcome labels or establish conditional causal effects.

![All feature directions on both probe panels](within_family.png)

{mdtable(ws[ws.cohort=='eligible'],['panel','representation','feature','concordance','leave_one_family_min','leave_one_family_max','two_sided_conditional_p','max24_adjusted_p'])}

Conditional label reassignment preserves each family's failure count. The maximum-departure adjustment covers all24 raw/residual feature-panel combinations within each cohort. Exact/Monte Carlo counts are recorded in `conditional_permutation.json`; Monte Carlo uses a frozen seed and plus-one counts. These are retrospective descriptive randomisation diagnostics, relying on an exchangeability assumption across constructions/initialisations. They do not eliminate the small-family problem, shared images, prior research selection or multiplicity across the research history. There are **{len(robust)}** feature/representation combinations meeting the descriptive same-direction, adjusted-p and family-omission consistency check on both panels. Even such a pattern is only a development clue, not a passed independent gate.

## What do the already-existing forecasters show within families?

{mdtable(r,['model','concordance'])}

These use the previously saved development out-of-fold risk scores, with no new training. Their hyperparameters and banks were selected using development outcomes, so this is optimistic development evidence rather than an independent test. A tiny within-family sample can produce coarse changes in concordance. It cannot establish robust within-setting seed discrimination or overturn the completed held-out result.

## My recommended decision

1. Keep the original and fresh results, including their ties, panel loss and any smaller-control improvements. Do not rewrite the primary endpoint as a success.
2. Explain the project as a tested investigation of the incremental value of layer information. The current evidence does not justify claiming sensitivity-specific novelty, a causal shortcut detector or journal readiness.
3. Present this diagnosis to the actual supervisor before further compute. If continuing, require a clearly different, development-justified hypothesis; changing learners, budgets or features against the consumed evaluation set is not fresh validation.
4. If no such hypothesis is defensible within the timeframe, finish a rigorous comparative/negative-result FYP with the working audit application and reproducible evidence. More runs of an unchanged representation are not the default next step.

No manuscript version or proposal was changed, and no follow-on training is launched. The diagnosis may suggest reasons for the old result but cannot identify a unique explanation or prove mathematical redundancy.

## Files and reproduction

New code: `layer_allocation/development_diagnostics`. Exact PowerShell commands: `instructions/STEP_BY_STEP.txt` in that folder. Numerical outputs: `layer_allocation/runs/{CFG['attempt']}`. Report and figures: `layer_allocation/reports/{CFG['attempt']}`. Read `plan_freeze.json`, `semantic_read_audit.json`, `preprocessing_audit.json`, `preservation_after.json`, `conditional_permutation.json`, all score/pair tables and `tests.xml`. The results are conditional on the fixed constructions that passed the official generator; missing high-alignment constructions remain disclosed.
'''
    (R/'REPORT.md').write_text(text,encoding='utf8');verify()
    write(O/'completion.json',{'finished':True,'time_ns':time.time_ns(),'analysis_seconds':read(O/'analysis_complete.json')['seconds'],'source_hashes':source_hashes(),'report_sha256':sha(R/'REPORT.md'),'decision_sha256':sha(R/'decision.json')})
    manifest={str(p.relative_to(BASE)):sha(p) for folder in [O,R] for p in folder.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix!='.log'}
    write(O/'artifact_hashes.json',manifest)
    print(json.dumps({'report':str(R/'REPORT.md'),'M_R2_median':median,'R2_at_least_0_8_count':count,'mixed_families':mixed,'stable_within_feature_count':len(robust)}),flush=True)

if __name__=='__main__':report()
