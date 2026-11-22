# GradientGuard manuscript v1.3 — editorial revision record

Date: 22 November 2026. This is an editorial revision of the supplied v1.2 manuscript, not a new experiment or an independent validation. No classifier was trained, forecaster fitted, prediction freeze regenerated or scientific threshold changed. A reviewer score cannot be guaranteed through editorial changes.

## Main changes

| Review concern | Revision | Remaining boundary |
|---|---|---|
| Template provenance | Editable DOCX generated using the official IEEE Access Word template's styles and page geometry; two-column body, single-column wide objects, publication-date and DOI assignment lines | Publisher identifiers remain unassigned. IEEE performs final production formatting; the generated document is still an author draft |
| Author information | Didula Fernando, IIT Sri Lanka, corresponding-author email, project-focused biography | Latest author instruction supersedes the earlier two-author arrangement. Supervisor is acknowledged, not represented as a coauthor; authorship and acknowledgment should be agreed with the supervisor |
| Supervisor | Maninda Edirisooriya identified as lecturer and project supervisor with supplied institutional email | No invented degrees, appointments or research history |
| Funding | Explicit statement that the project currently has no funding | Based on author confirmation |
| Competing interests | A visible confirmation-required declaration | Author has not supplied this fact; it cannot ethically be replaced with an invented “none” statement |
| Availability | Clearly labelled Data and Code Availability statement; repository, archival commit, excluded binaries and reconstruction boundary | Public code is not represented as a full binary archive |
| Ethics | Explicit description of public benchmark use and absence of new recruitment/private patient data | Does not invent institutional approval or a formal exemption |
| AI disclosure | Identifies assistance, affected sections and software, author responsibility, plus an in-text AI-system citation and reference | Author should check the disclosure against the actual assistance used and the submission form |
| Reference [12] in v1.2 | Replaced generic TMLR listing URL with the publisher-supplied direct OpenReview URL; references renumbered by first citation | The reviewer’s claim that the identifier is wrong is not supported by the publisher record. Direct OpenReview content returned an access challenge during verification |
| Numbering and conversion | Restored eight native Word tables, three figures, nine native equation objects, captions, cross-references and consistent section numbering | Word pagination may vary if fonts or rendering software differ |
| Readability | Explained the uncentred error-product diagnostic, corrected GPU allocation terminology, unpacked the analytical fixture and clarified metric abbreviations | No scientific result was strengthened by wording |
| Statistical claims | Explicitly states that archived AP/AUROC/Brier and regression comparisons lack paired-family intervals; these are descriptive estimates | No new intervals or significance tests were calculated |
| Contribution framing | Positions the contribution as a reproducible exploratory measurement/outcome investigation and bounded candidate representation | Does not claim sensitivity-specific superiority, successful primary-endpoint improvement or broad external validity |

## Evidence that remains unchanged

- 96 completed classifier trajectories; primary held-out source-normal cohort: 36 models, nine families, 12 failures.
- Original nine audits: E, logistic B, logistic L6 and logistic L8 each discover eight failures.
- Retrospective twelve audits: E/B discover eight; L6/L8 discover ten.
- Family uncertainty includes zero; an influential-family omission reverses the comparison.
- Slopes enter the estimator, but add no primary-cohort discovery gain over L6.
- Matched structural controls and independent second-benchmark validation remain unexecuted. Editing cannot substitute for these experiments.

## Publisher verification and review corrections

1. Official submission guidance: https://ieeeaccess.ieee.org/authors/submission-guidelines/ (accessed 22 November 2026). Requires the journal template, matching editable source/PDF and author biographies; requires submitting-account ORCID linkage. This does not establish that every author must print an ORCID in the manuscript or that author photographs are mandatory.
2. Official Word template retrieved from https://ieeeaccess.ieee.org/wp-content/uploads/2025/08/Access-Template-2024.docx . The document internally identifies itself as the April 2024 template; the filename is not evidence that a newer template was omitted. The current publisher download was used.
3. TMLR accepted listing: https://www.jmlr.org/tmlr/papers/ and publisher BibTeX https://www.jmlr.org/tmlr/papers/bib/51TWh8tlSy.bib identify Michael Bohadana, Alon Schneider and Gilad Katz, “Neural Networks Performance Prediction using Weights and Gradients Analysis,” 2026, and explicitly link https://openreview.net/forum?id=51TWh8tlSy . The local publisher BibTeX is retained in `qa/closest_prior.bib`. No invented DOI, volume or page range was added to this unpaginated record.
4. Preprints and software references are not intrinsically invalid IEEE references. They should be labelled accurately. This revision does not claim a new systematic audit of all inherited scientific citations or a plagiarism/similarity certification.
5. A zero-crossing uncertainty interval or tied primary endpoint is a scientific limitation, not by itself a journal-format violation. It requires appropriately bounded claims; it cannot be turned into a positive result through prose.

## Before submission

1. Confirm the competing-interest declaration and replace the clearly identified outstanding statement with the actual declaration.
2. Agree authorship/acknowledgment with the supervisor and obtain permission to include the acknowledgment/contact details.
3. Verify the submitting account's ORCID and all submission-form declarations. No ORCID was invented or inserted here.
4. Author and supervisor should approve every claim, reference and AI-use disclosure; obtain any institutional review and similarity checks normally required.
5. Decide whether to submit a transparently exploratory paper with the current evidence or first complete the separately designed structural/independent validation. This draft cannot promise a 7/10 assessment or publication.

The author's age was intentionally omitted from the professional biography; the confirmed course, institution and project are included. No author photograph was supplied, and none was fabricated.
