# GradientGuard IEEE Access manuscript v1.2

21 November 2026. Authors requested for this draft: **Didula Fernando and Maninda Edirisooriya**. Institutional affiliations, contact details, authorship contributions/approval and any funding declaration need confirmation before submission. No institution, email, membership, grant, acceptance date or assigned DOI is invented.

## Deliverables

- `GradientGuard_IEEE_Access_v1_2.pdf`: rendered two-column manuscript.
- `GradientGuard_IEEE_Access_v1_2.tex` and `references.tex`: editable LaTeX source.
- `GradientGuard_IEEE_Access_v1_2_source.zip`: portable manuscript source/figure/evidence package, generated at delivery.
- `evidence/verified_results.json`: numerical claims, selected predictor settings and input hashes.
- `EDITORIAL_NOTES.md`: submission-readiness items and evidence boundaries for author/supervisor review.

The article reports the completed feasibility, checkpoint follow-up, 96-classifier outcome PoC and post-hoc diagnostics. V9 supplies related-work framing and prospective discussion only. Its proposed new families, structural controls and Waterbirds study are not presented as executed results. No classifier or forecaster was trained or retuned for this manuscript.

The positive ranking and larger-capacity findings are reported with the original primary result, full-cohort results, uncertainty, family influence and slope limitations. This preserves a credible exploratory interpretation.

## Build and verification

Use the Codex-bundled Python and Tectonic runtimes, or compatible Python with `reportlab`, `pypdf` and `pypdfium2` plus Tectonic 0.17.0. The authoring environment is separate from the preserved scientific `.venv`.

From this directory in PowerShell:

```powershell
.\build.ps1
```

The script prepares manuscript figures/tables read-only, compiles the PDF, checks numerical claims and source hashes, and renders page images under ignored `qa`. Required typesetting packages are cached under this folder. First-time network access may require the execution environment's permission mechanism. No global environment is changed.

## Template provenance

The current official IEEE Access template link was verified at <https://ieeeaccess.ieee.org/authors/preparing-your-article/>. Its direct ZIP download was blocked by the site's browser challenge. This draft therefore uses a public legacy `ieeeaccess.cls` / `IEEEtran.cls` mirror from <https://github.com/TMLlaboratory/latex-template-ieee-access>, with exact downloaded hashes recorded locally. The publisher-class files are not edited.

`spotcolor.sty` is a small, explicit XeTeX compatibility layer: it renders the class's declared full-tint blue as its stated CMYK process colour, avoiding pdfTeX-only spot-ink primitives. Manuscript-local overrides remove the mirror template's example volume/year, logos and DOI line. The source package supports editing and review; before journal submission, migrate/compare with the then-current official template and run the journal's submission checks. This draft is not represented as a fully validated current submission package.

Caches, temporary files, rendered QA PNGs and layout smoke files remain local. The final manuscript, source, vector figures, compact evidence tables and validation records are committed locally under the user's standing automatic-commit preference. A GitHub push is a separate publication action.
