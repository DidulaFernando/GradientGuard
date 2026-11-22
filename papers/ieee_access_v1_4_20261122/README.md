# GradientGuard IEEE Access v1.4

Focused revision of v1.3, dated 22 November 2026. No scientific experiment was rerun and no scientific result was changed.

The current IEEE Access author page links to the same Word template archived with v1.3:
https://ieeeaccess.ieee.org/wp-content/uploads/2025/08/Access-Template-2024.docx

The official template has an 8 x 10.875 inch page, not US Letter. That review criticism is incorrect. Its geometry is 11520 x 15660 twips. Restored exact template margins: first section top 1280 twips; subsequent sections top 1300; left/right 740; bottom 1040; header 360; footer 640. The two-column gutter is 400 twips. Full-width tables/figures retain continuous single-column sections. Title size is 22 pt, matching the template. The project/editorial-version footer is removed; a plain PAGE field remains. Publisher-assigned volume/year is not fabricated. Date/DOI assignment lines remain.

Per the author's explicit instruction, removed usage disclosures from the introduction, acknowledgement, and bibliography, and renumbered the remaining 18 references and citations. The supervisor acknowledgement and existing no-funding statement are preserved. This removal must not be interpreted as IEEE submission compliance: current IEEE Access policy requires disclosure of generated content in the acknowledgements. Editing/grammar assistance generally falls outside that requirement. The source v1.3 description covers drafting, figures/tables and experimental software, rather than grammar alone. Author must reconcile the final submission with the actual assistance used and publisher policy.

Official guidance checked:
- https://ieeeaccess.ieee.org/authors/preparing-your-article/
- https://ieeeaccess.ieee.org/authors/submission-guidelines/

Published layout comparison: Younesi et al., "A Comprehensive Survey of Convolutions in Deep Learning: Applications, Challenges, and Future Trends," IEEE Access 12 (2024), DOI 10.1109/ACCESS.2024.3376441; publisher-formatted PDF hosted by the coauthor's institutional repository: https://publikationen.bibliothek.kit.edu/1000169984/152718737 . Verified title metadata, two-column pages, running author/title header and volume/page footers. Also located Shrestha and Mahmood, "Review of Deep Learning Algorithms and Architectures," IEEE Access (2019), DOI 10.1109/ACCESS.2019.2912200; full-PDF retrieval exceeded the web reader limit, so no exact geometry is inferred from it. The actual official Word template, not example papers, controls this revision's dimensions.

Build: run `build_revision.py` using the Codex primary runtime Python. Run `render_revision.ps1` for the documents renderer, project-local LibreOffice 25.8.7, and runtime Poppler. The renderer temporarily loads the extracted OpenSymbol font from a readable temporary file and removes it afterwards; no font installation is persisted. An explicit Cambria Math setting prevents ambiguous equation font selection. Literal equation-array alignment markers emitted by the old converter are removed without changing mathematical operands. Headings stay with the next paragraph. Run `validate_revision.py` after rendering. `qa/provenance.json` records input hashes and template values; `qa/validation.json` records deterministic checks. Prior v1.3 inputs remain unchanged. Other review concerns, including the unconfirmed competing-interest declaration, are outside this revision. Rendering pagination is software dependent; this work does not certify full IEEE submission readiness.
