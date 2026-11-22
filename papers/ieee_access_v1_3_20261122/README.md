# GradientGuard IEEE Access v1.3

Deliverable: `GradientGuard_IEEE_Access_v1_3.docx`. A matching rendered PDF is provided alongside it. Read `REVISION_RESPONSE.md` for changes, verification and outstanding author declarations.

This is a new, separate editorial attempt. The matching v1.2 source/PDF and its evidence directory were hashed before and after building; all 721 prior files remained unchanged. The scientific virtual environment, legacy implementations and archived experiments were not edited or executed.

The new document uses the official publisher Word template as a style reference. It contains editable text, eight native tables and nine native Word equation objects. Three scientific figures were converted from the existing vector sources at four-times rendering scale; their scientific content was not changed.

`build_revision.py` recreates the DOCX from the preserved v1.2 source using project-local Pandoc and the official reference document. Rendering uses project-local LibreOffice extracted without installing into a global environment. These document tools are separate from the scientific numerical stack. `cache` and `tmp` are local tools only and are not committed. No scientific Python packages were installed or changed.

`qa` contains the input extraction, publisher BibTeX, preservation audit and final render checks. The publication date and DOI lines explicitly identify publisher assignment rather than inventing production metadata.

Remaining author fact: competing interests. This must be confirmed before treating the manuscript as submission-ready. No new experimental evidence, reference similarity certification or guaranteed reviewer score is claimed.
