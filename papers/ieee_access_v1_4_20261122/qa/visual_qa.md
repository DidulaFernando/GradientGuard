# v1.4 visual verification

All 15 final PDF pages were rendered with PDFium and visually inspected after the documents renderer completed. Checked title block, column flow, figure/table sizing, equation symbols, captions, section headings, references, acknowledgement, and biography. All pages have the same running header and a plain page-number footer. Headings are kept with following content; acknowledgement begins on page 15 with its text. Eight tables, three scientific figures and nine display equations remain. No scientific result was recomputed.

The missing OpenSymbol font in the extracted LibreOffice distribution initially caused brackets to render as icons. The final renderer loads the existing distribution's OpenSymbol font temporarily from a readable temporary path and removes it on completion. The final PDF contains the corrected glyphs. Native equation objects remain editable in Word. Legacy equation-array alignment markers are suppressed as formatting markup; mathematical operands remain unchanged.

The Word file uses the exact template page-size/margin values. LibreOffice's PDF export rounds page width by about 0.54 pt and height by about 0.01 pt; deterministic validation allows less than one point of export rounding. This does not substitute for a complete IEEE submission-readiness review or resolve the other outstanding scientific/declaration issues.
