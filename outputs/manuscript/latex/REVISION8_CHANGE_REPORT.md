# Revision 8 change report

Revision 8 is a separate copy of revision 7. The TeX and PDFs in revision 7 were left unchanged. This update incorporates conservative Curie copy edits and the actionable text/location compliance changes described below; it does not change scientific results or their interpretation.

- Reduced the abstract from 206 to 196 whitespace-delimited words while retaining the explicit limits on reused folds, single-seed evidence, sensitivity diagnostics, causal interpretation, local biomass and competition evaluation.
- Added main-text citations to Supplementary Fig. S3, Figs. S7–S9, Fig. S11 and Table S2; preserved existing citations to S1, S2, S4–S6, S10 and Table S1. Removed the S3 caption's uppercase row references, consistent with the instruction not to refer to individual SI panels.
- Moved Code availability beneath Methods and capitalized the Data Availability heading. The code/repository statement remains explicitly honest: local archive only; public repository address is a placeholder.
- Added corresponding-author ORCID 0000-0002-5577-0024 (public record verified by project lead), author-provided equal-contribution statement, B.M.S. College of Engineering funding (no grant number supplied), and no-competing-interests statement. These are author-provided facts; final author approval is not represented as complete.
- Applied vetted language edits including “typically pretrained,” a few grammar changes, and the accepted “whereas” revisions. Curie changes that altered technical/statistical meaning or changed completed-study tense were not incorporated; notably, zero proportions and “Total” remain as in revision 7.

## Build and structural checks

Main manuscript and supplementary TeX compile to 11-page and 8-page PDFs. Removed the redundant bibliography-style directive; fresh BibTeX and subsequent LaTeX passes succeed. References are unchanged and no citation/reference markers remain unresolved. The source logs still contain underfull-box warnings and a figure/table-only page (page 5). No overfull boxes or undefined references were observed. Thus the literal zero-warning criterion remains unmet. Inline main figures/legends are retained; the primary first-submission guidance permits figures at their text positions, although the checklist has a different end-placement direction.

Root added lower-case bold panel labels to all 12 identified composite figure assets, removed uppercase row labels from Supplementary Fig. S3, and inspected rendered PDFs. Original revision-7 artwork and underlying plot/image content are preserved. Code availability is inside Methods; author contributions and Additional Information now follow References. No scientific data were regenerated.

See `REVISION8_CHECKS.json` for machine-readable counts, hashes, warning status, and author-pending facts. `cover_letter_DRAFT.md` is not submission-ready: reviewer suggestions/exclusions and final confirmations remain.
