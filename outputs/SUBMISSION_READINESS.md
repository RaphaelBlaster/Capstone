# Submission-readiness checklist (15 October 2026)

## Included and checked

- Revision-10 main manuscript and supplement PDFs plus editable LaTeX, bibliography, Springer Nature class/style files, figure assets, author metadata, and build checks.
- Exact submitted notebooks and source packages for the DINOv3 dual study, DINOv3 depth study, DINOv3 F1/F2 fine-tuning, DINO local-weighting comparison, and JEPA shuffle experiment.
- Saved metrics, fold assignments, raw held-out predictions, figure data, plots, hardware/update evidence, and available independent arithmetic/reproducibility checks.
- D0/D1/D2 fold-specific regression heads, hashes, DINOv3 metadata, and license copy.
- JEPA/DINOv3 explainability panels, attribution arrays, and diagnostic reports.

## Must be resolved before public release or journal submission

1. **Code Availability is still a placeholder in revision 10.** A public repository now exists at https://github.com/RaphaelBlaster/Capstone. Replace the manuscript placeholder with this URL and the selected commit/tag, then rebuild both PDFs.
2. **Photograph rights:** obtain and document permission to reproduce all source photographs used in the main text, supplement, and example/heatmap figures. Dataset access does not itself establish republication rights.
3. **Weights and licenses:** choose and approve a license for the project code, obtain any required permission and durable delivery path for trained checkpoints. The official DINOv3 base weights are omitted here; follow the supplied Meta license and official download provenance. Large trained backbones and the official base checkpoint are not in this Git package; their lineage/hashes are recorded where available. Selected D0/D1/D2 regression heads are included. The separate large release-asset upload previously failed with HTTP 401, so do not describe those larger weights as publicly available.
4. **Final author sign-off:** confirm the revision-10 text, author order, corresponding author details, equal-contribution statement, B.M.S. College of Engineering support, and no-competing-interests declaration with all authors. These statements reflect prior user-provided information but still need coauthor approval.
5. **Data/code availability language:** verify the final statements against current dataset terms and the eventual repository/archive URL. The public repository is available at the URL above; verify dataset terms and do not imply that omitted datasets or checkpoints are downloadable there.

## Scientific interpretation to retain

All reported development estimates reuse a small set of grouped folds and a single seed. They are exploratory. Do not label them full OOF, independent test, competition scores, or confirmatory evidence. Do not claim that JEPA as a model family failed, that DINO depth alone caused the observed ranking, that local contributions are measured local biomass, or that an XAI map identifies causal evidence.
