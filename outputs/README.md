# Img2BioMass — Scientific Reports submission and reproducibility package

This is the complete, curated research package for the current revision-10 manuscript and its supporting analyses. It contains the manuscript and supplement, editable LaTeX, exact submitted training notebooks and source packages, the selected DINOv3 D2 regression heads, held-out prediction tables, fold assignments, training histories, metric calculations, figures, explainability artifacts, and verification reports.

## Start here

- `manuscript/Manuscript_revision10.pdf` — current main manuscript.
- `manuscript/Supplementary_Material_revision10.pdf` — current supplement.
- `manuscript/latex/` — editable source and figure assets used for those PDFs.
- `experiments/` — experiment-specific inputs, exact notebooks, predictions, metrics, plots, and verification evidence.
- `source/` — relevant source modules, build/run tools, and targeted tests.
- `protocols/` — locked experiment descriptions and evidence limits.
- `SHA256SUMS.txt` and `INVENTORY.json` — integrity manifest for this package.

The experiment folders distinguish frozen I-JEPA encoder/predictor transfer, DINOv3 D0/D1/D2, the JEPA–DINO dual study, DINOv3 unfreezing depths 2/4/8, matched DINOv3 F1/F2 continuation, DINO local weighting, and JEPA block reshuffling. Their scores are not interchangeable: folds, image counts, recipes, and checkpoints differ. Each experiment’s own config, predictions, and report are authoritative.

## Evidence limits

The manuscript reports exploratory development results from repeatedly used date-group folds and one seed. The principal 105-image analyses cover 105 of 357 labeled CSIRO photographs, folds 0 and 1, and nine acquisition-date groups. The three-fold shuffle study covers 167 of 357. These are not complete out-of-fold results, protected confirmation, or competition scores. The package preserves raw predictions so the reported metrics can be independently recomputed. No score is represented as evidence of generalization to unseen sites or seasons.

The JEPA analysis supports a bounded result for the tested adaptation, data, and compute. It does not establish that JEPA generally fails. The depth study changes trainable parameter count and the aggregate effect of mean-normalized anchoring together; it is not a depth-only causal comparison. XAI panels show model sensitivity or arithmetic contribution proxies, not measured biomass by pixel/cell and not causal explanations.

## Data and model weights

The original CSIRO and GrassClover image datasets are not duplicated here. Use the source datasets and their terms: [CSIRO Image2Biomass challenge](https://www.kaggle.com/competitions/csiro-biomass/data) and [GrassClover dataset](https://vision.eng.au.dk/grass-clover-dataset/). The package includes the fold/label/prediction tables and source-image-derived figures needed to inspect the reported analyses; image redistribution rights for publication still require confirmation.

The official DINOv3 base checkpoint is not included; its SHA-256, model revision, download provenance, and Meta license are recorded under `experiments/dinov3_D0_D1_D2/provenance/`. The D0/D1/D2 selected regression-head checkpoints remain in the experiment folder. Final selected checkpoints for the completed DINOv3 depth-2/4/8, matched F1/F2, JEPA dual-study, DINOv3 local-weighting, and JEPA block-shuffle experiments are in `trained_checkpoints/`; large depth-study files use Git LFS. The 5.14-GB grass-adapted I-JEPA encoder source, original pretrained backbones, and intermediate optimizer/resume checkpoints are not included. No complete checkpoint archive or raw training-data archive is claimed.

## Reproduction

The exact submitted notebooks and embedded source packages are retained in each experiment’s `notebook/` folder. They expect the documented Kaggle datasets, official pretrained weights, and offline dependency wheels as mounted inputs; they are not standalone CPU scripts. Source modules and targeted unit tests are included separately in `source/`. No new training is triggered by opening this package.

## Journal release status

This package is mirrored in the public repository [RaphaelBlaster/Capstone](https://github.com/RaphaelBlaster/Capstone). The manuscript’s current Code Availability paragraph still contains an illustrative placeholder and must be replaced with the repository URL and a durable version identifier before journal submission. Image-use permissions and weight redistribution terms must be confirmed for any further redistribution. The public code repository does not include the original datasets or large backbone checkpoints. The package does not claim that the paper is submission-ready or that Springer Nature has approved its contents.
