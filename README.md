# Capstone — Img2BioMass research outputs

This repository preserves the current curated Img2BioMass research and journal-submission package under [`outputs/`](outputs/). The folder structure and filenames inside the verified package are preserved.

Start with [`outputs/README.md`](outputs/README.md) and [`outputs/SUBMISSION_READINESS.md`](outputs/SUBMISSION_READINESS.md). The package includes the revision-10 manuscript and supplement, editable LaTeX, experiment-specific code and exact notebook sources, saved predictions, fold assignments, metric calculations, training histories, figures, explainability artifacts, and verification reports. Training and metric files remain with their corresponding experiment outputs.

## Scope and reproducibility

The `outputs/` directory is the curated research package, not a copy of every source dataset or every remote Kaggle file. Its inventory and SHA-256 manifest record the included files. Raw CSIRO and GrassClover image datasets, the official DINOv3 base checkpoint, and large trained backbone checkpoints are not included. Selected DINOv3 D0/D1/D2 regression heads are included. See `outputs/README.md` for the exact artifact and licensing limits.

Reported development metrics have the fold, image-count, seed, and other limitations described in each experiment report. They should not be presented as competition scores or full out-of-fold results unless the relevant report explicitly verifies that status.
