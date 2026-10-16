# Capstone — Img2BioMass research outputs

This repository preserves the current curated Img2BioMass research and journal-submission package under [`outputs/`](outputs/). The folder structure and filenames inside the verified package are preserved.

Start with [`outputs/README.md`](outputs/README.md) and [`outputs/SUBMISSION_READINESS.md`](outputs/SUBMISSION_READINESS.md). The package includes the revision-10 manuscript and supplement, editable LaTeX, experiment-specific code and exact notebook sources, saved predictions, fold assignments, metric calculations, training histories, figures, explainability artifacts, and verification reports. Training and metric files remain with their corresponding experiment outputs.

## Scope and reproducibility

The `outputs/` directory contains the curated manuscript/reproducibility package, experiment metrics and training histories, saved Kaggle stdout logs under `kaggle_run_logs/`, plus 31 final selected checkpoints across the completed studies. The `.ipynb` files are source snapshots; the saved run logs are separate files. Larger DINOv3 depth checkpoints are stored with Git LFS. Its inventory and SHA-256 manifests record included artifacts. Raw CSIRO and GrassClover image datasets, the official DINOv3 base checkpoint, the 5.14-GB grass-adapted I-JEPA encoder, and intermediate optimizer/resume checkpoints are not included. See `outputs/README.md` and `outputs/trained_checkpoints/README.md` for artifact scope and licenses.

Reported development metrics have the fold, image-count, seed, and other limitations described in each experiment report. They should not be presented as competition scores or full out-of-fold results unless the relevant report explicitly verifies that status.
