# Selected trained checkpoints

This folder contains 31 final selected checkpoint files for the completed DINOv3 depth-2/4/8 study, matched DINOv3 F1/F2 study, JEPA predictor-specialization study, DINOv3 local-weighting study, and JEPA block-shuffle study. Every copied checkpoint was checked against the SHA-256 manifest from its saved experiment output; see `CHECKPOINT_MANIFEST.json`.

DINOv3 depth checkpoints larger than GitHub's ordinary Git file limit use Git LFS. After cloning, run `git lfs install` and `git lfs pull` to fetch them. The official DINOv3 ViT-B/16 base checkpoint is not included; its exact source, SHA-256 and license are recorded under `../experiments/dinov3_D0_D1_D2/provenance/` and `../experiments/dinov3_D0_D1_D2/licenses/`.

The JEPA predictor checkpoints require the grass-adapted I-JEPA encoder used in the experiment. That 5.14-GB adapted-source checkpoint is not included in this Git repository. The raw CSIRO and GrassClover image datasets are also not included. Intermediate optimizer/resume checkpoints and CPU/GPU smoke-test weights are omitted; this folder contains final selected model states only.

DINOv3-derived files are distributed under the supplied Meta DINOv3 license. I-JEPA-derived files are subject to the included I-JEPA CC BY-NC 4.0 license. Check each license before reuse; no blanket license is granted for this project or source data.
