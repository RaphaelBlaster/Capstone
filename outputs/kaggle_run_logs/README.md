# Saved Kaggle run logs

These are raw text stdout logs collected from saved Kaggle notebook runs and copied byte-for-byte from the archived output artifacts. They are stored separately from the `.ipynb` source snapshots, which contain no saved cell outputs. The logs retain progress output and messages actually available from each saved run; they are not reconstructed from metric tables.

`KAGGLE_LOG_MANIFEST.json` records each source artifact, kernel slug/ID where verified, saved version, run classification, byte size, SHA-256, and copy check. The `training/` directory contains the principal completed study runs. `gpu_smoke/` and `gpu_diagnostics/` contain bounded verification runs, including the preserved RED failure and GREEN resume-fix logs. `historical_notebooks/` holds three older DINOv3 competition notebook logs; those recipes and scores are separate from the manuscript's controlled experiments.

These logs are evidence of the saved notebook output, not by themselves proof of every hardware or metric claim. Use each experiment's report, status, and verification files alongside them. Some earlier, cancelled, unavailable, or unsaved Kaggle sessions have no recoverable output log and are not represented here.
