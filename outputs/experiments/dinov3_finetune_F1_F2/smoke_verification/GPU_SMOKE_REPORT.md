# DINO F1/F2 GPU smoke — NO-GO for full training

The exact reviewed model package ran as private, offline Kaggle Version 1 on NVIDIA RTX PRO 6000 Blackwell Server Edition. Full training and F3 were not launched. The bounded smoke performed 10 optimizer updates in total, including the interrupted/resumed test.

| Check | Actual result |
| --- | --- |
| Reviewed package / saved notebook | Exact package SHA and code/settings match |
| Hardware | RTX PRO 6000 Blackwell, capability 12.0, 97,887 MiB |
| FP8 | Preliminary forward/backward passed; 48 backbone + 1 head eligible FP8 linears, BF16 other operations, FP32 master weights |
| Starting D2 parity, folds 0/1 | Max prediction difference 1.4211e-14 g; tested patch-grid difference 0; fold R² differences 0 |
| F1 updates | 2 per fold; head changed, backbone unchanged |
| F2 updates | 2 per fold; all 28 intended final-two-block/norm tensors changed, other weights unchanged; finite nonzero gradients |
| Serialized checkpoints | Selected/latest files preserved; 9 telemetry hashes independently checked |
| Independent smoke metric calculations | Match saved calculations within 1e-10; not a new development score |
| Interrupted/resumed F2 | **FAIL**: backbone max difference 3.9562583e-6; head 2.1241605e-5; predictions 0.143447876 g |

The cause of resume divergence is unverified. The predeclared requirement was bitwise equality. Its failed gate must be resolved or explicitly reviewed before another full-training authorization; no tolerance was widened and no architecture was changed.

Actual peak allocated/reserved GPU memory was 1,066,982,912 / 1,203,765,248 bytes. Each two-update arm, including validation and atomic checkpoint saves, took 3.01–3.58 seconds. Notebook logs extend to 108.06 seconds; quota usage was 108.370 seconds. Remaining quota is 104989.455 seconds (29.16 hours), reserved zero. The smoke stopped before its final resource summary. A conditional estimate derived from per-arm measurements is 82.46 minutes including a 20% margin and 30-minute allowance; this is not a guaranteed runtime or a passed full-run gate.

Raw logs: `../smoke_launch/watch/execution.log`; readable actual diagnostic lines: `ACTUAL_GPU_TEST.log`. Saved checkpoint files: `dino_finetune_gpu_smoke/fold_0/`, `fold_1/`, `resume_fold0_F2/`. Individual checkpoint SHA-256 values and update/gradient evidence are in the corresponding `UPDATES_*.json`, `metrics.json`, and `RESUME.json`. Downloaded file inventory is `DOWNLOADED_FILES_SHA256.json`. The failed run did not emit its global GPU file inventory; selected/latest hashes were checked individually against telemetry instead.

Independent GPT-6 Medium verdict: `../../dino_finetune_verifier/GPU_VERDICT.md` (repository path: `outputs/dino_finetune_verifier/GPU_VERDICT.md`). All existing JEPA/DINO results and checkpoints remain preserved.
