# DINOv3 unfreezing-depth comparison — 11 October 2026

User authorizes a fast, convolution-free depth comparison. Reuse official DINOv3
ViT-B/16 (12 blocks), the saved fold-specific D2 density head, and complete left/right
1000-pixel views resized to512. New arms independently initialize from the SAME
pretrained backbone and saved D2 head, never from the previously fine-tuned F2.

Compare final2,4,8 blocks plus final normalization; no convolution, LoRA, predictor
changes, new pooling or auxiliary F3. Frozen earlier blocks and dormant zero QKV
biases remain unchanged. Same date-group folds0/1 (308/49 and301/56 train/validation),
seed42+fold, identical epoch/sample/accumulation schedule, identity augmentation
matching executed D2, raw five-target weighted biomass loss and D2 consistency.

Fixed20 epochs, batch2 photographs, accumulation8, AdamW betas(.9,.999),eps1e-8,
backboneLR2e-6/headLR3e-5, backboneWD0/headWD.01, cosine by epoch, clip1,
FP8 eligible linears/BF16 other operations/FP32 master weights on actual RTXPRO6000.
Use existing strict deterministic CUDA policy. L2-SP coefficient1e6 is fixed,
penalty=sum squared departure / number of trainable backbone parameters. Increasing
depth therefore decreases per-parameter regularization strength under this fixed
mean-normalized rule; this study tests depth under that rule, not depth independently
of every effective regularization property. No hyperparameter selection on private
competition scores. Fixed final epoch; do not choose validation-best checkpoints.

Preserve all old notebooks/checkpoints. Separate smoke and full directories; retain
initial,latest,selected checkpoints, optimizer/RNG/cursor, exact sample schedules,
per-epoch logs and original-image predictions. Re-run depth2 as a matched reference;
historical F1/F2 stay contextual evidence and are not silently relabeled.

Before full epochs, each depth must pass actual FP8 gradients/intended updates,
frozen-prefix proof, cache-reencoded initial parity, and exact interrupted/resumed
optimizer/weights/RNG/prediction equality. Smoke uses32training photos/2validation
photos, two optimizer updates at the same depth; no scientific score from this probe.
Independent exact-package static review must pass before dispatch. Full execution
is conditional on every in-notebook smoke gate and a conservative measured runtime
budget fitting remaining execution cap/quota. Failure stops; no automatic repair/retry.

Report pooled raw weighted R2 on105reused images, all targets, paired date-bootstrap
differences against depth2, compute, trainable parameter counts and high-Total
residuals. Exploratory development evaluation, NOT fullOOF or competition scores.
