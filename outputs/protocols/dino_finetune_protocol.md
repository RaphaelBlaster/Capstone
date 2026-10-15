# DINOv3 matched continuation protocol — 10 October 2026

Implementation only. No Kaggle submission, GPU smoke or full training authorized now.
Frozen D2 remains the verified 0.7800650684 exploratory reference; no new scientific scores exist.

## Repository inspection and fixed settings

Actual D2 is `DINODensityHead`: shared Linear768→256/GELU/Linear256→3, softplus positive Green/Dead/Clover primitives, equal-area mean over two complete views and 8×8 cells. GDM=Green+Clover, Total=GDM+Dead. Official `x_norm_patchtokens` at 512 are [2B,1024,768]; CLS/four register tokens excluded. Differentiable disjoint4×4-token averaging gives [B,2,8,8,768]. Coarse4×4 cells average embeddings before the same nonlinear head.

Exact unchanged objective: raw five-target weighted MSE with weights .1/.1/.1/.2/.5, divided by saved training-fold scalar scale², plus .10 times mean-square disagreement of the three primitive fine/coarse image-level predictions, divided by scale². `head_loss(...,'D2')` is reused verbatim. No log targets or local biomass labels.

The executed D2 DINO cache used identity orientation only. F1/F2 therefore use the SAME unrotated/unshuffled complete1000×1000 left/right halves, PIL BICUBIC512 and ImageNet normalization. They do not add rotations. Current official backbone remains eval during differentiable training to preserve D2's rotary-position behavior (eval does not disable gradients). Training does not call frozen `extract_dense`, use cached features, detach tokens, or enclose F2 forward in no_grad.

Start every arm independently from authentic official B16 weights and fold-specific D2 `head_selected.pt` (200epochs/final40SWA). Strict hash checks and state loads. Preserve saved head.scale: fold0 26.252408981323242; fold1 26.9482479095459 (exact tensor authoritative). No head reinitialization or scale refit. Fresh AdamW state because D2 selected head is SWA, not the previous optimizer's last iterate.

F1: all backbone frozen, head trainable. F2: only final2 of12 blocks and patch-output `norm` plus same head. No LoRA, new pooling, JEPA modification or repeated SSL.

Fixed continuation config: 20epochs, two original photographs/microbatch, accumulation8/effective16 except final partial window, deterministic same image permutations per epoch(seed42+fold), no held-out dates/images in training. Every original training photograph once/epoch, no drop-last. Both arms have identical CSIRO optimizer-step counts: fold0 308train/49validation,20steps/epoch=400; fold1 301train/56validation,19steps/epoch=380. 1560 total updates across four runs.

HeadLR3e-5; backboneLR2e-6; cosine schedule by epoch, no restart; AdamW betas(.9,.999),eps1e-8. HeadWD.01; backboneWD0 so shrink-to-zero does not replace L2-SP. Global trainable gradient norm clip1. FP32 master weights; future GPU eligible linear FP8/bfloat16 other. No fallback to T4/fullFP32 training. CPU checks labeled functional only.

L2-SP coefficient **1e6**, fixed before any run: mean squared change across ALL unfrozen backbone scalar parameters relative to detached immutable official initialization, P-normalized. Applied exactly once per optimizer update. D2 loss accumulates in proportion to actual photographs per microbatch/window, including partial final windows. No L2-SP on head/F1. The1e6 coefficient is a fixed scale calibration for the P-normalized term (about14M trainable scalars); coefficient1 makes the per-parameter pull extremely small. It is not a proven optimum or evidence of effective generalization. Initial L2-SP gradient is necessarilyzero; the CPU probe also verifies nonzero pull after an update.

Endpoint: **fixed final epoch20**, not validation-best or new SWA. This is a declared difference from D2's original200/final40SWA endpoint; both continuation arms share it, and start from the saved SWA head. The saved D2 head provides warm initialization; no unequal extra warm-up stage.

## F3 prepared separately, disabled

`composition_fractions` verifies finite nonnegative measured dry grass/clover/weeds, positive drytotal, matching component sum. Genuine261-row CSV yields fractions; weeds are not Dead, grass is not CSIRO Green. A separate Linear768→3 receives mean official normalized patch tokens of a COMPLETE GrassClover photograph resized to512, never cropped. Soft-label cross entropy, provisional fixed auxiliary coefficient.1. Planned one auxiliary microbatch per CSIRO optimizer update; CSIRO schedule/initialization stay identical to F2, while auxiliary exposures/compute are separately reported. No F3 runner is enabled.

Training-only auxiliary gradient gate requires finite/nonzero gradients on shared trainable backbone, and weighted auxiliary-to-biomass gradient norm ratio ≤10. This is a fail-closed sanity boundary, not a guarantee of a balanced optimal objective. Real-image coverage, task-loss/gradient/runtime validation, usage rights, independent review and user execution authorization remain necessary. Current preparation manifest reports exactly which images are locally present/missing. Tiny gradient checks alone do not satisfy those requirements.

## Persistence, reporting and verification

New independent directories; never modify old checkpoint/notebook/cache. Atomic initial and latest checkpoints; latest includes trainable backbone delta, head, optimizer, RNG, epoch/window cursor and losses. Commit after each optimizer update; abrupt failures can replay only the uncommitted window. Full schedule/config/source hashes lock resumption. Strict selected checkpoint reload before final predictions. Delta checkpoint reconstruction requires the exact preserved official backbone plus recorded delta.

Progress bars count actual optimizer updates. Per-epoch training loss/L2-SP/development score recorded; initialization/full final parameter hashes and updated names/gradient maxima prove actual scope. Fixed final predictions, fold/pooled metric parity, per-target errors, 10000 paired acquisition-date draws, scatter/residual/training figures and upper-Total-quintile residuals come from completed saved predictions only. High-mass subset is descriptive; no fitted correction.

105/357 reused dates0/1, nine dates, one seed: exploratory evidence, never fullOOF/competition/independent confirmation. F1−F0 is extra continuation; F2−F1 tests encoder adaptation plus anchoring, not anchoring's separate causal contribution. F3−F2 would test added auxiliary training including its extra compute, not isolate labels versus extra compute.

Static notebook default is prepare and enable_gpu=False. Smoke/full require independent exact-package PASS plus new explicit user authorization; full also requires actual independently verified RTX/FP8 smoke. No dispatch mechanism exists in the new builder. Static review cannot verify an unsubmitted notebook, actual GPU precision, runtime quota or future optimizer updates.

## Authentic-backbone correction verified during CPU smoke

All twelve official B16 qkv.bias_mask buffers AND qkv.bias parameters are entirely zero. These wholly dormant bias tensors cannot receive gradients. F2 keeps final-block qkv.bias frozen and excludes them from L2-SP/optimizer/update requirements; all other final2block and normalization weights train. The new FP8 setup requires this exact zero state, preserves mask buffers across generic linear conversion and verifies the biases remain zero/frozen. Dropping masked-bias multiplication is mathematically identical only under this verified zero state; any partial/nonzero mask state is rejected. Prior frozen D2 starts from identical zero biases, so its initial function is preserved. Existing historical precision/extractor code is unchanged. Actual GPU proof of the new guard remains pending.
