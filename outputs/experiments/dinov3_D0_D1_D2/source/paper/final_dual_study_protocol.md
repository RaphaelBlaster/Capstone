# Locked dual study: implementation record

Contract DUAL-JEPA-DINO-2026-10-09-r1 is authoritative. Its original notebook
SHA is 4d2f7ef85d89c26fff489e6848063927262662f74ff053d1839f16784f93a355.
Completed prior notebooks and checkpoints are retained. This study reuses the
2400-step grass-adapted I-JEPA student, predictor and EMA teacher; it does not
repeat Stage1. Stage2 learns only predictor layers/mask token, retaining official
fixed sinusoidal positions and both frozen encoders. Paired actual mask arrays
and complete20slot schedules are stored per fold. Controls and ten native10x10
jigsaws have identical parent exposures and update counts (385/377 per arm).
Stage3 strictly reloads specialized predictors into four fixed42-token semantic
readouts, then trains the existing raw-loss gated head200epochs/final40SWA.

DINO uses authentic official ViT-B16,512 input per complete view,768-dimensional
spatial-only tokens and a pinned official source. D0 global pooling, D1 shared
nonlinear local density proxy, D2 same inference with lambda0.10 cross-scale
penalty. Frozen weights receive no supervised adaptation. J0/J1 optional
references are omitted to prioritize the contract's bounded primary comparisons;
historical scores are not substituted as matched controls.

The two reused development folds contain105 of357 original photographs/nine
acquisition-date groups/one seed. Augmentation creates exposures, not independent
subjects. Use one global raw weightedR2 and paired10000date-group bootstrap
intervals. No fullOOF, protected confirmation, competition score, causal mechanism,
backbone superiority, novel-algorithm or journal-acceptance claim follows.

Offline quota ruling: Kaggle API access cannot occur inside an internet-disabled
notebook without secrets/network. The external authenticated dispatcher therefore
checks authoritative current quota/reservation/job state immediately before the
only write, and rejects full dispatch if current quota is below the reviewed
notebook's conservative budget bound or measured cost+20% margin. A static
snapshot cannot authorize dispatch. Runtime separately checks measured costs
against that bound and the21600-second session cap. Isolated two-update GPU
smoke precedes independent GPT-6 Medium P0 review; no long run before PASS.

Notebook cells0–13 expose the specified gate and stage order. Eligible linear
FP8 conversion is idempotent, master weights/loss stayFP32, attention/normsBF16.
Progress bars, atomic predictor/head/cache checkpoints and full RNG/cursors are
implemented. A session is temporary; saved files require Kaggle Output readback.
P1 after execution is separate, mandatory before independently verified results
enter the manuscript. Check verifier artifacts for unresolved requirements.

Official I-JEPA code/revision52c1ae95d05f743e000e8f10a1f3a79b10cff048 and its
CC BY-NC4.0 license retained; DINOv3 official license retained in embedded source.
Cite Meta I-JEPA, Meta DINOv3 and Arystanova et al.,2026,
doi10.3390/agriculture16131401. Learned local contributions are unvalidated
spatial proxies, never measured cell-level ground truth. Dataset and weight
redistribution rights require documentation before journal submission.
