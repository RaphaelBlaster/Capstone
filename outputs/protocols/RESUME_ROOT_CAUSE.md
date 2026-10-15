# DINOv3 FP8 checkpoint-resume root-cause investigation

Status: actual GREEN GPU regression PASS; independent PASS_GPU_RESUME issued.
Full F1/F2 has now been submitted after the required release gates; F3 is disabled. No scientific accuracy result is
produced by these tests. The user now authorizes starting F1/F2 after the required
independent PASS_GPU_RESUME gate. The first verifier hit its usage limit before
issuing that verdict; its earlier static review is not a final GPU verdict.

## Evidence and controlled comparisons

The original smoke reported a 0.1434478759765696 g prediction discrepancy between
an uninterrupted run and a separately initialized interrupted/resumed run. The
15 original checkpoints remain preserved, with SHA256 identities recorded in
`outputs/dino_resume_rca/preserved_dataset/PRESERVED_SHA256.json`. CPU inspection
found that the two trajectories already differed after their first optimizer
update, before a resume occurred. Both trajectories used the same 16 training
photographs, two excluded validation photographs, initialization and schedule.

The actual RED diagnostic ran on NVIDIA RTX PRO 6000 Blackwell, PyTorch
2.11.0+cu128, CUDA 12.8, TorchAO 0.17.0, driver 580.178.04. It retained the exact
original trainer/package and ran seven bounded optimizer updates. R4 resumes
from the uninterrupted branch's own step-1 checkpoint; R5 repeats the complete
uninterrupted trajectory. Both comparisons end at optimizer step 2 and use
the identical held-out photographs. Training never uses the frozen feature cache.

| Test | Actual RED result |
| --- | --- |
| R0: repeat live inference | Exact, 0 g |
| R1: same-instance state serialization/reload | All state and predictions exact |
| R2: fresh official/FP8 reconstruction | Parameters, buffers, head and predictions exact |
| R3: optimizer and RNG restoration | Exact; optimizer step 2, epoch 2, cursor 0 |
| R4: own-step-1 checkpoint continuation | FAIL: backbone max 1.974403858e-7, head 7.450580597e-9; predictions 0 g |
| R5: repeated uninterrupted training | FAIL: backbone max 4.770234227e-6, head 2.204626799e-5; predictions max 0.19678878784 g |
| R6: alternate precision diagnostic | Not run; precision unchanged |

The same saved GPU regression test reported four passes and two failures. RED
Kaggle completion means the diagnostic finished; it does not mean parity passed.
The first observed difference was in accumulated gradients **before clipping**
at update 1, first parameter `backbone.blocks.10.norm1.weight`. All eight preceding
microbatch input hashes and dense feature hashes matched. Resuming A's own
step-1 checkpoint likewise gives matching step-2 forward traces but differing
backward gradients. Serialization alone therefore cannot explain the failure.

## Isolated cause and smallest candidate fix

An isolated repeat of the actual BF16 attention operation, with identical Q/K/V
and identical upstream gradient, produced maximum dQ difference
1.862645149230957e-9; dK/dV were exact. The profiler identified PyTorch Flash
Attention backward, including the sequence-parallel backward reduction. Enabling
strict deterministic algorithms made dQ/dK/dV repeats exactly equal. This is
direct evidence of GPU backward nondeterminism. The isolated operation does not
alone prove that Flash Attention is the only source across all future workloads.

The production diff enables `torch.use_deterministic_algorithms(True)`
before CUDA model work and sets a supported `CUBLAS_WORKSPACE_CONFIG` before
CUDA/cuBLAS initialization. Unsupported workspace settings fail explicitly.
FP8 eligible linears, BF16 attention, FP32 master weights, DINOv3 architecture,
D2 head, folds, losses, L2-SP coefficient, learning rates and schedule are
unchanged. No tolerance is widened. Package member comparison changes only
`biomass_jepa/dino_finetune.py` and its hash inventory.

## Release evidence locations

- Original checkpoint audit: `outputs/dino_resume_rca/AUDIT.md` and `checkpoint_audit.json`.
- RED actual GPU data: `outputs/dino_resume_rca/red_retry/results/dino_resume_red/resume_diagnostic.json`.
- RED raw execution log: `outputs/dino_resume_rca/red_retry/watch/execution.log`.
- Minimal diff: `outputs/dino_resume_rca/MINIMAL_SOURCE_DIFF.patch`.
- Immutable GREEN package/notebook/hashes: `outputs/dino_resume_rca/green/launch/`.
- Independent reviews: `outputs/dino_resume_rca/RED_RCA_REVIEW.md` and `GREEN_STATIC_REVIEW.md`.

## Actual GREEN result and limits

Kaggle diagnostic Version 3 completed on the actual RTX PRO 6000. The unchanged
six-test regression went from RED four passes/two failures to GREEN six passes.
R0–R5 all pass at exact zero tensor/prediction tolerance. R4 and R5 have maximum
absolute and relative prediction differences of zero for every target. Independent
CPU comparison of the saved latest checkpoints confirms identical trainable
weights, heads, optimizer moments/step counters/group LRs, RNG, epoch/window cursor,
schedule/signatures and gradient evidence. Matched forward and pre/post-clip
gradient traces are exact. All 28 intended F2 backbone tensors update; frozen
weights remain unchanged, and F1 changes only its head.

The standalone verifier reconstructs official DINO plus FP8 conversion and directly
forms patch grids, without the trainer's forward helper or cached features. Its
three checkpoint predictions differ from the saved decimal CSVs by at most
7.105427357601002e-15 g (predeclared bound 1e-10 g). D2 initial head prediction
parity is 1.4210854715202004e-14 g for both saved folds. The A and resumed C
selected checkpoint files even have the identical SHA256
`dd9e894603f90b32dd02abb5451b8122bf8f6e780588849535b80ef928b8d309`.

The cause supported by the complete RED/GREEN experiment is a missing strict
deterministic CUDA execution policy, with Flash Attention backward independently
observed as a nondeterministic operation. The policy removes the measured
divergence without changing precision or scientific settings. This does not
establish that this is the sole possible source on other software or hardware.
The initial environment snapshot shows deterministic algorithms false because
it precedes configure_precision; the executed revised trainer enables them
before FP8 model forwards. The full diagnostic remains 16 training photographs,
two held-out photographs and two updates per completed branch, not a full run.

GREEN diagnostic body: 43.6983 seconds and seven updates; peak allocated
1,209,069,056 bytes (1.126 GiB), reserved 1,434,451,968 bytes (1.336 GiB).
Measured F2 two-update branch times are 4.8338 and 4.6411 seconds; F1 is
3.4469 seconds. These include instrumentation and checkpoint work, and are not
a guaranteed full-run runtime. After collection, quota is 104,775.493 seconds
(29.1043 hours), zero reserved. Full schedules require 400/380 updates per arm
on folds 0/1. A conservative provisional budget using maximum measured branch
time for all 1,560 updates, plus 900 seconds for validation/startup and 20% margin,
is about 93.4 minutes; full evaluation runtime is not empirically established.

CPU targeted checks: 16 passed, six GPU-evidence tests skipped locally. Full
repository CPU suite: 366 passed, 19 failed because historical output notebooks
are missing, six skipped. No clean full-suite claim. All original 15 checkpoint
hashes remain unchanged. The CPU evidence checker initially paired snapshot list
positions; resumed training re-commits loaded step 1 at the epoch boundary. It
was corrected to match optimizer step IDs, then passed. This was an evidence
alignment issue; no production change or GPU rerun was needed.

New package: `18756d12b22828abf4266cbc47012aea4875f13ab39025b03ce6f4f719dab8a0`.
New trainer: `60a8b97d5c3b222b7d7cb5f0f3a971a454d0897e0042ef1f2d61297fc037563f`.
Actual GREEN notebook: `356e75092b909b288558c42acda861c13e7b40e1fddff7ccafde058ec6671d8b`.
Wrapper and GPU regression hashes are unchanged between RED and GREEN; full
hash inventory is in green/launch/build_manifest.json.

Saved results: `outputs/dino_resume_rca/green/results/dino_resume_green/`.
Actual raw log: `outputs/dino_resume_rca/green/watch/execution.log`.
Root checkpoint verification: `outputs/dino_resume_rca/ROOT_GPU_RESUME_VERIFICATION.json`.
Combined RED/GREEN summary: `outputs/dino_resume_rca/resume_diagnostic.json`.
Independent /root/dual_verifier issued PASS_GPU_RESUME and separately approved the exact new full notebook and guarded dispatcher. A second existing verifier completed the review after the first hit its usage limit. The stale prepared notebook was never dispatched.

Full F1/F2 Version1: meghmayur1186/img2biomass-dino-finetune-f1-f2, ID138038538; exact saved cells/settings read back. Notebook SHA30885af0d676666adc2a93eebe0fd9af2b1ba79e66c432a3fcdbfc3061957005; package remains18756d12. Fresh owned-last-run inventory and changed/restarted job checks plus pinned source/diagnostic states and zero-reservation quota preceded the one push. Full training scores and P1 remain unverified. Receipt/watch: outputs/dino_resume_rca/full_launch/.
