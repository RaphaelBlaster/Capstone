# Implementation differences and evidence — 10 October 2026

**F1/F2 implemented and independently reviewed. NO Kaggle job/GPU smoke/full training submitted.**

| Item | Actual D2 / proposal | New implementation |
|---|---|---|
| Backbone | Authentic official frozen B16 | Same strict initialization; F1 frozen; F2 final2blocks+patchnorm |
| Head | D2 DINODensityHead with saved fold scale | Identical class/state/buffers strictly loaded; no new architecture |
| Images | Two complete1000square views, BICUBIC512, ImageNet normalization | Identical online differentiable preprocessing; identity only as executed D2 |
| Dense readout | Official1024normalized patches,4×4 pooling→8×8 perview | Unchanged differentiable pooling; CLS/registers excluded |
| Loss | Raw five-target weighted MSE/scale² +.10 cross-scale primitive consistency | Reused verbatim; F2 addsP-normalized fixed L2-SP |
| Initialization | Saved200epoch/final40SWA fold D2 | Same head plus official backbone independently reloaded perarm; fresh optimizer |
| New duration/endpoint | Proposal20epochs; originalD2used40SWA | Matched20epochs with fixed FINAL EPOCH endpoint, explicitly declared change |
| Optimizer | Proposal enc2e-6/head3e-5 | AdamW(.9,.999),eps1e-8; headWD.01/backboneWD0; cosine20;clip1 |
| Anchoring | Coefficient not fixed by user | Fixed1e6 ×mean scalar squared shift, calibrated for≈14Mparameters; not proven optimal |
| Dormant bias | Proposal all finalblock parameters | Authentic allzero QKVbiasmasks mean biases cannotlearn; keep exactlyzero/frozen, preserve mask buffers throughfutureFP8conversion; reject unsupportedstates |
| Folds/exposure | Originalfolds0/1 | Same308/49 and301/56; batch2/accum8;400/380stepsperarm; same schedule |
| F3 | Suggested composition supervision | Separate261valid labels and differentiable auxiliaryloss/gate; disabled,260images locally missing |
| GPU verification | Requested actualsubmitted code/FP8 proof | UNVERIFIED because user forbids GPU/submission; static/CPU PASS is scoped |

New implementation files: biomass_jepa/dino_finetune.py, tools/build_dino_finetune.py, tests/test_dino_finetune.py, tests/test_dino_finetune_notebook.py. Historical D2/JEPA code and saved notebooks remain unchanged. Preservation comparison against preimplementation handoff:157 existing code/result/user-notebook files match bytewise; both D2 head hashes match saved metrics.

## Completed verification

48 targeted branch/regression checks pass, including16 newbehavior tests independently rerun by GPT-6 Medium. Test-first records in outputs/dino_finetune include missing-feature, F3/helper, builder and real masked-bias correction failures before fixes. Full repository suite:366passed,19failed,2warnings. All19failures are legacy absent-output-notebook FileNotFoundError, detailed by name in TEST_RESULTS.json; do not call the full suite green.

Authentic official B16 CPU512 smoke used a real photo from fold1 as TRAINING for fold0, two optimizer updates,28intended active parameter tensors updated, frozenparameters unchanged, headupdated. Initial differentiable/frozen dense output parity is bitwise exact. L2-SP has nonzero second-step gradient and immutableanchors. Saved delta/head strict reload reproduces predictions bitwise. CPUversuspriorFP8cache drift=1.18225098g on one trainingphoto; this is a precision diagnostic, not a newvalidation score. CPUfunctional checkpoint is kept separately, not a selected research model.

## Immutable package and limitations

Notebook SHA256: `e70f4ce68737e0990316517ea70c6e2c3462febcacb02b9c14524b593045d55f`
Embedded package SHA256: `f390af06ce0065e308736dd9b5af38a54dc9d3fe15864a098e5fcec95a27fe16`
Independent verdict: PASS_STATIC_CPU_SCOPE; actual submittednotebook/GPUFP8/resourcequota/metrics remainUNVERIFIED. DefaultMODEprepare and GPUdisabled. Runtime smoke/full require new explicit authorization and independentlyverified exactsource; fullalso requiresactualGPUruntimePASS. Progressbars andatomic optimizer/RNG/delta checkpointing included. Noresult/performance/novelty claim.

Fixed1e6 anchoring is a predeclared arbitrary scale calibration, not empiricalproof that anchoring improvestransfer. Allfuture105photo results remainreused explorations. F2−F1 testsencoderlearningwith anchoring; it doesnot separatelyidentify anchoring benefit. F3−F2 adds auxiliarytrainingandcompute, not justbiologylabels isolatedfromcost.
