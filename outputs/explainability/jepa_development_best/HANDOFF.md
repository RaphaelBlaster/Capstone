# Development-best XAI handoff

Completed script: tools/explain_best_jepa.py. Three-image CPU FP32 sensitivity run completed successfully in about2m44s. The earlier interrupted attempt produced no scientific result; this completed run wrote all artifacts described below.

Run: `PYTHONPATH=/workspace/.img2biomass-cpu-deps:. python tools/explain_best_jepa.py > outputs/best_jepa_xai_2026_10_11/run.log 2>&1`.

Checkpoint lineage: paper/five_target_independent_selection.json all_results[0] cached_context, development global weightedR2 .8340391905617833,105images. Its signature exactly equals recovered inference_manifest.json. Both jepa_encoder.pt and fold_1_best.pt hashed and matched manifest before model reconstruction. Both datasets private firstaccount: img2biomass-jepa-v6-weights sharedSSLbase and img2biomass-jepa-head-v9-weights fold1. Selecting these weights now follows developmentbest, not competitionbest. Historical architecture is DINOv3-L encoder plus JEPA-style latent-predictor context readout, not official I-JEPA H14. SourcebaseSHA15afb4a... verified, fold1 state epoch19 score.8387037369255395.

All three images are existing fixed random examples from outputs/jepa_xai_3/images. Their current fold is1. HOWEVER historical foldsSHA4ffc3c... differs current foldsSHA5f00ab... . Original Kaggle capstone saved9 source/output404; sources13/14/16404 too. Original historical held-out status is UNVERIFIED. Script labels this plainly: these are descriptive sensitivity examples, not verified historicalOOF samples. Never claim historicalheldout or originalGPUparity.

Script reconstructs exact changingparameter state through existing biomass_jepa.infer.reconstruct_model (strictinitialencoder/predictor and exactchangingkeyset), freezes all weights, CPUFP32 inputgradientTotal, |grad×normalizedinput| heatmap, top-three versus random-three 4x4cell meanRGB occlusion, repeated same-state prediction equality; saves PNG/PDF/NPZ/JSON. Completed checks confirmed finite nonzero gradients, finite outputs, and exact repeated predictions for all three images. No training, GPUdispatch, architecturechanges. Existing CPUdeps emitted a nonfatal warning that httpcore2 is missing during HF API import; timm pretrained=False required no model lookup. Historical held-out status remains unverified; see report limitations.

Outputs may contain preserved downloaded weights1.7GB; exclude weights from user figureZIP.
