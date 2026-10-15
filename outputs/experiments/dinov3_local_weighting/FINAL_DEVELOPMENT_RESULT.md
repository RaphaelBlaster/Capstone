# Final DINO development result —10 October 2026

Best observed score remains **0.7800650684336633**: frozen official DINOv3
ViT-B/16 at512 per complete view, local contributions plus cross-scale consistency
(D2). This is105 of357 original photographs, two repeatedly reused development
folds, nine acquisition dates and one seed. It is not full OOF or a competition
score, and selecting the highest development score is not protected confirmation.

| Arm | Pooled global weighted R2 |
| --- | ---: |
| D0 global pooling |0.7659124500760509 |
| D1 uniform local contributions |0.7798813752570810 |
| D2 contributions plus cross-scale consistency |0.7800650684336633 |
| D3 learned normalized local pooling |0.7608191087125595 |

The new D3 model did not improve the observed score. D3 minus its repeated D1
control was -0.019062266544521478. Independently generated10000 paired acquisition-
date-group bootstrap draws (seed42) yielded95% interval[-0.07279355094422288,
0.007985241973672513]. The independent architectural verifier's separate seed9801
interval also includedzero. These intervals condition on fixed fitted models and
reused development groups. No causal explanation for the negative result is
established; D3 also adds98,819 parameters, so this is not capacity-isolated.

Both D1 fold models and predictions reproduced prior Version2 exactly. All four
new full heads completed200epochs/2000optimizersteps/final40SWA. Saved selected
states equal latest40-SWA states. Runtime proved actual NVIDIA RTX PRO6000
Blackwell Server Edition and eligibleFP8; both heads passed actual smoke/reload.
No new SSL or backbone training/extraction was performed: the pinned saved dense
cache was reused. Actual notebook execution completed in about63seconds.

Private/offline notebook: meghmayur1186/img2biomass-dino-local-weighting,
ID137985504 Version1. Exact submitted notebookSHAbd889c27f4f248eaaa37fb90dadd7d39957a68120566d5e6d67cd980e489a8e9;
embeddedZIP86f51ffe133585593fa285e4625aa412d44225cd5b01d72fc7c8b05d77dbb46b.
Independent P1 PASS/VALIDATED_EXPLORATORY_NEGATIVE_RESULT. Forty files collected,
35 savedinventorySHA256entries verified without mismatches. Prior and new weights,
raw paired predictions and training logs preserved. No competition submission.

Evidence: outputs/dino_weighting/final_score.json, independent_root_metrics.json,
P1_VERDICT.md, P1_REPRODUCIBILITY_CHECKS.json and results/dino_weighting/.
Source private branch research/dino-local-weighting. Complete original dataset/
weight GitHub release backup remains separately blocked by401; source push is not
asserted to constitute that complete bulk backup.
