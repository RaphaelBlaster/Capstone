# Spatial permutation ablation
Partial development only: reused folds 0/1/2, one training seed; not protected confirmation, full OOF, competition scores or a preregistered study. Paired date-group bootstrap conditions on fixed trained models and excludes training-seed uncertainty and prior model-selection bias. Dates are proxies, not verified sites. Synthetic permutations add no independent biological observations. SSL latent loss cannot establish downstream benefit from more SSL steps.

- control: independently_rescored
- shuffle_4: independently_rescored
- shuffle_10: independently_rescored

- shuffle_4_minus_control: Positive improvement is not established.
- shuffle_10_minus_control: Positive improvement is not established.
- shuffle_4_minus_shuffle_10: Positive improvement is not established.

Data availability: CSIRO competition data remain subject to the original dataset terms; no private test labels were used.
Code availability: the controlled experiment and independent analysis live in the private Img2BioMass repository; public release status is not asserted.
Prior art: Meta I-JEPA official source; Noroozi & Favaro, ECCV 2016 (jigsaw learning). This is not a claim of novel jigsaw learning.
Meta reference code/checkpoint licensing, including CC BY-NC 4.0, must accompany any authorized distribution.
