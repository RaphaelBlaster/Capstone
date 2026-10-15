# JEPA predictor explanation: three random held-out photographs

Completed 2026-10-10T13:45:55.941346+05:30. Saved J3 shuffle-specialized predictor and fixed final-SWA head from Kaggle dual-study Version2; frozen grass-adapted encoder. No training or GPU dispatch.

Sample: uniform without replacement from the 105 held-out originals, seed42, without inspecting prediction errors. All three happen to belong to fold1. Each image SHA matches the locked original; the head signature matches the exact specialized predictor SHA.

| Image | True Total (g) | Saved FP8 Total (g) | CPU FP32 Total (g) |
| --- | ---: | ---: | ---: |
| ID567744300 | 22.83 | 14.15 | 14.41 |
| ID1211362607 | 67.90 | 73.07 | 73.19 |
| ID1103883611 | 95.70 | 52.92 | 53.81 |

## Verified observations

- The semantic predictor readout exposes only88/256 patches (34.375%) of each224x224 view to the student encoder. Four hidden corner blocks cover65.625%. This is the intended masked readout, not a newly discovered implementation failure.
- Every hidden-pixel gradient is exactly zero. Replacing all hidden pixels with normalized zero (ImageNet mean RGB) leaves all five predictions exactly unchanged for all three photographs. This verifies an information restriction of this readout; it does not prove the restriction causes the aggregate score gap.
- ID1211362607 has Green66.56g and Dead1.34g, but the saved predictor assigns33.27g Green and38.85g Dead. A relatively close Total estimate therefore hides a large composition error. This confusion persists in CPU FP32.
- ID567744300 has no measured Clover, yet the model predicts6.79g; Green is underestimated (4.74g versus20.64g). ID1103883611 also underestimates Green (33.42g versus82.78g) and Total (52.92g versus95.70g). These are observations on three examples, not a population estimate.

## How to read the panels

Original photograph; actual visible/hidden patch mask; Green, Dead, Clover and Total local sensitivity maps. Maps show absolute gradient times ImageNet-normalized input, with each target independently clipped at its visible-pixel99th percentile. Bright pixels mean greater local sensitivity under this calculation, not attention, species segmentation, measured local biomass, or necessarily positive evidence. Raw signed channel gradients, GDM gradients and unnormalized heatmaps are saved in NPZ files.

## Perturbation check and limits

Five visible14x14 patches are replaced by mean RGB, either highest Total sensitivity or a seeded random set. The two sets may overlap. This is a diagnostic perturbation, not a label-preserving training augmentation. In ID1211362607 random-five occlusion changes Total more than top-five (-37.48g versus-20.01g), so the gradient ranking is not consistently predictive of finite occlusion effects. We cannot use these maps as a complete causal explanation.

CPU FP32 and saved FP8 predictions are not identical: maximum absolute difference across15 predictions is1.75864g (GDM of ID1103883611); maximum Total difference is0.89399g. The explanations apply to CPU FP32 inference with the saved weights, not exact FP8 gradients. The large observed composition/underestimation errors persist across these numerical differences.

Three images cannot establish why JEPA is weaker overall. Masked information, target composition ambiguity and representation/readout suitability are hypotheses for a controlled comparison; no new experiment or causal claim follows from these panels.

## Files

- `source_data.csv`: truths, saved/CPU predictions, precision differences and occlusion changes for allfive targets.
- `results.json`: image/checkpoint hashes, exact patch indices and complete predictions.
- `ID*_explanation.png`: real-image panels.
- `ID*_attributions.npz`: gradients, masks and heatmaps.
- `run.log`: execution progress.
- Reproduce with `PYTHONPATH=/workspace/.img2biomass-cpu-deps:/workspace/.img2biomass-status-deps:. python tools/explain_jepa_predictor.py` from the checkout. Python/PyTorch versions and artifact hashes are in `REPRODUCIBILITY_CHECKS.json`.
