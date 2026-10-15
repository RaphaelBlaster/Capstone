# Img2BioMass DINOv3 D2 research release

This private research release contains the exact self-contained Kaggle training notebook and supporting source, the selected D2 regression heads for folds 0 and 1, the official frozen DINOv3 ViT-B/16 base checkpoint, saved metrics/predictions/training logs, and the metric/XAI figures. The two `head_selected.pt` files are regression heads only; load them with the supplied DINOv3 base checkpoint and the source in `source/`.

## Result scope

D2 independently rescored raw global weighted R² is **0.7800650684** on 105 of 357 photographs, reused date-group folds 0/1, nine acquisition-date groups, and one seed. This is exploratory development evaluation, not a competition score or complete out-of-fold result. D2 minus D1 is **0.0001836932**, with a conditional date-group bootstrap 95% interval of **[-0.0000974862, 0.0004333366]**; this does not establish a D2 improvement.

The depth 2/4/8 and JEPA explanation images in `figures/depth_and_explainability/` belong to separate analyses; they are not additional D2 folds or score evidence.

## Reproduction

- Exact saved notebook: `source/capstone_FINAL_DUAL_JEPA_DINO.ipynb`
- Saved Kaggle source/settings readback: `source/kaggle_actual_saved_source.json`
- Training modules, builder, runner, and targeted tests: `source/`
- Fold metrics, predictions, logs, aggregate CSVs, and figures: `results/`
- Checkpoint hashes and source provenance: `provenance/`

The notebook expects the pinned CSIRO data, GrassClover adaptation source, offline dependency wheels, and DINOv3 checkpoint as Kaggle inputs. This release intentionally does not contain raw datasets or cached feature tensors.

## DINOv3 terms

The official DINOv3 checkpoint and source are distributed under Meta's DINOv3 License Agreement. A copy is included at `licenses/DINOv3_LICENSE.md`; anyone receiving or using those materials must comply with its terms. The official source revision and checkpoint hash are recorded in `provenance/dinov3_metadata.json`.
