# Evidence-backed DINOv3 improvement research

Research completed 10 October 2026, IST. This is a research recommendation, not an implemented experiment or a GPU launch. Existing models, notebooks and checkpoints remain preserved.

**Recommendation:** test limited supervised adaptation of the actual DINOv3 encoder first; then test whether external GrassClover self-supervised adaptation adds benefit to the same supervised recipe. Keep the existing biomass head, image geometry and metric fixed. GrassClover provides a defensible additional source of plant imagery, but adapting on it is a hypothesis, not a guaranteed improvement.

## What our saved results already establish

The current official DINOv3 ViT-B/16 encoder is frozen. Its 512-pixel complete views produce genuine patch tokens; the current cache averages each 4-by-4 token region into an 8-by-8 grid. D0/D1/D2/D3 change the supervised readout, not encoder weights. A new encoder adaptation is therefore a materially different question from the latest pooling tests.

| Current arm | Raw pooled global weighted R² |
| --- | ---: |
| Frozen DINOv3, global head D0 | 0.765912 |
| Frozen DINOv3, local contribution head D1 | 0.779881 |
| D1 with cross-scale consistency D2 | 0.780065 |
| Learned local weighting D3 | 0.760819 |

All results use 105/357 original photographs, two reused date-grouped folds, nine acquisition dates and one seed. These are partial development results, not competition results or full OOF. The paired intervals for D1−D0 and D2−D1 include zero. D3 did not establish an improvement.

Fresh arithmetic on the saved D2 predictions gives:

- Total biomass accounts for **63.39%** of metric-weighted squared error; GDM accounts for19.07%, Green9.49%, Dead5.38%, and Clover2.67%.
- Among the21 highest-Total observations (Total≥84.2g), mean prediction error is−18.78g and MAE25.61g. Among the21 lowest (Total≤28g), mean error is+4.65g.
- Clover is exactly zero in47/105 development observations and135/357 original labels. Dead is zero in1/105 development observations and40/357 labels, showing that the reused development sample is not a complete description of the dataset.
- D2 Dead target R² is0.2398; Clover has positive mean error+2.81g. Component identification remains imperfect, but Total/GDM dominate the current aggregate error.

This motivates tests targeting representation suitability and high-biomass error. It does not diagnose the cause of those errors. Do not fit a correction to these held-out residuals and then report its performance on the same observations.

Earlier DINO-initialized custom JEPA adaptation was already investigated in this project: historical V6 full357 OOF0.77454 versus matched V7 fresh-predictor0.73605, with a paired interval including zero. That was a different backbone/readout/objective recipe. New official-DINO SSL would not be our first use of GrassClover or proof that adaptation generally works.

## Primary literature and what it supports

### 1. Limited encoder adaptation: first priority

[Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR2022](https://arxiv.org/abs/2202.10054) find that full fine-tuning can improve in-distribution performance while damaging out-of-distribution performance. Training the head before fine-tuning (LP-FT) mitigates the problem in their classification experiments. Their theory concerns simplified linear networks; it does not guarantee improvement for our nonlinear biomass head.

[Li et al., *Explicit Inductive Bias for Transfer Learning with Convolutional Networks*, ICML2018](https://proceedings.mlr.press/v80/li18a.html) support regularizing trainable weights toward their pretrained values (L2-SP), rather than relying only on initialization and early stopping. This is evidence from CNN transfer, not a DINOv3 biomass result. Ordinary AdamW weight decay toward zero is not equivalent to anchoring weights to their pretrained values.

A reasonable first test is head warm-up followed by a small adaptation capacity: either low-rank adapters in late attention projections or a narrowly unfrozen final block. These are alternatives, not additions to stack together immediately. LoRA keeps the base tensors fixed but changes the encoder's effective computation; it is therefore genuine encoder adaptation, unlike training only the biomass head. There is no theorem that the required update has low rank or that late blocks alone are optimal.

**New directly relevant publication:** [Sen, Dhankhar and Bayat, *Dynamic LoRA Fine-Tuning of DINOv3 for Multi-Component Pasture Biomass Estimation*, Sensors2026, published20August2026](https://doi.org/10.3390/s26165285). Its full text was read through [EuropePMC](https://europepmc.org/articles/PMC13517550), not only its abstract. Table1 reports frozen0.72CV versus uniform LoRA0.73 and dynamic LoRA0.75; the complete pipeline reports approximately0.81CV,0.77public and0.641private. These are author-reported results, not independently reproduced here.

Treat this as relevant supporting evidence with limitations. The complete pipeline uses larger backbones, progressive resolution, metadata, alignment, TTA and ensembling. Section3.4 feeds NDVI/height/species/state into the prediction representation; its availability in a competition inference pipeline is not established by the article. Table1 labels the final result an L+H ensemble, while Table5 attributes a similar result to ViT-L+dynamic LoRA; the comparison details require clarification before interpreting that table as an isolated causal effect. An exact reproducible fold assignment was not located during this review. Do not copy its entire pipeline or claim dynamic LoRA as a new invention.

### 2. GrassClover self-supervised adaptation: strongest external-data research question

[Khanna et al., *ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts*, ICML2025](https://proceedings.mlr.press/v267/khanna25a.html) continue self-supervised pretraining on a target domain, unfreeze1–2ViT blocks and use LoRA in other attention projections. Their DINOv2 experiments demonstrate gains on remote-sensing tasks; some ablations also show poor outcomes, so parameter allocation matters. They retain the pretraining objective and train new DINO/iBOT projection heads where released backbone checkpoints lack them. Their [official code](https://github.com/samar-khanna/ExPLoRA) is DINOv2/MAE code, not a drop-in DINOv3 implementation.

This gives a defensible rationale for using our **30,663 verified GrassClover originals**: many additional images of the relevant plant structures can supply representation-learning signals without fabricated biomass labels. Restricting updates can preserve useful pretrained information while allowing adaptation. Whether Denmark's crops, camera systems and growth conditions transfer to CSIRO's Australian quadrats must be tested. Image count alone does not guarantee useful diversity or transfer.

[The official DINOv3 report](https://arxiv.org/abs/2508.10104), sections4.1–4.2, documents dense features degrading during extended SSL even while global metrics improve. Its Gram objective preserves relationships between normalized patch features rather than fixing every individual feature vector:

`L_Gram = ||X_student X_studentᵀ − X_anchor X_anchorᵀ||²`.

For a model whose biomass head depends on patch features, protecting spatial feature quality is justified. A possible adaptation recipe combines DINO's global self-distillation, iBOT's masked-patch objective and a pretrained-reference Gram penalty. The exact short-run recipe, coefficient and adapter placement remain to be validated. A loss that merely matches a frozen model's identical outputs on the identical image can be zero at initialization; that alone is not a useful adaptation objective. DINO/iBOT projection-head warm-up and anti-collapse mechanisms cannot be omitted silently.

The full ExPLoRA research schedule uses very large effective batches and substantial compute. Fewer trainable parameters do not mean negligible GPU cost: gradients through adapters in early layers still require intermediate activations. Do not promise the original paper's gains from a few minutes of reduced training. A short pilot must measure actual cost and downstream feature utility.

[MedDINOv3](https://arxiv.org/abs/2509.02379) provides additional DINOv3-specific domain-adaptation precedent, but uses3.87million CT slices and changes a segmentation architecture. It supports feasibility, not expected gains from30K pasture images.

### 3. GrassClover's real labels: useful auxiliary task, separate from CSIRO

The [GrassClover dataset paper, CVPR Workshops2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVPPP/Skovsen_The_GrassClover_Image_Dataset_for_Semantic_and_Hierarchical_Species_Understanding_CVPRW_2019_paper.html) and [official dataset page](https://theailab.dk/grassclover-dataset/) provide real biomass-composition samples, synthetic segmentation labels and31,600unlabeled source images. Our locked adaptation manifest contains30,663usable originals; this review does not infer why those counts differ.

The local re-hosted training CSV contains **261 labeled photographs**, with dry grass, dry clover, dry weeds and dry total, plus composition fractions. Its174test rows have only filenames/year/harvest metadata. The original paper describes435biomass samples,0.5×0.5m quadrats and cutting at5cm, and explicitly distinguishes visible canopy cover from harvested composition. CSIRO describes0.70×0.30m quadrats and different target definitions. Area conversion alone cannot reconcile harvesting and biological target differences.

A shared encoder with a **separate GrassClover composition head** could learn grass/clover discrimination from actual additional labels. Keep its outputs and normalization separate from the five CSIRO masses. `dry_grass` means dry matter in the grass species fraction; it is not an observed CSIRO Green label. GrassClover has no separate Dead target, and weeds cannot simply be relabeled Clover or Green. Missing labels must remain missing.

This route provides additional labeled observations for an auxiliary task, not618homogeneous CSIRO observations. It is worth testing if clover/grass confusion remains material after encoder adaptation, but it does not directly resolve the weak Dead target. Synthetic masks are weaker evidence for natural-image mass because occlusion and canopy area do not identify dry mass or hidden vegetation. Do not turn a segmentation coverage map into ground-truth local biomass.

### 4. FINO metadata adaptation: interesting, conditional priority

[Gardès et al., *Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have*, June2026 preprint](https://arxiv.org/abs/2606.05107), now linked from Meta's [official FINO branch](https://github.com/facebookresearch/dinov3/tree/FINO), adapts DINOv3 with DINO+iBOT+SIGReg and metadata guidance. Metadata can be encouraged or adversarially suppressed. The paper compares four scientific domains and finds that ordinary SSL adaptation can hurt a domain near DINO's original distribution; it explicitly caps iWildCam adaptation after observing degradation.

This is not evidence that adding dates automatically helps our biomass task. GrassClover's261label table supplies year/harvest number, but reliable metadata across the entire30Kimage corpus have not been verified. Acquisition date, site and harvest can encode genuine growing conditions and biomass, so suppressing them indiscriminately may remove useful biological information. Prefer plain, controlled adaptation before introducing a metadata branch. FINO is a preprint here; no peer-reviewed venue was verified.

## Ranked next actions

| Priority | Candidate | Justification | Principal uncertainty |
| --- | --- | --- | --- |
| 1 | Warmed-up head + small encoder adaptation | Current DINO is frozen; LP-FT/PEFT offer ways to adapt without unconstrained full-model updates | Small labeled set can still overfit; no promised gain |
| 2 | External GrassClover DINO/iBOT adaptation, then identical supervised recipe | Extra relevant images; ICML ExPLoRA precedent; DINOv3 supports protecting dense features | Transfer mismatch, SSL instability and forgetting |
| 3 | Separate supervised GrassClover composition auxiliary task |261additional real composition labels can target grass/clover ambiguity | Target mismatch; no Dead labels; cross-domain interference |
| 4 | Last-layer versus intermediate-layer patch features | DINOv3 appendixB.2 shows some geometric tasks peak before the final layer | Evidence is from7B and other tasks, not ourB16biomass |
| 5 | Less aggressive spatial token averaging before the nonlinear head | Current32×32tokens are averaged to8×8cells; testing the order/resolution of pooling measures retained heterogeneity | DINO attention already mixes context; finer readouts can overfit or cost more |
| Conditional | FINO metadata guidance | Direct DINOv3 scientific-domain adaptation precedent | Full-corpus metadata and the informative/spurious assignment are unverified |

Do not combine these into one first run. Intermediate-layer/finer-token tests require new feature extraction; they are not already available in the current cache. Increasing resolution/model size may help, but changes cost and several factors; the current strongest literature does not establish monotonic gains for our task. Adding another learned pooling gate is low priority after the completed negative D3 result.

A Clover presence/conditional-mass head is an additional data-motivated hypothesis because zeros are common. However, Clover accounts for only2.67% of current weighted error, and the conditional expectation under squared error need not be exactly zero even for a zero-valued observation. Strictly positive Softplus outputs alone are not proof of a bug. Do not prioritize it ahead of Total/GDM errors or treat an arbitrary presence threshold as justified.

## Proposed controlled comparison

The paper question should be: **Does external plant-domain self-supervised adaptation improve image-only biomass regression beyond limited supervised adaptation of the same pretrained DINOv3 encoder?**

| Arm | Encoder initialization | Encoder learning | Readout |
| --- | --- | --- | --- |
| F0 | Official DINOv3B/16 | Frozen | Existing D2 |
| F1 | Same official DINOv3B/16 | Limited supervised adaptation after head warm-up | Same D2 |
| F2 | Same official checkpoint, then GrassClover SSL | Same limited supervised adaptation asF1 | Same D2 |

F1−F0 tests supervised representation adaptation. F2−F1 tests the added external adaptation stage. It establishes the utility of that complete external training stage, not whether the objective, extra images or extra compute individually caused a gain. A later equal-compute data/objective control would be needed for those finer claims.

Use exactly the same predefined two date folds, input512 per complete view, labels, regression head, raw metric/loss and downstream schedule. F1/F2 must use equal downstream optimizer-step budgets and selection rules; warm up their heads using their respective feature spaces so neither arm starts with an incompatible random head. The existing F0model can be preserved and reused if its complete signature matches. Keep regression quarter-turn augmentations only and validate original photographs. Unlabeled SSL crop/view generation needs its own explicitly specified protocol; never attach a photograph's gram label to an area-changing crop or force biologically different left/right halves to have identical embeddings.

Start from the official DINOv3 checkpoint, not an I-JEPA checkpoint or a historical DINO+JEPA artifact. For limited adaptation, choose one small adapter placement/rank or one narrow unfreezing scope; do not sweep all options on the same reused validation set. Low encoder learning rates, head warm-up, pretrained-reference regularization and short schedules are starting hypotheses that need a concrete pilot—not verified optima.

All external-image overlaps must be audited. CSIRO held-out/test images and labels must not enter fold-specific specialization or optimization. External-only SSL can be shared across folds if it uses no CSIRO held-out data and the manifests/checkpoint selection are locked. Record the external exposure count and its additional compute explicitly.

Primary endpoint remains the paired difference in pooled raw globally weightedR², with date-group bootstrap intervals and full prediction CSVs. Also report weighted error decomposition, Dead/Clover performance and high-Total errors. Measure representation drift and actual effective encoder updates; a falling SSL loss is not evidence of improved biomass regression. Old checkpoints, complete optimizer/RNG resumes, progress bars and exact source hashes must be retained.

These folds have been reused many times. Any new result remains exploratory partial development evidence. Dates are grouping proxies, not verified independent sites. Higher performance does not by itself establish unseen-condition generalization, original invention or journal acceptance.

## Retrieval and evidence record

Primary HTML/PDF/XML and official code documentation are saved under `outputs/dino_improvement_research_2026_10_10/`, with hashes and access limits in `SOURCE_MANIFEST.json`. ExPLoRA and L2-SP publication venues were verified from PMLR; LP-FT's venue is recorded by arXiv. The new Sensors paper was read in full through EuropePMC. The Agriculture wide+tiles paper's bibliographic details and abstract were verified through Crossref; its publisher full text returned403, so no full-text architectural claim from that paper is used here. Its1196-image Kazakhstan study is not a CSIRO score.

Hugging Face's connector search was unavailable, so public paper-search APIs and direct primary sources were used. Google returned a script-only search page, DuckDuckGo a bot challenge, and Bing irrelevant results; none of those pages was used as scientific evidence. No browser capability, new model training, quota claim or GPU dispatch is asserted.
