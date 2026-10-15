# Dual study development results



Contract DUAL-JEPA-DINO-2026-10-09-r1. All five arms completed both folds0/1,200 head epochs and final40 SWA.

Evaluation uses105 of357 original photographs,9 acquisition-date groups,one seed, and reused development folds. These are neither full OOF nor competition scores.



branch,arm,global_weighted_r2_raw
jepa,control,0.5489117543819693
jepa,shuffle,0.5567129773256263
dinov3,D0,0.7659124500760509
dinov3,D1,0.779881375257081
dinov3,D2,0.7800650684336633


Paired intervals use10000 fixed-seed date-group resamples and are conditional on these trained models. They exclude seed variation and do not establish unseen-condition generalization.

J-shuffle versus J-control tests predictor specialization. D2 versus D1 tests the prespecified0.10 consistency penalty. JEPA versus DINO also changes resolution, representation, head and compute, so it does not isolate backbone superiority.

The ten jigsaws are artificial augmentations, not additional independently labelled photographs. No local biomass ground truth or measured spatial density is inferred.

Figures use saved observations and training logs. DINO local heatmaps are omitted unless actual contribution arrays are separately preserved; no heatmaps were invented.

Independent P1 architectural and checkpoint verification remains required. Favorable development scores alone do not guarantee novelty or Scientific Reports acceptance.

Cite Meta I-JEPA; Meta DINOv3; Arystanova et al.(2026), Agriculture16:1401, doi:10.3390/agriculture16131401. Dataset/model redistribution licenses require documentation before publication.