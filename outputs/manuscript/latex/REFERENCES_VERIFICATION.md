# Revision 6 primary-reference review — 11 October 2026

I read the current `revision_6/manuscript.tex` and `revision_6/references.bib` without editing either. The companion BibTeX file contains seven focused additions. I checked arXiv records through the arXiv API, the DINO ICCV record through CVF/Crossref, and the CSIRO competition page directly; DOI metadata for GrassClover, grouped cross-validation, and the bootstrap paper was checked through Crossref. The sources below are primary papers or the official task page. I cite DINOv2 by its verified arXiv record; an OpenReview endpoint presented a challenge page, so I did not rely on a journal-version record.

## Recommended insertion points

- **Transformer and ViT foundations:** cite `vaswani2017attention` and `dosovitskiy2021vit` after the Methods sentence that identifies the DINOv3 backbone as ViT-B/16 and describes its transformer blocks and patch grid. These support the architecture background, not a claim that the present method is novel.
- **DINO lineage:** cite `caron2021dino` and `oquab2024dinov2` in the Introduction immediately after the DINOv3 sentence, as context for earlier self-distillation vision-transformer work. Keep `simeoni2025dinov3` as the direct source for DINOv3. Its arXiv record identifies a 2025 technical report; do not label it peer reviewed based on this record.
- **Optimizer:** cite `loshchilov2019adamw` at the first Methods mention of AdamW. The source is the AdamW paper, not a generic optimizer survey.
- **Bootstrap:** cite `efron1979bootstrap` after the first description of the paired bootstrap. Keep the text clear that this study resamples the observed acquisition-date groups conditional on saved predictions; the citation does not validate the small number of groups or convert the interval into independent confirmation.
- **Grouped validation:** cite `roberts2017crossvalidation` after the Methods explanation of date-grouped folds, alongside the existing caveat that recorded dates are proxies rather than verified sites. The paper concerns validation under structured dependence; it does not establish that this split removes all dependence here.

The existing methods sources for L2-SP and jigsaw permutation are already present and cited where they support those choices. The manuscript already cites the primary I-JEPA paper and the CSIRO task page. For dataset provenance, retain `Liao2025PastureDataset` (the arXiv record gives the matching title, authors, and 27 October 2025 first-posted date) and `skovsen2019grassclover` (the DOI record confirms the CVPR Workshops paper metadata).

## Existing entries checked

- `assran2023ijepa`: arXiv:2301.08243 has the cited title and author list. The entry is a valid primary-source citation for I-JEPA.
- `simeoni2025dinov3`: arXiv:2508.10104 has title *DINOv3*, year 2025, and the listed author order. The current entry accurately marks it as an arXiv technical report.
- `csiro2025image2biomass`: the official Kaggle page resolves to “CSIRO - Image2Biomass Prediction.” It is appropriate for competition task/data/evaluation context; the Liao dataset paper remains the scholarly source for the larger image dataset.
- `skovsen2019grassclover`: DOI 10.1109/CVPRW.2019.00325 resolves to the listed title, authors, CVPRW 2019 venue, and pages 2676–2684.
- `Liao2025PastureDataset`: arXiv:2510.22916v1 confirms the listed title, authors, and initial 2025 posting. Retain the versioned URL if revision 6 is specifically citing v1.

## Exclusions and cleanup

Revision 6’s manuscript text currently does not cite Mandal, Sen, Tonu, or Arystanova. Its `references.bib` still contains unused entries `Sen2026DynamicLoRA`, `Tonu2026VisionBased`, and `Arystanova2026WideTiles`; remove those entries when preparing the release bibliography so the excluded pasture-method papers are not carried forward. I found no Mandal entry in this revision 6 bibliography. The additions file deliberately includes none of these papers and makes no novelty or priority claim.

## Network sources

- arXiv API records: https://export.arxiv.org/api/query?id_list=1706.03762,2010.11929,2104.14294,2304.07193,1711.05101,2301.08243,2508.10104,2510.22916
- DINO ICCV 2021 paper: https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html
- Official CSIRO competition page: https://www.kaggle.com/competitions/csiro-biomass/overview
- DOI metadata: https://doi.org/10.1109/CVPRW.2019.00325, https://doi.org/10.1111/ecog.02881, and https://doi.org/10.1214/aos/1176344552
