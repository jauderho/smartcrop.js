# Content-aware cropping: state of the art (synthesis, 2026-11-22)

Sources were read by three research passes over the papers listed at the end.
Numbers came from fetched pages, often through a summarising fetcher, so they
should be spot-checked against the paper tables before they are cited
externally. "Inference" marks our own reasoning; everything else is reported
by the papers.

## Distinctions

- **Aesthetic vs subject-preserving.** GAIC, S2CNet, ProCrop, CROP, Venus
  and CropReward aim at aesthetics. GenCrop and HCIC aim at keeping a given
  subject, such as a person. smartcrop's use case, thumbnails and hero
  strips, needs both.
- **Free-form vs fixed aspect.** Cropper reports both on FCDB: IoU 0.667
  free-form against 0.756 aspect-ratio-aware. Our harness is aspect-ratio
  aware, so only aspect-aware numbers compare with ours.
- **Specialised predictors vs prompted VLMs.**
  - Small CNN rankers: GAIC (1.8M params), S2CNet (3.9M), HCIC (19M).
  - VLM methods: Cropper (Gemini, in-context), CROP (7B), COMEX (0.8–2B),
    Venus.
  - Learned reward models: CropReward (4.6B).
- **Out of scope:** seam carving, outpainting and generative retargeting.
  GenCrop and ProCrop use outpainting only to make training data.

## Per approach

| Method      | Status                    | Constraints                   | Code / weights / licence                   | Reported results                                | Cost                        |
| ----------- | ------------------------- | ----------------------------- | ------------------------------------------ | ----------------------------------------------- | --------------------------- |
| GAIC        | CVPR'19 / TPAMI           | aspect via grid anchors       | code; Drive weights; licence unclear       | GAICv2 SRCC ≈0.85, Acc1/5 ≈64                   | 125–212 FPS (GPU)           |
| S2CNet      | AAAI'24                   | aspect; multi-object graph    | code; GAIC weights only; no licence        | GAICv2 SRCC 0.861, Acc5 64.0                    | 163 FPS, needs Faster R-CNN |
| HCIC        | ECCV'22                   | person box (manually checked) | code, no weights, no licence               | FCDB+FLMS human subset IoU 0.747                | not reported                |
| GenCrop     | AAAI'24                   | subject mask                  | BSD-3 code, no weights                     | FCDB 0.654                                      | 5.7 ms (GPU)                |
| ProCrop     | arXiv; AAAI'26 per README | none                          | 512 MB .pth, Apache-2.0                    | FLMS 0.843, GAICv2 Acc1/5 85.4                  | retrieval stack             |
| Cropper     | CVPR'25                   | free-form, subject, aspect    | no code; uses hosted Gemini                | FCDB aspect 0.756, GAICD Acc̄5 84.3 / SRCC 0.904 | ~3 VLM calls × ~6 s         |
| CROP        | arXiv May '26             | composition reasoning         | none                                       | FCDB 0.745, FLMS 0.871                          | 7B                          |
| COMEX       | ACM MM'26                 | explainable                   | none                                       | FCDB 0.7225                                     | 0.8–2B                      |
| Venus       | CVPR'26                   | aesthetic guidance            | weights on HF; data gated; licence unknown | FCDB 0.68–0.71 (reported by others)             | MLLM                        |
| PhotoGuider | arXiv Nov '26             | 60 features + LightGBM, gates | repo and demo; licence unknown             | FCDB 0.759 fast / 0.766 full                    | DINOv3/SAM/CLIP features    |
| CropReward  | arXiv Oct '26             | reward model                  | promised, not released                     | GAICD-v2 PLCC 0.766 / SRCC 0.716 vs humans      | 4.6B                        |

## Findings that matter for smartcrop

1. **Where we stand on FCDB.** Our tuned pure-JS scorer reaches 0.752, and
   0.757 with a U²-Netp saliency map. That is level with Cropper's
   aspect-aware 0.756 and close to PhotoGuider's 0.759–0.766. The protocol is
   comparable to Cropper's; test-split sizes differ slightly (345 vs 348).
   FCDB IoU is near its ceiling for every published method, so large IoU
   gains are not available there.
2. **IoU is a weak proxy for preference.** CropReward and HCIC both say so.
   Ranking human-scored candidates (GAICD SRCC and Acc) is the stronger test.
   For a personal collection, measure how often a crop is preferred over the
   original image and over the classic crop.
3. **Which features carry the signal.** PhotoGuider's ablation ranks them:
   - Subject: saliency kept in the crop.
   - Composition: thirds and centre alignment, balance, tightness, negative
     space, boundary placement.
   - Integrity and aesthetics, which matter less.

   HCIC adds person-partition features and a content-preserving heatmap.

4. **Nothing small and licensed is ready to ship.** No learned model under
   30 MB has a clear licence and weights. GAIC's MobileNet ranker is the
   closest, but its licence is unclear and it needs a custom RoD op.
5. **GAICD carries the human signal.** About 2,636 training images, each with
   about 86 MOS-scored candidates, are already in our download.

## Adobe

The paper bodies for Fang 2014 and Good View Hunting could not be read
(HTTP 403, and PDF extraction was unavailable), so the details below come
from abstracts, project pages and citing papers.

- **Fang, Lin, Měch & Shen, ACM MM 2014.** This is the FLMS paper.
  - It combines three models: a learned composition model (saliency
    layout), boundary simplicity (gradients along the crop edges) and
    content preservation (salient mass kept).
  - It uses a hard floor on content preservation, then a linear mix of
    composition and boundary simplicity, over dense candidates.
  - FLMS IoU 0.740, free-form.
- **Good View Hunting, CVPR 2018 (Wei et al., Stony Brook and Adobe).**
  - The CPC dataset has about 10.8K images and about 1M crowdsourced view
    pairs. Its VEN model reaches GAICD SRCC 0.621 and Acc1/5 40.5.
  - No licence is stated for the code, weights or data, so we do not use
    them.
- **Products** (methods undocumented):
  - Firefly Services AutoCrop detects and preserves foreground objects. It
    offers priority-object and full-subject modes, the latter ignoring the
    aspect ratio as a fallback.
  - Dynamic Media Smart Crop derives one crop per aspect ratio from a focal
    point.
  - Lightroom's crop "Auto" only straightens the horizon.
- **GenCrop (Stanford and Adobe, 2023).** Uses outpainting to make training
  labels and a subject mask as input. See the table above.
- **Adopted in our scorer:**
  - a hard content-preservation floor;
  - boundary-simplicity features;
  - an explicit area prior;
  - a consistent subject anchor across aspect ratios, for the site
    pipeline.

## Deep-feature (CNN) plug-in: research for a later optional feature

Goal: lift top-1 crop choice by adding learned image features to the learned
scorer, as an optional local ONNX plug-in.

- **Real gap.** Our split is GAICD-v2. There, GAIC (MobileNetV2) reports SRCC
  about 0.85 with Acc5-mean about 64 and Acc10-mean about 80. S2CNet reports
  SRCC 0.861. Ours is 0.739, with Acc10-mean 68.3.
- **What works in GAIC.**
  - Backbone: a frozen-shape CNN at a 256 px short side.
  - Pooling: 9×9 RoIAlign of the crop plus RoDAlign of the discarded area,
    reduced to about 8 channels by a 1×1 conv, with multi-scale features.
  - Ablations: RoD adds about +0.01 SRCC and +2 Acc1/5; multi-scale adds
    about +0.01. MobileNetV2 beats VGG16 and ShuffleNetV2. Stride-32 and
    ResNet-50 features did worse.
  - Global CLIP aesthetics rank crops poorly: AesCLIP gets 0.27 SRCC zero-shot
    (CropReward). Features must be crop-local.
- **Candidate backbones.**
  1. MobileNetV2, ONNX model zoo: 13 MB fp32, Apache-2.0 repo, with an
     ImageNet-weights caveat. It is the best-evidenced choice. Concatenate
     the stride 8, 16 and 32 blocks (448 channels) at stride 16.
  2. DINOv2 ViT-S/14, onnx-community: 88 MB fp32 or 24 MB int8,
     Apache-2.0. Strong semantic patch features, but unproven for cropping.
- **Design.**
  - Run the ONNX backbone and export only the feature map.
  - Do RoI/RoD pooling in JS; optionally PCA the map to about 32 channels.
  - Concatenate with the 141 hand features and train a small head.
- **Expected gain** (inference): with a frozen backbone, about +0.04 to +0.09
  SRCC and +4 to +9 Acc10-mean. Reaching GAIC's level probably needs
  fine-tuning the backbone.
- **Licence caveat (applies to the shipped weights too).** GAICD's dataset
  terms are unstated: the code is MIT, and the images are crawled from Flickr.
  FCDB is a Flickr-sourced research set. Weights trained on them are
  research-use until the dataset authors are asked.

**Built (2026-11-23).** We used MobileNetV2 taps at strides 8, 16 and 32 (448
channels at stride 16), a PCA to 32 channels, and 3×3 RoI and RoD pooling
plus the global mean. The 141 hand features and 608 CNN features feed a
32-32 MLP. Five other configurations (K=5, a 64-32 head, other L2 values)
were within noise on val.

Results:

- GAICD val SRCC 0.755 → 0.807.
- GAICD test SRCC 0.739 → 0.795, Acc1/5 50.4 → 54.0.
- FCDB 0.765 → 0.769, FLMS 0.925 → 0.931.
- The only subset that dropped is FCDB faces (0.803 → 0.800, noise).

The gap to GAIC (about 0.85) remains, and would probably need a fine-tuned
backbone.

## Recommendation

- **Lightweight production baseline (in progress).** A learned linear or
  tiny-MLP composition ranker, in pure JS, trained on GAICD train MOS with a
  pairwise ranking loss. Its features come from maps smartcrop already
  computes: saliency recall and density, border cut, centroid against thirds
  and centre, balance, geometry, a horizon row, and face headroom and cut.
  It ships as a few KB of weights, with optional U²-Netp and YuNet maps as
  extra inputs.
  - _Inference:_ this is PhotoGuider's design minus the heavy feature
    extractors.
- **Highest quality.** An offline helper that asks a VLM for subject and crop
  proposals, Cropper-style, then lets the learned ranker refine the choice.
  This is a build-time tool for sites like jauderho.com, not part of the
  library.
- **Evaluation protocol for a personal collection:**
  - Run the original against the improved crop side by side, blind and in
    random order, and record which the reviewer prefers.
  - Report the preference rate and a "no worse than centre" rate.
  - Tag images by content (people, landscape, other) and report each group,
    so one kind of photo does not hide failures on another.

## Sources

CropReward arXiv 2610.34221 · Cropper CVPR 2025 (arXiv 2408.07790) · ProCrop
arXiv 2505.22490 · CROP arXiv 2605.12545 · Venus CVPR 2026 · COMEX arXiv
2608.07570 · PhotoGuider arXiv 2611.02897 · GenCrop arXiv 2312.12080 ·
S2CNet arXiv 2401.08086 (AAAI 2024) · HCIC arXiv 2207.10269 (ECCV 2022) ·
GAIC (CVPR 2019 / TPAMI 2020) · CACNet (CVPR 2021).

Not accessed: Venus CVPR page (HTTP 403), CACNet paper (HTTP 403), and any
LICENSE file for S2CNet or HCIC.
