# smartcrop.js crop-quality improvement plan

Status: approved; phase 0 in progress. Last updated 2026-11-22.

## Why

On the jauderho.com hero strips (full-width 2.2:1 crops of 3:2 photos), the
owner hand-corrected half the crops. Measured against those hand-set values
(the original motivation only; they are not used as ground truth for tuning):

| Method (9 hand-corrected photos) | Within ±10% | Mean error |
| -------------------------------- | ----------: | ---------: |
| smartcrop 3.0.0                  |         0/9 |       35.7 |
| Always center the strip          |         7/9 |        5.6 |

The strip sinks to textured foreground (ground, water, grass) and cuts off
the sky, the sun, or the top of the subject. Diagnosis so far:

- The score is dominated by Laplacian _detail_, which rewards texture rather
  than subjects. Faces, sky and smooth objects score low.
- None of the existing options rescue it (best: `ruleOfThirds: false`, 1/9).
- The `cie()` luminance bug (upstream PR #142) is real but has almost no effect
  here (mean error 35.7 → 35.6). Fix it, but it is not the cause.
- The skin detector is tuned to one skin tone range (fairness concern raised
  by Twitter's 2021 cropping study) and false-fires on wood, sand and food.
- The algorithm is the same one libvips `attention` and the Python port use,
  so it has had no meaningful algorithmic change since ~2014.

## Upstream feedback (jwagner/smartcrop.js and siblings)

Grouped from 45 relevant issues and PRs:

- Wrong subject / busy background or shadows win (#141, #90, #78, #36, #84).
- Prefer the centre when unsure; symmetry (#67, #82, #57) — matches our data.
- Skin-tone bias and skin false positives (#9, #103, #79, #73).
- Requests for real detectors or saliency (#33, #55, #112, #122; cli #22).
- `cie()` luminance typo (#77, PR #142).
- Output 1 px smaller than requested (#123; cli #23).
- Wants a focal point / CSS `object-position` instead of a box (#45, #81, #99)
  — exactly what jauderho.com computes by hand.
- Slow at low `minScale` on large images (#46); multiscale ideas (#70).

## Principles

1. **Measure first.** Every change is judged on a harness, never on a few
   eyeballed images. A change ships only if it beats the current default
   _and_ the center baseline on held-out data.
2. **Core stays small and model-free.** Pure JS, runs in browser and Node.
   ML models are optional plug-ins feeding the existing `boost` mechanism or a
   new importance-map input.
3. **Original vs improved, no site "truth".** The baseline is the original
   algorithm (the `classic` preset reproduces upstream 2.0.5 exactly). The
   jauderho.com hand-set positions are not ground truth: they are shown only
   as a labelled reference and are never used to score or select candidates.
   Selection uses public benchmarks (FCDB train); FCDB test and FLMS are
   held out. The site photos are compared visually, original vs improved.
   behavior; it is regenerated only for an intended, measured change.

On the public FCDB benchmark the original algorithm barely beats a plain
centre crop (IoU 0.718 vs 0.701, aspect-ratio-aware, minScale 0.5).

## Evaluation harness (phase 0)

| Set                                      | Use                                                          | Metric                           |
| ---------------------------------------- | ------------------------------------------------------------ | -------------------------------- |
| jauderho.com strips (private, 18)        | Visual review, original vs improved, via the website session | change vs original (no truth)    |
| FCDB test (348 imgs, 1 human crop each)  | Main tuning target, aspect-ratio-aware                       | IoU, boundary displacement (BDE) |
| FLMS (500 imgs, ~10 expert crops each)   | Held-out check                                               | best-of-N IoU, BDE               |
| GAICD test (500 imgs, scored candidates) | Ranking quality: smartcrop scores each candidate             | Acc1/5, Acc1/10, SRCC            |

- Aspect-ratio-aware protocol: take the ratio from the ground-truth box and
  ask smartcrop for its best crop at that ratio (as Cropper, CVPR 2025, does).
- Always report the center-crop baseline and the current default.
- Datasets are downloaded by a script into a gitignored folder; never committed.
- Reference points from the literature (FCDB IoU): learned scorers like GAIC
  reach ~0.67–0.69, VLM methods ~0.75. smartcrop's own number is unknown.

## Phases

Each phase ends with a harness run, a short results note, and a contact-sheet
review by the website session.

### Phase 1 — Fix and re-tune what exists (pure JS)

1. Fix the `cie()` luminance coefficients and the 1 px rounding loss (#123).
   Add a `position` (focal point / CSS `object-position` %) to the result.
2. Automated search over existing weights and priors (detail, saturation,
   skin, edge penalty, outside importance, rule of thirds, scan resolution),
   tuned on an FCDB train split, checked on FCDB test and FLMS.
3. Add an explicit, tunable **center prior** (the data says it matters).
4. Skin detector: off by default, or gated (only with low saturation of
   neutral colours, never on low-colour images). Fairness check on a
   balanced face set before deciding the default.

### Phase 2 — Better model-free signals (pure JS)

1. **Colour-contrast saliency** (frequency-tuned, Achanta 2009): distance of
   blurred Lab colour from the image mean. One blur; complements detail.
2. **Image-signature saliency** (sign of a 64×64 DCT): cheap pop-out detector.
   Experimental; keep only if the harness says so.
3. **Border cut-off penalty**: penalise crop edges that pass through
   high-importance blobs (cut-off heads, buildings, vehicles).
4. **Detail normalisation**: dampen uniformly textured regions (foliage,
   gravel, water) by local contrast normalisation, so texture stops winning.
5. **Content-type gating**: low colour count / flat fills ⇒ graphics or
   screenshot ⇒ skin off, centre bias up, avoid cutting text blocks.

### Phase 3 — Optional detectors (plug-ins, not core)

1. A generic `importanceMap` input so any saliency model can feed the scorer.
2. **Face boost adapter** (BlazeFace via MediaPipe or `@vladmandic/human`):
   boxes with headroom margin; prefer windows containing all faces, then the
   largest (fallback chain, as imgix does).
3. **Salient-object plug-in**: U²-Netp (4.6 MB ONNX, Apache-2.0) via
   onnxruntime; MODNet (26 MB, Apache-2.0) as an alternative. Avoid RMBG
   (non-commercial) and ISNet (AGPL).

### Phase 4 — Only if phases 1–3 plateau

- A small GAIC-style learned candidate scorer (ONNX), trained on GAICD.
- Offline/site-build-time helpers (e.g. a vision model locating the subject
  and passing it as a boost) — outside the library.

## Decisions (2026-11-22)

- **Evaluation:** start from the original code as the baseline and show how
  far research-backed improvements get on their own, through all phases,
  before any human tuning. Site hand-set positions are reference only.

- **Rollout:** new behaviour ships as an opt-in preset during 3.x; current
  crops stay the default until the harness proves the new ones, then the
  default flips in 4.0.
- **Datasets:** FCDB, FLMS and GAICD are downloaded into a gitignored local
  folder for tuning and evaluation; never committed or redistributed.
- **Skin detector:** default decided by measurement (harness plus a
  fairness check on a balanced face set).

## Acceptance bar before showing results (set 2026-11-22)

The owner delegated training and judging. Results are shown only when the
best configuration meets all of these. Selection never uses FCDB test, FLMS
or GAICD test.

- **GAICD test ranking:** clearly beats the classic scorer (SRCC 0.369,
  Acc1/5 22.4) and the large-centred-crop baseline (SRCC 0.432, Acc1/5 47.4).
  Target: SRCC ≥ 0.6 and Acc1/5 ≥ 55.
- **FCDB test and FLMS (aspect-aware IoU):** better than classic. This was
  first "at least as good as the tuned configuration (0.752 / 0.932)". It was
  revised 2026-11-22 after GAICD val showed that the IoU-tuned classic
  weights rank human-scored crops worse than the original does (SRCC 0.26).
  They fit FCDB's single box rather than crop quality, so they are not a fair
  bar.
- **No regression on any content subset** (people, faces, landscape, sky,
  other) compared with classic.
- **Visual check of the gallery** (strips, examples, FCDB) shows no class of
  obviously worse crops, such as cut faces or a lost horizon or sky.
- The pipeline runs locally, and only the downloaded weights are needed.

## Results (2026-11-22, held-out test sets, measured once after selection)

The learned scorer (`scorer: 'learned'`, `PRESETS.next`) is a 117-feature,
32-32 MLP trained pairwise on GAICD train MOS plus FCDB-train windows. It is
selected on GAICD val and an FCDB-train holdout. "+fs" adds the local ONNX
face (YuNet) and saliency (U²-Netp) maps.

| | GAICD SRCC | Acc1/5 | Acc10 mean | FCDB IoU | FLMS IoU |
|---|---|---|---|---|---|
| centre crop | 0.432 | 47.4 | 48.4 | 0.701 | 0.894 |
| original (classic) | 0.369 | 22.4 | 33.2 | 0.718 | 0.918 |
| learned, pure JS | 0.623 | 48.0 | 61.5 | 0.744 | 0.917 |
| **learned + faces/saliency** | **0.739** | **50.4** | **68.3** | **0.765** | **0.925** |

FCDB per subset, original → learned+fs:
- people 0.761 → 0.795
- faces 0.764 → 0.803
- landscape 0.693 → 0.749
- sky 0.705 → 0.753
- other 0.705 → 0.755

FLMS is level or up in every subset; sky is 0.9245 → 0.9238.

Rejected after measurement:
- Blending in the IoU-tuned classic score: GAICD val SRCC fell to 0.26–0.70.
- A 5-seed ensemble: +0.005 val SRCC for 5× the weights.
- A content-preservation floor: no val gain.

Open:
- Acc1/5 stays below the 55 target. Learned CNN features (GAIC-style) are the likely route, which needs downloaded weights.
- The pure-JS model sometimes loses a small subject, such as a person or the church in the examples. The +fs config fixes most of these.

## Sources

Research surveys run 2026-11-22 (academic methods, production systems,
upstream issues). Key references: GAIC (Zeng et al., CVPR 2019 / TPAMI 2020),
CACNet (CVPR 2021), Cropper (arXiv 2408.07790), FCDB (Chen et al., WACV 2017),
Twitter cropping fairness study (arXiv 2105.08667), frequency-tuned saliency
(Achanta et al., CVPR 2009), image signature (Hou et al., TPAMI 2012), libvips
`smartcrop.c`, Cloudinary `g_auto`, imgix crop modes, MediaPipe face detector.
