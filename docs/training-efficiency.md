# Training efficiency and GitHub Actions training

Goal: faster training of the learned scorer, and a way to retrain without the
owner's MacBook. Timings are from the MacBook (8 cores) with the data already on
disk; nothing was downloaded. Anything about GitHub runners is an estimate and
the workflow has not been run (it was only linted with actionlint).

## 1. Measured profile

Training run being profiled: `train.ts --cnn 3 --fcdb 0.5 --epochs 30 --l2 3e-3`
(the shipped cnn head). GAICD train has 2636 images, 231,489 candidate windows;
val 200 images, 17,345; FCDB train 1369 images (about 296k windows).

| stage                                                     | measured                                                      | notes                                                                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dataset fetch                                             | not measured (no downloads)                                   | archives on disk: GAICD 308 MB, FCDB 321 MB, FLMS 204 MB; images 315 + 244 + 204 MB. FCDB restore pulls from Flickr, so it is the unknown.                              |
| plug-in inference (faces + saliency + cnn)                | 0.21 s/image wall with onnxruntime defaults, 0.58 CPU-s/image | per image: YuNet 15 ms, U2-Netp saliency 168 ms (80% of it), MobileNetV2 20 ms. Needed images: 2636 + 200 + 1369 = 4205, so about 15 min. Results are cached per image. |
| feature dump (hand + cnn pooling, 749 columns)            | gaicd-val 17k windows in 1.5 s on 8 procs                     | gaicd-train + fcdb-train about 1 min. Not a bottleneck, and dumps are already compact Float32 `.bin` (+ small `.json`).                                                 |
| training                                                  | about 19-20 s/epoch before                                    | 30 epochs: 524 s logged in `improve-cnn.txt` (3 runs in parallel); LambdaRank 894-908 s (scores every image's windows each step)                                        |
| selection (val SRCC + FCDB holdout each epoch)            | 0.7 s/epoch                                                   | 4% of an epoch before, 7% after                                                                                                                                         |
| load + standardise 1.5 GB of dumps                        | about 1 s                                                     | 1.8 GB RSS per training process                                                                                                                                         |
| shrink                                                    | seconds                                                       |                                                                                                                                                                         |
| eval (`box-eval.ts` on the FCDB holdout, cached plug-ins) | 8 s                                                           |                                                                                                                                                                         |

Critical path: training is 85-95% of the compute of a run. One config costs
about 9 min, so the "many hours" came from sweeps (the `bench/out/*.sh`
scripts: dozens of configs, two waves of three processes each), repeated dumps
for every variant, and the cold inference pass. The fixed per-run cost
(inference, dump) is small once cached.

Inner loop, per training pair: 2 forward passes and 2 backward passes through
749-32-32-1 in scalar Float64 JS, about 2 GMAC/s: add-latency bound in the
forward dot products and memory bound in the layer-0 gradient accumulation.

## 2. What changed, and the speedups

All changes keep results bit-identical or are pure additions.

| change                                                                                                                                                                                                                                           | file                                                                                             | effect                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | ---------------------------------- |
| Forward pass computes 4 output units at once with independent accumulators (each still summed in the original order), reads layer 0 straight from the dump (no copy to a Float64 vector)                                                         | `bench/learned/train.ts`                                                                         | bit-identical                      |
| Backward for both rows of a pair is fused: each gradient element still receives `+= dA*xA` then `+= dB*xB` in the original order, but the buffers are touched once                                                                               | `bench/learned/train.ts`                                                                         | bit-identical                      |
| Training time is printed (`time: train Xs, val/holdout eval Ys`)                                                                                                                                                                                 | `bench/learned/train.ts`                                                                         | log only                           |
| Content-addressed dump cache: `<name>.key` = SHA-256 of the feature code, plug-in code, model pins, package versions, annotations, projection and dump args; unchanged dumps are skipped (`--force` recomputes). `--outdir` relocates the output | `bench/learned/dump.ts`, `scripts/weights-fingerprint.ts` (`featureKey`)                         | stage skipped when unchanged       |
| `precompute.ts --shard k/n` for parallel processes; `SMARTCROP_ORT_THREADS` caps onnxruntime threads; `SMARTCROP_CACHE_DIR` relocates the plug-in cache                                                                                          | `bench/precompute.ts`, `plugins/onnx/runtime.ts`, `plugins/onnx/cnn.ts`, `bench/plugin-cache.ts` | 1.4x on inference                  |
| Ensemble members trained as separate processes and merged (`combine.ts`)                                                                                                                                                                         | `bench/learned/combine.ts`                                                                       | K members in about the time of one |
| `gate.ts`, `projection-from-shipped.ts`, `weights-fingerprint.ts`                                                                                                                                                                                | new                                                                                              | for the workflow (section 3)       |

Measured:

| stage                            | before                           | after                                                              |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------ |
| training epoch (cnn 3, fcdb 0.5) | about 19-20 s                    | about 9-11 s (about 2x; the machine was busy for part of the runs) |
| 30-epoch run                     | about 9 min                      | about 4.5-5.5 min                                                  |
| hand-feature model, 2 epochs     | 8 s                              | 6 s                                                                |
| `--lambda 0.5` epoch             | about 30 s                       | about 15 s                                                         |
| plug-in inference, 96 images     | 20.1 s (1 proc, default threads) | 13.9 s (8 procs x 1 thread); 14.8 s (4 x 2); 55.9 s (1 x 1)        |
| ensemble of K                    | K x one run, sequential          | about one run per 4 members (4 cores), same file                   |

Not done on purpose: eval every N epochs (changes which epoch is selected, saves
only 7%); Float32 or SIMD kernels (not bit-identical); data-parallel SGD
(changes the result); skipping the existing dumps' recompute (an existing dump
has no `.key`, so it is recomputed once, about 1 min).

Equivalence evidence (all on the real data, from the repo checkout):

- `train.ts --cnn 3 --fcdb 0.5 --epochs 2 --l2 3e-3 --write --out X.json`:
  the new and the original `train.ts` (from `git show HEAD:`) produce
  byte-identical model files (`cmp`), including the note with val SRCC 0.7906,
  Acc1/5 47.0, Acc1/10 70.0 and holdout IoU 0.7700. Repeated after formatting.
- The same for the hand-feature model (`--hidden 32,32 --epochs 2`): identical.
- `combine.ts` of seed 1 and seed 2 runs equals `train.ts --ensemble 2` (model
  and cnn fields identical).
- `dump.ts` into a scratch dir (8 procs) is byte-identical to the existing
  `gaicd-val-faces+saliency+cnn.{bin,json}`; the second run is skipped by its key.
- `projection-from-shipped.ts` writes a file identical to
  `bench/out/cnn/projection.bin`.
- `gate.ts` on the shipped model reproduces the logged metrics (val SRCC 0.8072,
  Acc1/5 55.0, Acc1/10 71.5, holdout IoU 0.7780) and fails on itself at margin 0;
  on `exp-ema.json` it passes (1.5857 vs 1.5852).
- `bun check` passes (no type errors), `bun test` passes (30 tests, including the
  classic golden). Changed files were formatted individually, not with a
  repo-wide `bun fmt`; `dump.ts`, `precompute.ts`, `plugins/onnx/cnn.ts`,
  `train.ts` were already unformatted at HEAD, so their diffs include whitespace.

## 3. GitHub Actions design

### Decisions recorded (from the owner)

1. Datasets (GAICD, FCDB, FLMS) are downloaded fresh in every run with
   `scripts/fetch-datasets.ts` and used only for the duration of that run. They
   and everything derived from them (plug-in caches, feature dumps) are never put
   in `actions/cache` or in artifacts.
2. Trained weights (full-precision JSON, packed weights, metrics) may be stored in
   `actions/cache` and artifacts, keyed by a training fingerprint.
3. ONNX models may be cached, keyed on the pinned sha256 values in
   `scripts/fetch-models.ts`. Bun's package cache may be cached.
4. The workflow never commits, pushes or opens PRs; the owner reviews the artifact
   and commits the weights and manifest.
5. Release/deploy is gated on weights freshness (below).

### `.github/workflows/train.yml` (workflow_dispatch only)

Inputs: `cnn` (K, 0 = hand model), `hidden`, `l2`, `lr`, `epochs`, `fcdb`, `ppi`,
`lambda`, `ema`, `cosine`, `ensemble`, `seed`, `refit_pca`, `gate_margin`,
`parallel`, `force`. They are validated as plain numbers and reach the shell only
through `env`.

It is one job, `ubuntu-latest`, `timeout-minutes: 345`. Because no stage needs an
artifact hand-off, no dataset or derived data ever leaves the runner:

1. checkout, bun, `bun install --frozen-lockfile` (bun cache keyed on `bun.lock`).
2. Fingerprint: `scripts/weights-fingerprint.ts --config <inputs>`.
3. `actions/cache/restore` of `out/weights` under `weights-<fingerprint>`. On a
   hit (and no `force`) every training step is skipped, the stored gate table is
   shown, and the stored weights are uploaded again as the artifact.
4. ONNX models: cache key = SHA-256 of the sorted pinned sha256 lines of
   `fetch-models.ts`; `fetch-models.ts` still verifies every file.
5. Datasets: `fetch-datasets.ts gaicd fcdb flms --clean-archives`; a check that the
   splits are complete (GAICD 2636/200, FCDB train 1369).
6. cnn projection: default reuses the projection in the shipped cnn weights
   (`projection-from-shipped.ts`), so the new head is comparable with the shipped
   one and no PCA is fitted; `refit_pca` runs `cnn-pca.ts`.
7. Inference: 4 `precompute.ts` shards in parallel, 1 onnxruntime thread each, for
   gaicd train, gaicd val, fcdb train (test splits and FLMS are never read).
8. Dumps (`--procs 4`), then the images and plug-in caches are deleted.
9. Training: one `train.ts --seed s --write --out m<s>.json` process per ensemble
   member, `parallel` (default 4) at a time, about 1.8 GB each.
10. `combine.ts`, then the gate, then `shrink.ts --out` (packs to f16, losslessly,
    writing into the output dir rather than `src/`), then the manifest.
11. Summary table (shipped vs new), artifact upload, cache save.

Gate (`bench/learned/gate.ts`): the shipped head (`src/learned-cnn-weights.ts`, or
`learned-weights.ts` for `cnn: 0`) and the new model are scored on the same
validation data train.ts selects on: GAICD val and, when `fcdb > 0`, the FCDB-train
holdout (last 15%). The metric is val SRCC + holdout IoU. The step fails the run if
new <= shipped + `gate_margin`. Test data is never read. A failed model is uploaded
for inspection but not cached.

Outputs: artifact `trained-weights-<fp12>` (retention 30 days) holding
`model-full.json`, `learned-cnn-weights.ts` (or `learned-weights.ts`),
`weights-manifest.json`, `gate.json`/`gate.md`, `config.json`, `fingerprint.txt`,
`shrink.log` and member logs. A step refuses to upload anything else or any file
over 20 MB.

There is no separate cleanup job: no dataset or derived data is passed between
jobs, so there is nothing to delete; the runner is destroyed at the end. If the
design ever splits into jobs that pass data, use `retention-days: 1` and a final
`if: always()` job calling `gh api -X DELETE /repos/{repo}/actions/artifacts/{id}`.

Resumability: no checkpointing inside `train.ts`. A member takes minutes, and a
whole run is estimated at 1-2.5 h of a 6 h limit, so a mid-training checkpoint
(Adam moments, RNG state, best net) would add risk for little. The unit of resume
is the run; a finished fingerprint is free thanks to the weights cache.

### Fingerprint and freshness gate

`scripts/weights-fingerprint.ts` hashes (SHA-256): `src/smartcrop.ts`,
`src/learned.ts`, `src/cnn.ts`, `plugins/onnx/*.ts`, `bench/plugin-cache.ts`,
`bench/precompute.ts`, `bench/learned/*.ts` (not `_*.ts` scratch files),
`scripts/fetch-datasets.ts` and `scripts/mat5.ts` (dataset identities),
the pinned model sha256 values in `scripts/fetch-models.ts`, the
onnxruntime-node, onnxruntime-common and @napi-rs/canvas versions from
`bun.lock`, the shipped projection (unless `refitPca`), and the canonicalised
config (flags and seed; `"3e-3"` equals `0.003`). The weight files themselves,
docs, tests, eval scripts, `parallel`, `gate_margin` and `force` are excluded.

- Stability: run twice with the same config and also with a re-ordered, re-typed
  config: same hash. On a temp copy (`--root`): same hash for a copy; a comment
  appended to `src/cnn.ts` changes it; a change to `src/learned-cnn-weights.ts` or
  `src/entry-cjs.ts` does not; changing a config value or a pinned model sha256
  changes it.
- Recording: `src/weights-manifest.json` (new file, not imported by the library, so
  the bundle is unchanged) holds, per head (`cnn`, `hand`): fingerprint, config,
  sha256 of the weights file, timestamp and gate metrics. The workflow writes it into
  the artifact; the owner commits it with the weights. `weights-fingerprint.ts
--write --config '...'` records one by hand. I did not edit
  `src/learned-weights.ts` or `src/learned-cnn-weights.ts`, and did not create a
  manifest, because I cannot know which config produced the shipped weights.
- Check: `bun scripts/weights-fingerprint.ts --check` recomputes each head's
  fingerprint with the config stored in the manifest, and also checks the weights
  file hash. It fails with an explicit "STALE ... run the Train workflow" message,
  or when the manifest is missing. Exercised on a temp copy: missing manifest ->
  exit 1; recorded -> exit 0; one appended line in `bench/learned/train.ts` -> exit 1.
- Wiring: `.github/workflows/weights-fresh.yml` (`workflow_call` and
  `workflow_dispatch`) runs the check and, on failure, writes the stale message and
  links the latest successful Train runs to the job summary.
  `release.yml` (which replaced `publish.yml`) has `weights-fresh` as a job and `release` `needs` it; its
  trigger (manual only) and the disabled npm publishing are unchanged, and
  `scripts/release.sh` stays disabled. `tests.yml` is not gated (see risks).

## 4. Estimates

Full run, per stage (K = 1 member unless noted):

|                         | MacBook (before)     | MacBook (after)      | Actions 4 vCPU (estimate)                                |
| ----------------------- | -------------------- | -------------------- | -------------------------------------------------------- |
| fetch                   | unknown              | unknown              | unknown, 10-90 min; Flickr and Google Drive are the risk |
| inference (4205 images) | about 15 min         | about 10 min         | about 15-20 min (assumes cores 1.5-2x slower)            |
| dumps                   | about 1-2 min        | about 1-2 min        | about 3-5 min                                            |
| train 30 epochs         | about 9 min          | about 5 min          | about 8-15 min                                           |
| ensemble of 4           | about 36 min         | about 6-8 min        | about 12-25 min (4 cores, memory contention)             |
| gate + shrink           | under 1 min          | under 1 min          | about 1-2 min                                            |
| total, K = 1            | about 25 min + fetch | about 17 min + fetch | about 0.6-1.2 h incl. fetch                              |
| total, K = 4            | about 55 min + fetch | about 20 min + fetch | about 1-2.5 h                                            |

A config sweep: before, 3 processes at a time (about 33 min for the six-config
`improve-cnn.sh`); after, about 17 min for the same six. The workflow runs one
config per dispatch.

## 5. Open risks and decisions for the owner

Decisions:

1. Release gating: I gated only the release workflow (`release.yml`). `tests.yml` (push and PR CI) is not
   gated: it would turn every PR red until weights are retrained after any edit
   to a hashed file. Say if you want it there, perhaps as a non-blocking job.
2. First manifest: there is none. The publish workflow will fail its gate until
   one exists. Either run the Train workflow once (recommended, since the
   shipped weights predate this code and the exact epochs/seed are not recorded in
   the weights note) and commit what it produces, or bless the current weights
   with `bun scripts/weights-fingerprint.ts --write --config '<json>'` if you
   know the config. Blessing asserts something I cannot verify.
3. Any edit to a hashed file (including whitespace or a comment in
   `bench/learned/*.ts`) makes weights stale, as specified. This change itself edits
   `train.ts`, `dump.ts` and others, so it needs a retrain or a re-bless. Hashing
   with comments and whitespace stripped would soften this at the cost of
   complexity.
4. FLMS is downloaded because the decision says so, but neither training nor the
   gate uses it. Dropping it saves about 200 MB of download.
5. Dataset licence: you decided to download the datasets on runners per run. The
   licence terms of GAICD and FCDB remain unstated; trained weights remain
   research-use only. The decision to store weights in caches and artifacts of a
   public repository makes those weights downloadable by anyone logged in to
   GitHub while they live (30 days); a private repository or a shorter retention
   narrows that. I did not resolve it.
6. Artifact retention: 30 days, "normal". Caches are evicted after 7 days idle or
   at the 10 GB repo cap; the artifact, then a commit, is the durable copy.

Risks:

- The workflow is untested end to end. Verified: actionlint, every script on local
  data, equivalence of the pieces. Not verified: the fetch on a runner (Google
  Drive quota and Flickr dead links are plausible failures; the completeness check
  fails the run if splits are short), runner timings, the cache and artifact steps.
- Linux onnxruntime may differ numerically from macOS, so cnn features, and
  therefore weights, are not bit-comparable with laptop runs even at an equal
  fingerprint. The gate compares both models on the same runner features, so it
  stays fair.
- Dump row order depends on `--procs` (4 on the runner, 8 locally by default), which
  changes the initial shuffle order; results differ slightly between a laptop run
  and a runner run with the same seed.
- `ensemble` above 4 runs in waves; with `parallel: 4` all four cores and about
  7.5 GB RAM are used, so runner memory (16 GB) is fine, disk (about 7 GB at peak
  of about 14 GB free) is fine.
- Runs are serialised with `concurrency: train`.
- Dataset contents can drift from the local ones (dead Flickr links); the split
  counts are checked, but not the image contents.
- `parallel` and `gate_margin` are not in the fingerprint (they do not change the
  weights); `force` replaces the cache entry only when the gate passes.
