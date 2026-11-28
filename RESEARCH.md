# Research notes: saliency, CNN features, backends

Experiments run November 2026 on the learned scorer (hand features + optional
CNN head). Everything below is measured, not estimated.

## How things were measured

- **Full crop search** (`bench/box-eval.ts`): IoU of the chosen crop with the
  human crop on FCDB test, GAICD test (top-MOS boxes) and FLMS. Paired per-image
  difference vs a baseline retrained the same way, 95% bootstrap interval,
  2 head seeds averaged.
- **Held-out ranking** (cheaper, used to screen): GAICD test SRCC and Acc1/5,
  FCDB test top-1 MOS proxy.
- Seed noise is about 0.01 on SRCC and FCDB, and 1–2 points on Acc1/5. Only
  differences whose interval excludes 0 count as wins.

## What worked

### 1. IS-Net + UNISAL saliency (hand model, no CNN)

The map is the mean of IS-Net (salient objects, 1024² box-averaged to 256²) and
UNISAL (where people look, MIT1003 fine-tune, 384×288 / 288×384). The hand head
is retrained on that map.

| Saliency (hand head)   | FCDB               | GAICD              | FLMS               |
| ---------------------- | ------------------ | ------------------ | ------------------ |
| U²-Netp (baseline)     | 0.763              | 0.836              | 0.927              |
| UNISAL alone           | 0.769 (+0.006)     | 0.843 (+0.007)     | 0.929 (+0.003)     |
| IS-Net alone           | 0.759 (−0.004)     | 0.825 (−0.011)     | 0.928              |
| **IS-Net + UNISAL**    | **0.773 (+0.010)** | **0.843 (+0.007)** | **0.931 (+0.004)** |
| BiRefNet-lite + UNISAL | 0.776 (+0.013)     | 0.841 (+0.006)     | 0.929 (+0.002)     |

- IS-Net + UNISAL is significant on all three sets. It is built as an opt-in
  (`saliencyMap(img, { model: 'isnet-unisal' })` plus `learnedModel`) on branch
  `saliency-isnet-unisal`, which is not committed yet.
- Costs about 6× the inference time of U²-Netp and about 205 MB of models.
  UNISAL is converted from PyTorch on fetch; it is not hosted.
- The shipped weights gain nothing from the new map. The head must be retrained
  on it.
- BiRefNet-lite + UNISAL is slightly higher on FCDB but not on GAICD or FLMS,
  and BiRefNet-lite is larger (224 MB) and slower than IS-Net.

### 2. Fine-tuning MobileNetV2 end to end (CNN head)

The pipeline is ported to PyTorch: onnx2torch backbone → learned 448→32
projection (initialised from the PCA) → the same RoI/RoD pooling as
`src/cnn.ts` → MLP head. It is trained with RankNet pairs on GAICD + FCDB
train, with BatchNorm frozen. The fine-tuned maps are then cached and the
standard head is retrained with `train.ts`.

| CNN head                                                                  | GAICD SRCC | Acc1/5   | FCDB      | Crop search GAICD |
| ------------------------------------------------------------------------- | ---------- | -------- | --------- | ----------------- |
| Frozen MobileNetV2 (shipped)                                              | 0.793      | 56.2     | 0.766     | 0.834             |
| Fine-tuned, plain (run 1)                                                 | 0.812      | 57.5     | 0.764     | 0.838             |
| **Fine-tuned + flip aug + EMA 0.998 (run 2, the crop-tuned MobileNetV2)** | **0.821**  | **59.9** | **0.770** | **0.845**         |
| Run 2 recipe, seed 2                                                      | 0.821      | 58.2     | 0.768     | –                 |

- Run 2 in full crop search vs the frozen baseline: GAICD +0.015 [+0.012,
  +0.019], FCDB +0.003 (n.s.), FLMS ±0. That is about +0.010 GAICD over what
  ships. The second seed reproduces it.
- The two ingredients that mattered are random horizontal flips of the image
  and boxes, and an EMA of the weights for evaluation and saving.
- Model size and speed are unchanged. Unlike UNISAL, the 14 MB fine-tuned ONNX
  cannot be regenerated on fetch without the datasets, so it is hosted on
  Hugging Face and pinned by sha256 in `scripts/fetch-models.ts`.

## What did not work

All compared on held-out ranking, frozen MobileNetV2 at 0.797 / 55.3 / 0.764
unless stated otherwise.

**Other frozen backbones** (timm, same taps/PCA/pooling) were none better
beyond noise:

| Backbone                              | SRCC          | Acc1/5      | FCDB          |
| ------------------------------------- | ------------- | ----------- | ------------- |
| MobileNetV3-L                         | 0.794         | 52.3        | 0.757         |
| MobileNetV4-M                         | 0.776         | 53.2        | 0.761         |
| ConvNeXt-T (32 / 64 standardised PCA) | 0.798 / 0.797 | 53.5 / 55.8 | 0.772 / 0.767 |
| DINOv2 ViT-S/14 reg                   | 0.771         | 53.4        | 0.761         |
| SigLIP 2 ViT-B/16                     | 0.787         | 51.6        | 0.760         |

**Fine-tuned MobileNetV3-L** ties fine-tuned V2 on GAICD (0.813) and is worse on
FCDB (0.757). It overfits after one epoch.

**Frozen MobileNetV2 pipeline changes:**

| Change                                           | SRCC                  | Acc1/5             | FCDB                  |
| ------------------------------------------------ | --------------------- | ------------------ | --------------------- |
| Input 320 px / 384 px (vs 256)                   | 0.791 / 0.787         | 57.6 / 55.2        | 0.761 / 0.766         |
| Flip-averaged feature map (test-time)            | 0.792                 | 54.6               | 0.759                 |
| PCA 48 ch / 64 ch standardised / 32 standardised | 0.794 / 0.785 / 0.794 | 54.5 / 53.7 / 53.9 | 0.758 / 0.757 / 0.758 |
| Pooling grid K = 2 / 4 / 5 (vs 3)                | 0.784 / 0.788 / 0.797 | 57.9 / 55.8 / 59.1 | 0.758 / 0.760 / 0.763 |

**Fine-tuning recipe variants** (all tie run 2, none beat it):

| Variant                                          | SRCC  | Acc1/5 | FCDB  |
| ------------------------------------------------ | ----- | ------ | ----- |
| + scale (2^±0.25) and colour (±20%) augmentation | 0.816 | 58.2   | 0.771 |
| Backbone LR 4e-5 (vs 2e-5)                       | 0.819 | 59.0   | 0.764 |
| EMA 0.999, 10 epochs                             | 0.820 | 59.2   | 0.768 |
| Weight average of 4 fine-tunes ("soup")          | 0.821 | 58.2   | 0.773 |
| K=5 pooling on run 2 maps                        | 0.822 | 58.7   | 0.769 |

The soup's full crop search (GAICD +0.013) is no better than run 2 alone.

**Backends.** On GitHub runners onnxruntime beats PyTorch for every model
(median ms, onnxruntime / PyTorch):

| Model       | Linux x64   | Linux arm64 |
| ----------- | ----------- | ----------- |
| YuNet       | 4.8 / 56    | 9.0 / 72    |
| MobileNetV2 | 3.3 / 16    | 6.8 / 65    |
| U²-Netp     | 184 / 394   | 261 / 574   |
| IS-Net      | 1159 / 3191 | 1964 / 3413 |

macOS runners have no MPS. On Apple silicon, CoreML (small models) and PyTorch
MPS (large models) are faster, and the plug-ins already use both there.

**Claude as a teacher (pilot).** Claude (Opus, in a Claude Code session) judged
120 GAICD-val crop pairs from 30 images. Each pair was shown twice, with sides
swapped and rows shuffled, and Claude had no access to the answer key.

| Pairs                    | Claude       | Frozen MobileNetV2 head | Fine-tuned (run 2) head |
| ------------------------ | ------------ | ----------------------- | ----------------------- |
| All 120                  | 63%          | 91%                     | 93%                     |
| Clear MOS gap (≥ 0.6)    | 73%          | 98%                     | 98%                     |
| Close MOS gap (0.25–0.6) | 52% (chance) | 83%                     | 88%                     |

- Claude was self-consistent across the two orders 90% of the time, with no
  left/right bias (50% A).
- It reliably catches cut-off subjects and strip-like crops, and nothing finer.
- Its pairwise labels would be noise for RankNet training, so it is not a
  useful teacher for crop ranking.
- The heads chose their checkpoint on GAICD val, so their numbers are slightly
  optimistic. The gap is far larger than that bias.

## Takeaways

- Backbone choice barely matters at this data size. About 4,000 training
  images (2,636 GAICD + 1,369 FCDB) cap what a frozen feature extractor can add, and bigger or newer
  backbones do not help.
- Adapting the backbone to the task (fine-tuning) is the only CNN lever that
  moved results, and it plateaus at GAICD SRCC ≈ 0.82 across recipes.
- Better saliency helps the no-CNN path more than any CNN change helps the CNN
  path on FCDB.

## Possible avenues

Ordered roughly by expected value per effort.

1. **Combine the two wins.** Fine-tuned CNN + IS-Net/UNISAL saliency, with the
   head retrained on both. Untested: the shipped CNN head gained nothing from
   the new map, but a head retrained on it might.
2. **Fine-tune jointly with saliency features.** Run 2 was trained with U²-Netp
   hand features as the fixed side input.
3. **Better use of the data.**
   - Listwise or MOS-regression loss instead of RankNet pairs.
   - Hard-pair mining (close-MOS pairs).
   - Using all GAICD candidates per step.
4. **More data.** Additional crop datasets (e.g. CPC, SACD, GAIC extensions)
   or pseudo-labels from a large teacher. This is likely the biggest lever,
   given the backbone results.
5. **Distillation.** Use a large frozen model (ConvNeXt/DINOv2/SigLIP) or a
   fine-tuned large model as a teacher for crop scores, and train MobileNetV2
   to match it. Inference cost stays the same. A general VLM (Claude) is too
   weak for this: see the pilot above.
6. **Unfreeze BatchNorm** with multi-image batches. Every run used batch-1 with
   frozen BN.
7. **Higher-resolution taps.** Add the stride-4 MobileNetV2 tap, or pool from
   the stride-8 map directly for small crops.
8. **Distil IS-Net + UNISAL into a small saliency model**, as was done for the
   JS saliency from U²-Netp. That would cut its 6× cost.
