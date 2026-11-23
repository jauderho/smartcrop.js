// Evaluate smartcrop against ground-truth crop boxes (FCDB / FLMS).
//
// Aspect-ratio-aware protocol (as in the Cropper paper): for every
// ground-truth box, smartcrop is asked for a crop with that box's aspect
// ratio. smartcrop's width/height only define the aspect; the largest crop of
// that aspect that fits the image is "scale 1", and minScale/maxScale let it
// search smaller crops. Metrics are IoU and BDE (mean absolute difference of
// the four box edges, x edges normalized by image width, y edges by height).
// With several references per image (FLMS) the best reference counts (max
// IoU / min BDE per image); the mean over all boxes is reported too.
// Baselines: largest centered crop of the same aspect ("center"), and the
// same centered crop at a fixed scale ("centerScaled").
//
// Datasets come from `bun scripts/fetch-datasets.ts`.
//
// Usage: bun bench/box-eval.ts [--dataset fcdb|flms] [--split test|train|all]
//          [--limit N] [--library path] [--options '{json}'] [--json]
//          [--center-scale 0.8] [--concurrency 4]
// --options is merged over { minScale: 0.5, maxScale: 1 }; e.g.
// --options '{"minScale":1}' is the library default (largest crop only).
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { join, resolve } from 'node:path';
import { imageKey, parseWith, pluginOptions } from './plugin-cache';
import { indexBySubset, loadTags, SUBSETS } from './subsets';

type Box = [number, number, number, number];
type Ann = {
  image: string;
  width: number;
  height: number;
  boxes: Box[];
  split: string;
};

const args = process.argv.slice(2);
const take = (name: string, fallback?: string) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 ? args.splice(i, 2)[1] : fallback;
};
const asJson = args.includes('--json');
if (asJson) args.splice(args.indexOf('--json'), 1);
const dataset = take('dataset', 'fcdb')!;
const split = take('split', 'test')!;
const limit = Number(take('limit', '0'));
// --holdout keeps only the last 15% of the split (the learned scorer's FCDB
// holdout; safe for selecting options without touching test sets).
const holdout = args.includes('--holdout');
if (holdout) args.splice(args.indexOf('--holdout'), 1);
const libPath = take('library');
const extraOptions = JSON.parse(take('options', '{}')!);
// --with faces,saliency feeds the ONNX plug-ins (plugins/onnx) into the crop
// options (boost / importanceMap); weights still come from --options.
const withPlugins = parseWith(take('with'));
const faceMinScore = Number(take('face-min-score', '0.6'));
const centerScale = Number(take('center-scale', '0.8'));
const conc = Math.max(1, Number(take('concurrency', '4')));

const dir = join(import.meta.dir, 'datasets', dataset);
const annFile = Bun.file(join(dir, 'annotations.json'));
if (!(await annFile.exists())) {
  console.error(
    `missing ${dir}/annotations.json; run: bun scripts/fetch-datasets.ts ${dataset}`
  );
  process.exit(2);
}
let anns = ((await annFile.json()) as Ann[]).filter(
  (a) => split === 'all' || a.split === split
);
if (holdout) anns = anns.slice(Math.floor(anns.length * 0.85));
if (limit > 0) anns = anns.slice(0, limit);

const lib = await import(
  resolve(libPath ?? join(import.meta.dir, '../src/smartcrop.ts'))
);
const smartcrop = lib.default ?? lib;
const canvasFactory = (w: number, h: number) => createCanvas(w, h);
const options = { minScale: 0.5, maxScale: 1, ...extraOptions };

const iou = (a: Box, b: Box) => {
  const iw = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
};
const bde = (a: Box, b: Box, w: number, h: number) =>
  (Math.abs(a[0] - b[0]) / w +
    Math.abs(a[0] + a[2] - b[0] - b[2]) / w +
    Math.abs(a[1] - b[1]) / h +
    Math.abs(a[1] + a[3] - b[1] - b[3]) / h) /
  4;

// Largest centered box with the aspect of `ref`, shrunk by `scale`.
const centered = (ref: Box, w: number, h: number, scale: number): Box => {
  const s = Math.min(w / ref[2], h / ref[3]) * scale;
  const cw = ref[2] * s;
  const ch = ref[3] * s;
  return [(w - cw) / 2, (h - ch) / 2, cw, ch];
};

type Method = 'smartcrop' | 'center' | 'centerScaled';
const methods: Method[] = ['smartcrop', 'center', 'centerScaled'];
type Per = {
  image: string;
  iou: number;
  bde: number;
  boxIou: number;
  boxBde: number;
};
const results: Record<Method, Per[]> = {
  smartcrop: [],
  center: [],
  centerScaled: [],
};
let failures = 0;

async function evalImage(a: Ann) {
  const img = await loadImage(join(dir, a.image));
  if (img.width !== a.width || img.height !== a.height) {
    throw new Error(
      `size mismatch ${a.image}: ${img.width}x${img.height} vs ${a.width}x${a.height}`
    );
  }
  const preds: Record<Method, Box[]> = {
    smartcrop: [],
    center: [],
    centerScaled: [],
  };
  const cache = new Map<string, Box>();
  const pluginOpts = withPlugins.size
    ? await pluginOptions(
        dataset,
        imageKey(a.image),
        withPlugins,
        a,
        async () => {
          const c = createCanvas(img.width, img.height);
          const ctx = c.getContext('2d');
          ctx.drawImage(img, 0, 0);
          return ctx.getImageData(0, 0, img.width, img.height);
        },
        faceMinScore
      )
    : {};
  for (const gt of a.boxes) {
    const key = `${gt[2]}x${gt[3]}`;
    let sc = cache.get(key);
    if (!sc) {
      const { topCrop } = await smartcrop.crop(img, {
        width: gt[2],
        height: gt[3],
        canvasFactory,
        ...options,
        ...pluginOpts,
      });
      sc = [topCrop.x, topCrop.y, topCrop.width, topCrop.height] as Box;
      cache.set(key, sc);
    }
    preds.smartcrop.push(sc);
    preds.center.push(centered(gt, a.width, a.height, 1));
    preds.centerScaled.push(centered(gt, a.width, a.height, centerScale));
  }
  for (const m of methods) {
    const ious = a.boxes.map((gt, i) => iou(preds[m][i], gt));
    const bdes = a.boxes.map((gt, i) =>
      bde(preds[m][i], gt, a.width, a.height)
    );
    results[m].push({
      image: a.image,
      iou: Math.max(...ious),
      bde: Math.min(...bdes),
      boxIou: ious.reduce((s, v) => s + v, 0) / ious.length,
      boxBde: bdes.reduce((s, v) => s + v, 0) / bdes.length,
    });
  }
}

const t0 = performance.now();
let next = 0;
await Promise.all(
  Array.from({ length: conc }, async () => {
    while (next < anns.length) {
      const a = anns[next++];
      try {
        await evalImage(a);
      } catch (e) {
        failures++;
        console.error(`skip ${a.image}: ${(e as Error).message}`);
      }
    }
  })
);
const secs = (performance.now() - t0) / 1000;

const mean = (xs: number[]) =>
  xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length);
const summarize = (rs: Per[]) => ({
  iou: +mean(rs.map((r) => r.iou)).toFixed(4),
  bde: +mean(rs.map((r) => r.bde)).toFixed(4),
  boxIou: +mean(rs.map((r) => r.boxIou)).toFixed(4),
  boxBde: +mean(rs.map((r) => r.boxBde)).toFixed(4),
});
const n = results.smartcrop.length;
await loadTags(dataset);
const idx = indexBySubset(
  dataset,
  results.smartcrop.map((r) => r.image)
);
const pick = (rs: Per[], ix: number[]) => ix.map((i) => rs[i]);
const subsets = Object.fromEntries(
  SUBSETS.map((s) => {
    const sc = pick(results.smartcrop, idx[s]);
    const ce = pick(results.center, idx[s]);
    return [
      s,
      {
        n: sc.length,
        smartcrop: { iou: summarize(sc).iou, bde: summarize(sc).bde },
        center: { iou: summarize(ce).iou, bde: summarize(ce).bde },
      },
    ];
  })
);
const summary = {
  dataset,
  split,
  images: n,
  boxes: anns.reduce((s, a) => s + a.boxes.length, 0),
  failures,
  smartcrop: summarize(results.smartcrop),
  center: summarize(results.center),
  centerScaled: { scale: centerScale, ...summarize(results.centerScaled) },
  subsets,
  options,
  ...(withPlugins.size ? { with: [...withPlugins] } : {}),
};

if (!asJson) {
  console.table({
    smartcrop: summary.smartcrop,
    center: summary.center,
    [`center@${centerScale}`]: summary.centerScaled,
  });
  console.table(
    Object.fromEntries(
      Object.entries(subsets).map(([k, v]) => [
        k,
        {
          n: v.n,
          'smartcrop iou': v.smartcrop.iou,
          'smartcrop bde': v.smartcrop.bde,
          'center iou': v.center.iou,
          'center bde': v.center.bde,
        },
      ])
    )
  );
  console.log(
    `${n} images, ${summary.boxes} boxes, ${failures} failures, ${secs.toFixed(1)}s`
  );
}
console.log(JSON.stringify(summary));
