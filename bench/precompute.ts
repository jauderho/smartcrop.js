// Run the ONNX plug-ins (faces, saliency) once per image and cache results in
// bench/out/cache/<dataset>/. Resumable: existing cache entries are skipped.
//
// Usage: bun bench/precompute.ts <fcdb|flms|gaicd|examples|strips> [--split test|train|all]
//          [--limit N] [--labels path/to/strip-labels.json]
// `strips` caches for the 512-px 3:2 center-cropped canvas used by strip-eval.
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { readdir } from 'node:fs/promises';
import { join, relative } from 'node:path';
import {
  hasCache,
  imageKey,
  loadFaces,
  loadSaliency,
  toImageData,
} from './plugin-cache';

const args = process.argv.slice(2);
const take = (name: string, fallback?: string) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 ? args.splice(i, 2)[1] : fallback;
};
const split = take('split', 'all')!;
const limit = Number(take('limit', '0'));
const labelsPath = take(
  'labels',
  join(import.meta.dir, 'private/jauderho-strip.json')
)!;
const dataset = args[0];

type Item = { key: string; path: string; strip?: boolean };
let items: Item[] = [];
if (dataset === 'gaicd') {
  const dir = join(import.meta.dir, 'datasets', dataset);
  const anns = (await Bun.file(join(dir, 'candidates.json')).json()) as {
    image: string;
    split: string;
  }[];
  items = anns
    .filter((a) => split === 'all' || a.split === split)
    .map((a) => ({ key: imageKey(a.image), path: join(dir, a.image) }));
} else if (dataset === 'fcdb' || dataset === 'flms') {
  const dir = join(import.meta.dir, 'datasets', dataset);
  const anns = (await Bun.file(join(dir, 'annotations.json')).json()) as {
    image: string;
    split: string;
  }[];
  items = anns
    .filter((a) => split === 'all' || a.split === split)
    .map((a) => ({ key: imageKey(a.image), path: join(dir, a.image) }));
} else if (dataset === 'examples') {
  const root = join(import.meta.dir, '../examples/images');
  const walk = async (d: string): Promise<string[]> =>
    (await readdir(d, { withFileTypes: true })).flatMap((e) =>
      e.isDirectory()
        ? []
        : /\.(jpe?g|png)$/i.test(e.name)
          ? [join(d, e.name)]
          : []
    );
  const dirs = [
    root,
    ...(await readdir(root, { withFileTypes: true }))
      .filter((e) => e.isDirectory())
      .map((e) => join(root, e.name)),
  ];
  for (const d of dirs)
    for (const p of await walk(d))
      items.push({
        key: relative(root, p)
          .replace(/[\\/]/g, '_')
          .replace(/\.[^.]+$/, ''),
        path: p,
      });
} else if (dataset === 'strips') {
  const labels = await Bun.file(labelsPath).json();
  const ids = [
    ...Object.keys(labels.truth),
    ...Object.keys(labels.softTruth ?? {}),
    ...(labels.unreviewed ?? []),
  ];
  items = ids.map((id) => ({
    key: id,
    path: join(labels.imageDir, `${id}.jpg`),
    strip: true,
  }));
} else {
  console.error('usage: bun bench/precompute.ts <fcdb|flms|gaicd|examples|strips>');
  process.exit(2);
}
if (limit > 0) items = items.slice(0, limit);

const WIDTH = 512;
let tf = 0;
let ts = 0;
let nf = 0;
let ns = 0;
const t0 = performance.now();
for (const [n, it] of items.entries()) {
  const needF = !(await hasCache(dataset, it.key, 'faces'));
  const needS = !(await hasCache(dataset, it.key, 'saliency'));
  if (needF || needS) {
    const img = await loadImage(it.path);
    let data: ReturnType<typeof toImageData> | undefined;
    const getData = async () => {
      if (data) return data;
      if (it.strip) {
        // Mirror strip-eval: center-crop to 3:2, scale to WIDTH.
        let sw = img.width;
        let sh = Math.round((img.width * 2) / 3);
        if (sh > img.height) {
          sh = img.height;
          sw = Math.round((img.height * 3) / 2);
        }
        const sx = Math.round((img.width - sw) / 2);
        const sy = Math.round((img.height - sh) / 2);
        const h = Math.round((WIDTH * sh) / sw);
        const c = createCanvas(WIDTH, h);
        const ctx = c.getContext('2d');
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, WIDTH, h);
        data = ctx.getImageData(0, 0, WIDTH, h);
      } else {
        data = toImageData(img, (ctx) => ctx.drawImage(img, 0, 0));
      }
      return data;
    };
    let t = performance.now();
    if (needF) {
      await loadFaces(dataset, it.key, getData);
      tf += performance.now() - t;
      nf++;
    }
    t = performance.now();
    if (needS) {
      await loadSaliency(dataset, it.key, getData);
      ts += performance.now() - t;
      ns++;
    }
  }
  if ((n + 1) % 25 === 0 || n + 1 === items.length) {
    console.log(
      `${dataset} ${n + 1}/${items.length} ${((performance.now() - t0) / 1000).toFixed(0)}s`
    );
  }
}
console.log(
  `done: faces ${nf} imgs ${nf ? (tf / nf).toFixed(0) : '-'} ms/img, saliency ${ns} imgs ${ns ? (ts / ns).toFixed(0) : '-'} ms/img`
);
