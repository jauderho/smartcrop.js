// Compute learned-scorer features for every candidate crop of a dataset split
// and cache them under bench/out/learned/<name>.{json,bin}.
//
//   gaicd  candidates from bench/datasets/gaicd/candidates.json, target = MOS (1-5)
//   fcdb   windows at each ground-truth box's aspect, target = 1 + 4 * IoU with
//          the closest ground-truth box (same 1-5 range, used as extra signal)
//
// Usage: bun bench/learned/dump.ts [--dataset gaicd|fcdb] [--split train|val|test]
//          [--with faces,saliency,cnn] [--cnnk 3] [--procs 8] [--limit N]
// With cnn, each row is the hand features followed by the pooled deep features
// (RoI KxK, RoD KxK, global; src/cnn.ts) and the file is named ...+cnn (K != 3:
// ...+cnn<K>).
//
// The parent skips work when <name>.key matches the content-addressed key of
// everything that determines the features (code, model pins, inputs, args);
// --force recomputes.
//
// The parent spawns --procs worker processes (each takes every procs-th image)
// and concatenates their output. .bin is Float32 (rows of FEATURE_COUNT);
// .json lists {image, n, off (row offset), mos[]} per image.
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import smartcrop from '../../src/smartcrop';
import { FEATURE_COUNT, FEATURE_NAMES } from '../../src/learned';
import { cnnDim } from '../../src/cnn';
import {
  imageKey,
  parseWith,
  pluginOptions,
  PROJECTION_FILE,
} from '../plugin-cache';
import { featureKey } from '../../scripts/weights-fingerprint';

const args = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};
const dataset = opt('dataset', 'gaicd')!;
const split = opt('split', 'train')!;
const withSet = parseWith(opt('with'));
const limit = Number(opt('limit', '0'));
const procs = Number(opt('procs', '8'));
const shard = opt('shard');
const OUT = resolve(
  opt('outdir') ?? resolve(import.meta.dir, '../out/learned')
);
const cnnK = withSet.has('cnn') ? Number(opt('cnnk', '3')) : 0;
const CNN_CHANNELS = 32;
if (cnnK) process.env.SMARTCROP_CNN_DUMP = '1'; // pool features, no head model
const DIM = FEATURE_COUNT + (cnnK ? cnnDim(cnnK, CNN_CHANNELS) : 0);
const label = [...withSet]
  .map((p) => (p === 'cnn' && cnnK !== 3 ? `cnn${cnnK}` : p))
  .join('+');
const name = `${dataset}-${split}${withSet.size ? '-' + label : ''}`;
const DIR = resolve(import.meta.dir, '../datasets', dataset);

type Cand = [number, number, number, number, number]; // x1 y1 x2 y2 target (internal)
type Entry = { image: string; width: number; height: number; cands: Cand[] };

const loadEntries = async (): Promise<Entry[]> => {
  let entries: Entry[];
  if (dataset === 'gaicd') {
    const all = JSON.parse(
      await readFile(join(DIR, 'candidates.json'), 'utf8')
    );
    entries = all
      .filter((e: any) => e.split === split)
      .map((e: any) => ({
        image: e.image,
        width: e.width,
        height: e.height,
        // stored as [x, y, w, h, mos]
        cands: e.candidates.map(([x, y, w, h, m]: number[]) => [
          x,
          y,
          x + w,
          y + h,
          m,
        ]),
      }));
  } else {
    const all = JSON.parse(
      await readFile(join(DIR, 'annotations.json'), 'utf8')
    );
    entries = all
      .filter((a: any) => a.split === split)
      .map((a: any) => ({
        image: a.image,
        width: a.width,
        height: a.height,
        cands: fcdbCandidates(a),
      }));
  }
  return limit ? entries.slice(0, limit) : entries;
};

const iou = (a: number[], b: number[]) => {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const i = iw * ih;
  return (
    i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
  );
};

// Windows at the aspect of each ground-truth box, scales 0.5-1, <= 7 positions per axis.
function fcdbCandidates(a: {
  width: number;
  height: number;
  boxes: [number, number, number, number][];
}): Cand[] {
  const gts = a.boxes.map(([x, y, w, h]) => [x, y, x + w, y + h]);
  const out: Cand[] = [];
  const seen = new Set<string>();
  const aspects = new Set(a.boxes.map(([, , w, h]) => `${w}x${h}`));
  const add = (x1: number, y1: number, x2: number, y2: number) => {
    const c = [x1, y1, x2, y2].map(Math.round);
    const key = c.join(',');
    if (seen.has(key) || c[2] - c[0] < 8 || c[3] - c[1] < 8) return;
    seen.add(key);
    out.push([...c, 1 + 4 * Math.max(...gts.map((g) => iou(c, g)))] as Cand);
  };
  for (const asp of aspects) {
    const [bw, bh] = asp.split('x').map(Number);
    const s0 = Math.min(a.width / bw, a.height / bh);
    for (let s = 0.5; s <= 1.0001; s += 0.1) {
      const cw = bw * s0 * s;
      const ch = bh * s0 * s;
      const nx = Math.max(1, Math.min(7, Math.round((a.width - cw) / 24) + 1));
      const ny = Math.max(1, Math.min(7, Math.round((a.height - ch) / 24) + 1));
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const x =
            nx > 1 ? ((a.width - cw) * i) / (nx - 1) : (a.width - cw) / 2;
          const y =
            ny > 1 ? ((a.height - ch) * j) / (ny - 1) : (a.height - ch) / 2;
          add(x, y, x + cw, y + ch);
        }
      }
    }
  }
  for (const g of gts) add(...(g as [number, number, number, number]));
  return out;
}

const canvasFactory = (w: number, h: number) => createCanvas(w, h);

async function work(entries: Entry[], k: number, n: number, file: string) {
  const rows: Float32Array[] = [];
  const meta: { image: string; n: number; off: number; mos: number[] }[] = [];
  let off = 0;
  for (let i = k; i < entries.length; i += n) {
    const e = entries[i];
    const path = join(
      DIR,
      e.image.startsWith('images/') ? e.image : `images/${split}/${e.image}`
    );
    const img = await loadImage(path);
    const po = withSet.size
      ? await pluginOptions(
          dataset,
          imageKey(e.image),
          withSet,
          e,
          async () => {
            const c = createCanvas(img.width, img.height);
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            return ctx.getImageData(0, 0, img.width, img.height);
          }
        )
      : {};
    const f = await smartcrop._features(
      img,
      e.cands.map(([x1, y1, x2, y2]) => ({
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
      })),
      {
        width: img.width,
        height: img.height,
        canvasFactory,
        // Built-in JS saliency only when asked for (--with jssal), so the
        // other dumps keep their meaning.
        jsSaliency: withSet.has('jssal'),
        ...po,
        ...(cnnK ? { cnnK } : {}),
      }
    );
    rows.push(f);
    meta.push({
      image: e.image,
      n: e.cands.length,
      off,
      mos: e.cands.map((c) => c[4]),
    });
    off += e.cands.length;
  }
  const buf = new Float32Array(off * DIM);
  let p = 0;
  for (const r of rows) {
    buf.set(r, p);
    p += r.length;
  }
  await Bun.write(file + '.bin', buf);
  await Bun.write(file + '.json', JSON.stringify({ D: DIM, images: meta }));
}

await mkdir(OUT, { recursive: true });
const entries = await loadEntries();
if (shard) {
  const [k, n] = shard.split('/').map(Number);
  await work(entries, k, n, join(OUT, `${name}.shard${k}`));
  process.exit(0);
}
const fileSha = async (p: string) =>
  existsSync(p)
    ? new Bun.CryptoHasher('sha256')
        .update(await Bun.file(p).bytes())
        .digest('hex')
    : 'none';
const key = await featureKey({
  dataset,
  split,
  with: [...withSet].sort().join(','),
  cnnK,
  limit,
  inputs: await fileSha(
    join(DIR, dataset === 'gaicd' ? 'candidates.json' : 'annotations.json')
  ),
  projection: cnnK ? await fileSha(PROJECTION_FILE) : 'none',
});
if (
  !args.includes('--force') &&
  existsSync(join(OUT, `${name}.bin`)) &&
  existsSync(join(OUT, `${name}.json`)) &&
  existsSync(join(OUT, `${name}.key`)) &&
  (await readFile(join(OUT, `${name}.key`), 'utf8')).trim() === key
) {
  console.log(
    `${name}: up to date (key ${key.slice(0, 12)}), skipping; --force to recompute`
  );
  process.exit(0);
}
const t0 = performance.now();
const childArgs = args.filter(
  (a, i) => a !== '--procs' && args[i - 1] !== '--procs'
);
await Promise.all(
  Array.from({ length: procs }, async (_, k) => {
    const p = Bun.spawn(
      ['bun', import.meta.path, ...childArgs, '--shard', `${k}/${procs}`],
      {
        stdout: 'inherit',
        stderr: 'inherit',
      }
    );
    if ((await p.exited) !== 0) throw new Error(`shard ${k} failed`);
  })
);
// Merge back into input order (shard k holds images k, k + procs, ...), so
// the dump is identical whatever --procs is.
const shards: { images: any[]; bin: Float32Array }[] = [];
for (let k = 0; k < procs; k++) {
  const base = join(OUT, `${name}.shard${k}`);
  const m = JSON.parse(await readFile(base + '.json', 'utf8'));
  shards.push({
    images: m.images,
    bin: new Float32Array(await Bun.file(base + '.bin').arrayBuffer()),
  });
  await rm(base + '.json');
  await rm(base + '.bin');
}
const total = shards.reduce((s, x) => s + x.bin.length, 0);
const all = new Float32Array(total);
const metas: any[] = [];
let off = 0;
const count = shards.reduce((s, x) => s + x.images.length, 0);
for (let i = 0; i < count; i++) {
  const sh = shards[i % procs];
  const im = sh.images[Math.floor(i / procs)];
  all.set(sh.bin.subarray(im.off * DIM, (im.off + im.n) * DIM), off * DIM);
  metas.push({ ...im, off });
  off += im.n;
}
await Bun.write(join(OUT, `${name}.bin`), all);
await Bun.write(
  join(OUT, `${name}.json`),
  JSON.stringify({ D: DIM, names: FEATURE_NAMES, images: metas })
);
await Bun.write(join(OUT, `${name}.key`), key + '\n');
console.log(
  `${name}: ${metas.length} images, ${off} candidates, ${((performance.now() - t0) / 1000).toFixed(1)}s`
);
