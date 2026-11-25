// Training data for the pure-JS saliency model: per-cell features
// (src/saliency.ts) and the cached U2-Netp map pooled to the same grid.
// Usage: bun bench/jssal/build.ts <gaicd|fcdb> <split> [--procs 8]
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { GRID, SAL_DIM, saliencyFeatures } from '../../src/saliency';
import { imageKey, loadSaliency } from '../plugin-cache';

const [dataset, split] = process.argv.slice(2);
const opt = (n: string, d: string) => {
  const i = process.argv.indexOf(`--${n}`);
  return i > -1 ? process.argv[i + 1] : d;
};
const procs = Number(opt('procs', '8'));
const shard = opt('shard', '');
const root = join(import.meta.dir, '../..');
const DIR = join(root, 'bench/datasets', dataset);
const OUT = join(root, 'bench/out/jssal');
const list: string[] = (
  await Bun.file(
    join(DIR, dataset === 'gaicd' ? 'candidates.json' : 'annotations.json')
  ).json()
)
  .filter((e: any) => e.split === split)
  .map((e: any) => join(DIR, e.image));
const name = `${dataset}-${split}`;

const pool = (m: {
  width: number;
  height: number;
  data: ArrayLike<number>;
}) => {
  const out = new Float32Array(GRID * GRID);
  for (let gy = 0; gy < GRID; gy++)
    for (let gx = 0; gx < GRID; gx++) {
      const y0 = Math.floor((gy * m.height) / GRID);
      const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * m.height) / GRID));
      const x0 = Math.floor((gx * m.width) / GRID);
      const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * m.width) / GRID));
      let s = 0;
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++) s += m.data[y * m.width + x];
      out[gy * GRID + gx] = s / ((y1 - y0) * (x1 - x0));
    }
  return out;
};

if (shard) {
  const [k, n] = shard.split('/').map(Number);
  const xs: Float32Array[] = [];
  const ys: Float32Array[] = [];
  for (let i = k; i < list.length; i += n) {
    const img = await loadImage(list[i]);
    const c = createCanvas(img.width, img.height);
    c.getContext('2d').drawImage(img, 0, 0);
    const data = c.getContext('2d').getImageData(0, 0, img.width, img.height);
    xs.push(saliencyFeatures(data));
    const sal = await loadSaliency(
      dataset,
      imageKey(list[i]),
      async () => data as any
    );
    ys.push(pool(sal));
  }
  const cat = (a: Float32Array[]) => {
    const o = new Float32Array(a.reduce((s, x) => s + x.length, 0));
    let p = 0;
    for (const x of a) {
      o.set(x, p);
      p += x.length;
    }
    return o;
  };
  await Bun.write(join(OUT, `${name}.s${k}.x.bin`), cat(xs));
  await Bun.write(join(OUT, `${name}.s${k}.y.bin`), cat(ys));
  process.exit(0);
}
await mkdir(OUT, { recursive: true });
const t0 = performance.now();
await Promise.all(
  Array.from({ length: procs }, async (_, k) => {
    const p = Bun.spawn(
      ['bun', import.meta.path, dataset, split, '--shard', `${k}/${procs}`],
      {
        stderr: 'inherit',
      }
    );
    if ((await p.exited) !== 0) throw new Error(`shard ${k} failed`);
  })
);
// Merge in input order (image i is in shard i % procs).
const per = GRID * GRID;
const X = new Float32Array(list.length * per * SAL_DIM);
const Y = new Float32Array(list.length * per);
const shards = await Promise.all(
  Array.from({ length: procs }, async (_, k) => ({
    x: new Float32Array(
      await Bun.file(join(OUT, `${name}.s${k}.x.bin`)).arrayBuffer()
    ),
    y: new Float32Array(
      await Bun.file(join(OUT, `${name}.s${k}.y.bin`)).arrayBuffer()
    ),
  }))
);
for (let i = 0; i < list.length; i++) {
  const s = shards[i % procs];
  const j = Math.floor(i / procs);
  X.set(
    s.x.subarray(j * per * SAL_DIM, (j + 1) * per * SAL_DIM),
    i * per * SAL_DIM
  );
  Y.set(s.y.subarray(j * per, (j + 1) * per), i * per);
}
for (let k = 0; k < procs; k++) {
  await rm(join(OUT, `${name}.s${k}.x.bin`));
  await rm(join(OUT, `${name}.s${k}.y.bin`));
}
await Bun.write(join(OUT, `${name}.x.bin`), X);
await Bun.write(join(OUT, `${name}.y.bin`), Y);
console.log(
  `${name}: ${list.length} images, ${((performance.now() - t0) / 1000).toFixed(1)}s`
);
