// Fit the fixed PCA projection (448 -> C channels) of the cnn plug-in on
// GAICD-train backbone feature maps, and write bench/out/cnn/projection.bin
// (Float32 LE: uint32 C, mean[448], proj[448 x C]; values pre-rounded to
// float16 so the shipped f16 weights are exactly what the features used).
//
// Usage: bun bench/learned/cnn-pca.ts [--images 300] [--channels 32]
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { decodeF16, encodeF16 } from '../../src/learned';
import { cnnRawMap, RAW_CHANNELS } from '../../plugins/onnx/cnn';
import { PROJECTION_FILE } from '../plugin-cache';

const args = process.argv.slice(2);
const opt = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};
const nImages = Number(opt('images', '300'));
const C = Number(opt('channels', '32'));
const dir = join(import.meta.dir, '../datasets/gaicd');
const all = (await Bun.file(join(dir, 'candidates.json')).json()) as {
  image: string;
  split: string;
}[];
const train = all.filter((a) => a.split === 'train');
const step = train.length / nImages;
const pick = Array.from(
  { length: nImages },
  (_, i) => train[Math.floor(i * step)]
);

const R = RAW_CHANNELS;
const rows: Float32Array[] = [];
let n = 0;
const t0 = performance.now();
for (const a of pick) {
  const img = await loadImage(join(dir, a.image));
  const c = createCanvas(img.width, img.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const m = await cnnRawMap(ctx.getImageData(0, 0, img.width, img.height));
  rows.push(m.data);
  n += m.width * m.height;
}
console.log(
  `${nImages} maps, ${n} pixels, ${((performance.now() - t0) / 1000).toFixed(1)}s`
);

const mean = new Float64Array(R);
for (const d of rows) for (let i = 0; i < d.length; i++) mean[i % R] += d[i];
for (let k = 0; k < R; k++) mean[k] /= n;
const cov = new Float64Array(R * R);
const v = new Float64Array(R);
for (const d of rows) {
  for (let o = 0; o < d.length; o += R) {
    for (let k = 0; k < R; k++) v[k] = d[o + k] - mean[k];
    for (let a = 0; a < R; a++) {
      const va = v[a];
      for (let b = a; b < R; b++) cov[a * R + b] += va * v[b];
    }
  }
}
for (let a = 0; a < R; a++)
  for (let b = a; b < R; b++) {
    cov[a * R + b] /= n;
    cov[b * R + a] = cov[a * R + b];
  }

// Cyclic Jacobi eigen-decomposition.
const V = new Float64Array(R * R);
for (let i = 0; i < R; i++) V[i * R + i] = 1;
for (let sweep = 0; sweep < 30; sweep++) {
  let off = 0;
  for (let p = 0; p < R; p++)
    for (let q = p + 1; q < R; q++) off += cov[p * R + q] ** 2;
  if (off < 1e-18) break;
  for (let p = 0; p < R; p++) {
    for (let q = p + 1; q < R; q++) {
      const apq = cov[p * R + q];
      if (Math.abs(apq) < 1e-15) continue;
      const th = (cov[q * R + q] - cov[p * R + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const cs = 1 / Math.sqrt(t * t + 1);
      const sn = t * cs;
      for (let k = 0; k < R; k++) {
        const akp = cov[k * R + p];
        const akq = cov[k * R + q];
        cov[k * R + p] = cs * akp - sn * akq;
        cov[k * R + q] = sn * akp + cs * akq;
      }
      for (let k = 0; k < R; k++) {
        const apk = cov[p * R + k];
        const aqk = cov[q * R + k];
        cov[p * R + k] = cs * apk - sn * aqk;
        cov[q * R + k] = sn * apk + cs * aqk;
      }
      for (let k = 0; k < R; k++) {
        const vkp = V[k * R + p];
        const vkq = V[k * R + q];
        V[k * R + p] = cs * vkp - sn * vkq;
        V[k * R + q] = sn * vkp + cs * vkq;
      }
    }
  }
}
const order = Array.from({ length: R }, (_, i) => i).sort(
  (a, b) => cov[b * R + b] - cov[a * R + a]
);
const total = order.reduce((s, i) => s + cov[i * R + i], 0);
let acc = 0;
const proj = new Float32Array(R * C);
for (let c = 0; c < C; c++) {
  const e = order[c];
  acc += cov[e * R + e];
  for (let k = 0; k < R; k++) proj[k * C + c] = V[k * R + e];
}
console.log(
  `top ${C} components explain ${((100 * acc) / total).toFixed(1)}% of variance`
);

// Round to float16 (what ships).
const all16 = decodeF16(encodeF16([...mean, ...proj]));
const out = new Uint8Array(4 + all16.byteLength);
new Uint32Array(out.buffer, 0, 1)[0] = C;
out.set(new Uint8Array(all16.buffer), 4);
await mkdir(dirname(PROJECTION_FILE), { recursive: true });
await Bun.write(PROJECTION_FILE, out);
console.log(`wrote ${PROJECTION_FILE}`);
