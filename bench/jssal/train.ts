// Train the pure-JS saliency MLP (src/saliency.ts) to reproduce U2-Netp maps.
// Train: gaicd-train + fcdb-train cells. Validate: gaicd-val (per-image
// Pearson r with the U2-Netp map). Writes bench/out/jssal/model.json.
//
// Usage: bun bench/jssal/train.ts [--hidden 32,16] [--epochs 8] [--lr 2e-3] [--seed 1]
import { join } from 'node:path';
import {
  CONTEXT_DIM,
  GRID,
  SAL_DIM,
  saliencyProbs,
  withContext,
  type SaliencyModel,
} from '../../src/saliency';

const opt = (n: string, d: string) => {
  const i = process.argv.indexOf(`--${n}`);
  return i > -1 ? process.argv[i + 1] : d;
};
const hidden = opt('hidden', '32,16').split(',').map(Number);
const epochs = Number(opt('epochs', '8'));
const lr0 = Number(opt('lr', '2e-3'));
let seed = Number(opt('seed', '1'));
const out = opt('out', join(import.meta.dir, '../out/jssal/model.json'));
const OUT = join(import.meta.dir, '../out/jssal');
const rnd = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
// --context m.json: train the next auto-context stage on top of model m.
const ctxPath = opt('context', '');
const ctx: SaliencyModel | null = ctxPath
  ? await Bun.file(ctxPath).json()
  : null;
const D = SAL_DIM + (ctx ? CONTEXT_DIM : 0);
const load = async (n: string) => {
  let x = new Float32Array(
    await Bun.file(join(OUT, `${n}.x.bin`)).arrayBuffer()
  );
  const y = new Float32Array(
    await Bun.file(join(OUT, `${n}.y.bin`)).arrayBuffer()
  );
  if (ctx) {
    const per = GRID * GRID;
    const imgs = y.length / per;
    const ext = new Float32Array(imgs * per * D);
    for (let i = 0; i < imgs; i++) {
      const base = x.subarray(i * per * SAL_DIM, (i + 1) * per * SAL_DIM);
      ext.set(withContext(base, saliencyProbs(base, ctx)), i * per * D);
    }
    x = ext;
  }
  return { x, y };
};
const sets = await Promise.all(['gaicd-train', 'fcdb-train'].map(load));
const val = await load('gaicd-val');
const X = new Float32Array(sets.reduce((s, d) => s + d.x.length, 0));
const Y = new Float32Array(sets.reduce((s, d) => s + d.y.length, 0));
{
  let p = 0;
  let q = 0;
  for (const d of sets) {
    X.set(d.x, p);
    Y.set(d.y, q);
    p += d.x.length;
    q += d.y.length;
  }
}
const N = Y.length;
// Standardise.
const mean = new Array(D).fill(0);
const std = new Array(D).fill(0);
for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) mean[j] += X[i * D + j];
for (let j = 0; j < D; j++) mean[j] /= N;
for (let i = 0; i < N; i++)
  for (let j = 0; j < D; j++) std[j] += (X[i * D + j] - mean[j]) ** 2;
for (let j = 0; j < D; j++) std[j] = Math.sqrt(std[j] / N) || 1;

const dims = [D, ...hidden, 1];
const L = dims.length - 1;
const W = dims.slice(0, -1).map((din, l) => {
  const dout = dims[l + 1];
  const w = new Float32Array(din * dout);
  const s = Math.sqrt(2 / din);
  for (let i = 0; i < w.length; i++) w[i] = (rnd() * 2 - 1) * s * Math.sqrt(3);
  return w;
});
const Bs = dims.slice(1).map((d) => new Float32Array(d));
const mW = W.map((w) => new Float32Array(w.length));
const vW = W.map((w) => new Float32Array(w.length));
const mB = Bs.map((b) => new Float32Array(b.length));
const vB = Bs.map((b) => new Float32Array(b.length));
const gW = W.map((w) => new Float32Array(w.length));
const gB = Bs.map((b) => new Float32Array(b.length));
const acts = dims.map((d) => new Float32Array(d));
const deltas = dims.map((d) => new Float32Array(d));

const forward = (x: Float32Array, o: number, stdz: boolean) => {
  const a0 = acts[0];
  for (let j = 0; j < D; j++)
    a0[j] = stdz ? (x[o + j] - mean[j]) / std[j] : x[o + j];
  for (let l = 0; l < L; l++) {
    const din = dims[l];
    const dout = dims[l + 1];
    const w = W[l];
    const a = acts[l];
    const z = acts[l + 1];
    for (let k = 0; k < dout; k++) {
      let s = Bs[l][k];
      const r = k * din;
      for (let i = 0; i < din; i++) s += w[r + i] * a[i];
      z[k] = l < L - 1 ? (s > 0 ? s : 0) : s;
    }
  }
  return acts[L][0];
};

const evalVal = () => {
  const per = GRID * GRID;
  const imgs = val.y.length / per;
  let rSum = 0;
  let bce = 0;
  for (let im = 0; im < imgs; im++) {
    const p = new Float32Array(per);
    for (let k = 0; k < per; k++) {
      const z = forward(val.x, (im * per + k) * D, true);
      p[k] = 1 / (1 + Math.exp(-z));
      const t = val.y[im * per + k];
      bce -= t * Math.log(p[k] + 1e-7) + (1 - t) * Math.log(1 - p[k] + 1e-7);
    }
    const t = val.y.subarray(im * per, (im + 1) * per);
    const mp = p.reduce((a, b) => a + b, 0) / per;
    const mt = t.reduce((a, b) => a + b, 0) / per;
    let c = 0;
    let vp = 0;
    let vt = 0;
    for (let k = 0; k < per; k++) {
      c += (p[k] - mp) * (t[k] - mt);
      vp += (p[k] - mp) ** 2;
      vt += (t[k] - mt) ** 2;
    }
    rSum += vt > 0 && vp > 0 ? c / Math.sqrt(vp * vt) : 0;
  }
  return { r: rSum / imgs, bce: bce / val.y.length };
};

const order = Uint32Array.from({ length: N }, (_, i) => i);
const BATCH = 256;
let step = 0;
const b1 = 0.9;
const b2 = 0.999;
let best = { r: -1, model: null as SaliencyModel | null, epoch: -1 };
for (let ep = 0; ep < epochs; ep++) {
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  const lr = lr0 * 0.5 * (1 + Math.cos((Math.PI * ep) / epochs));
  let loss = 0;
  for (let b0 = 0; b0 < N; b0 += BATCH) {
    for (const g of gW) g.fill(0);
    for (const g of gB) g.fill(0);
    const bn = Math.min(BATCH, N - b0);
    for (let bi = 0; bi < bn; bi++) {
      const idx = order[b0 + bi];
      const z = forward(X, idx * D, true);
      const p = 1 / (1 + Math.exp(-z));
      const t = Y[idx];
      loss -= t * Math.log(p + 1e-7) + (1 - t) * Math.log(1 - p + 1e-7);
      deltas[L][0] = p - t;
      for (let l = L - 1; l >= 0; l--) {
        const din = dims[l];
        const dout = dims[l + 1];
        const w = W[l];
        const a = acts[l];
        const dz = deltas[l + 1];
        const da = deltas[l];
        da.fill(0);
        for (let k = 0; k < dout; k++) {
          const d = dz[k];
          if (d === 0) continue;
          gB[l][k] += d;
          const r = k * din;
          for (let i = 0; i < din; i++) {
            gW[l][r + i] += d * a[i];
            da[i] += d * w[r + i];
          }
        }
        if (l > 0) for (let i = 0; i < din; i++) if (a[i] <= 0) da[i] = 0;
      }
    }
    step++;
    const c1 = 1 - b1 ** step;
    const c2 = 1 - b2 ** step;
    const upd = (
      p: Float32Array,
      g: Float32Array,
      m: Float32Array,
      v: Float32Array
    ) => {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] / bn;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
      }
    };
    for (let l = 0; l < L; l++) {
      upd(W[l], gW[l], mW[l], vW[l]);
      upd(Bs[l], gB[l], mB[l], vB[l]);
    }
  }
  const v = evalVal();
  console.log(
    `epoch ${ep} train bce ${(loss / N).toFixed(4)} val bce ${v.bce.toFixed(4)} val r ${v.r.toFixed(4)}`
  );
  if (v.r > best.r)
    best = {
      r: v.r,
      epoch: ep,
      model: {
        ...(ctx ? { context: ctx } : {}),
        mean: [...mean],
        std: [...std],
        layers: W.map((w, l) => ({
          w: Array.from(w),
          b: Array.from(Bs[l]),
          inDim: dims[l],
          outDim: dims[l + 1],
        })),
      },
    };
}
await Bun.write(out, JSON.stringify(best.model));
console.log(`best epoch ${best.epoch} val r ${best.r.toFixed(4)} -> ${out}`);
