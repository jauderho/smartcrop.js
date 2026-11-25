// Pure-JS saliency: a tiny per-cell model distilled from U2-Netp (see
// bench/jssal). The image is resampled to GRID*CELL square pixels (aspect is
// ignored, as U2-Netp does), each CELL x CELL block becomes one cell, and an
// MLP maps hand features of the cell and its surroundings to a 0..1 saliency.

export const GRID = 32;
export const CELL = 4;
const SIZE = GRID * CELL;

export const SAL_FEATURES = [
  'L',
  'a',
  'b',
  'chroma',
  'ft',
  'ftNorm',
  'cs2',
  'cs5',
  'cs2Norm',
  'grad',
  'grad3',
  'gradNorm',
  'skin',
  'x',
  'y',
  'r2',
  'border',
  'ftBlur',
  'gradBlur',
  'lumaDev',
  'bdMin',
  'bdMean',
  'bdMinBlur',
  'bdMinNorm',
] as const;
export const SAL_DIM = SAL_FEATURES.length;

import { decodeF16 } from './learned';
import { SALIENCY_PACKED } from './saliency-weights';

export interface PackedSaliencyModel {
  context?: PackedSaliencyModel;
  mean: number[];
  std: number[];
  layers: { w: string; b: string; inDim: number; outDim: number }[];
}

export const unpackSaliency = (p: PackedSaliencyModel): SaliencyModel => ({
  ...(p.context ? { context: unpackSaliency(p.context) } : {}),
  mean: p.mean,
  std: p.std,
  layers: p.layers.map((l) => ({
    inDim: l.inDim,
    outDim: l.outDim,
    w: Array.from(decodeF16(l.w)),
    b: Array.from(decodeF16(l.b)),
  })),
});

export interface SaliencyModel {
  /** Previous stage (auto-context); its map feeds this stage's features. */
  context?: SaliencyModel;
  mean: number[];
  std: number[];
  layers: { w: number[]; b: number[]; inDim: number; outDim: number }[];
}

type RGBA = { width: number; height: number; data: ArrayLike<number> };

const lab = (r: number, g: number, b: number, out: Float32Array, o: number) => {
  const lin = (c: number) => {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const R = lin(r);
  const G = lin(g);
  const B = lin(b);
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const x = f((0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047);
  const y = f(0.2126 * R + 0.7152 * G + 0.0722 * B);
  const z = f((0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883);
  out[o] = 116 * y - 16;
  out[o + 1] = 500 * (x - y);
  out[o + 2] = 200 * (y - z);
};

/** Area-average resample of RGBA to SIZE x SIZE RGB (Float32, 0..255). */
const resample = (img: RGBA): Float32Array => {
  const out = new Float32Array(SIZE * SIZE * 3);
  const { width: W, height: H, data } = img;
  for (let y = 0; y < SIZE; y++) {
    const y0 = Math.floor((y * H) / SIZE);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * H) / SIZE));
    for (let x = 0; x < SIZE; x++) {
      const x0 = Math.floor((x * W) / SIZE);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * W) / SIZE));
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let yy = y0; yy < y1; yy++) {
        let i = (yy * W + x0) * 4;
        for (let xx = x0; xx < x1; xx++, i += 4) {
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          n++;
        }
      }
      const o = (y * SIZE + x) * 3;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
    }
  }
  return out;
};

/** Mean of a GRID x GRID plane over a (2r+1)^2 neighbourhood (clamped). */
const boxMean = (p: Float32Array, r: number): Float32Array => {
  const out = new Float32Array(p.length);
  for (let y = 0; y < GRID; y++)
    for (let x = 0; x < GRID; x++) {
      let s = 0;
      let n = 0;
      for (let yy = Math.max(0, y - r); yy <= Math.min(GRID - 1, y + r); yy++)
        for (
          let xx = Math.max(0, x - r);
          xx <= Math.min(GRID - 1, x + r);
          xx++
        ) {
          s += p[yy * GRID + xx];
          n++;
        }
      out[y * GRID + x] = s / n;
    }
  return out;
};

const maxOf = (p: Float32Array) => {
  let m = 0;
  for (const v of p) if (v > m) m = v;
  return m || 1;
};

/** Per-cell features, GRID*GRID rows of SAL_DIM values (row-major cells). */
export function saliencyFeatures(img: RGBA): Float32Array {
  const rgb = resample(img);
  const n = GRID * GRID;
  // Pixel Lab and luma gradient, then pooled per cell.
  const px = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE * SIZE; i++)
    lab(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], px, i * 3);
  const L = new Float32Array(n);
  const A = new Float32Array(n);
  const B = new Float32Array(n);
  const grad = new Float32Array(n);
  const skin = new Float32Array(n);
  const lumaDev = new Float32Array(n);
  for (let cy = 0; cy < GRID; cy++)
    for (let cx = 0; cx < GRID; cx++) {
      let l = 0;
      let a = 0;
      let b = 0;
      let g = 0;
      let s = 0;
      let l2 = 0;
      for (let y = cy * CELL; y < (cy + 1) * CELL; y++)
        for (let x = cx * CELL; x < (cx + 1) * CELL; x++) {
          const i = y * SIZE + x;
          const li = px[i * 3];
          l += li;
          l2 += li * li;
          a += px[i * 3 + 1];
          b += px[i * 3 + 2];
          const lx = x + 1 < SIZE ? px[(i + 1) * 3] : li;
          const ly = y + 1 < SIZE ? px[(i + SIZE) * 3] : li;
          g += Math.abs(lx - li) + Math.abs(ly - li);
          // classic smartcrop skin colour distance (normalised rgb)
          const r = rgb[i * 3];
          const gg = rgb[i * 3 + 1];
          const bb = rgb[i * 3 + 2];
          const mag = Math.sqrt(r * r + gg * gg + bb * bb) || 1;
          const d = Math.sqrt(
            (r / mag - 0.78) ** 2 +
              (gg / mag - 0.57) ** 2 +
              (bb / mag - 0.44) ** 2
          );
          s += Math.max(0, 1 - d);
        }
      const k = cy * GRID + cx;
      const m = CELL * CELL;
      L[k] = l / m;
      A[k] = a / m;
      B[k] = b / m;
      grad[k] = g / m;
      skin[k] = s / m;
      lumaDev[k] = Math.sqrt(Math.max(0, l2 / m - (l / m) ** 2));
    }
  let mL = 0;
  let mA = 0;
  let mB = 0;
  for (let k = 0; k < n; k++) {
    mL += L[k];
    mA += A[k];
    mB += B[k];
  }
  mL /= n;
  mA /= n;
  mB /= n;
  const ft = new Float32Array(n);
  for (let k = 0; k < n; k++)
    ft[k] = Math.hypot(L[k] - mL, A[k] - mA, B[k] - mB);
  const dist = (p: [Float32Array, Float32Array, Float32Array]) => {
    const o = new Float32Array(n);
    for (let k = 0; k < n; k++)
      o[k] = Math.hypot(L[k] - p[0][k], A[k] - p[1][k], B[k] - p[2][k]);
    return o;
  };
  const cs2 = dist([boxMean(L, 2), boxMean(A, 2), boxMean(B, 2)]);
  const cs5 = dist([boxMean(L, 5), boxMean(A, 5), boxMean(B, 5)]);
  // Background prior: colour distance to each image border strip (2 cells),
  // using the strip's colour that is nearest to the cell (border strips are
  // usually background; objects rarely touch all four borders).
  const strips: number[][][] = [[], [], [], []];
  for (let i = 0; i < GRID; i++)
    for (let d = 0; d < 2; d++) {
      strips[0].push([L[d * GRID + i], A[d * GRID + i], B[d * GRID + i]]);
      const yb = (GRID - 1 - d) * GRID + i;
      strips[1].push([L[yb], A[yb], B[yb]]);
      const xl = i * GRID + d;
      strips[2].push([L[xl], A[xl], B[xl]]);
      const xr = i * GRID + GRID - 1 - d;
      strips[3].push([L[xr], A[xr], B[xr]]);
    }
  const bdMin = new Float32Array(n);
  const bdMean = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let mn = Infinity;
    let sum = 0;
    for (const st of strips) {
      // distance to the nearest colour in this strip
      let near = Infinity;
      for (const c of st) {
        const d = (L[k] - c[0]) ** 2 + (A[k] - c[1]) ** 2 + (B[k] - c[2]) ** 2;
        if (d < near) near = d;
      }
      near = Math.sqrt(near);
      if (near < mn) mn = near;
      sum += near;
    }
    bdMin[k] = mn;
    bdMean[k] = sum / 4;
  }
  const bdMinBlur = boxMean(bdMin, 2);
  const bdMax = maxOf(bdMin);
  const grad3 = boxMean(grad, 1);
  const ftBlur = boxMean(ft, 2);
  const gradBlur = boxMean(grad, 3);
  const ftMax = maxOf(ft);
  const csMax = maxOf(cs2);
  const gMax = maxOf(grad);
  const out = new Float32Array(n * SAL_DIM);
  for (let cy = 0; cy < GRID; cy++)
    for (let cx = 0; cx < GRID; cx++) {
      const k = cy * GRID + cx;
      const x = (cx + 0.5) / GRID;
      const y = (cy + 0.5) / GRID;
      const row = [
        L[k] / 100,
        A[k] / 100,
        B[k] / 100,
        Math.hypot(A[k], B[k]) / 100,
        ft[k] / 100,
        ft[k] / ftMax,
        cs2[k] / 100,
        cs5[k] / 100,
        cs2[k] / csMax,
        grad[k] / 50,
        grad3[k] / 50,
        grad[k] / gMax,
        skin[k],
        x,
        y,
        (x - 0.5) ** 2 + (y - 0.5) ** 2,
        Math.min(x, 1 - x, y, 1 - y),
        ftBlur[k] / ftMax,
        gradBlur[k] / gMax,
        lumaDev[k] / 50,
        bdMin[k] / 100,
        bdMean[k] / 100,
        bdMinBlur[k] / 100,
        bdMin[k] / bdMax,
      ];
      out.set(row, k * SAL_DIM);
    }
  return out;
}

/** Per-cell probabilities of an MLP over `dim`-wide feature rows. */
const mlpMap = (f: Float32Array, dim: number, model: SaliencyModel) => {
  const n = GRID * GRID;
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let x = new Float32Array(dim);
    for (let j = 0; j < dim; j++)
      x[j] = (f[k * dim + j] - model.mean[j]) / model.std[j];
    model.layers.forEach((ly, li) => {
      const h = new Float32Array(ly.outDim);
      for (let o = 0; o < ly.outDim; o++) {
        let s = ly.b[o];
        for (let i = 0; i < ly.inDim; i++) s += ly.w[o * ly.inDim + i] * x[i];
        h[o] = li < model.layers.length - 1 ? (s > 0 ? s : 0) : s;
      }
      x = h;
    });
    out[k] = 1 / (1 + Math.exp(-x[0]));
  }
  return out;
};

export const CONTEXT_DIM = 6;

/**
 * Auto-context: append features of the previous stage's map (the cell, its
 * blurred surroundings at three radii, the map's mean, and the cell relative
 * to the map's maximum) to every base feature row.
 */
export function withContext(
  base: Float32Array,
  prev: Float32Array
): Float32Array {
  const n = GRID * GRID;
  const b1 = boxMean(prev, 1);
  const b2 = boxMean(prev, 3);
  const b3 = boxMean(prev, 7);
  const mean = prev.reduce((a, v) => a + v, 0) / n;
  const mx = maxOf(prev);
  const dim = SAL_DIM + CONTEXT_DIM;
  const out = new Float32Array(n * dim);
  for (let k = 0; k < n; k++) {
    out.set(base.subarray(k * SAL_DIM, (k + 1) * SAL_DIM), k * dim);
    out.set(
      [prev[k], b1[k], b2[k], b3[k], mean, prev[k] / mx],
      k * dim + SAL_DIM
    );
  }
  return out;
}

/** Per-cell probabilities of a (possibly multi-stage) model, unnormalised. */
export function saliencyProbs(
  base: Float32Array,
  model: SaliencyModel
): Float32Array {
  if (!model.context) return mlpMap(base, SAL_DIM, model);
  const prev = saliencyProbs(base, model.context);
  return mlpMap(withContext(base, prev), SAL_DIM + CONTEXT_DIM, model);
}

/** Saliency map (GRID x GRID, 0..1) predicted by `model`. */
export function predictSaliency(img: RGBA, model: SaliencyModel) {
  const data = saliencyProbs(saliencyFeatures(img), model);
  let min = Infinity;
  let max = -Infinity;
  for (const v of data) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const r = max - min || 1;
  for (let k = 0; k < data.length; k++) data[k] = (data[k] - min) / r;
  return { width: GRID, height: GRID, data };
}

let builtIn: SaliencyModel | undefined;
/** The built-in model's map (lazily unpacked weights). */
export const builtInSaliency = (img: RGBA) =>
  predictSaliency(img, (builtIn ??= unpackSaliency(SALIENCY_PACKED)));
