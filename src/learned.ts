// Learned composition scorer: feature extraction and model evaluation.
//
// Everything works on a coarse "score grid" (gw x gh cells of `factor` pixels
// of the prescaled image). Each candidate crop is described by a fixed-length
// feature vector computed in O(1) per map with summed-area tables, and scored
// by a small model whose weights (LEARNED_PACKED) were trained on
// human-rated crops (GAICD). See bench/learned/train.ts.
import { LEARNED_PACKED } from "./learned-weights";

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Grid maps, all gw*gh, values roughly 0-1. null = not available. */
export interface LearnedMaps {
  gw: number;
  gh: number;
  /** Pixels per grid cell. */
  factor: number;
  /** Size of the (prescaled) image in pixels. */
  width: number;
  height: number;
  /** detail (texture damped), skin, saturation, saliency, importance map, boost. */
  maps: (Float32Array | null)[];
  /** Mean luminance per cell, 0-255. */
  luma: Float32Array;
  /** Boost boxes (e.g. faces) in image pixels. */
  boosts: Box[];
}

export const MAP_NAMES = [
  "detail",
  "skin",
  "saturation",
  "saliency",
  "importance",
  "boost",
  "gradient", // luminance gradient magnitude
] as const;

/**
 * Per-map features. `lift` = mean density relative to the image mean. Bands
 * are 10% of the crop size. "outer" = just outside the crop edge.
 */
export const PER_MAP_FEATURES = [
  "recall", // mass inside / total mass
  "lift", // mean density inside / image mean (cap 5)
  "bandLift", // density in the inner border band (cap 5)
  "outerFrac", // mass in the strips just outside the edges / total mass
  "cutSum", // sum over edges of min(inner lift, outer lift): content crossing the edge
  "cutMax", // max over edges of the same
  "cdx", // |centroid x - crop centre| / crop width
  "cdy",
  "thirds", // distance of the centroid to the nearest thirds intersection (crop units)
  "balLR", // |left - right| / (left + right) mass
  "balTB",
  "edgeLift", // density on the 1-cell lines along the four crop edges (cap 5)
] as const;

export const GLOBAL_FEATURES = [
  "hasImportance", // presence flags
  "hasBoost",
  "area", // crop area / image area
  "area2",
  "logArea", // log(area ratio)
  "offX", // |crop centre - image centre| / image size * 2
  "offY",
  "aspect", // log(crop aspect / image aspect)
  "absAspect",
  "touchL", // crop edge on image border
  "touchR",
  "touchT",
  "touchB",
  "horizon", // horizon confidence inside the crop
  "horizon13", // confidence * distance of the horizon to the 1/3 line (crop units)
  "horizon23",
  "horizon12",
  "horizonThirds", // confidence * distance to the nearest third line
  "horizonIn", // horizon inside the crop
  "faceCut", // fraction of boost-box area in boxes that are partly cut
  "faceIn", // a boost box is mostly inside
  "headroom", // (top of topmost box - crop top) / crop height
  "headroomDev", // |headroom - 0.12|
  "faceSize", // largest contained box area / crop area
  // Boundary simplicity (Fang et al. 2014): gradient energy in 2-cell bands
  // along each crop edge relative to the image mean (cap 5); good crops have
  // edges through simple, low-gradient regions.
  "gradL",
  "gradR",
  "gradT",
  "gradB",
  "gradMax",
  "gradMean",
] as const;

/**
 * "Subject box" features: a bounding box of the main subject (saliency map,
 * importance map, boost boxes) against the crop. Margins are in crop units,
 * negative = the subject is cut by that edge (clamped to +-0.5).
 */
export const SUBJECT_SOURCES = ["saliency", "importance", "boost"] as const;
export const SUBJECT_FEATURES = [
  "inside", // fraction of the subject box area inside the crop
  "mL",
  "mR",
  "mT",
  "mB",
  "minMargin",
  "cx", // subject box centre relative to the crop centre (crop units)
  "cy",
  "size", // subject box area / crop area (cap 2)
] as const;

export const FEATURE_NAMES: string[] = [
  ...MAP_NAMES.flatMap((m) => PER_MAP_FEATURES.map((f) => `${m}.${f}`)),
  ...GLOBAL_FEATURES,
  ...SUBJECT_SOURCES.flatMap((m) =>
    SUBJECT_FEATURES.map((f) => `subject.${m}.${f}`),
  ),
];
export const FEATURE_COUNT = FEATURE_NAMES.length;
const PER = PER_MAP_FEATURES.length;
/** Index of a feature by name. */
export const featureIndex = (name: string): number =>
  FEATURE_NAMES.indexOf(name);

export interface LearnedContext extends LearnedMaps {
  stride: number;
  sums: (Float64Array | null)[];
  sumsX: (Float64Array | null)[];
  sumsY: (Float64Array | null)[];
  totals: number[];
  horizonRow: number; // in pixels
  horizonConf: number;
  boostTop: number;
  /** Subject box (image pixels) per SUBJECT_SOURCES entry, null = none. */
  subjects: (Box | null)[];
}

const integral = (
  grid: Float32Array,
  gw: number,
  gh: number,
  mode: 0 | 1 | 2,
): Float64Array => {
  const stride = gw + 1;
  const out = new Float64Array(stride * (gh + 1));
  for (let y = 0; y < gh; y++) {
    let row = 0;
    for (let x = 0; x < gw; x++) {
      let v = grid[y * gw + x];
      if (mode === 1) v *= x + 0.5;
      else if (mode === 2) v *= y + 0.5;
      row += v;
      out[(y + 1) * stride + x + 1] = out[y * stride + x + 1] + row;
    }
  }
  return out;
};

// Box spanning the 5th-95th percentile of the mass of cells above 40% of the
// map maximum, in image pixels.
const subjectBox = (
  g: Float32Array,
  gw: number,
  gh: number,
  factor: number,
): Box | null => {
  let max = 0;
  for (let i = 0; i < g.length; i++) if (g[i] > max) max = g[i];
  if (max <= 0) return null;
  const thr = max * 0.4;
  const col = new Float64Array(gw);
  const row = new Float64Array(gh);
  let total = 0;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = g[y * gw + x];
      if (v >= thr) {
        col[x] += v;
        row[y] += v;
        total += v;
      }
    }
  }
  const span = (a: Float64Array): [number, number] => {
    let acc = 0;
    let lo = 0;
    let hi = a.length;
    for (let i = 0; i < a.length; i++) {
      acc += a[i];
      if (acc >= total * 0.05) {
        lo = i;
        break;
      }
    }
    acc = 0;
    for (let i = a.length - 1; i >= 0; i--) {
      acc += a[i];
      if (acc >= total * 0.05) {
        hi = i + 1;
        break;
      }
    }
    return [lo, Math.max(hi, lo + 1)];
  };
  const [x0, x1] = span(col);
  const [y0, y1] = span(row);
  return {
    x: x0 * factor,
    y: y0 * factor,
    width: (x1 - x0) * factor,
    height: (y1 - y0) * factor,
  };
};

export const learnedContext = (m: LearnedMaps): LearnedContext => {
  const { gw, gh, factor } = m;
  const sums: (Float64Array | null)[] = [];
  const sumsX: (Float64Array | null)[] = [];
  const sumsY: (Float64Array | null)[] = [];
  const totals: number[] = [];
  for (const g of m.maps) {
    if (g) {
      const s = integral(g, gw, gh, 0);
      sums.push(s);
      sumsX.push(integral(g, gw, gh, 1));
      sumsY.push(integral(g, gw, gh, 2));
      totals.push(s[s.length - 1]);
    } else {
      sums.push(null);
      sumsX.push(null);
      sumsY.push(null);
      totals.push(0);
    }
  }
  // Horizon: the row with the strongest coherent vertical luminance step
  // (|mean over the row of L[y+1] - L[y-1]|), relative to the other rows.
  const e = new Float32Array(gh);
  let mean = 0;
  let peak = 0;
  let row = 1;
  for (let y = 1; y < gh - 1; y++) {
    let s = 0;
    for (let x = 0; x < gw; x++) {
      s += m.luma[(y + 1) * gw + x] - m.luma[(y - 1) * gw + x];
    }
    e[y] = Math.abs(s) / gw;
    mean += e[y];
    if (e[y] > peak) {
      peak = e[y];
      row = y;
    }
  }
  mean /= Math.max(1, gh - 2);
  const conf =
    gh > 4
      ? (Math.min(8, peak / (mean + 1e-6)) / 8) * Math.min(1, peak / 32)
      : 0;
  let boostTop = Infinity;
  for (const b of m.boosts) boostTop = Math.min(boostTop, b.y);
  return {
    ...m,
    stride: gw + 1,
    sums,
    sumsX,
    sumsY,
    totals,
    horizonRow: (row + 0.5) * factor,
    horizonConf: conf,
    boostTop,
    subjects: [
      m.maps[3] ? subjectBox(m.maps[3], gw, gh, factor) : null,
      m.maps[4] ? subjectBox(m.maps[4], gw, gh, factor) : null,
      unionBox(m.boosts),
    ],
  };
};

const unionBox = (boxes: Box[]): Box | null => {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const b of boxes) {
    if (b.width <= 0 || b.height <= 0) continue;
    x0 = Math.min(x0, b.x);
    y0 = Math.min(y0, b.y);
    x1 = Math.max(x1, b.x + b.width);
    y1 = Math.max(y1, b.y + b.height);
  }
  return x1 > x0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : null;
};

const rect = (
  s: Float64Array,
  stride: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): number =>
  x1 > x0 && y1 > y0
    ? s[y1 * stride + x1] -
      s[y0 * stride + x1] -
      s[y1 * stride + x0] +
      s[y0 * stride + x0]
    : 0;

const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

const THIRD = [1 / 3, 2 / 3];

/**
 * Write the feature vector of crop (x, y, w, h) (image pixels) into
 * out[off .. off + FEATURE_COUNT).
 */
export const learnedFeatures = (
  c: LearnedContext,
  x: number,
  y: number,
  w: number,
  h: number,
  out: Float32Array,
  off: number,
): void => {
  const { gw, gh, factor: F, stride } = c;
  const gx0 = clamp(Math.round(x / F), 0, gw - 1);
  const gy0 = clamp(Math.round(y / F), 0, gh - 1);
  const gx1 = clamp(Math.round((x + w) / F), gx0 + 1, gw);
  const gy1 = clamp(Math.round((y + h) / F), gy0 + 1, gh);
  const cw = gx1 - gx0;
  const ch = gy1 - gy0;
  const area = cw * ch;
  const tx = Math.max(1, Math.round(cw * 0.1));
  const ty = Math.max(1, Math.round(ch * 0.1));
  const ix0 = gx0 + tx;
  const ix1 = gx1 - tx;
  const iy0 = gy0 + ty;
  const iy1 = gy1 - ty;
  const innerArea = ix1 > ix0 && iy1 > iy0 ? (ix1 - ix0) * (iy1 - iy0) : 0;
  const midX = gx0 + Math.round(cw / 2);
  const midY = gy0 + Math.round(ch / 2);
  const cells = gw * gh;

  let o = off;
  for (let m = 0; m < c.sums.length; m++) {
    const S = c.sums[m];
    if (!S) {
      for (let k = 0; k < PER; k++) out[o + k] = 0;
      o += PER;
      continue;
    }
    const T = c.totals[m];
    const sum = rect(S, stride, gx0, gy0, gx1, gy1);
    if (T <= 1e-9) {
      for (let k = 0; k < PER; k++) out[o + k] = 0;
      out[o + 8] = 0.47;
      o += PER;
      continue;
    }
    const meanDensity = T / cells;
    const lift = (mass: number, a: number) =>
      a > 0 ? Math.min(5, mass / a / meanDensity) : 0;
    const bandMass = sum - rect(S, stride, ix0, iy0, ix1, iy1);
    // Edge strips: inner (inside the crop) and outer (just outside).
    const inL = rect(S, stride, gx0, gy0, Math.min(gx1, gx0 + tx), gy1);
    const inR = rect(S, stride, Math.max(gx0, gx1 - tx), gy0, gx1, gy1);
    const inT = rect(S, stride, gx0, gy0, gx1, Math.min(gy1, gy0 + ty));
    const inB = rect(S, stride, gx0, Math.max(gy0, gy1 - ty), gx1, gy1);
    const outL =
      gx0 > 0 ? rect(S, stride, Math.max(0, gx0 - tx), gy0, gx0, gy1) : 0;
    const outR =
      gx1 < gw ? rect(S, stride, gx1, gy0, Math.min(gw, gx1 + tx), gy1) : 0;
    const outT =
      gy0 > 0 ? rect(S, stride, gx0, Math.max(0, gy0 - ty), gx1, gy0) : 0;
    const outB =
      gy1 < gh ? rect(S, stride, gx0, gy1, gx1, Math.min(gh, gy1 + ty)) : 0;
    const aV = Math.min(tx, cw) * ch;
    const aH = Math.min(ty, ch) * cw;
    const cut = (inside: number, outside: number, a: number) =>
      Math.min(lift(inside, a), lift(outside, a));
    const cL = gx0 > 0 ? cut(inL, outL, aV) : 0;
    const cR = gx1 < gw ? cut(inR, outR, aV) : 0;
    const cT = gy0 > 0 ? cut(inT, outT, aH) : 0;
    const cB = gy1 < gh ? cut(inB, outB, aH) : 0;

    const ew = Math.min(1, cw);
    const eh = Math.min(1, ch);
    const edgeMass =
      rect(S, stride, gx0, gy0, gx0 + ew, gy1) +
      rect(S, stride, gx1 - ew, gy0, gx1, gy1) +
      rect(S, stride, gx0, gy0, gx1, gy0 + eh) +
      rect(S, stride, gx0, gy1 - eh, gx1, gy1);
    out[o + 11] = lift(edgeMass, Math.max(1, 2 * (cw + ch) - 4));
    out[o] = sum / T;
    out[o + 1] = lift(sum, area);
    out[o + 2] = lift(bandMass, area - innerArea);
    out[o + 3] = (outL + outR + outT + outB) / T;
    out[o + 4] = cL + cR + cT + cB;
    out[o + 5] = Math.max(cL, cR, cT, cB);
    if (sum > 1e-9) {
      const u =
        (rect(c.sumsX[m]!, stride, gx0, gy0, gx1, gy1) / sum - gx0) / cw;
      const v =
        (rect(c.sumsY[m]!, stride, gx0, gy0, gx1, gy1) / sum - gy0) / ch;
      out[o + 6] = Math.abs(u - 0.5);
      out[o + 7] = Math.abs(v - 0.5);
      let d = Infinity;
      for (const a of THIRD)
        for (const b of THIRD) d = Math.min(d, Math.hypot(u - a, v - b));
      out[o + 8] = d;
      const left = rect(S, stride, gx0, gy0, midX, gy1);
      const right = sum - left;
      const top = rect(S, stride, gx0, gy0, gx1, midY);
      out[o + 9] = Math.abs(left - right) / sum;
      out[o + 10] = Math.abs(top - (sum - top)) / sum;
    } else {
      out[o + 6] = 0;
      out[o + 7] = 0;
      out[o + 8] = 0.47;
      out[o + 9] = 0;
      out[o + 10] = 0;
    }
    o += PER;
  }

  // Global features.
  const W = c.width;
  const H = c.height;
  out[o++] = c.sums[4] ? 1 : 0;
  out[o++] = c.sums[5] ? 1 : 0;
  const ar = (w * h) / (W * H);
  out[o++] = ar;
  out[o++] = ar * ar;
  out[o++] = Math.log(Math.max(ar, 1e-4));
  out[o++] = (Math.abs(x + w / 2 - W / 2) / W) * 2;
  out[o++] = (Math.abs(y + h / 2 - H / 2) / H) * 2;
  const la = Math.log(w / h / (W / H));
  out[o++] = la;
  out[o++] = Math.abs(la);
  const eps = 0.5;
  out[o++] = x <= eps ? 1 : 0;
  out[o++] = x + w >= W - eps ? 1 : 0;
  out[o++] = y <= eps ? 1 : 0;
  out[o++] = y + h >= H - eps ? 1 : 0;

  const p = (c.horizonRow - y) / h;
  const inside = p >= 0.02 && p <= 0.98 ? 1 : 0;
  const hc = inside * c.horizonConf;
  const d13 = Math.abs(p - 1 / 3);
  const d23 = Math.abs(p - 2 / 3);
  out[o++] = hc;
  out[o++] = hc * d13;
  out[o++] = hc * d23;
  out[o++] = hc * Math.abs(p - 0.5);
  out[o++] = hc * Math.min(d13, d23);
  out[o++] = inside;

  let cutArea = 0;
  let totalArea = 0;
  let largest = 0;
  let topBox = Infinity;
  for (const b of c.boosts) {
    const a = b.width * b.height;
    if (a <= 0) continue;
    const iw = Math.min(x + w, b.x + b.width) - Math.max(x, b.x);
    const ih = Math.min(y + h, b.y + b.height) - Math.max(y, b.y);
    const frac = iw > 0 && ih > 0 ? (iw * ih) / a : 0;
    totalArea += a;
    if (frac > 0.02 && frac < 0.98) cutArea += a;
    if (frac >= 0.5) {
      topBox = Math.min(topBox, b.y);
      largest = Math.max(largest, a);
    }
  }
  out[o++] = totalArea > 0 ? cutArea / totalArea : 0;
  out[o++] = topBox < Infinity ? 1 : 0;
  const head = topBox < Infinity ? clamp((topBox - y) / h, -0.5, 1) : 0;
  out[o++] = head;
  out[o++] = topBox < Infinity ? Math.abs(head - 0.12) : 0;
  out[o++] = largest / (w * h);

  const G = c.sums[6];
  if (G && c.totals[6] > 1e-9) {
    const mean = c.totals[6] / cells;
    const t2 = 2;
    const e = (m: number, a: number) => Math.min(5, m / Math.max(1, a) / mean);
    const gL = e(
      rect(G, stride, gx0, gy0, Math.min(gx1, gx0 + t2), gy1),
      Math.min(t2, cw) * ch,
    );
    const gR = e(
      rect(G, stride, Math.max(gx0, gx1 - t2), gy0, gx1, gy1),
      Math.min(t2, cw) * ch,
    );
    const gT = e(
      rect(G, stride, gx0, gy0, gx1, Math.min(gy1, gy0 + t2)),
      Math.min(t2, ch) * cw,
    );
    const gB = e(
      rect(G, stride, gx0, Math.max(gy0, gy1 - t2), gx1, gy1),
      Math.min(t2, ch) * cw,
    );
    out[o++] = gL;
    out[o++] = gR;
    out[o++] = gT;
    out[o++] = gB;
    out[o++] = Math.max(gL, gR, gT, gB);
    out[o++] = (gL + gR + gT + gB) / 4;
  } else {
    for (let k = 0; k < 6; k++) out[o++] = 0;
  }

  for (const b of c.subjects) {
    if (!b) {
      for (let k = 0; k < SUBJECT_FEATURES.length; k++) out[o++] = 0;
      continue;
    }
    const iw = Math.min(x + w, b.x + b.width) - Math.max(x, b.x);
    const ih = Math.min(y + h, b.y + b.height) - Math.max(y, b.y);
    out[o++] = iw > 0 && ih > 0 ? (iw * ih) / (b.width * b.height) : 0;
    const mL = clamp((b.x - x) / w, -0.5, 0.5);
    const mR = clamp((x + w - b.x - b.width) / w, -0.5, 0.5);
    const mT = clamp((b.y - y) / h, -0.5, 0.5);
    const mB = clamp((y + h - b.y - b.height) / h, -0.5, 0.5);
    out[o++] = mL;
    out[o++] = mR;
    out[o++] = mT;
    out[o++] = mB;
    out[o++] = Math.min(mL, mR, mT, mB);
    out[o++] = (b.x + b.width / 2 - x) / w - 0.5;
    out[o++] = (b.y + b.height / 2 - y) / h - 0.5;
    out[o++] = Math.min(2, (b.width * b.height) / (w * h));
  }
};

export interface LearnedModel {
  mean: number[];
  std: number[];
  /**
   * Dense layers, ReLU between them, last layer has one output. Layer l has
   * b[l].length outputs and w[l] is row-major (outputs x inputs). One layer
   * is a linear model.
   */
  layers: { w: number[]; b: number[] }[];
  /**
   * Optional extra ensemble members (same architecture and standardisation).
   * The score is the mean of `layers` and every member.
   */
  members?: { w: number[]; b: number[] }[][];
}

/** Standardise one feature, clipped to +-6 sigma. */
export const z = (v: number, mean: number, std: number): number =>
  Math.max(-6, Math.min(6, (v - mean) / std));

/** Score one feature vector (raw, unstandardised). */
export const learnedScore = (
  f: Float32Array,
  off: number,
  model: LearnedModel = defaultModel(),
): number => {
  const { mean, std, layers } = model;
  if (layers.length === 0) return 0; // untrained
  const D = mean.length; // FEATURE_COUNT, or more for the cnn head
  const x0 = new Float64Array(D);
  for (let i = 0; i < D; i++) x0[i] = z(f[off + i], mean[i], std[i]);
  const nets = model.members ? [layers, ...model.members] : [layers];
  let sum = 0;
  for (const net of nets) sum += mlp(x0, net);
  return sum / nets.length;
};

const mlp = (
  x0: Float64Array,
  layers: { w: number[]; b: number[] }[],
): number => {
  let x = x0;
  for (let l = 0; l < layers.length; l++) {
    const { w, b } = layers[l];
    const n = x.length;
    const y = new Float64Array(b.length);
    for (let j = 0; j < y.length; j++) {
      let a = b[j];
      for (let i = 0; i < n; i++) a += w[j * n + i] * x[i];
      y[j] = l < layers.length - 1 && a < 0 ? 0 : a;
    }
    x = y;
  }
  return x[0];
};

// ---- compact weight encoding ----------------------------------------------
// Means, stds and biases are float32; weight matrices are float32 ('f32'),
// float16 ('f16') or int8 with one float32 scale per output row ('q8'). All little-endian,
// concatenated and base64-encoded.

export interface PackedModel {
  kind: "f32" | "f16" | "q8";
  /** outputs per layer; layer 0's input is FEATURE_COUNT (or the cnn head's input width) */
  sizes: number[];
  /** number of ensemble nets (1 + members) */
  nets: number;
  data: string;
}

const f16bits = (v: number): number => {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >> (1 - exp);
    return sign | ((mant + 0x1000) >> 13);
  }
  if (exp >= 31) return sign | 0x7c00;
  const r = sign | (exp << 10) | (mant >> 13);
  return mant & 0x1000 ? r + 1 : r; // round half up on the dropped bits
};
const f16value = (h: number): number => {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
};

const toBase64 = (bytes: Uint8Array): string => {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
};
const fromBase64 = (b64: string): Uint8Array => {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
};

/** base64 of float16 little-endian values. */
export const encodeF16 = (v: ArrayLike<number>): string => {
  const dv = new DataView(new ArrayBuffer(2 * v.length));
  for (let i = 0; i < v.length; i++) dv.setUint16(2 * i, f16bits(v[i]), true);
  return toBase64(new Uint8Array(dv.buffer));
};
/** Inverse of encodeF16. */
export const decodeF16 = (b64: string): Float32Array => {
  const b = fromBase64(b64);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const out = new Float32Array(b.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = f16value(dv.getUint16(2 * i, true));
  return out;
};

export const packModel = (
  model: LearnedModel,
  kind: PackedModel["kind"],
): PackedModel => {
  const nets = [model.layers, ...(model.members ?? [])];
  const sizes = model.layers.map((l) => l.b.length);
  const f32: number[] = [...model.mean, ...model.std];
  for (const net of nets) for (const l of net) f32.push(...l.b);
  // Layout: [f32 block][per-net, per-layer weights].
  const view = new DataView(new ArrayBuffer(4 * f32.length));
  f32.forEach((v, i) => view.setFloat32(4 * i, v, true));
  const chunks: Uint8Array[] = [new Uint8Array(view.buffer)];
  for (const net of nets) {
    for (const l of net) {
      const rows = l.b.length;
      const cols = l.w.length / rows;
      if (kind === "f32") {
        const dv = new DataView(new ArrayBuffer(4 * l.w.length));
        l.w.forEach((v, i) => dv.setFloat32(4 * i, v, true));
        chunks.push(new Uint8Array(dv.buffer));
      } else if (kind === "f16") {
        const dv = new DataView(new ArrayBuffer(2 * l.w.length));
        l.w.forEach((v, i) => dv.setUint16(2 * i, f16bits(v), true));
        chunks.push(new Uint8Array(dv.buffer));
      } else {
        const dv = new DataView(new ArrayBuffer(4 * rows + l.w.length));
        for (let r = 0; r < rows; r++) {
          let m = 0;
          for (let c = 0; c < cols; c++)
            m = Math.max(m, Math.abs(l.w[r * cols + c]));
          const scale = m / 127 || 1;
          dv.setFloat32(4 * r, scale, true);
          for (let c = 0; c < cols; c++) {
            dv.setInt8(
              4 * rows + r * cols + c,
              Math.round(l.w[r * cols + c] / scale),
            );
          }
        }
        chunks.push(new Uint8Array(dv.buffer));
      }
    }
  }
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const bytes = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  return { kind, sizes, nets: nets.length, data: toBase64(bytes) };
};

export const unpackModel = (
  p: PackedModel,
  /** input width of layer 0 (the cnn head has more than FEATURE_COUNT) */
  D: number = FEATURE_COUNT,
): LearnedModel => {
  const bytes = fromBase64(p.data);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 0;
  const f32 = (n: number) => {
    const a: number[] = [];
    for (let i = 0; i < n; i++, o += 4) a.push(dv.getFloat32(o, true));
    return a;
  };
  const mean = f32(D);
  const std = f32(D);
  const biases = Array.from({ length: p.nets }, () =>
    p.sizes.map((n) => f32(n)),
  );
  const nets = biases.map((bs) =>
    bs.map((b, l) => {
      const rows = p.sizes[l];
      const cols = l === 0 ? D : p.sizes[l - 1];
      const w: number[] = new Array(rows * cols);
      if (p.kind === "f32") {
        for (let i = 0; i < w.length; i++, o += 4)
          w[i] = dv.getFloat32(o, true);
      } else if (p.kind === "f16") {
        for (let i = 0; i < w.length; i++, o += 2)
          w[i] = f16value(dv.getUint16(o, true));
      } else {
        const scales = f32(rows);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++, o++)
            w[r * cols + c] = dv.getInt8(o) * scales[r];
        }
      }
      return { w, b };
    }),
  );
  return {
    mean,
    std,
    layers: nets[0],
    ...(nets.length > 1 ? { members: nets.slice(1) } : {}),
  };
};

let cachedModel: LearnedModel | undefined;
/** The shipped weights, unpacked on first use. */
export const defaultModel = (): LearnedModel =>
  (cachedModel ??= unpackModel(LEARNED_PACKED));
