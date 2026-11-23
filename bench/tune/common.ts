// Shared helpers for bench/tune.ts: metrics (identical to box-eval.ts),
// seeded RNG, search-space handling.
import { BALANCED_SUBSETS, indexBySubset, SUBSETS } from '../subsets';

export type Box = [number, number, number, number];
export type Ann = {
  image: string;
  width: number;
  height: number;
  boxes: Box[];
  split: string;
};
export type Config = Record<string, unknown>;

export const iou = (a: Box, b: Box) => {
  const iw = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
};
export const bde = (a: Box, b: Box, w: number, h: number) =>
  (Math.abs(a[0] - b[0]) / w +
    Math.abs(a[0] + a[2] - b[0] - b[2]) / w +
    Math.abs(a[1] - b[1]) / h +
    Math.abs(a[1] + a[3] - b[1] - b[3]) / h) /
  4;
export const centered = (
  ref: Box,
  w: number,
  h: number,
  scale: number
): Box => {
  const s = Math.min(w / ref[2], h / ref[3]) * scale;
  const cw = ref[2] * s;
  const ch = ref[3] * s;
  return [(w - cw) / 2, (h - ch) / 2, cw, ch];
};
export const mean = (xs: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / Math.max(1, xs.length);
};

// Per-image [iou, bde] with the best reference counting (max IoU, min BDE).
export function scoreBoxes(preds: Box[], a: Ann): [number, number] {
  const ious = a.boxes.map((gt, i) => iou(preds[i], gt));
  const bdes = a.boxes.map((gt, i) => bde(preds[i], gt, a.width, a.height));
  return [Math.max(...ious), Math.min(...bdes)];
}

export function centerBaseline(anns: Ann[], scale = 1) {
  const r = anns.map((a) =>
    scoreBoxes(
      a.boxes.map((gt) => centered(gt, a.width, a.height, scale)),
      a
    )
  );
  return { iou: mean(r.map((x) => x[0])), bde: mean(r.map((x) => x[1])) };
}

export function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffled<T>(xs: T[], rand: () => number): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export type Param =
  | {
      type: 'float';
      min: number;
      max: number;
      scale?: 'linear' | 'log';
      offset?: number;
      zeroProb?: number;
    }
  | { type: 'int'; min: number; max: number }
  | { type: 'bool' }
  | { type: 'choice'; values: unknown[] }
  | { type: 'fixed'; value: unknown };
export type Space = { fixed?: Config; params: Record<string, Param> };

const round = (v: number) => Number(v.toPrecision(4));
export function fromUnit(p: Param, u: number): unknown {
  u = Math.min(1, Math.max(0, u));
  switch (p.type) {
    case 'float': {
      if (p.scale === 'log') {
        const o = p.offset ?? 0.1;
        return round((p.min + o) * ((p.max + o) / (p.min + o)) ** u - o + 0);
      }
      return round(p.min + (p.max - p.min) * u);
    }
    case 'int':
      return Math.round(p.min + (p.max - p.min) * u);
    default:
      throw new Error('not unit-mapped');
  }
}
export function toUnit(p: Param, v: number): number {
  if (p.type === 'float' && p.scale === 'log') {
    const o = p.offset ?? 0.1;
    return (
      Math.log((v + o) / (p.min + o)) / Math.log((p.max + o) / (p.min + o))
    );
  }
  if (p.type === 'float' || p.type === 'int')
    return (v - p.min) / (p.max - p.min);
  throw new Error('not unit-mapped');
}
export function sample(space: Space, rand: () => number): Config {
  const c: Config = { ...(space.fixed ?? {}) };
  for (const [k, p] of Object.entries(space.params)) {
    if (p.type === 'fixed') c[k] = p.value;
    else if (p.type === 'bool') c[k] = rand() < 0.5;
    else if (p.type === 'choice')
      c[k] = p.values[Math.floor(rand() * p.values.length)];
    else {
      const r = rand();
      const z = p.type === 'float' && rand() < (p.zeroProb ?? 0);
      c[k] = z ? p.min : fromUnit(p, r);
    }
  }
  return c;
}
// Single-parameter alternatives for coordinate descent.
export function neighbours(
  space: Space,
  c: Config,
  key: string,
  delta: number
): Config[] {
  const p = space.params[key];
  const out: unknown[] = [];
  if (p.type === 'bool') out.push(!c[key]);
  else if (p.type === 'choice')
    out.push(...p.values.filter((v) => v !== c[key]));
  else if (p.type === 'float' || p.type === 'int') {
    const u = toUnit(p, c[key] as number);
    for (const d of [-delta, delta]) {
      const v = fromUnit(p, u + d);
      if (v !== c[key]) out.push(v);
    }
  }
  return [...new Set(out)].map((v) => ({ ...c, [key]: v }));
}
export const keyOf = (c: Config) =>
  JSON.stringify(
    Object.fromEntries(Object.entries(c).sort(([a], [b]) => (a < b ? -1 : 1)))
  );

// Per-subset summary of per-image scores (subsets come from bench/subsets.ts).
export type SubStat = { n: number; iou: number; bde: number };
export type SetResult = {
  iou: number;
  bde: number;
  n: number;
  secs: number;
  sub: Record<string, SubStat>;
  /** equal-weight mean of IoU over the balanced subsets (people, landscape, other) */
  bal: number;
};
export function summarizeSet(
  dataset: string,
  images: string[],
  iou: ArrayLike<number>,
  bde: ArrayLike<number>,
  secs = 0
): SetResult {
  const idx = indexBySubset(dataset, images);
  const sub: Record<string, SubStat> = {};
  for (const s of SUBSETS) {
    const ix = idx[s];
    sub[s] = {
      n: ix.length,
      iou: mean(ix.map((i) => iou[i])),
      bde: mean(ix.map((i) => bde[i])),
    };
  }
  const bs = BALANCED_SUBSETS.filter((s) => sub[s].n > 0);
  return {
    iou: mean(iou),
    bde: mean(bde),
    n: iou.length,
    secs,
    sub,
    bal: bs.length ? mean(bs.map((s) => sub[s].iou)) : mean(iou),
  };
}
// Center-crop baseline with per-subset results.
export function centerSet(dataset: string, anns: Ann[], scale = 1): SetResult {
  const r = anns.map((a) =>
    scoreBoxes(
      a.boxes.map((gt) => centered(gt, a.width, a.height, scale)),
      a
    )
  );
  return summarizeSet(
    dataset,
    anns.map((a) => a.image),
    r.map((x) => x[0]),
    r.map((x) => x[1])
  );
}
