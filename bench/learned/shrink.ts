// Shrink a trained learned-scorer model for shipping: try weight encodings
// from smallest to largest (q8, f16, f32) and keep the smallest that is
// lossless on validation data, then write src/learned-weights.ts.
//
// Lossless means, on every check set:
//   - the top-scored candidate of every image is unchanged, and
//   - mean SRCC moves by at most --srcc-tol (default 0.001).
// Check sets: GAICD val (plain, +saliency, +faces+saliency) and the FCDB-train
// holdout (last 15%), all from bench/out/learned (bench/learned/dump.ts).
// Test splits are never used.
//
// A model.json with a `cnn: {k, channels}` field (train.ts --cnn) is the cnn
// head: it is checked on the +faces+saliency+cnn[K] sets only (same criteria),
// packed together with the fixed PCA projection (bench/out/cnn/projection.bin,
// already float16-exact) and written to src/learned-cnn-weights.ts.
//
// Usage: bun bench/learned/shrink.ts [model.json] [--dry-run] [--srcc-tol 0.001]
//          [--top-tol 0] [--out src/learned-weights.ts]
// model.json is {note, model} as written by train.ts --write (default:
// bench/out/learned/models/latest.json). --top-tol allows that fraction of
// images to change their top choice (0 = identical).
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  encodeF16,
  learnedScore,
  packModel,
  unpackModel,
  type LearnedModel,
  type PackedModel,
} from '../../src/learned';
import type { PackedCnn } from '../../src/cnn';
import { PROJECTION_FILE } from '../plugin-cache';

const args = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args.splice(i, 2)[1] : d;
};
const dryRun = args.includes('--dry-run');
if (dryRun) args.splice(args.indexOf('--dry-run'), 1);
const srccTol = Number(opt('srcc-tol', '0.001'));
const topTol = Number(opt('top-tol', '0'));
const OUT = resolve(import.meta.dir, '../out/learned');
const outOpt = opt('out');
const modelPath = args[0] ?? resolve(OUT, 'models/latest.json');

const { note, model, cnn } = JSON.parse(await readFile(modelPath, 'utf8')) as {
  note: string;
  model: LearnedModel;
  cnn?: { k: number; channels: number };
};
const outPath =
  outOpt ??
  resolve(
    import.meta.dir,
    cnn ? '../../src/learned-cnn-weights.ts' : '../../src/learned-weights.ts'
  );

type DataSet = {
  D: number;
  X: Float32Array;
  imgs: { off: number; n: number; mos: number[] }[];
};
const load = async (name: string, holdout = false): Promise<DataSet | null> => {
  if (!existsSync(`${OUT}/${name}.bin`)) return null;
  const m = JSON.parse(await readFile(`${OUT}/${name}.json`, 'utf8'));
  const X = new Float32Array(
    await Bun.file(`${OUT}/${name}.bin`).arrayBuffer()
  );
  const imgs = m.images as DataSet['imgs'];
  return {
    D: m.D,
    X,
    imgs: holdout ? imgs.slice(Math.floor(imgs.length * 0.85)) : imgs,
  };
};
const checks: [string, DataSet][] = [];
const variants = cnn
  ? [`-faces+saliency+cnn${cnn.k === 3 ? '' : cnn.k}`]
  : ['', '-saliency', '-faces+saliency'];
for (const v of variants) {
  const g = await load(`gaicd-val${v}`);
  if (g) checks.push([`gaicd-val${v}`, g]);
  const f = await load(`fcdb-train${v}`, true);
  if (f) checks.push([`fcdb-holdout${v}`, f]);
}
if (!checks.length) {
  console.error(
    'no cached features in bench/out/learned; run bench/learned/dump.ts first'
  );
  process.exit(1);
}

const ranks = (a: number[]) => {
  const idx = a.map((_, i) => i).sort((p, q) => a[p] - a[q]);
  const r = new Array<number>(a.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2;
    i = j + 1;
  }
  return r;
};
const pearson = (x: number[], y: number[]) => {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0,
    sxx = 0,
    syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
};
const run = (m: LearnedModel, ds: DataSet) => {
  if (m.mean.length !== ds.D) throw new Error('model/feature width mismatch');
  let srcc = 0;
  const top: number[] = [];
  for (const im of ds.imgs) {
    const s: number[] = [];
    for (let i = 0; i < im.n; i++)
      s.push(learnedScore(ds.X, (im.off + i) * ds.D, m));
    srcc += pearson(ranks(s), ranks(im.mos));
    let best = 0;
    for (let i = 1; i < im.n; i++) if (s[i] > s[best]) best = i;
    top.push(best);
  }
  return { srcc: srcc / ds.imgs.length, top };
};

const base = checks.map(([, ds]) => run(model, ds));
let chosen: PackedModel | null = null;
for (const kind of ['q8', 'f16', 'f32'] as const) {
  const packed = packModel(model, kind);
  const m = unpackModel(packed, model.mean.length);
  let ok = true;
  const rows: string[] = [];
  checks.forEach(([name, ds], i) => {
    const r = run(m, ds);
    const changed =
      r.top.filter((t, k) => t !== base[i].top[k]).length / r.top.length;
    const dSrcc = r.srcc - base[i].srcc;
    const pass = changed <= topTol && Math.abs(dSrcc) <= srccTol;
    ok &&= pass;
    rows.push(
      `  ${name.padEnd(26)} top changed ${(100 * changed).toFixed(2)}%  srcc ${r.srcc.toFixed(4)} (${dSrcc >= 0 ? '+' : ''}${dSrcc.toFixed(4)}) ${pass ? 'ok' : 'FAIL'}`
    );
  });
  console.log(
    `${kind}: ${JSON.stringify(packed).length} bytes ${ok ? 'lossless' : 'lossy'}`
  );
  console.log(rows.join('\n'));
  if (ok) {
    chosen = packed;
    break;
  }
}
if (!chosen)
  throw new Error(
    'no encoding passed (f32 should always pass; check the inputs)'
  );

let projection = '';
if (cnn) {
  const buf = await Bun.file(PROJECTION_FILE).arrayBuffer();
  if (new Uint32Array(buf, 0, 1)[0] !== cnn.channels)
    throw new Error('projection channels differ from the model');
  projection = encodeF16(new Float32Array(buf, 4)); // already float16-exact
}
const packedCnn: PackedCnn | null = cnn
  ? {
      k: cnn.k,
      channels: cnn.channels,
      inChannels: 448,
      projection,
      head: chosen,
    }
  : null;
const src = cnn
  ? `// Weights of the optional deep-feature (cnn) head of the learned scorer. Trained by
// bench/learned/train.ts --cnn, packed by bench/learned/shrink.ts: the head is
// '${chosen.kind}' (smallest encoding that keeps every validation top-1 choice),
// the fixed PCA projection of the plug-in (plugins/onnx/cnn.ts) is float16.
// Only used together with a featureMap; not imported by the library itself.
// ${note}
import type { PackedCnn } from './cnn';

export const LEARNED_CNN_PACKED: PackedCnn = ${JSON.stringify(packedCnn)};
`
  : `// Weights of the learned composition scorer. Trained by bench/learned/train.ts,
// encoded by bench/learned/shrink.ts as '${chosen.kind}' (smallest encoding that
// keeps every validation top-1 choice; see packModel in learned.ts).
// ${note}
import type { PackedModel } from './learned';

export const LEARNED_PACKED: PackedModel = ${JSON.stringify(chosen)};
`;
if (dryRun) {
  console.log(
    `dry run: would write ${outPath} (${src.length} bytes, ${chosen.kind})`
  );
} else {
  await Bun.write(outPath, src);
  console.log(`wrote ${outPath} (${src.length} bytes, ${chosen.kind})`);
}
