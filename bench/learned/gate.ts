// Validation gate: does a newly trained model beat the shipped one?
//
// Both models are scored on the same selection data train.ts uses: the GAICD
// val split and (when the model was trained with --fcdb) the last 15% of
// fcdb-train as holdout. The selection score is val SRCC + holdout IoU of the
// top-scored window (just SRCC without a holdout). Test splits are never read.
// Exit code 1 when new <= shipped + margin.
//
// Usage: bun bench/learned/gate.ts <model.json> [--cnn 3] [--fcdb 0.5]
//          [--margin 0] [--dumps DIR] [--report out.json] [--markdown out.md]
// model.json is {note, model[, cnn]} as written by train.ts --write --out or
// combine.ts. --cnn K compares against the shipped cnn head
// (src/learned-cnn-weights.ts), otherwise against the hand-feature model
// (src/learned-weights.ts). --margin m demands a lead of at least m.
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  defaultModel,
  learnedScore,
  unpackModel,
  type LearnedModel,
} from '../../src/learned';
import { LEARNED_CNN_PACKED } from '../../src/learned-cnn-weights';

const args = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};
const modelPath = args[0];
if (!modelPath || modelPath.startsWith('--')) {
  console.error(
    'usage: gate.ts <model.json> [--cnn K] [--fcdb w] [--margin m]'
  );
  process.exit(2);
}
const cnnK = Number(opt('cnn', '0'));
const fcdbW = Number(opt('fcdb', '0'));
const margin = Number(opt('margin', '0'));
const OUT = resolve(opt('dumps') ?? resolve(import.meta.dir, '../out/learned'));
const suffix = cnnK ? `-faces+saliency+cnn${cnnK === 3 ? '' : cnnK}` : '';

type DataSet = {
  D: number;
  X: Float32Array;
  imgs: { off: number; n: number; mos: number[] }[];
};
const load = async (name: string, holdout = false): Promise<DataSet> => {
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

// Metrics as in train.ts evaluate().
const ranks = (v: number[]) => {
  const idx = v.map((_, i) => i).sort((a, b) => v[a] - v[b]);
  const r = new Array<number>(v.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
};
const pearson = (a: number[], b: number[]) => {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0,
    saa = 0,
    sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
};
const evaluate = (m: LearnedModel, ds: DataSet) => {
  let srcc = 0,
    a15 = 0,
    a110 = 0,
    iou = 0;
  for (const im of ds.imgs) {
    const s: number[] = [];
    for (let i = 0; i < im.n; i++)
      s.push(learnedScore(ds.X, (im.off + i) * ds.D, m));
    srcc += pearson(ranks(s), ranks(im.mos));
    const order = s.map((_, i) => i).sort((a, b) => s[b] - s[a]); // stable
    const mosSorted = [...im.mos].sort((a, b) => b - a);
    const hit = (N: number) =>
      im.mos[order[0]] >= mosSorted[Math.min(N, im.n) - 1] ? 1 : 0;
    a15 += hit(5);
    a110 += hit(10);
    iou += (im.mos[order[0]] - 1) / 4;
  }
  const n = ds.imgs.length;
  return {
    srcc: srcc / n,
    acc1_5: (100 * a15) / n,
    acc1_10: (100 * a110) / n,
    top1: iou / n,
  };
};

const { model: next } = JSON.parse(await readFile(modelPath, 'utf8')) as {
  model: LearnedModel;
};
const shipped: LearnedModel = cnnK
  ? unpackModel(LEARNED_CNN_PACKED.head, next.mean.length)
  : defaultModel();

const val = await load(`gaicd-val${suffix}`);
const hold = fcdbW > 0 ? await load(`fcdb-train${suffix}`, true) : null;
if (next.mean.length !== val.D)
  throw new Error(`model has ${next.mean.length} inputs, dumps ${val.D}`);

const measure = (m: LearnedModel) => {
  const v = evaluate(m, val);
  const h = hold ? evaluate(m, hold) : null;
  return {
    ...v,
    holdoutIoU: h ? h.top1 : null,
    score: v.srcc + (h ? h.top1 : 0),
  };
};
const a = measure(shipped);
const b = measure(next);
const pass = b.score > a.score + margin;

const f = (x: number | null, d = 4) => (x == null ? '-' : x.toFixed(d));
const sgn = (x: number | null, y: number | null, d = 4) =>
  x == null || y == null
    ? '-'
    : `${x - y >= 0 ? '+' : ''}${(x - y).toFixed(d)}`;
const md = [
  `| metric (GAICD val${hold ? ' + FCDB-train holdout' : ''}) | shipped | new | delta |`,
  '|---|---|---|---|',
  `| val SRCC | ${f(a.srcc)} | ${f(b.srcc)} | ${sgn(b.srcc, a.srcc)} |`,
  `| val Acc1/5 | ${f(a.acc1_5, 1)} | ${f(b.acc1_5, 1)} | ${sgn(b.acc1_5, a.acc1_5, 1)} |`,
  `| val Acc1/10 | ${f(a.acc1_10, 1)} | ${f(b.acc1_10, 1)} | ${sgn(b.acc1_10, a.acc1_10, 1)} |`,
  `| holdout IoU | ${f(a.holdoutIoU)} | ${f(b.holdoutIoU)} | ${sgn(b.holdoutIoU, a.holdoutIoU)} |`,
  `| **selection score** | ${f(a.score)} | ${f(b.score)} | ${sgn(b.score, a.score)} |`,
  '',
  `Gate (new score > shipped + ${margin}): **${pass ? 'PASS' : 'FAIL'}**. Validation data only; test splits are not used.`,
].join('\n');
console.log(md);
const rep = opt('report');
if (rep)
  await Bun.write(
    rep,
    JSON.stringify(
      { shipped: a, next: b, margin, pass, cnnK, fcdbW },
      null,
      2
    ) + '\n'
  );
const mdPath = opt('markdown');
if (mdPath) await Bun.write(mdPath, md + '\n');
process.exit(pass ? 0 : 1);
