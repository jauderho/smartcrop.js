// Paired validation of two crop configurations on every held-out corpus:
// FCDB test, FLMS and GAICD test (boxes = top-MOS candidates), ~1,345 images.
// Reports per set, per content subset and pooled: mean IoU of each config,
// the mean paired difference (B - A) with a 95% bootstrap CI, win/tie/loss
// counts, and a two-sided sign-test p-value. Run only after selection.
//
// Usage: bun bench/validate.ts --a '{json}' [--a-with faces,saliency]
//          --b '{json}' [--b-with ...] [--sets fcdb:test,flms:all,gaicd:test]
//          [--label-a name] [--label-b name]
import { join } from 'node:path';
import { loadTags, subsetsOf, SUBSETS } from './subsets';

const args = process.argv.slice(2);
const take = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};
const cfg = (k: 'a' | 'b') => ({
  options: take(k, '{}')!,
  with: take(`${k}-with`, ''),
  label: take(`label-${k}`, k.toUpperCase())!,
});
const A = cfg('a');
const B = cfg('b');
const sets = take('sets', 'fcdb:test,flms:all,gaicd:test')!
  .split(',')
  .map((s) => s.split(':'));
const out = join(import.meta.dir, 'out/validate');

type Per = { image: string; iou: number; bde: number };
async function run(
  dataset: string,
  split: string,
  c: typeof A,
  tag: string
): Promise<Per[]> {
  const file = join(out, `${dataset}-${split}-${tag}.json`);
  const cmd = [
    'bun',
    join(import.meta.dir, 'box-eval.ts'),
    '--dataset',
    dataset,
    '--split',
    split,
    '--json',
    '--options',
    c.options,
    '--per-image',
    file,
    ...(c.with ? ['--with', c.with] : []),
  ];
  const p = Bun.spawn(cmd, { stdout: 'ignore', stderr: 'inherit' });
  if ((await p.exited) !== 0) throw new Error(`failed: ${cmd.join(' ')}`);
  return Bun.file(file).json();
}

const mulberry = (a: number) => () => {
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
// Two-sided sign test (normal approximation with continuity correction).
const signP = (w: number, l: number) => {
  const n = w + l;
  if (!n) return 1;
  const z = (Math.abs(w - l) - 1) / Math.sqrt(n);
  const erf = (x: number) => {
    const t = 1 / (1 + 0.3275911 * x);
    return (
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp(-x * x)
    );
  };
  return Math.max(0, Math.min(1, 1 - erf(Math.max(0, z) / Math.SQRT2)));
};
function stats(pairs: [number, number][]) {
  const n = pairs.length;
  const d = pairs.map(([a, b]) => b - a);
  const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const rnd = mulberry(7);
  const boots: number[] = [];
  for (let r = 0; r < 10000; r++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += d[Math.floor(rnd() * n)];
    boots.push(s / n);
  }
  boots.sort((x, y) => x - y);
  const win = d.filter((x) => x > 1e-4).length;
  const loss = d.filter((x) => x < -1e-4).length;
  return {
    n,
    a: mean(pairs.map((p) => p[0])),
    b: mean(pairs.map((p) => p[1])),
    diff: mean(d),
    lo: boots[250],
    hi: boots[9749],
    win,
    tie: n - win - loss,
    loss,
    p: signP(win, loss),
  };
}
const fmt = (name: string, s: ReturnType<typeof stats>) =>
  `| ${name} | ${s.n} | ${s.a.toFixed(4)} | ${s.b.toFixed(4)} | ${s.diff >= 0 ? '+' : ''}${s.diff.toFixed(4)} [${s.lo.toFixed(4)}, ${s.hi.toFixed(4)}] | ${s.win}/${s.tie}/${s.loss} | ${s.p < 0.001 ? '<0.001' : s.p.toFixed(3)} |`;

const rows: string[] = [
  `A = ${A.label}: ${A.options}${A.with ? ` with ${A.with}` : ''}`,
  `B = ${B.label}: ${B.options}${B.with ? ` with ${B.with}` : ''}`,
  '',
  '| set | n | A IoU | B IoU | B - A [95% CI] | win/tie/loss | sign p |',
  '|---|---|---|---|---|---|---|',
];
const pooled: [number, number][] = [];
const bySubset: Record<string, [number, number][]> = {};
for (const [dataset, split] of sets) {
  const [ra, rb] = await Promise.all([
    run(dataset, split, A, 'a'),
    run(dataset, split, B, 'b'),
  ]);
  const mb = new Map(rb.map((r) => [r.image, r.iou]));
  const pairs = ra
    .filter((r) => mb.has(r.image))
    .map((r) => [r.iou, mb.get(r.image)!] as [number, number]);
  rows.push(fmt(`${dataset} ${split}`, stats(pairs)));
  pooled.push(...pairs);
  await loadTags(dataset);
  for (const r of ra) {
    if (!mb.has(r.image)) continue;
    for (const s of subsetsOf(dataset, r.image))
      (bySubset[s] ??= []).push([r.iou, mb.get(r.image)!]);
  }
}
rows.push(fmt('**pooled**', stats(pooled)));
for (const s of SUBSETS)
  if (bySubset[s]?.length)
    rows.push(fmt(`subset ${s} (fcdb+flms)`, stats(bySubset[s])));
console.log(rows.join('\n'));
