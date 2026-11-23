// Stage-by-stage comparison: original → tuned pure JS → + saliency map → + faces.
// Runs box-eval for each stage on FCDB test and FLMS and prints a table.
//
// Usage: bun bench/stages.ts [--tuned '{json}'] [--map-weights 4,8,16]
import { join } from 'node:path';

const args = process.argv.slice(2);
const take = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 ? args.splice(i, 2)[1] : fallback;
};
const tuned = JSON.parse(take('tuned', '{}'));
const mapWeights = take('map-weights', '4,8,16').split(',').map(Number);

type Stage = { name: string; options: object; with?: string };
const stages: Stage[] = [
  { name: 'original (2.0.5)', options: {} },
  { name: 'tuned pure JS', options: tuned },
  ...mapWeights.map((w) => ({
    name: `tuned + saliency w${w}`,
    options: { ...tuned, importanceMapWeight: w },
    with: 'saliency',
  })),
  ...mapWeights.map((w) => ({
    name: `tuned + saliency w${w} + faces`,
    options: { ...tuned, importanceMapWeight: w },
    with: 'saliency,faces',
  })),
];

const sets = [
  ['fcdb', 'test'],
  ['flms', 'all'],
] as const;

async function run(dataset: string, split: string, s: Stage) {
  const cmd = [
    'bun',
    join(import.meta.dir, 'box-eval.ts'),
    '--dataset',
    dataset,
    '--split',
    split,
    '--json',
    '--options',
    JSON.stringify(s.options),
    ...(s.with ? ['--with', s.with] : []),
  ];
  const p = Bun.spawn(cmd, { stdout: 'pipe', stderr: 'inherit' });
  const out = await new Response(p.stdout).text();
  if ((await p.exited) !== 0) throw new Error(`failed: ${cmd.join(' ')}`);
  return JSON.parse(out.trim().split('\n').pop()!);
}

const SUB = ['people', 'landscape', 'other'] as const;
const cells = (r: any) =>
  [
    r.smartcrop.iou,
    ...SUB.map((k) => r.subsets?.[k]?.smartcrop.iou ?? '-'),
  ].join(' | ');
const centerCells = (r: any) =>
  [r.center.iou, ...SUB.map((k) => r.subsets?.[k]?.center.iou ?? '-')].join(
    ' | '
  );
const sub = (n: string) => SUB.map((k) => `${n} ${k}`).join(' | ');
const rows: string[] = [
  `| stage | FCDB test IoU | ${sub('FCDB')} | FCDB BDE | FLMS IoU | ${sub('FLMS')} | FLMS BDE |`,
  `|---|---|---|---|---|---|---|---|---|---|---|`,
];
let center: string | undefined;
for (const s of stages) {
  const [f, l] = await Promise.all(sets.map(([d, sp]) => run(d, sp, s)));
  center ??= `| center crop | ${centerCells(f)} | ${f.center.bde} | ${centerCells(l)} | ${l.center.bde} |`;
  const row = `| ${s.name} | ${cells(f)} | ${f.smartcrop.bde} | ${cells(l)} | ${l.smartcrop.bde} |`;
  console.error(row);
  rows.push(row);
}
rows.splice(2, 0, center!);
console.log(rows.join('\n'));
