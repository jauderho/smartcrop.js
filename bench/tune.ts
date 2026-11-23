// Tune smartcrop's scoring options against FCDB (train objective) and report
// held-out FCDB test / FLMS / private strip results.
//
// Usage: bun bench/tune.ts [--random 60] [--refine 2] [--refine-starts 3]
//          [--train-limit 500] [--space bench/tune/space.json] [--seed 1]
//          [--concurrency 4] [--top 5] [--out bench/out/tune]
//          [--with faces,saliency] [--objective overall|balanced]
// --with feeds the cached ONNX plug-ins (bench/plugin-cache.ts) into every
// tuned config (not into the original/default baselines); with saliency,
// importanceMapWeight is added to the search space (range 0..8).
// --objective balanced maximises the equal-weight mean of IoU over the
// people / landscape / other subsets (bench/subsets.ts) on FCDB train.
// Strategy: seeded random search, then coordinate descent (--refine rounds,
// step in unit space 0.2/round#) around the best --refine-starts configs.
// Decoded images live in worker shards and are reused across all configs.
// Search space: JSON file, any numeric Options key can be added (see space.json).
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import smartcrop from '../src/smartcrop';
import { parseWith } from './plugin-cache';
import { BALANCED_SUBSETS, loadTags, SUBSETS } from './subsets';
import {
  centerSet,
  keyOf,
  mean,
  neighbours,
  rng,
  sample,
  shuffled,
  summarizeSet,
  type Ann,
  type Config,
  type SetResult,
  type Space,
} from './tune/common';

const args = process.argv.slice(2);
const take = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 ? args.splice(i, 2)[1] : fallback;
};
const nRandom = Number(take('random', '60'));
const nRefine = Number(take('refine', '2'));
const nStarts = Number(take('refine-starts', '3'));
const trainLimit = Number(take('train-limit', '0'));
const spacePath = resolve(
  take('space', join(import.meta.dir, 'tune/space.json'))
);
const seed = Number(take('seed', '1'));
const conc = Math.max(1, Number(take('concurrency', '4')));
const nTop = Number(take('top', '5'));
const withPlugins = parseWith(take('with', ''));
const objective = take('objective', 'overall');
if (objective !== 'overall' && objective !== 'balanced')
  throw new Error(`--objective must be overall|balanced, got ${objective}`);
// Bare names (no slash) go under bench/out/tune/<name>.
const outArg = take('out', '');
const outDir = !outArg
  ? join(import.meta.dir, 'out/tune')
  : outArg.includes('/')
    ? resolve(outArg)
    : join(import.meta.dir, 'out/tune', outArg);
mkdirSync(outDir, { recursive: true });
const space: Space = await Bun.file(spacePath).json();
if (withPlugins.has('saliency') && !space.params.importanceMapWeight)
  space.params.importanceMapWeight = { type: 'float', min: 0, max: 8 };
const runsPath = join(outDir, 'runs.jsonl');
writeFileSync(runsPath, '');

// ---- worker pool ----------------------------------------------------------
const workers = Array.from(
  { length: conc },
  () => new Worker(new URL('./tune/worker.ts', import.meta.url))
);
let msgId = 0;
const call = (w: Worker, msg: object) =>
  new Promise<any>((res, rej) => {
    const id = ++msgId;
    const onMsg = (e: MessageEvent) => {
      if (e.data.id !== id) return;
      w.removeEventListener('message', onMsg);
      res(e.data);
    };
    w.addEventListener('message', onMsg);
    w.addEventListener('error', rej, { once: true });
    w.postMessage({ id, ...msg });
  });
const setInfo = new Map<string, { dataset: string; images: string[] }>();
async function loadSet(set: string, dataset: string, anns: Ann[]) {
  const dir = join(import.meta.dir, 'datasets', dataset);
  const shards = workers.map((_, k) => anns.filter((_, i) => i % conc === k));
  const r = await Promise.all(
    workers.map((w, k) =>
      call(w, {
        type: 'load',
        set,
        dir,
        dataset,
        use: [...withPlugins],
        anns: shards[k],
      })
    )
  );
  setInfo.set(set, { dataset, images: r.flatMap((x) => x.images as string[]) });
  const skipped = r.reduce((s, x) => s + x.skipped, 0);
  if (skipped) console.error(`${set}: skipped ${skipped} images`);
}
// plain: evaluate without the plug-ins (original / default baselines).
async function evalSet(
  set: string,
  options: Config,
  plain = false
): Promise<SetResult> {
  const t0 = performance.now();
  const r = await Promise.all(
    workers.map((w) =>
      call(w, { type: 'eval', set, options, noPlugins: plain })
    )
  );
  const all = r.flatMap((x) => Array.from(x.res as Float64Array));
  const iou: number[] = [];
  const bde: number[] = [];
  for (let i = 0; i < all.length; i += 2) {
    iou.push(all[i]);
    bde.push(all[i + 1]);
  }
  const info = setInfo.get(set)!;
  return summarizeSet(
    info.dataset,
    info.images,
    iou,
    bde,
    (performance.now() - t0) / 1000
  );
}

// ---- data -----------------------------------------------------------------
const annsOf = async (d: string) =>
  (await Bun.file(
    join(import.meta.dir, 'datasets', d, 'annotations.json')
  ).json()) as Ann[];
await loadTags('fcdb');
await loadTags('flms');
const fcdb = await annsOf('fcdb');
let trainAnns = fcdb.filter((a) => a.split === 'train');
if (trainLimit > 0 && trainLimit < trainAnns.length) {
  trainAnns = shuffled(trainAnns, rng(seed ^ 0x5eed)).slice(0, trainLimit);
}
const testAnns = fcdb.filter((a) => a.split === 'test');
const flmsAnns = await annsOf('flms');
console.error(`loading ${trainAnns.length} train images on ${conc} workers...`);
let t0 = performance.now();
await loadSet('train', 'fcdb', trainAnns);
console.error(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

// ---- search ---------------------------------------------------------------
type Run = {
  config: Config;
  iou: number;
  bde: number;
  score: number; // the objective: overall IoU or balanced subset IoU
  res: SetResult;
  secs: number;
  phase: string;
};
const scoreOf = (r: SetResult) => (objective === 'balanced' ? r.bal : r.iou);
const runs = new Map<string, Run>();
const defaultCfg: Config = { minScale: 0.5, maxScale: 1 };
const libDefaultCfg: Config = { minScale: 1, maxScale: 1 };
const startedAt = performance.now();
let done = 0;
let planned = 1 + nRandom;
let best = 0;
async function run(config: Config, phase: string, plain = false): Promise<Run> {
  const key = keyOf(config) + (plain ? '#plain' : '');
  const hit = runs.get(key);
  if (hit) return hit;
  const r = await evalSet('train', config, plain);
  const score = scoreOf(r);
  const rec: Run = {
    config,
    iou: r.iou,
    bde: r.bde,
    score,
    res: r,
    secs: r.secs,
    phase,
  };
  runs.set(key, rec);
  done++;
  if (phase !== 'default') best = Math.max(best, score);
  appendFileSync(
    runsPath,
    JSON.stringify({
      phase,
      config,
      trainIou: +r.iou.toFixed(5),
      trainBde: +r.bde.toFixed(5),
      score: +score.toFixed(5),
      subsets: Object.fromEntries(
        SUBSETS.map((s) => [
          s,
          { n: r.sub[s].n, iou: +r.sub[s].iou.toFixed(4) },
        ])
      ),
      trainImages: r.n,
      secs: +r.secs.toFixed(2),
    }) + '\n'
  );
  const elapsed = (performance.now() - startedAt) / 1000;
  const eta = (elapsed / done) * Math.max(0, planned - done);
  console.error(
    `[${done}/~${planned}] ${phase.padEnd(7)} ${objective} ${score.toFixed(4)} (iou ${r.iou.toFixed(4)}) best ${best.toFixed(4)} ` +
      `${r.secs.toFixed(1)}s/cfg ETA ${(eta / 60).toFixed(1)}m`
  );
  return rec;
}

const def = await run(defaultCfg, 'default', true);
const libDef = await run(libDefaultCfg, 'default', true);
planned += 1;
const rand = rng(seed);
for (let i = 0; i < nRandom; i++) await run(sample(space, rand), 'random');

const ranked = () =>
  [...runs.values()]
    .filter((r) => r.phase !== 'default')
    .sort((a, b) => b.score - a.score);
const refineKeys = Object.keys(space.params).filter(
  (k) => space.params[k].type !== 'fixed'
);
const starts = ranked().slice(0, nStarts);
planned += nRefine * starts.length * refineKeys.length * 2;
for (let r = 1; r <= nRefine; r++) {
  const delta = 0.2 / r;
  for (let s = 0; s < starts.length; s++) {
    let cur = starts[s];
    for (const key of refineKeys) {
      let bestN = cur;
      for (const c of neighbours(space, cur.config, key, delta)) {
        const rr = await run(c, `refine${r}`);
        if (rr.score > bestN.score) bestN = rr;
      }
      cur = bestN;
    }
    starts[s] = cur;
  }
}
console.error(
  `search done: ${done} configs in ${((performance.now() - startedAt) / 60000).toFixed(1)}m`
);

// ---- held-out -------------------------------------------------------------
type Row = {
  name: string;
  config: Config | null;
  train?: Run;
  test: SetResult;
  flms: SetResult;
  strip: any;
};
const top = ranked().slice(0, nTop);
const entries: { name: string; config: Config | null; train?: Run }[] = [
  { name: 'center', config: null },
  { name: 'default (minScale 0.5)', config: defaultCfg, train: def },
  { name: 'libdefault (minScale 1)', config: libDefaultCfg, train: libDef },
  ...top.map((t, i) => ({ name: `top${i + 1}`, config: t.config, train: t })),
];

await loadSet('test', 'fcdb', testAnns);
await loadSet('flms', 'flms', flmsAnns);

// Strip set (strip-eval.ts pipeline), prepared once. The hand-set labels are
// NOT used as truth: only the photo ids and aspect are read, and strips are
// reported as the change in Y position versus the original (default) config.
type Labels = {
  imageDir: string;
  aspect: number;
  truth: Record<string, number>;
  softTruth?: Record<string, number>;
  unreviewed?: string[];
};
const labelsFile = Bun.file(
  join(import.meta.dir, 'private/jauderho-strip.json')
);
const labels: Labels | null = (await labelsFile.exists())
  ? await labelsFile.json()
  : null;
const STRIP_W = 512;
const strips: { id: string; canvas: { height: number } }[] = [];
if (labels) {
  const ids = [
    ...Object.keys(labels.truth),
    ...Object.keys(labels.softTruth ?? {}),
  ];
  for (const id of ids) {
    try {
      const img = await loadImage(join(labels.imageDir, `${id}.jpg`));
      let sw = img.width;
      let sh = Math.round((img.width * 2) / 3);
      if (sh > img.height) {
        sh = img.height;
        sw = Math.round((img.height * 3) / 2);
      }
      const sx = Math.round((img.width - sw) / 2);
      const sy = Math.round((img.height - sh) / 2);
      const h = Math.round((STRIP_W * sh) / sw);
      const canvas = createCanvas(STRIP_W, h);
      canvas.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, STRIP_W, h);
      strips.push({ id, canvas });
    } catch {
      console.error(`strip: cannot load ${id}`);
    }
  }
}
const canvasFactory = (w: number, h: number) => createCanvas(w, h);
// Y position (CSS object-position %) per strip photo; null config = centred.
async function stripPositions(config: Config | null): Promise<number[]> {
  const out: number[] = [];
  for (const s of strips) {
    let pred = 50;
    if (config) {
      const stripH = Math.round(STRIP_W / labels!.aspect);
      const { topCrop } = await smartcrop.crop(s.canvas as never, {
        width: STRIP_W,
        height: stripH,
        minScale: 1,
        maxScale: 1,
        canvasFactory,
        ...config,
      });
      const slack = s.canvas.height - topCrop.height;
      pred = slack > 0 ? (100 * topCrop.y) / slack : 50;
    }
    out.push(pred);
  }
  return out;
}
const origStrip = strips.length ? await stripPositions(defaultCfg) : [];
async function stripChange(config: Config | null) {
  if (!strips.length) return null;
  const d = (await stripPositions(config)).map((p, i) =>
    Math.abs(p - origStrip[i])
  );
  return {
    n: d.length,
    meanChange: mean(d),
    maxChange: Math.max(...d),
    moved5: d.filter((x) => x > 5).length,
  };
}

const slim = (r: SetResult) => ({
  n: r.n,
  iou: +r.iou.toFixed(5),
  bde: +r.bde.toFixed(5),
  bal: +r.bal.toFixed(5),
  sub: r.sub,
});
const rows: Row[] = [];
for (const e of entries) {
  const plain = e.name !== 'center' && !e.name.startsWith('top');
  const test = e.config
    ? await evalSet('test', e.config, plain)
    : centerSet('fcdb', testAnns);
  const flms = e.config
    ? await evalSet('flms', e.config, plain)
    : centerSet('flms', flmsAnns);
  const strip = await stripChange(e.config);
  rows.push({ ...e, test, flms, strip });
  appendFileSync(
    runsPath,
    JSON.stringify({
      phase: 'heldout',
      name: e.name,
      config: e.config,
      test: slim(test),
      flms: slim(flms),
      strip,
    }) + '\n'
  );
  console.error(
    `held-out ${e.name}: test ${test.iou.toFixed(4)} flms ${flms.iou.toFixed(4)}`
  );
}

// Ablation: top config with each differing term reset to the default config.
const ablation: {
  term: string;
  from: unknown;
  to: unknown;
  test: SetResult;
  flms: SetResult;
}[] = [];
if (top.length) {
  const defs = {
    ...(smartcrop as any).DEFAULTS,
    ...(withPlugins.has('saliency') ? { importanceMapWeight: 0 } : {}),
    ...defaultCfg,
  } as Config;
  for (const [k, v] of Object.entries(top[0].config)) {
    if (!(k in defs) || JSON.stringify(defs[k]) === JSON.stringify(v)) continue;
    const cfg = { ...top[0].config, [k]: defs[k] };
    const test = await evalSet('test', cfg);
    const flms = await evalSet('flms', cfg);
    ablation.push({ term: k, from: v, to: defs[k], test, flms });
    appendFileSync(
      runsPath,
      JSON.stringify({
        phase: 'ablation',
        term: k,
        config: cfg,
        test: slim(test),
        flms: slim(flms),
      }) + '\n'
    );
    console.error(`ablation ${k}: test ${test.iou.toFixed(4)}`);
  }
}
workers.forEach((w) => w.terminate());

// ---- summary --------------------------------------------------------------
const f4 = (x: number) => x.toFixed(4);
const trainCenter = centerSet('fcdb', trainAnns);
const stripCell = (s: any) =>
  s
    ? `${s.meanChange.toFixed(1)} / ${s.maxChange.toFixed(1)} / ${s.moved5} of ${s.n}`
    : 'n/a';
const subHead = SUBSETS.join(' | ');
const subCells = (r: SetResult) =>
  SUBSETS.map((s) => (r.sub[s].n ? f4(r.sub[s].iou) : '-')).join(' | ');
const subCounts = (r: SetResult) =>
  SUBSETS.map((s) => `${s} ${r.sub[s].n}`).join(', ');
const md: string[] = [];
md.push(`# smartcrop tuning summary`, '');
md.push(
  `Search: ${nRandom} random + ${nRefine} refine round(s) x ${starts.length} starts, seed ${seed}, ` +
    `${done} configs evaluated on FCDB train (${trainAnns.length} images${trainLimit ? `, subsampled` : ''}), ` +
    `${((performance.now() - startedAt) / 60000).toFixed(1)} min total.`,
  `Objective: ${objective}${objective === 'balanced' ? ` (equal-weight mean of IoU over ${BALANCED_SUBSETS.join(', ')})` : ''}. ` +
    `Plug-ins: ${withPlugins.size ? [...withPlugins].join(', ') : 'none'} (applied to tuned configs only; center/default/libdefault are pure JS).`,
  ''
);
md.push(
  'Per-subset columns are IoU over the images in each content subset (rules in bench/subsets.ts; subsets overlap, untagged images are only in overall). balanced = mean of people, landscape, other.',
  ''
);
md.push('## Train (FCDB train; the optimisation objective)', '');
md.push(`Subset sizes: ${subCounts(trainCenter)}`, '');
md.push(
  `| config | overall IoU | BDE | balanced | ${subHead} |`,
  `|---|---|---|---|${SUBSETS.map(() => '---').join('|')}|`
);
md.push(
  `| center | ${f4(trainCenter.iou)} | ${f4(trainCenter.bde)} | ${f4(trainCenter.bal)} | ${subCells(trainCenter)} |`
);
for (const r of rows.filter((r) => r.train))
  md.push(
    `| ${r.name} | ${f4(r.train!.iou)} | ${f4(r.train!.bde)} | ${f4(r.train!.res.bal)} | ${subCells(r.train!.res)} |`
  );
md.push('', '## Held-out (never used for selection)', '');
md.push(
  'Strip column (private photos, "change vs original"): mean / max absolute change in Y position (percentage points) versus the original default config, and number of photos moved by more than 5 points. Hand-set labels are not used.',
  ''
);
for (const [title, pick] of [
  ['FCDB test', (r: Row) => r.test],
  ['FLMS', (r: Row) => r.flms],
] as const) {
  md.push(`### ${title}`, '', `Subset sizes: ${subCounts(pick(rows[0]))}`, '');
  md.push(
    `| config | overall IoU | BDE | balanced | ${subHead} |${title === 'FCDB test' ? ' strip change vs original (mean / max / moved>5) |' : ''}`,
    `|---|---|---|---|${SUBSETS.map(() => '---').join('|')}|${title === 'FCDB test' ? '---|' : ''}`
  );
  for (const r of rows) {
    const x = pick(r);
    md.push(
      `| ${r.name} | ${f4(x.iou)} | ${f4(x.bde)} | ${f4(x.bal)} | ${subCells(x)} |${title === 'FCDB test' ? ` ${stripCell(r.strip)} |` : ''}`
    );
  }
  md.push('');
}
md.push('## Ablation of top1 (each term reset to default, others kept)', '');
md.push(
  `| term (top1 -> default) | test IoU | delta | test bal | delta | FLMS IoU | delta | FLMS bal | delta | test ${SUBSETS.join(' / ')} | FLMS ${SUBSETS.join(' / ')} |`,
  `|---|---|---|---|---|---|---|---|---|---|---|`
);
const t1 = rows.find((r) => r.name === 'top1');
const d4 = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(4);
const subList = (r: SetResult, base: SetResult) =>
  SUBSETS.map((s) =>
    r.sub[s].n ? d4(r.sub[s].iou - base.sub[s].iou) : '-'
  ).join(' / ');
if (t1)
  for (const a of ablation)
    md.push(
      `| ${a.term} (${JSON.stringify(a.from)} -> ${JSON.stringify(a.to)}) | ${f4(a.test.iou)} | ${d4(a.test.iou - t1.test.iou)} | ${f4(a.test.bal)} | ${d4(a.test.bal - t1.test.bal)} | ${f4(a.flms.iou)} | ${d4(a.flms.iou - t1.flms.iou)} | ${f4(a.flms.bal)} | ${d4(a.flms.bal - t1.flms.bal)} | ${subList(a.test, t1.test)} | ${subList(a.flms, t1.flms)} |`
    );
md.push('', 'Ablation subset columns are IoU deltas versus top1.', '');
md.push('## Configs', '');
for (const r of rows.filter((r) => r.config))
  md.push(`- ${r.name}: \`${JSON.stringify(r.config)}\``);
md.push('');
writeFileSync(join(outDir, 'summary.md'), md.join('\n'));
console.log(md.join('\n'));
