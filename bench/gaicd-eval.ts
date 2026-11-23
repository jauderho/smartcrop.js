// Standard GAICD ranking evaluation for a pluggable candidate scorer.
//
// For each test image the scorer scores every annotated candidate crop; the
// predicted scores are compared with the human MOS (GAIC, Zeng et al.).
// Metrics, per image then averaged over images:
//   SRCC, PCC     Spearman (average ranks for ties) / Pearson of score vs MOS
//   Acc{k}/{N}    of the top-k predicted crops (k = 1..4), the fraction among
//                 the N best-MOS crops (N = 5, 10). Ties in MOS rank are
//                 broken by including every crop tied with the N-th best.
//                 Reported x100. AccN is the mean over k = 1..4.
// Predicted ties are broken by candidate order (stable sort).
//
// A scorer module exports
//   async function scoreCandidates(imagePath, width, height,
//     candidates: [x1, y1, x2, y2][], options): Promise<number[]>
// with candidates in the annotation width/height pixel space (x2/y2
// exclusive), higher score = better crop.
//
// Data: `bun scripts/fetch-datasets.ts gaicd` (bench/datasets/gaicd/).
//
// Usage: bun bench/gaicd-eval.ts [--scorer path] [--split test|val]
//          [--limit N] [--options '{json}'] [--json]
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

type Entry = {
  image: string;
  width: number;
  height: number;
  split: string;
  candidates: [number, number, number, number, number][]; // x y w h mos
};
type Scorer = (
  imagePath: string,
  width: number,
  height: number,
  candidates: [number, number, number, number][],
  options: Record<string, unknown>,
) => Promise<number[]>;

const args = process.argv.slice(2);
const opt = (n: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : undefined;
};
const scorerPath = resolve(opt("scorer") ?? "bench/gaicd-scorers/smartcrop.ts");
const split = opt("split") ?? "test";
const limit = Number(opt("limit")) || Infinity;
const options = JSON.parse(opt("options") ?? "{}") as Record<string, unknown>;
const asJson = args.includes("--json");
if (split !== "test" && split !== "val") {
  throw new Error("--split must be test or val");
}

const DIR = resolve(import.meta.dir, "datasets/gaicd");

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / (a.length || 1);

// Average ranks (1-based), ties share the mean rank.
function ranks(v: number[]): number[] {
  const idx = v.map((_, i) => i).sort((a, b) => v[a] - v[b]);
  const r = new Array<number>(v.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}

function pearson(a: number[], b: number[]): number {
  const ma = mean(a);
  const mb = mean(b);
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  // Constant input: correlation undefined, count as 0.
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

// Set of crops among the n best by MOS, including ties with the n-th.
function topSet(mos: number[], n: number): Set<number> {
  const sorted = [...mos].sort((a, b) => b - a);
  const cut = sorted[Math.min(n, sorted.length) - 1];
  const s = new Set<number>();
  mos.forEach((m, i) => m >= cut && s.add(i));
  return s;
}

function accs(pred: number[], mos: number[], n: number): number[] {
  const best = topSet(mos, n);
  const order = pred
    .map((_, i) => i)
    .sort((a, b) => pred[b] - pred[a] || a - b);
  const out: number[] = [];
  for (let k = 1; k <= 4; k++) {
    out.push(order.slice(0, k).filter((i) => best.has(i)).length / k);
  }
  return out;
}

const { scoreCandidates } = (await import(scorerPath)) as {
  scoreCandidates: Scorer;
};
const entries = (
  JSON.parse(await readFile(join(DIR, "candidates.json"), "utf8")) as Entry[]
)
  .filter((e) => e.split === split)
  .slice(0, limit);

const acc5: number[][] = [[], [], [], []];
const acc10: number[][] = [[], [], [], []];
const srcc: number[] = [];
const pcc: number[] = [];
const t0 = performance.now();
for (const [n, e] of entries.entries()) {
  const boxes = e.candidates.map(
    ([x, y, w, h]) => [x, y, x + w, y + h] as [number, number, number, number],
  );
  const mos = e.candidates.map((c) => c[4]);
  const pred = await scoreCandidates(
    join(DIR, e.image),
    e.width,
    e.height,
    boxes,
    options,
  );
  if (pred.length !== mos.length || pred.some((v) => !Number.isFinite(v))) {
    throw new Error(`bad scores for ${e.image}`);
  }
  srcc.push(pearson(ranks(pred), ranks(mos)));
  pcc.push(pearson(pred, mos));
  accs(pred, mos, 5).forEach((v, k) => acc5[k].push(v));
  accs(pred, mos, 10).forEach((v, k) => acc10[k].push(v));
  if (!asJson && (n + 1) % 100 === 0)
    console.error(`${n + 1}/${entries.length}`);
}

const pct = (a: number[]) => mean(a) * 100;
const result = {
  scorer: scorerPath.replace(resolve(".") + "/", ""),
  split,
  images: entries.length,
  options,
  srcc: mean(srcc),
  pcc: mean(pcc),
  acc5: acc5.map(pct),
  acc5Mean: mean(acc5.map(pct)),
  acc10: acc10.map(pct),
  acc10Mean: mean(acc10.map(pct)),
  seconds: (performance.now() - t0) / 1000,
};

if (asJson) {
  console.log(JSON.stringify(result, null, 2));
} else {
  const f = (v: number) => v.toFixed(2);
  console.log(
    `${result.scorer}  split=${split}  images=${result.images}  ${result.seconds.toFixed(1)}s`,
  );
  console.log(`SRCC ${result.srcc.toFixed(4)}  PCC ${result.pcc.toFixed(4)}`);
  console.log(
    `Acc5   ${result.acc5.map(f).join("  ")}  mean ${f(result.acc5Mean)}   (k=1..4 of best 5)`,
  );
  console.log(
    `Acc10  ${result.acc10.map(f).join("  ")}  mean ${f(result.acc10Mean)}   (k=1..4 of best 10)`,
  );
}
