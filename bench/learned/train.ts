// Train the learned composition scorer with a pairwise ranking (RankNet) loss.
//
// Training pairs are drawn within an image: crop i is rated higher than crop j
// when target_i - target_j > 0.25 (GAICD MOS, or 1 + 4 * IoU for FCDB windows).
// Loss: log(1 + exp(-(s_i - s_j))) + L2. Optimiser: Adam. Hyper-parameters are
// chosen on the GAICD val split by mean SRCC (optionally plus FCDB holdout IoU);
// test is never used here (use bench/gaicd-eval.ts).
//
// Usage: bun bench/learned/train.ts [--sets gaicd-train[,more]] [--val gaicd-val]
//          [--fcdb 0.5] [--hidden 32,32] [--l2 1e-3] [--lr 3e-3] [--epochs 40]
//          [--ppi 64] [--seed 1] [--grid] [--write]
// --fcdb w mixes fcdb-train windows into the loss with weight w; the last 15%
// of fcdb-train images are held out to report the IoU of the top-scored window.
// --grid runs a small hyper-parameter grid and (with --write) writes the best.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { packModel, type LearnedModel } from "../../src/learned";

const args = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};
const OUT = resolve(import.meta.dir, "../out/learned");

type DataSet = {
  D: number;
  X: Float32Array; // standardised, N x D
  imgs: { off: number; n: number; mos: number[] }[];
};

async function load(name: string) {
  const m = JSON.parse(await readFile(`${OUT}/${name}.json`, "utf8"));
  const X = new Float32Array(
    await Bun.file(`${OUT}/${name}.bin`).arrayBuffer(),
  );
  return { D: m.D as number, X, imgs: m.images as DataSet["imgs"] };
}

const mulberry = (a: number) => () => {
  a |= 0;
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const sets = (opt("sets", "gaicd-train") as string).split(",");
const valNames = (opt("val", "gaicd-val") as string).split(",");
const valName = valNames[0];
const fcdbW = Number(opt("fcdb", "0"));
const ppi = Number(opt("ppi", "64"));
const epochs = Number(opt("epochs", "40"));

const train = await Promise.all(sets.map(load));
const vals = await Promise.all(valNames.map(load));
const val = vals[0];
const D = train[0].D;
for (const d of [...train, ...vals])
  if (d.D !== D) throw new Error("feature count mismatch; re-run dump.ts");

const fcNames = (opt("fcdbsets", "fcdb-train") as string).split(",");
let fcs: DataSet[] = [];
let fcHolds: DataSet[] = [];
if (fcdbW > 0) {
  for (const n of fcNames) {
    const raw = await load(n);
    if (raw.D !== D)
      throw new Error(`${n} feature count mismatch; re-run dump.ts`);
    const cut = Math.floor(raw.imgs.length * 0.85);
    fcs.push({ ...raw, imgs: raw.imgs.slice(0, cut) });
    fcHolds.push({ ...raw, imgs: raw.imgs.slice(cut) });
  }
}

// Standardisation statistics from the GAICD training features.
const mean = new Float64Array(D);
const std = new Float64Array(D);
{
  let n = 0;
  for (const t of train) {
    for (let r = 0; r < t.X.length / D; r++, n++) {
      for (let k = 0; k < D; k++) mean[k] += t.X[r * D + k];
    }
  }
  for (let k = 0; k < D; k++) mean[k] /= n;
  for (const t of train) {
    for (let r = 0; r < t.X.length / D; r++) {
      for (let k = 0; k < D; k++) std[k] += (t.X[r * D + k] - mean[k]) ** 2;
    }
  }
  for (let k = 0; k < D; k++) {
    std[k] = Math.sqrt(std[k] / n);
    if (std[k] < 1e-6) std[k] = 1; // constant (absent map): weight stays ~0
  }
}
const standardise = (ds: DataSet) => {
  const X = ds.X;
  const n = X.length / D;
  for (let r = 0; r < n; r++) {
    for (let k = 0; k < D; k++) {
      X[r * D + k] = Math.max(
        -6,
        Math.min(6, (X[r * D + k] - mean[k]) / std[k]),
      );
    }
  }
};
[...train, ...vals, ...fcs].forEach(standardise);
// --drop <regex>: ablation, zero every feature whose name matches.
const dropRe = opt("drop");
if (dropRe) {
  const names: string[] = JSON.parse(
    await readFile(`${OUT}/${valName}.json`, "utf8"),
  ).names;
  const cols = names
    .map((n, k) => (new RegExp(dropRe).test(n) ? k : -1))
    .filter((k) => k >= 0);
  console.log(`dropping ${cols.length} features matching /${dropRe}/`);
  for (const ds of [...train, ...vals, ...fcs]) {
    for (let r = 0; r < ds.X.length / D; r++)
      for (const k of cols) ds.X[r * D + k] = 0;
  }
}
// fcHold shares fc's buffer (already standardised).

// ---- model -----------------------------------------------------------------
// Dense net D -> hidden... -> 1 with ReLU (hidden = [] is a linear model).
type Net = { sizes: number[]; W: Float64Array[]; B: Float64Array[] };
const newNet = (hidden: number[], rnd: () => number): Net => {
  const sizes = [D, ...hidden, 1];
  const W: Float64Array[] = [];
  const B: Float64Array[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const lim = Math.sqrt(hidden.length ? 6 / sizes[l] : 3 / D);
    W.push(
      new Float64Array(sizes[l] * sizes[l + 1]).map(
        () =>
          (rnd() * 2 - 1) *
          lim *
          (l === sizes.length - 2 && hidden.length ? 0.3 : 1),
      ),
    );
    B.push(new Float64Array(sizes[l + 1]));
  }
  return { sizes, W, B };
};
const newActs = (net: Net) => net.sizes.map((n) => new Float64Array(n));

const forward = (
  net: Net,
  X: Float32Array,
  r: number,
  acts: Float64Array[],
): number => {
  const o = r * D;
  const x0 = acts[0];
  for (let k = 0; k < D; k++) x0[k] = X[o + k];
  const L = net.W.length;
  for (let l = 0; l < L; l++) {
    const n = net.sizes[l];
    const m = net.sizes[l + 1];
    const w = net.W[l];
    const x = acts[l];
    const y = acts[l + 1];
    for (let j = 0; j < m; j++) {
      let a = net.B[l][j];
      const jo = j * n;
      for (let i = 0; i < n; i++) a += w[jo + i] * x[i];
      y[j] = l < L - 1 && a < 0 ? 0 : a;
    }
  }
  return acts[L][0];
};
const score1 = (net: Net, X: Float32Array, r: number) =>
  forward(net, X, r, SCRATCH(net));
let scratch: Float64Array[] | null = null;
const SCRATCH = (net: Net) => (scratch ??= newActs(net));

// ---- metrics ---------------------------------------------------------------
const ranks = (v: number[]) => {
  const idx = v.map((_, i) => i).sort((a, b) => v[a] - v[b]);
  const r = new Array(v.length);
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

function evaluate(p: Net | Net[], ds: DataSet) {
  const ps = Array.isArray(p) ? p : [p];
  let srcc = 0,
    a15 = 0,
    a110 = 0,
    a5 = 0,
    a10 = 0,
    iou = 0;
  for (const im of ds.imgs) {
    const s: number[] = [];
    scratch = newActs(ps[0]);
    for (let i = 0; i < im.n; i++) {
      let v = 0;
      for (const q of ps) v += score1(q, ds.X, im.off + i);
      s.push(v / ps.length);
    }
    srcc += pearson(ranks(s), ranks(im.mos));
    const order = s.map((_, i) => i).sort((a, b) => s[b] - s[a]); // stable
    const mosSorted = [...im.mos].sort((a, b) => b - a);
    const acc = (k: number, N: number) => {
      const thr = mosSorted[Math.min(N, im.n) - 1];
      let hit = 0;
      for (let i = 0; i < k; i++) if (im.mos[order[i]] >= thr) hit++;
      return hit / k;
    };
    a15 += acc(1, 5);
    a110 += acc(1, 10);
    a5 += (acc(1, 5) + acc(2, 5) + acc(3, 5) + acc(4, 5)) / 4;
    a10 += (acc(1, 10) + acc(2, 10) + acc(3, 10) + acc(4, 10)) / 4;
    iou += (im.mos[order[0]] - 1) / 4;
  }
  const n = ds.imgs.length;
  return {
    srcc: srcc / n,
    acc1_5: (100 * a15) / n,
    acc1_10: (100 * a110) / n,
    acc5: (100 * a5) / n,
    acc10: (100 * a10) / n,
    top1: iou / n, // for FCDB windows: IoU of the top-scored window
  };
}

// Mean of the metrics over the validation variants (plain / +saliency / ...).
function evalVal(p: Net | Net[]) {
  const rs = vals.map((v) => evaluate(p, v));
  const avg = (k: keyof (typeof rs)[0]) =>
    rs.reduce((s, r) => s + r[k], 0) / rs.length;
  return {
    srcc: avg("srcc"),
    acc1_5: avg("acc1_5"),
    acc1_10: avg("acc1_10"),
    acc5: avg("acc5"),
    acc10: avg("acc10"),
    top1: avg("top1"),
    per: rs.map((r) => +r.srcc.toFixed(4)),
  };
}

// ---- training --------------------------------------------------------------
type Cfg = { hidden: number[]; l2: number; lr: number; seed: number };

function fit(cfg: Cfg, verbose: boolean) {
  const rnd = mulberry(cfg.seed);
  const net = newNet(cfg.hidden, rnd);
  const L = net.W.length;
  const tensors = [...net.W, ...net.B];
  const m = tensors.map((t) => new Float64Array(t.length));
  const v = tensors.map((t) => new Float64Array(t.length));
  const g = tensors.map((t) => new Float64Array(t.length));
  let step = 0;

  type Item = { ds: DataSet; im: DataSet["imgs"][number]; wt: number };
  const items: Item[] = [];
  for (const ds of train)
    for (const im of ds.imgs) items.push({ ds, im, wt: 1 });
  for (const ds of fcs)
    for (const im of ds.imgs) items.push({ ds, im, wt: fcdbW });

  const actA = newActs(net);
  const actB = newActs(net);
  const deltas = net.sizes.map((n) => new Float64Array(n));
  const backward = (acts: Float64Array[], c: number) => {
    deltas[L][0] = c;
    for (let l = L - 1; l >= 0; l--) {
      const n = net.sizes[l];
      const mm = net.sizes[l + 1];
      const w = net.W[l];
      const gw = g[l];
      const gb = g[L + l];
      const x = acts[l];
      const dOut = deltas[l + 1];
      const dIn = deltas[l];
      if (l > 0) dIn.fill(0);
      for (let j = 0; j < mm; j++) {
        const d = dOut[j];
        if (d === 0) continue;
        gb[j] += d;
        const jo = j * n;
        for (let i = 0; i < n; i++) gw[jo + i] += d * x[i];
        if (l > 0) for (let i = 0; i < n; i++) dIn[i] += d * w[jo + i];
      }
      if (l > 0) for (let i = 0; i < n; i++) if (x[i] <= 0) dIn[i] = 0;
    }
  };

  const adam = (nPairs: number) => {
    step++;
    const b1c = 1 - 0.9 ** step;
    const b2c = 1 - 0.999 ** step;
    for (let t = 0; t < tensors.length; t++) {
      const decay = t < L ? cfg.l2 : 0;
      const T = tensors[t];
      for (let k = 0; k < T.length; k++) {
        const gr = g[t][k] / nPairs + decay * T[k];
        m[t][k] = 0.9 * m[t][k] + 0.1 * gr;
        v[t][k] = 0.999 * v[t][k] + 0.001 * gr * gr;
        T[k] -= (cfg.lr * (m[t][k] / b1c)) / (Math.sqrt(v[t][k] / b2c) + 1e-8);
        g[t][k] = 0;
      }
    }
  };

  let best = { srcc: -2 } as any;
  let bestNet: Net | null = null;
  let bestEpoch = 0;
  const BATCH = 512;
  const lr0 = cfg.lr;
  for (let ep = 0; ep < epochs; ep++) {
    // Linear decay of the learning rate in the second half.
    cfg.lr = ep >= epochs / 2 ? (lr0 * (epochs - ep)) / (epochs / 2) : lr0;
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    let inBatch = 0;
    let loss = 0;
    let cnt = 0;
    for (const { ds, im, wt } of items) {
      for (let q = 0; q < ppi; q++) {
        let a = 0,
          b = 0,
          tries = 0;
        do {
          a = Math.floor(rnd() * im.n);
          b = Math.floor(rnd() * im.n);
          tries++;
        } while (Math.abs(im.mos[a] - im.mos[b]) <= 0.25 && tries < 30);
        if (Math.abs(im.mos[a] - im.mos[b]) <= 0.25) continue;
        if (im.mos[a] < im.mos[b]) [a, b] = [b, a];
        const sa = forward(net, ds.X, im.off + a, actA);
        const sb = forward(net, ds.X, im.off + b, actB);
        const d = sa - sb;
        const sig = 1 / (1 + Math.exp(d)); // sigma(-d)
        loss += Math.log1p(Math.exp(-d));
        cnt++;
        backward(actA, -sig * wt);
        backward(actB, sig * wt);
        if (++inBatch === BATCH) {
          adam(BATCH);
          inBatch = 0;
        }
      }
    }
    if (inBatch) adam(inBatch);
    const r = evalVal(net);
    const h = fcHolds.length
      ? {
          top1:
            fcHolds.reduce((s, d) => s + evaluate(net, d).top1, 0) /
            fcHolds.length,
        }
      : null;
    const score = r.srcc + (h ? h.top1 : 0);
    if (verbose) {
      console.log(
        `ep ${ep} loss ${(loss / cnt).toFixed(4)} val srcc ${r.srcc.toFixed(4)} [${r.per}] acc1/5 ${r.acc1_5.toFixed(1)} acc1/10 ${r.acc1_10.toFixed(1)}` +
          (h ? ` fcdb-holdout IoU ${h.top1.toFixed(4)}` : ""),
      );
    }
    if (!bestNet || score > best.score) {
      best = { ...r, score, hold: h };
      bestNet = {
        sizes: net.sizes,
        W: net.W.map((x) => x.slice()),
        B: net.B.map((x) => x.slice()),
      };
      bestEpoch = ep;
    }
  }
  cfg.lr = lr0;
  return { p: bestNet!, best, epoch: bestEpoch };
}

// ---- output ----------------------------------------------------------------
// Full precision is kept; bench/learned/shrink.ts picks the smallest encoding
// that makes the same choices on validation data and writes the source.
const toModel = (p: Net): LearnedModel => ({
  mean: [...mean],
  std: [...std],
  layers: p.W.map((w, l) => ({ w: [...w], b: [...p.B[l]] })),
});
const write = async (model: LearnedModel, note: string) => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const path = opt("out") ?? resolve(OUT, "models", `${stamp}.json`);
  await Bun.write(path, JSON.stringify({ note, model }));
  console.log(`wrote full-precision model ${path}`);
  if (opt("out")) return; // staging run: don't touch the shipped weights
  await Bun.write(
    resolve(OUT, "models", "latest.json"),
    JSON.stringify({ note, model }),
  );
  const p = Bun.spawn(["bun", resolve(import.meta.dir, "shrink.ts"), path], {
    stdout: "inherit",
    stderr: "inherit",
  });
  if ((await p.exited) !== 0) throw new Error("shrink.ts failed");
};

const cfgs: Cfg[] = [];
const parseHidden = (v: string) => v.split(",").filter(Boolean).map(Number);
if (args.includes("--grid")) {
  for (const hidden of ["16", "32", "32,32", "64,64"]) {
    for (const l2 of [1e-4, 1e-3]) {
      cfgs.push({ hidden: parseHidden(hidden), l2, lr: 2e-3, seed: 1 });
    }
  }
} else {
  cfgs.push({
    hidden: parseHidden(opt("hidden", "32,32")!),
    l2: Number(opt("l2", "1e-3")),
    lr: Number(opt("lr", "2e-3")),
    seed: Number(opt("seed", "1")),
  });
}

let top: { p: Net; best: any; cfg: Cfg } | null = null;
for (const cfg of cfgs) {
  const t0 = performance.now();
  const r = fit({ ...cfg }, cfgs.length === 1);
  const tr = evaluate(r.p, train[0]);
  console.log(
    `hidden=[${cfg.hidden}] l2=${cfg.l2} lr=${cfg.lr}: best epoch ${r.epoch} | train srcc ${tr.srcc.toFixed(4)} | val srcc ${r.best.srcc.toFixed(4)} [${r.best.per}] acc1/5 ${r.best.acc1_5.toFixed(1)} acc1/10 ${r.best.acc1_10.toFixed(1)}` +
      (r.best.hold
        ? ` | fcdb holdout IoU ${r.best.hold.top1.toFixed(4)}`
        : "") +
      ` (${((performance.now() - t0) / 1000).toFixed(0)}s)`,
  );
  if (!top || r.best.score > top.best.score) top = { ...r, cfg };
}
// --ensemble K: retrain the selected config with K seeds and average them.
const K = Number(opt("ensemble", "1"));
if (top && K > 1) {
  const nets: Net[] = [top.p];
  for (let k = 1; k < K; k++) {
    const r = fit({ ...top.cfg, seed: top.cfg.seed + k }, false);
    nets.push(r.p);
    const v = evalVal(nets);
    console.log(
      `ensemble ${nets.length}: val srcc ${v.srcc.toFixed(4)} acc1/5 ${v.acc1_5.toFixed(1)} acc5 ${v.acc5.toFixed(1)} acc1/10 ${v.acc1_10.toFixed(1)}` +
        (fcHolds.length
          ? ` fcdb holdout IoU ${(fcHolds.reduce((s, d) => s + evaluate(nets, d).top1, 0) / fcHolds.length).toFixed(4)}`
          : ""),
    );
  }
  if (args.includes("--write")) {
    const v = evalVal(nets);
    const model = toModel(nets[0]);
    model.members = nets.slice(1).map((q) => toModel(q).layers);
    await write(
      model,
      `ensemble of ${K} x hidden=[${top.cfg.hidden}] l2=${top.cfg.l2} sets=${sets.join("+")} fcdb=${fcdbW}; val SRCC ${v.srcc.toFixed(3)}, val Acc1/5 ${v.acc1_5.toFixed(1)}, val Acc1/10 ${v.acc1_10.toFixed(1)}.`,
    );
  }
} else if (top && args.includes("--write")) {
  const model = toModel(top.p);
  const tr = evaluate(top.p, train[0]);
  await write(
    model,
    `hidden=[${top.cfg.hidden}] l2=${top.cfg.l2} sets=${sets.join("+")} fcdb=${fcdbW}; train SRCC ${tr.srcc.toFixed(3)}, val SRCC ${top.best.srcc.toFixed(3)}, val Acc1/5 ${top.best.acc1_5.toFixed(1)}, val Acc1/10 ${top.best.acc1_10.toFixed(1)}.`,
  );
}
