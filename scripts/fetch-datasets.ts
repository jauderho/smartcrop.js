// Download image-cropping research datasets into bench/datasets/ (gitignored).
//
// The datasets are for local research evaluation only: never commit or
// redistribute them. Everything downloaded is treated as untrusted data:
// archives are extracted into their own fresh directory by unzip/tar run from
// the repo root (never from inside the archive), only files matching expected
// names are copied out, and nothing from them is ever executed.
//
// Usage: bun scripts/fetch-datasets.ts [fcdb] [flms] [gaicd] [--force]
//          [--force-extract] [--concurrency N] [--clean-archives]
//
// Output, per dataset <name>, under bench/datasets/<name>/ (image paths are
// relative to that directory, boxes are [x, y, w, h] in stored-image pixels):
//   annotations.json  [{ image, width, height, boxes, split }]
//   candidates.json   (gaicd only) [{ image, width, height, split,
//                      candidates: [[x, y, w, h, mos], ...] }]
//   report.json       what was downloaded / what failed
//
// Sources:
//   FCDB  https://github.com/yiling-chen/flickr-cropping-dataset (JSON + Flickr
//         URLs); URLs that are dead fall back to the author's own copy linked
//         from that README (Google Drive FCDB.zip).
//   FLMS  http://fangchen.org/proj_page/FLMS_mm14/ (authors' page, via the
//         bcmi Awesome-Aesthetic-Evaluation-and-Cropping list).
//   GAICD https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
//         README, "journal version" dataset (Google Drive); all splits (train/val/test).
import { loadImage } from "@napi-rs/canvas";
import {
  copyFile,
  mkdir,
  readdir,
  readFile,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { readMat } from "./mat5";

const ROOT = resolve(import.meta.dir, "..");
const DATA = join(ROOT, "bench/datasets");
const DL = join(DATA, "_dl"); // downloaded archives
const RAW = join(DATA, "_raw"); // fresh extraction directories

const args = process.argv.slice(2);
const flag = (n: string) => args.includes(`--${n}`);
const cIdx = args.indexOf("--concurrency");
const CONC = Math.min(8, cIdx > -1 ? Number(args[cIdx + 1]) || 4 : 4);
const FORCE = flag("force");
const picked = args.filter((a) => ["fcdb", "flms", "gaicd"].includes(a));
const want = (n: string) => picked.length === 0 || picked.includes(n);

type Box = [number, number, number, number];
type Ann = {
  image: string;
  width: number;
  height: number;
  boxes: Box[];
  split: "train" | "val" | "test";
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function pool<T>(items: T[], fn: (t: T, i: number) => Promise<void>) {
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(CONC, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        await fn(items[i], i);
      }
    }),
  );
}

async function retry<T>(fn: () => Promise<T>, tries = 3): Promise<T> {
  for (let i = 1; ; i++) {
    try {
      return await fn();
    } catch (e) {
      if (i >= tries || (e as { fatal?: boolean }).fatal) throw e;
      await sleep(500 * 2 ** i);
    }
  }
}

class HttpError extends Error {
  fatal: boolean;
  constructor(
    public status: number,
    url: string,
  ) {
    super(`HTTP ${status} ${url}`);
    // 4xx (not 429) will not get better on retry.
    this.fatal = status >= 400 && status < 500 && status !== 429;
  }
}

// Download url to dest (via .part, so existing dest means complete).
async function download(url: string, dest: string, imageOnly = false) {
  if (existsSync(dest)) return;
  await retry(async () => {
    const res = await fetch(url, {
      redirect: "follow",
      signal: AbortSignal.timeout(imageOnly ? 30_000 : 600_000),
    });
    if (!res.ok) throw new HttpError(res.status, url);
    const type = res.headers.get("content-type") ?? "";
    if (imageOnly && !type.startsWith("image/")) {
      // Flickr serves an HTML or placeholder page for removed photos.
      throw Object.assign(new Error(`not an image (${type}) ${url}`), {
        fatal: true,
      });
    }
    await Bun.write(dest + ".part", res);
    await rename(dest + ".part", dest);
  });
}

// Google Drive: the first response may be a virus-scan interstitial form.
async function driveDownload(id: string, dest: string) {
  if (existsSync(dest)) return;
  const base = "https://drive.usercontent.google.com/download";
  const q = new URLSearchParams({ id, export: "download", confirm: "t" });
  await retry(async () => {
    let res = await fetch(`${base}?${q}`);
    if ((res.headers.get("content-type") ?? "").startsWith("text/html")) {
      const html = await res.text();
      const p = new URLSearchParams();
      for (const m of html.matchAll(
        /<input type="hidden" name="([a-z]+)" value="([^"]*)"/g,
      )) {
        p.set(m[1], m[2]);
      }
      if (!p.has("id")) {
        throw Object.assign(
          new Error(`Drive ${id}: needs interactive download (quota/login)`),
          { fatal: true },
        );
      }
      res = await fetch(`${base}?${p}`);
    }
    if (!res.ok) throw new HttpError(res.status, id);
    if ((res.headers.get("content-type") ?? "").startsWith("text/html")) {
      throw Object.assign(new Error(`Drive ${id}: got HTML, not a file`), {
        fatal: true,
      });
    }
    await Bun.write(dest + ".part", res);
    await rename(dest + ".part", dest);
  });
}

// Extract into a brand-new empty dir; cwd is the repo root, not the archive.
async function extract(archive: string, name: string, patterns: string[] = []) {
  const dir = join(RAW, name);
  await rm(dir, { recursive: true, force: true });
  await mkdir(dir, { recursive: true });
  const cmd = archive.endsWith(".zip")
    ? ["unzip", "-q", archive, ...patterns, "-d", dir]
    : ["tar", "xf", archive, "-C", dir];
  const p = Bun.spawn(cmd, { cwd: ROOT, stdout: "ignore", stderr: "inherit" });
  if ((await p.exited) !== 0) throw new Error(`extract failed: ${archive}`);
  return dir;
}

async function dims(path: string) {
  const img = await loadImage(path);
  return { width: img.width, height: img.height };
}

const dirSize = async (dir: string) => {
  let total = 0;
  for (const f of await readdir(dir)) total += (await stat(join(dir, f))).size;
  return total;
};

const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

function clampBox(b: Box, w: number, h: number): Box {
  const x = Math.max(0, Math.min(w - 1, b[0]));
  const y = Math.max(0, Math.min(h - 1, b[1]));
  return [x, y, Math.min(b[2], w - x), Math.min(b[3], h - y)];
}

async function done(name: string) {
  return !FORCE && existsSync(join(DATA, name, "annotations.json"));
}

async function writeOut(
  name: string,
  anns: Ann[],
  report: Record<string, unknown>,
) {
  await Bun.write(
    join(DATA, name, "annotations.json"),
    JSON.stringify(anns) + "\n",
  );
  await Bun.write(
    join(DATA, name, "report.json"),
    JSON.stringify(report, null, 2) + "\n",
  );
  const imgDir = join(DATA, name, "images");
  const size = existsSync(imgDir) ? await dirSize(imgDir).catch(() => 0) : 0;
  console.log(
    `[${name}] ${anns.length} annotated images, ${mb(size)} top-level images; report.json written`,
  );
}

// ---------------------------------------------------------------- FCDB
async function fcdb() {
  if (await done("fcdb")) return console.log("[fcdb] already done (--force)");
  const repo =
    "https://raw.githubusercontent.com/yiling-chen/flickr-cropping-dataset/master";
  const imgDir = join(DATA, "fcdb/images");
  await mkdir(imgDir, { recursive: true });
  type Entry = { url: string; crop: Box };
  const sets: [Ann["split"], Entry[]][] = [];
  for (const [split, f] of [
    ["train", "cropping_training_set.json"],
    ["test", "cropping_testing_set.json"],
  ] as const) {
    const res = await fetch(`${repo}/${f}`);
    if (!res.ok) throw new HttpError(res.status, f);
    sets.push([split, (await res.json()) as Entry[]]);
  }
  const all = sets.flatMap(([split, es]) =>
    es.map((e) => ({ ...e, split, name: basename(new URL(e.url).pathname) })),
  );
  console.log(`[fcdb] ${all.length} entries, downloading from Flickr...`);
  const dead: { url: string; reason: string }[] = [];
  let n = 0;
  await pool(all, async (e) => {
    try {
      await download(e.url, join(imgDir, e.name), true);
    } catch (err) {
      dead.push({ url: e.url, reason: (err as Error).message });
    }
    if (++n % 200 === 0) console.log(`[fcdb] ${n}/${all.length}`);
  });
  const recovered: string[] = [];
  if (dead.length) {
    console.log(
      `[fcdb] ${dead.length} Flickr URLs failed; trying the author's Drive copy`,
    );
    try {
      await mkdir(DL, { recursive: true });
      const zip = join(DL, "fcdb-drive.zip");
      await driveDownload("11R4przja3GE3j36wrtMSovvHMuw-oeOn", zip);
      const dir = await extract(zip, "fcdb-drive");
      for (const d of dead) {
        const name = basename(new URL(d.url).pathname);
        const src = join(dir, "FCDB", name);
        if (existsSync(src)) {
          await copyFile(src, join(imgDir, name));
          recovered.push(d.url);
        }
      }
      await rm(dir, { recursive: true, force: true });
    } catch (err) {
      console.log(`[fcdb] Drive fallback failed: ${(err as Error).message}`);
    }
  }
  const stillDead = dead.filter((d) => !recovered.includes(d.url));
  const anns: Ann[] = [];
  const unreadable: string[] = [];
  let clamped = 0;
  for (const e of all) {
    const p = join(imgDir, e.name);
    if (!existsSync(p)) continue;
    try {
      const { width, height } = await dims(p);
      const b = clampBox(e.crop, width, height);
      if (b.some((v, i) => v !== e.crop[i])) clamped++;
      anns.push({
        image: `images/${e.name}`,
        width,
        height,
        boxes: [b],
        split: e.split,
      });
    } catch {
      unreadable.push(e.name);
    }
  }
  await writeOut("fcdb", anns, {
    source: repo,
    entries: all.length,
    flickrFailed: dead.length,
    recoveredFromDrive: recovered.length,
    stillMissing: stillDead,
    unreadable,
    boxesClamped: clamped,
    train: anns.filter((a) => a.split === "train").length,
    test: anns.filter((a) => a.split === "test").length,
  });
  console.log(
    `[fcdb] flickr failed ${dead.length}, recovered ${recovered.length}, missing ${stillDead.length}, unreadable ${unreadable.length}, clamped ${clamped}`,
  );
}

// ---------------------------------------------------------------- FLMS
async function flms() {
  if (await done("flms")) return console.log("[flms] already done (--force)");
  const site = "http://fangchen.org/proj_page/FLMS_mm14/data";
  await mkdir(DL, { recursive: true });
  const imgTar = join(DL, "flms-images.tar");
  const gtTar = join(DL, "flms-gt.tar");
  await download(`${site}/radomir500_image/image.tar`, imgTar);
  await download(`${site}/radomir500_gt/release_data.tar`, gtTar);
  const imgRaw = await extract(imgTar, "flms-images");
  const gtRaw = await extract(gtTar, "flms-gt");
  const mat = readMat(
    Buffer.from(await readFile(join(gtRaw, "500_image_dataset.mat"))),
  );
  const gt = mat.img_gt;
  if (gt?.cls !== "struct") throw new Error("unexpected FLMS mat layout");
  const outDir = join(DATA, "flms/images");
  await mkdir(outDir, { recursive: true });
  const anns: Ann[] = [];
  let invalid = 0;
  for (const rec of gt.items) {
    const fn = rec.filename.cls === "char" ? rec.filename.text : "";
    const bb = rec.bbox;
    if (!/^[\w.-]+$/.test(fn) || bb.cls !== "num") continue;
    const src = join(imgRaw, "image", fn);
    if (!existsSync(src)) continue;
    await copyFile(src, join(outDir, fn));
    const { width, height } = await dims(src);
    const rows = bb.dims[0];
    const boxes: Box[] = [];
    // column-major (hmin, wmin, hmax, wmax); negative = invalid turker box.
    for (let r = 0; r < rows; r++) {
      const [hmin, wmin, hmax, wmax] = [0, 1, 2, 3].map(
        (c) => bb.data[c * rows + r],
      );
      if (hmin < 0 || wmin < 0 || hmax <= hmin || wmax <= wmin) {
        invalid++;
        continue;
      }
      boxes.push(
        clampBox([wmin, hmin, wmax - wmin, hmax - hmin], width, height),
      );
    }
    if (boxes.length) {
      anns.push({ image: `images/${fn}`, width, height, boxes, split: "test" });
    }
  }
  await rm(imgRaw, { recursive: true, force: true });
  await rm(gtRaw, { recursive: true, force: true });
  await writeOut("flms", anns, {
    source: site,
    images: anns.length,
    boxes: anns.reduce((s, a) => s + a.boxes.length, 0),
    invalidBoxesDropped: invalid,
  });
}

// --------------------------------------------------------------- GAICD
// Extraction reuses an existing bench/datasets/_dl/gaicd-journal.zip (no
// re-download). Run `bun scripts/fetch-datasets.ts gaicd --force-extract` to
// re-run extraction only.
async function gaicd() {
  if (!flag("force-extract") && (await done("gaicd"))) {
    return console.log("[gaicd] already done (--force, --force-extract)");
  }
  await mkdir(DL, { recursive: true });
  const zip = join(DL, "gaicd-journal.zip");
  if (existsSync(zip)) console.log("[gaicd] reusing existing zip");
  else console.log("[gaicd] downloading journal-version zip (~300 MB)...");
  await driveDownload("1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR", zip);
  const splits = ["test", "train", "val"] as const;
  const raw = await extract(
    zip,
    "gaicd",
    splits.flatMap((s) => [`GAIC/images/${s}/*`, `GAIC/annotations/${s}/*`]),
  );
  const anns: Ann[] = [];
  const cands: unknown[] = [];
  const counts: Record<string, { images: number; candidates: number }> = {};
  for (const split of splits) {
    const outDir = join(DATA, "gaicd/images", split);
    await mkdir(outDir, { recursive: true });
    counts[split] = { images: 0, candidates: 0 };
    for (const f of (await readdir(join(raw, "GAIC/images", split))).sort()) {
      const id = f.replace(/\.jpg$/i, "");
      const annPath = join(raw, "GAIC/annotations", split, `${id}.txt`);
      if (!/^\d+\.jpg$/i.test(f) || !existsSync(annPath)) continue;
      const src = join(raw, "GAIC/images", split, f);
      await copyFile(src, join(outDir, f));
      const { width, height } = await dims(src);
      // Each line: y1 x1 y2 x2 mos (rows then columns), MOS in [1, 5].
      const candidates = (await readFile(annPath, "utf8"))
        .split("\n")
        .map((l) => l.trim().split(/\s+/).map(Number))
        .filter((v) => v.length === 5 && v.every(Number.isFinite))
        .map(([y1, x1, y2, x2, mos]) => [x1, y1, x2 - x1, y2 - y1, mos]);
      if (!candidates.length) continue;
      const best = Math.max(...candidates.map((c) => c[4]));
      const image = `images/${split}/${f}`;
      cands.push({ image, width, height, split, candidates });
      // "boxes" = the top-MOS candidate crops (ties kept).
      anns.push({
        image,
        width,
        height,
        boxes: candidates
          .filter((c) => c[4] === best)
          .map((c) => c.slice(0, 4) as Box),
        split,
      });
      counts[split].images++;
      counts[split].candidates += candidates.length;
    }
  }
  await rm(raw, { recursive: true, force: true });
  await Bun.write(
    join(DATA, "gaicd/candidates.json"),
    JSON.stringify(cands) + "\n",
  );
  const total = Object.values(counts).reduce((s, c) => s + c.candidates, 0);
  console.log(
    "[gaicd] per split:",
    JSON.stringify(counts),
    "scored candidates:",
    total,
  );
  await writeOut("gaicd", anns, {
    source:
      "https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch (journal version, Google Drive 1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR)",
    split: "train, val and test",
    images: anns.length,
    perSplit: counts,
    scoredCandidates: total,
  });
}

for (const [n, fn] of [
  ["fcdb", fcdb],
  ["flms", flms],
  ["gaicd", gaicd],
] as const) {
  if (!want(n)) continue;
  try {
    await fn();
  } catch (e) {
    console.error(`[${n}] FAILED: ${(e as Error).message}`);
    process.exitCode = 1;
  }
}
if (flag("clean-archives")) await rm(DL, { recursive: true, force: true });
await rm(RAW, { recursive: true, force: true });
