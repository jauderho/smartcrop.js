// Build data for the visual crop gallery (bench/gallery/index.html).
//
// Usage: bun bench/gallery.ts <strips|examples|fcdb|flms>
//          [--presets classic,next] [--lib path] [--limit N]
// Output: bench/out/gallery-<set>/data.json and img/<id>.jpg (gitignored).
// Boxes in data.json are normalized 0-1 relative to the thumbnail.
// Display names (page and log only; data.json keys are unchanged):
//   classic -> "original (2.0.5)", next -> "improved",
//   truth -> "site (hand-set)". For strips, the hand-set positions are the
//   website's current values, not ground truth, so no error is computed.
const LABEL: Record<string, string> = {
  classic: "original (2.0.5)",
  next: "improved",
  center: "center",
  truth: "site (hand-set)",
};
import { createCanvas, loadImage, type Canvas } from "@napi-rs/canvas";
import { Glob } from "bun";
import { existsSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import { pluginOptions } from "./plugin-cache";
import { basename, join, relative, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const args = process.argv.slice(2);
const flag = (name: string) => {
  const i = args.indexOf(name);
  return i > -1 ? args.splice(i, 2)[1] : undefined;
};
const presetNames = (flag("--presets") ?? "classic,next").split(",");
const libPath = flag("--lib");
const limit = Number(flag("--limit") ?? Infinity);
const set = args[0];
if (!["strips", "examples", "fcdb", "flms"].includes(set ?? "")) {
  console.error(
    "usage: bun bench/gallery.ts <strips|examples|fcdb|flms> [--presets classic,next] [--lib path] [--limit N]",
  );
  process.exit(2);
}

const lib = await import(resolve(libPath ?? join(root, "src/smartcrop.ts")));
const smartcrop = lib.default ?? lib;
// A preset name ending in "+fs" also feeds the local ONNX plug-ins (faces,
// saliency), "+fsc" adds the cnn feature map too (needs the cnn head in
// src/learned-cnn-weights.ts), and "+fscv" also turns on consensus (box
// voting, T=0.2); cached under bench/out/cache/gallery-<set>/.
const baseName = (name: string) => name.replace(/\+fsc?v?$/, "");
const usesCnn = (name: string) => /\+fscv?$/.test(name);
const optionsFor = (name: string): Record<string, unknown> =>
  baseName(name) === "classic"
    ? {}
    : {
        ...(smartcrop.PRESETS?.[baseName(name)] ?? {}),
        ...(name.endsWith("+fscv") ? { consensus: 0.2 } : {}),
      };
const pluginsFor = async (
  name: string,
  canvas: ReturnType<typeof createCanvas>,
  w: number,
  h: number,
  key: string,
): Promise<Record<string, unknown>> =>
  name.endsWith("+fs") || usesCnn(name)
    ? pluginOptions(
        `gallery-${set}`,
        key,
        new Set(
          usesCnn(name)
            ? (["faces", "saliency", "cnn"] as const)
            : (["faces", "saliency"] as const),
        ),
        { width: w, height: h },
        async () => canvas.getContext("2d").getImageData(0, 0, w, h),
      )
    : {};

const canvasFactory = (w: number, h: number) => createCanvas(w, h);
const THUMB = 800;

type Box = { x: number; y: number; w: number; h: number };
type Method = Box & {
  pos?: number;
  iou?: number;
  soft?: boolean;
};

const outDir = join(root, "bench/out", `gallery-${set}`);

async function prepare(dir: string) {
  await rm(dir, { recursive: true, force: true });
  await mkdir(join(dir, "img"), { recursive: true });
}

async function saveThumb(canvas: Canvas, id: string) {
  const s = Math.min(1, THUMB / Math.max(canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * s));
  const h = Math.max(1, Math.round(canvas.height * s));
  const t = createCanvas(w, h);
  t.getContext("2d").drawImage(canvas, 0, 0, w, h);
  const file = `img/${id}.jpg`;
  await Bun.write(join(outDir, file), await t.encode("jpeg", 80));
  return file;
}

const nb = (
  c: { x: number; y: number; width: number; height: number },
  w: number,
  h: number,
): Box => ({ x: c.x / w, y: c.y / h, w: c.width / w, h: c.height / h });

function iou(a: Box, b: Box) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const i = ix * iy;
  return i / (a.w * a.h + b.w * b.h - i);
}

function centered(cw: number, ch: number, ratio: number): Box {
  let w = cw;
  let h = cw / ratio;
  if (h > ch) {
    h = ch;
    w = ch * ratio;
  }
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

const r1 = (n: number) => +n.toFixed(1);
const r4 = (n: number) => +n.toFixed(4);
const round = (b: Box): Box => ({
  x: r4(b.x),
  y: r4(b.y),
  w: r4(b.w),
  h: r4(b.h),
});

async function runStrips() {
  const labelsPath = join(root, "bench/private/jauderho-strip.json");
  if (!existsSync(labelsPath)) {
    console.log(`${labelsPath} not found; nothing to do.`);
    return;
  }
  const labels = await Bun.file(labelsPath).json();
  const WIDTH = 512;
  const ids: [string, number | null, boolean][] = [
    ...Object.entries<number>(labels.truth).map(
      ([i, t]): [string, number, boolean] => [i, t, false],
    ),
    ...Object.entries<number>(labels.softTruth ?? {}).map(
      ([i, t]): [string, number, boolean] => [i, t, true],
    ),
    ...(labels.unreviewed ?? []).map((i: string): [string, null, boolean] => [
      i,
      null,
      false,
    ]),
  ].slice(0, limit);
  await prepare(outDir);
  const items = [];
  for (const [id, truth, soft] of ids) {
    const img = await loadImage(join(labels.imageDir, `${id}.jpg`));
    let sw = img.width;
    let sh = Math.round((img.width * 2) / 3);
    if (sh > img.height) {
      sh = img.height;
      sw = Math.round((img.height * 3) / 2);
    }
    const sx = Math.round((img.width - sw) / 2);
    const sy = Math.round((img.height - sh) / 2);
    const h = Math.round((WIDTH * sh) / sw);
    const canvas = createCanvas(WIDTH, h);
    canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, WIDTH, h);
    const stripH = Math.round(WIDTH / labels.aspect);
    const stripBox = (pos: number): Box => ({
      x: 0,
      y: ((h - stripH) * pos) / 100 / h,
      w: 1,
      h: stripH / h,
    });
    const methods: Record<string, Method> = {};
    for (const name of presetNames) {
      const { topCrop } = await smartcrop.crop(canvas, {
        width: WIDTH,
        height: stripH,
        minScale: 1,
        maxScale: 1,
        canvasFactory,
        ...optionsFor(name),
        ...(await pluginsFor(name, canvas, WIDTH, h, id)),
      });
      const slack = h - topCrop.height;
      const pos = slack > 0 ? (100 * topCrop.y) / slack : 50;
      methods[name] = { ...round(nb(topCrop, WIDTH, h)), pos: r1(pos) };
    }
    methods.center = { ...round(stripBox(50)), pos: 50 };
    if (truth !== null)
      methods.truth = { ...round(stripBox(truth)), pos: truth, soft };
    items.push({
      id,
      img: await saveThumb(canvas, id),
      w: WIDTH,
      h,
      kind: truth === null ? "unreviewed" : soft ? "soft" : "truth",
      truth,
      methods,
    });
    console.log(
      id,
      Object.entries(methods)
        .map(([k, m]) => `${LABEL[k] ?? k}=${m.pos}`)
        .join(" "),
    );
  }
  await Bun.write(
    join(outDir, "data.json"),
    JSON.stringify({
      set,
      kind: "strips",
      presets: presetNames,
      tolerance: labels.tolerance,
      aspect: labels.aspect,
      items,
    }),
  );
}

async function runExamples() {
  const shapes: [string, number][] = [
    ["square", 1],
    ["wide", 2.2],
    ["portrait", 0.8],
  ];
  const files = [
    ...new Glob("examples/images/**/*.jpg").scanSync({ cwd: root }),
  ]
    .sort()
    .slice(0, limit);
  await prepare(outDir);
  const items = [];
  for (const f of files) {
    const img = await loadImage(join(root, f));
    const s = Math.min(1, THUMB / Math.max(img.width, img.height));
    const w = Math.round(img.width * s);
    const h = Math.round(img.height * s);
    const canvas = createCanvas(w, h);
    canvas.getContext("2d").drawImage(img, 0, 0, w, h);
    const name = relative("examples/images", f)
      .replace(/\.jpg$/, "")
      .replaceAll("/", "_");
    const file = await saveThumb(canvas, name);
    for (const [shape, ratio] of shapes) {
      const cw = ratio >= 1 ? 100 * ratio : 100;
      const ch = ratio >= 1 ? 100 : 100 / ratio;
      const methods: Record<string, Method> = {};
      for (const pname of presetNames) {
        const { topCrop } = await smartcrop.crop(canvas, {
          width: cw,
          height: ch,
          minScale: 1,
          canvasFactory,
          ...optionsFor(pname),
          ...(await pluginsFor(pname, canvas, w, h, name)),
        });
        methods[pname] = round(nb(topCrop, w, h) as Box);
      }
      const c = centered(w, h, ratio);
      methods.center = round({
        x: c.x / w,
        y: c.y / h,
        w: c.w / w,
        h: c.h / h,
      });
      items.push({
        id: `${name}@${shape}`,
        img: file,
        w,
        h,
        shape,
        ratio,
        methods,
      });
    }
    console.log(name);
  }
  await Bun.write(
    join(outDir, "data.json"),
    JSON.stringify({ set, kind: "examples", presets: presetNames, items }),
  );
}

async function runDataset() {
  const dsDir = join(root, "bench/datasets", set);
  const annPath = join(dsDir, "annotations.json");
  if (!existsSync(annPath)) {
    console.log(
      `${annPath} does not exist yet (run the dataset fetch script first); nothing to do.`,
    );
    return;
  }
  const ann: {
    image: string;
    width: number;
    height: number;
    boxes: number[][];
    split: string;
  }[] = (await Bun.file(annPath).json()).filter(
    (a: { split: string }) => a.split === "test",
  );
  await prepare(outDir);
  const items = [];
  let count = 0;
  for (const a of ann) {
    if (count >= limit) break;
    count++;
    const img = await loadImage(join(dsDir, a.image));
    const s = Math.min(1, THUMB / Math.max(img.width, img.height));
    const w = Math.round(img.width * s);
    const h = Math.round(img.height * s);
    const canvas = createCanvas(w, h);
    canvas.getContext("2d").drawImage(img, 0, 0, w, h);
    const id = basename(a.image).replace(/\.[^.]+$/, "") + "_" + count;
    const file = await saveThumb(canvas, id);
    // Annotation coordinates are in the original image space.
    const kx = w / a.width;
    const ky = h / a.height;
    for (const [bi, [gx, gy, gw, gh]] of a.boxes.entries()) {
      const gt: Box = { x: gx * kx, y: gy * ky, w: gw * kx, h: gh * ky };
      const methods: Record<string, Method> = {};
      for (const pname of presetNames) {
        const { topCrop } = await smartcrop.crop(canvas, {
          width: gt.w,
          height: gt.h,
          minScale: 0.5,
          maxScale: 1,
          canvasFactory,
          ...optionsFor(pname),
          ...(await pluginsFor(pname, canvas, w, h, id)),
        });
        const b = {
          x: topCrop.x,
          y: topCrop.y,
          w: topCrop.width,
          h: topCrop.height,
        };
        methods[pname] = {
          ...round({ x: b.x / w, y: b.y / h, w: b.w / w, h: b.h / h }),
          iou: r4(iou(b, gt)),
        };
      }
      const c = centered(w, h, gt.w / gt.h);
      methods.center = {
        ...round({ x: c.x / w, y: c.y / h, w: c.w / w, h: c.h / h }),
        iou: r4(iou(c, gt)),
      };
      methods.truth = round({
        x: gt.x / w,
        y: gt.y / h,
        w: gt.w / w,
        h: gt.h / h,
      });
      items.push({ id: `${id}#${bi}`, img: file, w, h, methods });
    }
    if (count % 20 === 0)
      console.log(`${count}/${Math.min(ann.length, limit)}`);
  }
  await Bun.write(
    join(outDir, "data.json"),
    JSON.stringify({ set, kind: "dataset", presets: presetNames, items }),
  );
}

if (set === "strips") await runStrips();
else if (set === "examples") await runExamples();
else await runDataset();
console.log(`wrote ${outDir}`);
