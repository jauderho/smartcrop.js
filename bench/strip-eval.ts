// Evaluate full-width strip crops against hand-picked positions.
//
// A labels file describes a set of photos, the strip aspect ratio, and the
// accepted vertical position for each photo as a CSS object-position Y%
// (100 * top / slack). The pipeline mirrors a site build: center-crop the
// photo to 3:2, scale to 512 px wide, and ask smartcrop for a full-width
// strip, so only the vertical offset varies.
//
// Usage: bun bench/strip-eval.ts <labels.json> [library-module] [--options '{json}']
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { join, resolve } from 'node:path';
import { imageKey, parseWith, pluginOptions } from './plugin-cache';

type Labels = {
  imageDir: string;
  aspect: number;
  tolerance: number;
  truth: Record<string, number>;
  softTruth?: Record<string, number>;
  unreviewed?: string[];
};

const args = process.argv.slice(2);
const optIdx = args.indexOf('--options');
const extraOptions = optIdx > -1 ? JSON.parse(args.splice(optIdx, 2)[1]) : {};
// --with faces,saliency feeds the ONNX plug-ins into the crop options.
const withIdx = args.indexOf('--with');
const withPlugins = parseWith(
  withIdx > -1 ? args.splice(withIdx, 2)[1] : undefined
);
const [labelsPath, libPath] = args;
if (!labelsPath) {
  console.error('usage: bun bench/strip-eval.ts <labels.json> [library]');
  process.exit(2);
}
const labels: Labels = await Bun.file(labelsPath).json();
const lib = await import(
  resolve(libPath ?? join(import.meta.dir, '../src/smartcrop.ts'))
);
const smartcrop = lib.default ?? lib;

const WIDTH = 512;
const canvasFactory = (w: number, h: number) => createCanvas(w, h);

export async function stripPosition(path: string) {
  const img = await loadImage(path);
  // Center-crop to exactly 3:2, then scale to WIDTH.
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
  canvas.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, WIDTH, h);

  const stripH = Math.round(WIDTH / labels.aspect);
  const pluginOpts = withPlugins.size
    ? await pluginOptions(
        'strips',
        imageKey(path),
        withPlugins,
        { width: WIDTH, height: h },
        async () => canvas.getContext('2d').getImageData(0, 0, WIDTH, h)
      )
    : {};
  const { topCrop } = await smartcrop.crop(canvas, {
    width: WIDTH,
    height: stripH,
    minScale: 1,
    maxScale: 1,
    canvasFactory,
    ...extraOptions,
    ...pluginOpts,
  });
  const slack = h - topCrop.height;
  return slack > 0 ? (100 * topCrop.y) / slack : 50;
}

const rows: {
  id: string;
  kind: string;
  truth: number | null;
  pred: number;
  err: number | null;
}[] = [];
const all: [string, string, number | null][] = [
  ...Object.entries(labels.truth).map(
    ([id, t]) => [id, 'truth', t] as [string, string, number]
  ),
  ...Object.entries(labels.softTruth ?? {}).map(
    ([id, t]) => [id, 'soft', t] as [string, string, number]
  ),
  ...(labels.unreviewed ?? []).map(
    (id) => [id, 'unreviewed', null] as [string, string, null]
  ),
];
for (const [id, kind, truth] of all) {
  const pred = await stripPosition(join(labels.imageDir, `${id}.jpg`));
  rows.push({
    id,
    kind,
    truth,
    pred: +pred.toFixed(1),
    err: truth === null ? null : +Math.abs(pred - truth).toFixed(1),
  });
}
console.table(rows);

const scored = rows.filter((r) => r.err !== null);
const hard = rows.filter((r) => r.kind === 'truth');
const pass = (rs: typeof rows) =>
  rs.filter((r) => r.err! <= labels.tolerance).length;
const mae = (rs: typeof rows) =>
  rs.reduce((s, r) => s + r.err!, 0) / Math.max(1, rs.length);
// Baseline that always centers the strip; a useful crop must beat it.
const centered = (rs: typeof rows) =>
  rs.map((r) => ({ ...r, err: Math.abs(50 - r.truth!) }));
const summary = (rs: typeof rows) => ({
  n: rs.length,
  pass: pass(rs),
  mae: +mae(rs).toFixed(1),
});
console.log(
  JSON.stringify({
    hard: summary(hard),
    all: summary(scored),
    centerBaseline: {
      hard: summary(centered(hard)),
      all: summary(centered(scored)),
    },
    options: extraOptions,
    ...(withPlugins.size ? { with: [...withPlugins] } : {}),
  })
);
