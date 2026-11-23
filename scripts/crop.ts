// Precompute crop positions with the best configuration: the learned scorer
// with the face, saliency and CNN plug-ins and consensus selection. Runs
// fully locally; fetch the models first with `bun scripts/fetch-models.ts`.
//
// Usage: bun scripts/crop.ts --aspect 2.2[,1.5,...] [--precrop 3:2]
//          [--width 512] <image paths...>
//        bun scripts/crop.ts --version [--precrop 3:2] [--width 512]
//
// For every image and aspect it asks for the largest crop of that aspect
// (full width or full height), so only one axis has slack. Output on stdout:
//   { "version": "...", "results": { "<path>": { "2.2": {
//       "x": 50, "y": 37.2,             // % of slack, as CSS object-position
//       "box": { x, y, width, height }  // in the (pre-cropped) frame's pixels
//   } } } }
// `version` changes whenever the weights, models or config change.
//
// --precrop W:H first takes the exact centre W:H crop of the source with
// integer-floor geometry (k = min(floor(w/W), floor(h/H)); size kW x kH;
// offset floor((w - kW) / 2), floor((h - kH) / 2)). EXIF orientation is NOT
// applied: pass upright images.
import { createCanvas, loadImage, type Canvas } from '@napi-rs/canvas';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import smartcrop from '../src/smartcrop';
import { LEARNED_PACKED } from '../src/learned-weights';
import { LEARNED_CNN_PACKED } from '../src/learned-cnn-weights';
import { detectFaces, faceBoosts } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';
import { cnnFeatureMap } from '../plugins/onnx/cnn';
import { modelDir } from '../plugins/onnx/runtime';
import pkg from '../package.json';

const CONFIG = { scorer: 'learned', consensus: 0.2 } as const;
const FACE_MIN_SCORE = 0.6;
const MODELS = [
  'face_detection_yunet_2023mar.onnx',
  'u2netp.onnx',
  'mobilenetv2-12.onnx',
];

const args = process.argv.slice(2);
const take = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 ? args.splice(i, 2)[1] : undefined;
};
const aspects = (take('aspect') ?? '').split(',').filter(Boolean);
const precrop = take('precrop');
const width = Number(take('width') ?? 512);
const printVersion = args.includes('--version');
const paths = args.filter((a) => a !== '--version');
if (
  !printVersion &&
  (!aspects.length || !paths.length || aspects.some((a) => !(+a > 0)))
) {
  console.error(
    'usage: bun scripts/crop.ts --aspect 2.2[,1.5] [--precrop 3:2] [--width 512] <images...>'
  );
  process.exit(2);
}
let pre: [number, number] | undefined;
if (precrop) {
  const [pw, ph] = precrop.split(':').map(Number);
  if (!(pw > 0 && ph > 0)) {
    console.error(`bad --precrop ${precrop}`);
    process.exit(2);
  }
  pre = [pw, ph];
}
for (const m of MODELS)
  if (!existsSync(join(modelDir, m))) {
    console.error(`missing ${m}: run bun scripts/fetch-models.ts`);
    process.exit(1);
  }

// Version: library + weights + models + config + options that move results.
const hash = createHash('sha256');
hash.update(JSON.stringify([CONFIG, FACE_MIN_SCORE, width, precrop ?? '']));
hash.update(JSON.stringify(LEARNED_PACKED));
hash.update(JSON.stringify(LEARNED_CNN_PACKED));
for (const m of MODELS) hash.update(await Bun.file(join(modelDir, m)).bytes());
const version = `${pkg.version}+${hash.digest('hex').slice(0, 12)}`;
if (printVersion) {
  console.log(version);
  process.exit(0);
}

const canvasFactory = (w: number, h: number) => createCanvas(w, h);

async function analyse(path: string) {
  const img = await loadImage(await Bun.file(path).bytes());
  let [sx, sy, sw, sh] = [0, 0, img.width, img.height];
  if (pre) {
    const k = Math.min(
      Math.floor(img.width / pre[0]),
      Math.floor(img.height / pre[1])
    );
    [sw, sh] = [k * pre[0], k * pre[1]];
    sx = Math.floor((img.width - sw) / 2);
    sy = Math.floor((img.height - sh) / 2);
  }
  // Analysis copy at `width` px wide; positions are fractions, so scale is free.
  const w = width;
  const h = Math.round((w * sh) / sw);
  const canvas: Canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h);
  const image = { width: w, height: h, data: data.data };

  const faces = (await detectFaces(image)).filter(
    (f) => f.score >= FACE_MIN_SCORE
  );
  const plugin = {
    boost: faceBoosts(faces, { imageWidth: w, imageHeight: h }),
    importanceMap: await saliencyMap(image),
    featureMap: await cnnFeatureMap(image),
    cnnModel: LEARNED_CNN_PACKED,
  };

  const out: Record<string, unknown> = {};
  const scale = sw / w;
  for (const a of aspects) {
    const r = +a;
    // Largest crop of aspect r inside the analysis frame.
    const cw = w / h > r ? Math.round(h * r) : w;
    const ch = w / h > r ? h : Math.round(w / r);
    const { topCrop } = await smartcrop.crop(
      canvas as any,
      {
        width: cw,
        height: ch,
        minScale: 1,
        maxScale: 1,
        canvasFactory,
        ...CONFIG,
        ...plugin,
      } as any
    );
    const pct = (off: number, slack: number) =>
      slack > 0
        ? +Math.min(100, Math.max(0, (100 * off) / slack)).toFixed(1)
        : 50;
    out[a] = {
      x: pct(topCrop.x, w - topCrop.width),
      y: pct(topCrop.y, h - topCrop.height),
      box: {
        x: Math.round(topCrop.x * scale),
        y: Math.round(topCrop.y * scale),
        width: Math.round(topCrop.width * scale),
        height: Math.round(topCrop.height * scale),
      },
    };
  }
  return out;
}

const results: Record<string, unknown> = {};
for (const p of paths) {
  try {
    results[p] = await analyse(p);
  } catch (e) {
    console.error(`${p}: ${(e as Error).message}`);
    process.exit(1);
  }
}
console.log(JSON.stringify({ version, results }));
