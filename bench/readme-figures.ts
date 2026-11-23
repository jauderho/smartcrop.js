// README before/after figures: classic vs learned (the 3.0 default) crops of
// the repository's sample images (examples/images, upstream's test corpus).
//
// Usage:
//   bun bench/readme-figures.ts --list [--aspect 1]
//       rank every sample image by how much the two crops differ (1 - IoU)
//   bun bench/readme-figures.ts --render <image> [<image>...] [--aspect 1]
//       write docs/img/compare-<name>-<aspect>.jpg for each image: the photo
//       with the boxes, then the classic crop, the learned crop (pure JS
//       default) and the learned crop with the optional local face and
//       saliency plug-ins (run `bun scripts/fetch-models.ts` first)
import { createCanvas, loadImage, type Image } from '@napi-rs/canvas';
import { basename, join } from 'node:path';
import smartcrop from '../src/smartcrop';
import { detectFaces, faceBoosts } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';

const root = join(import.meta.dir, '..');
const args = process.argv.slice(2);
const flag = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args.splice(i, 2)[1] : d;
};
const aspect = Number(flag('aspect', '1'));
const canvasFactory = (w: number, h: number) => createCanvas(w, h);

type Box = { x: number; y: number; width: number; height: number };
const crops = async (img: Image) => {
  const opts = { aspect, canvasFactory } as const;
  const classic = (
    await smartcrop.crop(img as any, { ...opts, ...smartcrop.PRESETS.classic })
  ).topCrop;
  const learned = (await smartcrop.crop(img as any, opts)).topCrop;
  return { classic, learned };
};
// Plug-in crop on a copy at most 800 px on the long side (as the gallery),
// box scaled back to the original.
const pluginCrop = async (img: Image): Promise<Box> => {
  const k = Math.min(1, 800 / Math.max(img.width, img.height));
  const w = Math.round(img.width * k);
  const h = Math.round(img.height * k);
  const c = createCanvas(w, h);
  c.getContext('2d').drawImage(img, 0, 0, w, h);
  const id = c.getContext('2d').getImageData(0, 0, w, h);
  const image = { width: w, height: h, data: id.data };
  const faces = (await detectFaces(image)).filter((f) => f.score >= 0.6);
  const b = (
    await smartcrop.crop(c as any, {
      aspect,
      canvasFactory,
      boost: faceBoosts(faces, { imageWidth: w, imageHeight: h }),
      importanceMap: await saliencyMap(image),
    })
  ).topCrop;
  return { x: b.x / k, y: b.y / k, width: b.width / k, height: b.height / k };
};
const iou = (a: Box, b: Box) => {
  const iw = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const ih = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  const i = Math.max(0, iw) * Math.max(0, ih);
  return i / (a.width * a.height + b.width * b.height - i);
};

if (args[0] === '--list') {
  const paths = [
    ...new Bun.Glob('examples/images/**/*.jpg').scanSync(root),
  ].sort();
  const rows: [number, string][] = [];
  for (const p of paths) {
    const { classic, learned } = await crops(await loadImage(join(root, p)));
    rows.push([1 - iou(classic, learned), p]);
  }
  rows.sort((a, b) => b[0] - a[0]);
  for (const [d, p] of rows) console.log(d.toFixed(3), p);
} else if (args[0] === '--render') {
  const H = 300; // figure height
  const GAP = 8;
  for (const p of args.slice(1)) {
    const img = await loadImage(join(root, p));
    const { classic, learned } = await crops(img);
    const plugin = await pluginCrop(img);
    const s = H / img.height;
    const ow = Math.round(img.width * s);
    const cw = Math.round(H * aspect);
    const fig = createCanvas(ow + 3 * (cw + GAP), H);
    const ctx = fig.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, fig.width, fig.height);
    ctx.drawImage(img, 0, 0, ow, H);
    // Darken outside nothing; just outline both boxes on the photo.
    const outline = (b: Box, color: string, dash: number[]) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.setLineDash(dash);
      ctx.strokeRect(
        b.x * s + 2,
        b.y * s + 2,
        b.width * s - 4,
        b.height * s - 4
      );
    };
    outline(classic, '#e53935', [12, 8]);
    outline(learned, '#43a047', []);
    outline(plugin, '#1e88e5', [4, 6]);
    ctx.setLineDash([]);
    const panel = (b: Box, x: number, color: string, label: string) => {
      ctx.drawImage(img, b.x, b.y, b.width, b.height, x, 0, cw, H);
      ctx.fillStyle = color;
      ctx.fillRect(x, H - 30, cw, 30);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 16px sans-serif';
      ctx.fillText(label, x + 10, H - 10);
    };
    panel(classic, ow + GAP, '#e53935', 'classic (2.x)');
    panel(learned, ow + GAP + cw + GAP, '#43a047', 'learned (3.0 default)');
    panel(plugin, ow + 3 * GAP + 2 * cw, '#1e88e5', '+ faces & saliency');
    const name = basename(p, '.jpg');
    const out = join(
      root,
      'docs/img',
      `compare-${name}-${String(aspect).replace('.', '_')}.jpg`
    );
    await Bun.write(out, await fig.encode('jpeg', 82));
    console.log(out);
  }
} else {
  console.error('usage: --list | --render <images...> [--aspect 1]');
  process.exit(2);
}
