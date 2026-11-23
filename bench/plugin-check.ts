// Visual sanity check for the ONNX plug-ins: writes annotated images and
// saliency maps to bench/out/plugin-check/.
// Usage: bun bench/plugin-check.ts <image>...
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { basename, join } from 'node:path';
import { detectFaces } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';

const out = join(import.meta.dir, 'out/plugin-check');
for (const path of process.argv.slice(2)) {
  const img = await loadImage(path);
  const c = createCanvas(img.width, img.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, img.width, img.height);
  let t = performance.now();
  const faces = await detectFaces(data);
  const tf = performance.now() - t;
  t = performance.now();
  const sal = await saliencyMap(data);
  const ts = performance.now() - t;
  console.log(
    basename(path),
    `${img.width}x${img.height}`,
    `faces=${faces.length} (${tf.toFixed(0)}ms)`,
    `saliency ${ts.toFixed(0)}ms`,
    faces.map((f) => f.score.toFixed(2)).join(',')
  );
  ctx.lineWidth = Math.max(2, img.width / 200);
  ctx.strokeStyle = '#0f0';
  for (const f of faces) ctx.strokeRect(f.x, f.y, f.width, f.height);
  const name = basename(path).replace(/\.\w+$/, '');
  await Bun.write(join(out, `${name}.faces.jpg`), c.toBuffer('image/jpeg'));
  // Side by side: thumbnail | saliency
  const S = 320;
  const m = createCanvas(S * 2, S);
  const mc = m.getContext('2d');
  mc.drawImage(img, 0, 0, S, S);
  const id = mc.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    const v = Math.round(sal.data[i] * 255);
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  mc.putImageData(id, S, 0);
  await Bun.write(join(out, `${name}.sal.png`), m.toBuffer('image/png'));
}
