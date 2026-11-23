// Cross-platform determinism probe. Runs every stage that feeds training on
// the repository's sample images (examples/images, no datasets) and records
// the exact outputs, so two machines (laptop vs GitHub runner) can be
// compared stage by stage: JPEG decode + resize, faces, saliency, CNN map,
// learned features, and the final crops.
//
// Usage: bun bench/platform-probe.ts [--n 12] [--out probe.json]
//        bun bench/platform-probe.ts --compare a.json b.json
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { join } from 'node:path';
import smartcrop from '../src/smartcrop';
import { LEARNED_CNN_PACKED } from '../src/learned-cnn-weights';
import { detectFaces, faceBoosts } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';
import { cnnFeatureMap } from '../plugins/onnx/cnn';

const root = join(import.meta.dir, '..');
const args = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args[i + 1] : d;
};

const b64 = (a: ArrayLike<number>) =>
  Buffer.from(new Float32Array(a as ArrayLike<number>).buffer).toString(
    'base64'
  );
const unb64 = (s: string) => {
  const b = Buffer.from(s, 'base64');
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const sha = (d: ArrayLike<number> | Uint8Array) =>
  new Bun.CryptoHasher('sha256')
    .update(d instanceof Uint8Array ? d : new Float32Array(d as any))
    .digest('hex')
    .slice(0, 16);

type Box = { x: number; y: number; width: number; height: number };
type Rec = {
  pixels: string; // sha of the 512-px RGBA analysis image
  faces: number[][];
  saliency: string; // base64 f32
  cnn: string; // base64 f32 (projected map)
  features: string; // base64 f32 (learned features of every candidate)
  crops: Record<string, Box>;
};

if (args[0] === '--compare') {
  const [a, b] = await Promise.all(
    [args[1], args[2]].map((p) => Bun.file(p).json())
  );
  console.log(`A: ${a.platform}\nB: ${b.platform}\n`);
  const stat = (name: string, get: (r: Rec) => Float32Array) => {
    let same = 0;
    let n = 0;
    let maxAbs = 0;
    let maxRel = 0;
    for (const k of Object.keys(a.images)) {
      if (!b.images[k]) continue;
      const x = get(a.images[k]);
      const y = get(b.images[k]);
      n++;
      if (x.length !== y.length) {
        maxAbs = Infinity;
        continue;
      }
      let eq = true;
      for (let i = 0; i < x.length; i++) {
        const d = Math.abs(x[i] - y[i]);
        if (d > 0) eq = false;
        maxAbs = Math.max(maxAbs, d);
        maxRel = Math.max(maxRel, d / Math.max(1e-6, Math.abs(x[i])));
      }
      if (eq) same++;
    }
    console.log(
      `${name.padEnd(10)} identical ${same}/${n}  max |diff| ${maxAbs.toExponential(2)}  max rel ${maxRel.toExponential(2)}`
    );
  };
  const keys = Object.keys(a.images);
  const px = keys.filter((k) => a.images[k].pixels === b.images[k]?.pixels);
  console.log(`pixels     identical ${px.length}/${keys.length}`);
  stat('faces', (r) => new Float32Array(r.faces.flat()));
  stat('saliency', (r) => unb64(r.saliency));
  stat('cnn', (r) => unb64(r.cnn));
  stat('features', (r) => unb64(r.features));
  for (const m of Object.keys(a.images[keys[0]].crops)) {
    let same = 0;
    let maxPx = 0;
    for (const k of keys) {
      const p = a.images[k].crops[m];
      const q = b.images[k]?.crops[m];
      if (!q) continue;
      const d = Math.max(
        ...(['x', 'y', 'width', 'height'] as const).map((f) =>
          Math.abs(p[f] - q[f])
        )
      );
      if (d === 0) same++;
      maxPx = Math.max(maxPx, d);
    }
    console.log(
      `crop ${m.padEnd(14)} identical ${same}/${keys.length}  max box diff ${maxPx.toFixed(2)} px`
    );
  }
  console.log('\ntimings (s):');
  for (const k of Object.keys(a.timings))
    console.log(
      `  ${k.padEnd(10)} A ${a.timings[k].toFixed(2)}  B ${b.timings[k]?.toFixed(2)}`
    );
  process.exit(0);
}

const N = Number(opt('n', '12'));
const paths = [...new Bun.Glob('examples/images/*.jpg').scanSync(root)]
  .sort()
  .slice(0, N);
const timings: Record<string, number> = {
  decode: 0,
  faces: 0,
  saliency: 0,
  cnn: 0,
  crops: 0,
};
const tick = async <T>(k: string, f: () => Promise<T>): Promise<T> => {
  const t = performance.now();
  const r = await f();
  timings[k] += (performance.now() - t) / 1000;
  return r;
};
const canvasFactory = (w: number, h: number) => createCanvas(w, h);
const images: Record<string, Rec> = {};
for (const p of paths) {
  const { c, w, h, image } = await tick('decode', async () => {
    const img = await loadImage(await Bun.file(join(root, p)).bytes());
    const w = 512;
    const h = Math.round((512 * img.height) / img.width);
    const c = createCanvas(w, h);
    c.getContext('2d').drawImage(img, 0, 0, w, h);
    const id = c.getContext('2d').getImageData(0, 0, w, h);
    return { c, w, h, image: { width: w, height: h, data: id.data } };
  });
  const faces = await tick('faces', () => detectFaces(image));
  const sal = await tick('saliency', () => saliencyMap(image));
  const cnn = await tick('cnn', () => cnnFeatureMap(image));
  const plug = {
    boost: faceBoosts(
      faces.filter((f) => f.score >= 0.6),
      { imageWidth: w, imageHeight: h }
    ),
    importanceMap: sal,
    featureMap: cnn,
    cnnModel: LEARNED_CNN_PACKED,
  };
  const crops: Record<string, Box> = {};
  let features: Float32Array = new Float32Array();
  await tick('crops', async () => {
    const base = { width: w, height: Math.round(w / 2.2), canvasFactory };
    const run = async (name: string, o: object) => {
      const r = await smartcrop.crop(c as any, { ...base, ...o } as any);
      crops[name] = r.topCrop;
      return r;
    };
    await run('classic', smartcrop.PRESETS.classic);
    features = (await run('learned', { emitFeatures: true }))
      .features as Float32Array;
    await run('learned+fsc', plug);
    await run('learned+fscv', { ...plug, consensus: 0.2 });
  });
  images[p.split('/').pop()!] = {
    pixels: sha(new Uint8Array(image.data.buffer)),
    faces: faces.map((f) => [f.x, f.y, f.width, f.height, f.score]),
    saliency: b64(sal.data),
    cnn: b64(cnn.data),
    features: b64(features),
    crops,
  };
}
const ort = (await import('onnxruntime-node/package.json')).version;
const out = {
  platform: `${process.platform}-${process.arch} bun ${Bun.version} ort ${ort} cpus ${navigator.hardwareConcurrency}`,
  timings,
  images,
};
await Bun.write(opt('out', 'probe.json')!, JSON.stringify(out));
console.log(out.platform, JSON.stringify(timings));
