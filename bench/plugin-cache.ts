// Shared helpers for the optional ONNX plug-ins in the benchmarks: per-image
// cache under bench/out/cache/<dataset>/<imageKey>.{faces.json,saliency.bin,cnn.bin}.
// The saliency .bin is: uint32 LE width, uint32 LE height, then Float32 LE data.
// The cnn .bin is: uint32 LE width, height, channels, then Float32 LE data
// (channels-last, projected with bench/out/cnn/projection.bin, see
// bench/learned/cnn-pca.ts).
import { createCanvas } from '@napi-rs/canvas';
import { mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import type { ImportanceMap } from '../src/smartcrop';
import type { FeatureMap } from '../src/cnn';
import { cnnFeatureMap, type CnnProjection } from '../plugins/onnx/cnn';
import { detectFaces, type Face } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';

export type ImageDataLike = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};
export type Plugin = 'faces' | 'saliency' | 'cnn';

// Cached with a low threshold; consumers filter with a minimum score.
export const CACHE_FACE_THRESHOLD = 0.5;

export const imageKey = (path: string) =>
  basename(path).replace(/\.[^.]+$/, '');

// SMARTCROP_CACHE_DIR relocates the cache (CI cache dir, scratch runs).
const dirFor = (dataset: string) =>
  join(
    process.env.SMARTCROP_CACHE_DIR ?? join(import.meta.dir, 'out/cache'),
    dataset
  );

export async function loadFaces(
  dataset: string,
  key: string,
  getData: () => Promise<ImageDataLike>
): Promise<Face[]> {
  const f = Bun.file(join(dirFor(dataset), `${key}.faces.json`));
  if (await f.exists()) return f.json();
  const faces = await detectFaces(await getData(), {
    scoreThreshold: CACHE_FACE_THRESHOLD,
  });
  await mkdir(dirFor(dataset), { recursive: true });
  await Bun.write(f, JSON.stringify(faces));
  return faces;
}

export async function loadSaliency(
  dataset: string,
  key: string,
  getData: () => Promise<ImageDataLike>
): Promise<ImportanceMap> {
  const f = Bun.file(join(dirFor(dataset), `${key}.saliency.bin`));
  if (await f.exists()) {
    const buf = await f.arrayBuffer();
    const [width, height] = new Uint32Array(buf, 0, 2);
    return { width, height, data: new Float32Array(buf.slice(8)) };
  }
  const map = await saliencyMap(await getData());
  const data = map.data as Float32Array;
  const out = new Uint8Array(8 + data.byteLength);
  new Uint32Array(out.buffer, 0, 2).set([map.width, map.height]);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 8);
  await mkdir(dirFor(dataset), { recursive: true });
  await Bun.write(f, out);
  return map;
}

export const PROJECTION_FILE = join(import.meta.dir, 'out/cnn/projection.bin');

/** Fixed PCA projection (Float32 LE: uint32 C, mean[448], proj[448 x C]). */
let projection: Promise<CnnProjection> | undefined;
export const loadProjection = (): Promise<CnnProjection> =>
  (projection ??= (async () => {
    const buf = await Bun.file(PROJECTION_FILE).arrayBuffer();
    const channels = new Uint32Array(buf, 0, 1)[0];
    const f = new Float32Array(buf, 4);
    return { channels, mean: f.slice(0, 448), proj: f.slice(448) };
  })());

export async function loadCnn(
  dataset: string,
  key: string,
  getData: () => Promise<ImageDataLike>
): Promise<FeatureMap> {
  const f = Bun.file(join(dirFor(dataset), `${key}.cnn.bin`));
  if (await f.exists()) {
    const buf = await f.arrayBuffer();
    const [width, height, channels] = new Uint32Array(buf, 0, 3);
    return { width, height, channels, data: new Float32Array(buf.slice(12)) };
  }
  const map = await cnnFeatureMap(await getData(), {
    projection: await loadProjection(),
  });
  const out = new Uint8Array(12 + map.data.byteLength);
  new Uint32Array(out.buffer, 0, 3).set([map.width, map.height, map.channels]);
  out.set(new Uint8Array(map.data.buffer), 12);
  await mkdir(dirFor(dataset), { recursive: true });
  await Bun.write(f, out);
  return map;
}

export const hasCache = (dataset: string, key: string, p: Plugin) =>
  Bun.file(
    join(
      dirFor(dataset),
      p === 'faces'
        ? `${key}.faces.json`
        : p === 'cnn'
          ? `${key}.cnn.bin`
          : `${key}.saliency.bin`
    )
  ).exists();

/** Parse "--with faces,saliency" into a set; throws on unknown names. */
export function parseWith(v: string | undefined): Set<Plugin> {
  const set = new Set<Plugin>();
  for (const p of (v ?? '').split(',').filter(Boolean)) {
    if (p !== 'faces' && p !== 'saliency' && p !== 'cnn')
      throw new Error(`unknown plugin ${p}`);
    set.add(p);
  }
  return set;
}

export const toImageData = (
  img: {
    width: number;
    height: number;
  },
  draw: (ctx: any) => void
): ImageDataLike => {
  const c = createCanvas(img.width, img.height);
  const ctx = c.getContext('2d');
  draw(ctx);
  return ctx.getImageData(0, 0, img.width, img.height) as ImageDataLike;
};

/**
 * Crop options contributed by the plug-ins. Missing cache entries are computed
 * on the fly (and cached).
 */
export async function pluginOptions(
  dataset: string,
  key: string,
  use: Set<Plugin>,
  size: { width: number; height: number },
  getData: () => Promise<ImageDataLike>,
  faceMinScore = 0.6
): Promise<Record<string, unknown>> {
  const opts: Record<string, unknown> = {};
  if (use.has('faces')) {
    const { faceBoosts } = await import('../plugins/onnx/faces');
    const faces = (await loadFaces(dataset, key, getData)).filter(
      (f) => f.score >= faceMinScore
    );
    opts.boost = faceBoosts(faces, {
      imageWidth: size.width,
      imageHeight: size.height,
    });
  }
  if (use.has('saliency')) {
    opts.importanceMap = await loadSaliency(dataset, key, getData);
  }
  if (use.has('cnn')) {
    opts.featureMap = await loadCnn(dataset, key, getData);
    // Head weights: a staged model (SMARTCROP_CNN_MODEL=path to a PackedCnn
    // json, for selection runs) or the shipped ones. bench/learned/dump.ts
    // pools features itself and sets cnnK instead.
    const staged = process.env.SMARTCROP_CNN_MODEL;
    if (!process.env.SMARTCROP_CNN_DUMP) {
      opts.cnnModel = staged
        ? await Bun.file(staged).json()
        : (await import('../src/learned-cnn-weights')).LEARNED_CNN_PACKED;
    }
  }
  return opts;
}
