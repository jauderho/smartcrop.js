// Shared helpers for the optional ONNX plug-ins in the benchmarks: per-image
// cache under bench/out/cache/<dataset>/<imageKey>.{faces.json,saliency.bin}.
// The saliency .bin is: uint32 LE width, uint32 LE height, then Float32 LE data.
import { createCanvas } from '@napi-rs/canvas';
import { mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import type { ImportanceMap } from '../src/smartcrop';
import { detectFaces, type Face } from '../plugins/onnx/faces';
import { saliencyMap } from '../plugins/onnx/saliency';

export type ImageDataLike = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};
export type Plugin = 'faces' | 'saliency';

// Cached with a low threshold; consumers filter with a minimum score.
export const CACHE_FACE_THRESHOLD = 0.5;

export const imageKey = (path: string) =>
  basename(path).replace(/\.[^.]+$/, '');

const dirFor = (dataset: string) => join(import.meta.dir, 'out/cache', dataset);

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

export const hasCache = (dataset: string, key: string, p: Plugin) =>
  Bun.file(
    join(
      dirFor(dataset),
      p === 'faces' ? `${key}.faces.json` : `${key}.saliency.bin`
    )
  ).exists();

/** Parse "--with faces,saliency" into a set; throws on unknown names. */
export function parseWith(v: string | undefined): Set<Plugin> {
  const set = new Set<Plugin>();
  for (const p of (v ?? '').split(',').filter(Boolean)) {
    if (p !== 'faces' && p !== 'saliency')
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
  return opts;
}
