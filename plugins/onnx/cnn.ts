// Deep-feature plug-in (MobileNetV2 via onnxruntime-node) for the learned
// scorer's `featureMap` option. Model: bench/models/mobilenetv2-12.onnx
// (`bun scripts/fetch-models.ts`), a frozen ImageNet backbone.
//
//   const featureMap = await cnnFeatureMap(imageData);
//   const cnnModel = LEARNED_CNN_PACKED; // src/learned-cnn-weights.ts
//   smartcrop.crop(img, { scorer: 'learned', featureMap, cnnModel, ... });
//
// The image is resized to a 256 px short side (dims rounded to multiples of
// 32), the stride 8, 16 and 32 feature maps are resampled to the stride-16 grid
// and concatenated (32 + 96 + 320 = 448 channels), then projected to C channels
// by a fixed linear map (PCA fitted on GAICD train images) so that crop
// pooling in JS (src/cnn.ts) stays cheap.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { FeatureMap, PackedCnn } from '../../src/cnn';
import { decodeF16 } from '../../src/learned';
import { getOrt, modelDir, resizeRGBA, type RGBAImage } from './runtime';
import { patchOnnx } from './onnx-patch';

type Session = import('onnxruntime-node').InferenceSession;

const SHORT_SIDE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
// Stride 8, 16, 32 tensors of mobilenetv2-12.onnx.
const TAPS = ['365', '426', '624'] as const;
const TAP_CHANNELS = [32, 96, 320];
export const RAW_CHANNELS = 448;

/** Fixed linear projection of the 448-channel map: y = (x - mean) . proj */
export interface CnnProjection {
  mean: Float32Array; // RAW_CHANNELS
  proj: Float32Array; // RAW_CHANNELS x channels, row-major
  channels: number;
}

/** Projection stored in the packed cnn weights (float16). */
export const projectionOf = (p: PackedCnn): CnnProjection => {
  const all = decodeF16(p.projection);
  return {
    mean: all.slice(0, p.inChannels),
    proj: all.slice(p.inChannels),
    channels: p.channels,
  };
};

const sessions = new Map<string, Promise<Session>>();
const getBackbone = (path: string): Promise<Session> => {
  let s = sessions.get(path);
  if (!s) {
    s = (async () => {
      const ort = await getOrt();
      const bytes = patchOnnx(new Uint8Array(await readFile(path)), [...TAPS]);
      return ort.InferenceSession.create(bytes, { logSeverityLevel: 3 });
    })();
    sessions.set(path, s);
  }
  return s;
};

const round32 = (v: number) => Math.max(32, Math.round(v / 32) * 32);

// Bilinear resample of a channels-first map (c x sh x sw) to the grid
// (dh x dw), written channels-last into dst at channel offset `co` (stride C).
const resampleInto = (
  src: Float32Array,
  c: number,
  sw: number,
  sh: number,
  dst: Float32Array,
  dw: number,
  dh: number,
  C: number,
  co: number
): void => {
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.max(0, (y + 0.5) * sy - 0.5);
    const y0 = Math.min(sh - 1, Math.floor(fy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.max(0, (x + 0.5) * sx - 0.5);
      const x0 = Math.min(sw - 1, Math.floor(fx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = fx - x0;
      const o = (y * dw + x) * C + co;
      for (let k = 0; k < c; k++) {
        const b = k * sw * sh;
        dst[o + k] =
          (src[b + y0 * sw + x0] * (1 - wx) + src[b + y0 * sw + x1] * wx) *
            (1 - wy) +
          (src[b + y1 * sw + x0] * (1 - wx) + src[b + y1 * sw + x1] * wx) * wy;
      }
    }
  }
};

/** The unprojected 448-channel stride-16 map. */
export async function cnnRawMap(
  image: RGBAImage,
  opts: { modelPath?: string } = {}
): Promise<FeatureMap> {
  const ort = await getOrt();
  const session = await getBackbone(
    opts.modelPath ?? join(modelDir, 'mobilenetv2-12.onnx')
  );
  const scale = SHORT_SIDE / Math.min(image.width, image.height);
  const W = round32(image.width * scale);
  const H = round32(image.height * scale);
  const rgba = resizeRGBA(image, W, H);
  const plane = W * H;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  const out = await session.run({
    input: new ort.Tensor('float32', input, [1, 3, H, W]),
  });
  const gw = W / 16;
  const gh = H / 16;
  const data = new Float32Array(gw * gh * RAW_CHANNELS);
  let co = 0;
  TAPS.forEach((name, t) => {
    const tensor = out[name];
    const [, c, h, w] = tensor.dims as number[];
    if (c !== TAP_CHANNELS[t]) throw new Error(`unexpected ${name} shape`);
    resampleInto(
      tensor.data as Float32Array,
      c,
      w,
      h,
      data,
      gw,
      gh,
      RAW_CHANNELS,
      co
    );
    co += c;
  });
  return { width: gw, height: gh, channels: RAW_CHANNELS, data };
}

/** Project a raw map with a fixed linear projection. */
export const projectMap = (
  raw: FeatureMap,
  p: CnnProjection
): FeatureMap => {
  const n = raw.width * raw.height;
  const C = p.channels;
  const R = raw.channels;
  const data = new Float32Array(n * C);
  for (let i = 0; i < n; i++) {
    const o = i * R;
    for (let k = 0; k < R; k++) {
      const v = raw.data[o + k] - p.mean[k];
      if (v === 0) continue;
      const po = k * C;
      for (let c = 0; c < C; c++) data[i * C + c] += v * p.proj[po + c];
    }
  }
  return { width: raw.width, height: raw.height, channels: C, data };
};

/**
 * Projected feature map of the whole image (grid coordinates relative to the
 * image). `projection` defaults to the one in src/learned-cnn-weights.ts.
 */
export async function cnnFeatureMap(
  image: RGBAImage,
  opts: { modelPath?: string; projection?: CnnProjection } = {}
): Promise<FeatureMap> {
  const projection =
    opts.projection ??
    projectionOf(
      (await import('../../src/learned-cnn-weights')).LEARNED_CNN_PACKED
    );
  return projectMap(await cnnRawMap(image, opts), projection);
}
