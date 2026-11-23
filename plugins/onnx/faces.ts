// Face detection plug-in (YuNet via onnxruntime-node) for smartcrop's `boost`
// option. Model: bench/models/face_detection_yunet_2023mar.onnx
// (`bun scripts/fetch-models.ts`).
//
//   const faces = await detectFaces(imageData);
//   smartcrop.crop(img, { ...opts, boost: faceBoosts(faces) });
import type { CropBoost } from '../../src/smartcrop';
import {
  getOrt,
  getSession,
  modelDir,
  resizeRGBA,
  type RGBAImage,
} from './runtime';
import { join } from 'node:path';

export interface Face {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
}

export interface DetectOptions {
  modelPath?: string;
  /** Long side of the (letterboxed) network input. Default 640. */
  size?: number;
  scoreThreshold?: number;
  nmsThreshold?: number;
}

const STRIDES = [8, 16, 32];
const SIZE = 640;

const iou = (a: Face, b: Face) => {
  const iw = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const ih = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (a.width * a.height + b.width * b.height - inter);
};

const nms = (faces: Face[], thr: number): Face[] => {
  const sorted = [...faces].sort((a, b) => b.score - a.score);
  const keep: Face[] = [];
  for (const f of sorted) {
    if (keep.every((k) => iou(k, f) <= thr)) keep.push(f);
  }
  return keep;
};

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export async function detectFaces(
  image: RGBAImage,
  opts: DetectOptions = {}
): Promise<Face[]> {
  // The distributed YuNet graph has a fixed 640x640 input: letterbox into it
  // (top-left aligned, zero padded).
  const size = opts.size ?? SIZE;
  const scoreThr = opts.scoreThreshold ?? 0.6;
  const nmsThr = opts.nmsThreshold ?? 0.3;
  const ort = await getOrt();
  const session = await getSession(
    opts.modelPath ?? join(modelDir, 'face_detection_yunet_2023mar.onnx')
  );

  const scale = Math.min(size / image.width, size / image.height);
  const rw = Math.max(1, Math.round(image.width * scale));
  const rh = Math.max(1, Math.round(image.height * scale));
  const rgba =
    rw === image.width && rh === image.height
      ? image.data
      : resizeRGBA(image, rw, rh);
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const i = (y * rw + x) * 4;
      const o = y * size + x;
      input[o] = rgba[i + 2]; // B
      input[plane + o] = rgba[i + 1]; // G
      input[2 * plane + o] = rgba[i]; // R
    }
  }
  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', input, [
      1,
      3,
      size,
      size,
    ]),
  });

  const found: Face[] = [];
  for (const s of STRIDES) {
    const cls = out[`cls_${s}`].data as Float32Array;
    const obj = out[`obj_${s}`].data as Float32Array;
    const bbox = out[`bbox_${s}`].data as Float32Array;
    const cols = size / s;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
      if (score < scoreThr) continue;
      const r = Math.floor(i / cols);
      const c = i % cols;
      const cx = (c + bbox[i * 4]) * s;
      const cy = (r + bbox[i * 4 + 1]) * s;
      const w = Math.exp(bbox[i * 4 + 2]) * s;
      const h = Math.exp(bbox[i * 4 + 3]) * s;
      found.push({
        x: (cx - w / 2) / scale,
        y: (cy - h / 2) / scale,
        width: w / scale,
        height: h / scale,
        score,
      });
    }
  }
  return nms(found, nmsThr).map((f) => {
    const x0 = Math.max(0, f.x);
    const y0 = Math.max(0, f.y);
    const x1 = Math.min(image.width, f.x + f.width);
    const y1 = Math.min(image.height, f.y + f.height);
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0, score: f.score };
  });
}

export interface FaceBoostOptions {
  /** Extra height above the box, as a fraction of box height. */
  headroom?: number;
  /** Extra width on each side, as a fraction of box width. */
  sidePad?: number;
  weight?: number;
  /** Image size to clamp to; if omitted, only clamped at 0. */
  imageWidth?: number;
  imageHeight?: number;
}

/**
 * Turn face boxes into smartcrop boosts, expanded upward (hair / top of head)
 * and sideways. Pass the image size to clamp to the image.
 */
export function faceBoosts(
  faces: Face[],
  opts: FaceBoostOptions = {}
): CropBoost[] {
  const { headroom = 0.4, sidePad = 0.1, weight = 1 } = opts;
  const W = opts.imageWidth ?? Infinity;
  const H = opts.imageHeight ?? Infinity;
  return faces.map((f) => {
    const x0 = Math.max(0, f.x - f.width * sidePad);
    const x1 = Math.min(W, f.x + f.width * (1 + sidePad));
    const y0 = Math.max(0, f.y - f.height * headroom);
    const y1 = Math.min(H, f.y + f.height);
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0, weight };
  });
}
