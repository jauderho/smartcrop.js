// Shared lazy loader for onnxruntime-node and model sessions.
import { join } from 'node:path';

type Ort = typeof import('onnxruntime-node');
type Session = import('onnxruntime-node').InferenceSession;

export const modelDir = join(import.meta.dir, '../../bench/models');

let ortPromise: Promise<Ort> | undefined;
export const getOrt = (): Promise<Ort> =>
  (ortPromise ??= import('onnxruntime-node'));

const sessions = new Map<string, Promise<Session>>();
/** Create (once) and cache an InferenceSession for a model file. */
export const getSession = (path: string): Promise<Session> => {
  let s = sessions.get(path);
  if (!s) {
    s = getOrt().then((ort) =>
      ort.InferenceSession.create(path, { logSeverityLevel: 3 })
    );
    sessions.set(path, s);
  }
  return s;
};

export interface RGBAImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

/** Bilinear resample of RGBA into dw x dh (area-ish via bilinear sampling). */
export const resizeRGBA = (
  src: RGBAImage,
  dw: number,
  dh: number
): Uint8ClampedArray => {
  // Copy first: some canvas backends expose slow typed-array views.
  const sd = new Uint8Array(
    src.data.buffer.slice(
      src.data.byteOffset,
      src.data.byteOffset + src.data.byteLength
    )
  );
  const out = new Uint8ClampedArray(dw * dh * 4);
  const sw = src.width; // hoisted: getters can be slow on canvas ImageData
  const sh = src.height;
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
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = sd[(y0 * sw + x0) * 4 + c];
        const b = sd[(y0 * sw + x1) * 4 + c];
        const d = sd[(y1 * sw + x0) * 4 + c];
        const e = sd[(y1 * sw + x1) * 4 + c];
        out[o + c] =
          (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
      }
    }
  }
  return out;
};
