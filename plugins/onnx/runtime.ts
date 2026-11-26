// Shared lazy loader for onnxruntime-node and model sessions.
import { join } from 'node:path';

type Ort = typeof import('onnxruntime-node');
export type Session = import('onnxruntime-node').InferenceSession;

// CoreML flags: MLProgram format (the older NeuralNetwork format is slower
// than CPU), static input shapes only (dynamic-shape graphs such as the cnn
// backbone stay on CPU instead of recompiling per size), CPU + GPU: the
// Neural Engine is slower for these models or fails to compile them.
const COREML_FLAGS = 0x010 | 0x008 | 0x020;

type Backend = 'coreml' | 'cpu';
/**
 * Inference backend: SMARTCROP_BACKEND=coreml|cpu.
 *   coreml  onnxruntime's CoreML provider, CPU for what CoreML can't run;
 *           ~2x faster than CPU (U2-Netp 93 vs 175 ms on an M2).
 *   cpu     onnxruntime on the CPU: the only backend whose outputs match the
 *           Linux Train workflow bit for bit (CoreML differs by ~1e-4).
 * Default: coreml on Apple silicon, else cpu.
 */
export const backend = (): Backend => {
  const b =
    process.env.SMARTCROP_BACKEND ||
    (process.platform === 'darwin' && process.arch === 'arm64'
      ? 'coreml'
      : 'cpu');
  if (b !== 'coreml' && b !== 'cpu')
    throw new Error(`SMARTCROP_BACKEND must be coreml or cpu, got ${b}`);
  return b;
};

/**
 * Session options. SMARTCROP_ORT_THREADS caps onnxruntime's intra-op threads
 * (default: onnxruntime's own choice, all cores); useful when several worker
 * processes share a machine.
 */
export const sessionOptions = (ep: Backend = 'cpu') => {
  const t = Number(process.env.SMARTCROP_ORT_THREADS);
  return {
    logSeverityLevel: 3 as const,
    executionProviders:
      ep === 'coreml'
        ? [{ name: 'coreml', coreMlFlags: COREML_FLAGS }, 'cpu']
        : ['cpu'],
    ...(t > 0 ? { intraOpNumThreads: t, interOpNumThreads: 1 } : {}),
  };
};

/** Session on the chosen backend; CPU if CoreML can't load the model. */
export const createSession = async (
  model: string | Uint8Array
): Promise<Session> => {
  const ort = await getOrt();
  // `as string`: create() has one overload per argument type.
  const open = (ep: Backend) =>
    ort.InferenceSession.create(model as string, sessionOptions(ep));
  const b = backend();
  return b === 'cpu' ? open('cpu') : open('coreml').catch(() => open('cpu'));
};

export const modelDir = join(import.meta.dir, '../../bench/models');

let ortPromise: Promise<Ort> | undefined;
export const getOrt = (): Promise<Ort> =>
  (ortPromise ??= import('onnxruntime-node'));

const sessions = new Map<string, Promise<Session>>();
/** Create (once) and cache an InferenceSession for a model file. */
export const getSession = (path: string): Promise<Session> => {
  let s = sessions.get(path);
  if (!s) {
    s = createSession(path);
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
