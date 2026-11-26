// Shared lazy loader for onnxruntime-node and model sessions.
import { statSync } from 'node:fs';
import { join } from 'node:path';
import { createMpsSession, type Session } from './mps';

export type { Session };
type Ort = typeof import('onnxruntime-node');

// CoreML flags: MLProgram format (the older NeuralNetwork format is slower
// than CPU), static input shapes only (dynamic-shape graphs such as the cnn
// backbone stay on CPU instead of recompiling per size), CPU + GPU: the
// Neural Engine is slower for these models or fails to compile them.
const COREML_FLAGS = 0x010 | 0x008 | 0x020;

type Backend = 'apple' | 'mps' | 'coreml' | 'cpu';
/**
 * Inference backend: SMARTCROP_BACKEND=apple|mps|coreml|cpu.
 *   mps     PyTorch on the Apple GPU (plugins/onnx/mps.ts), ~4x faster than
 *           CPU for IS-Net (276 vs 1068 ms on an M2) but slower than CoreML
 *           for small models (worker round trip). Needs uv (or
 *           SMARTCROP_MPS_PYTHON); a worker that won't start, or a model it
 *           can't convert or run, falls back to coreml.
 *   coreml  onnxruntime's CoreML provider, CPU for what CoreML can't run;
 *           ~2x faster than CPU (U2-Netp 93 vs 175 ms).
 *   apple   mps for models of MPS_MIN_BYTES or more, coreml for smaller ones.
 *   cpu     onnxruntime on the CPU: the only backend whose outputs match the
 *           Linux Train workflow bit for bit (others differ by ~1e-6-1e-4).
 * Default: apple on Apple silicon, else cpu.
 */
export const backend = (): Backend => {
  const b =
    process.env.SMARTCROP_BACKEND ||
    (process.platform === 'darwin' && process.arch === 'arm64'
      ? 'apple'
      : 'cpu');
  if (b !== 'apple' && b !== 'mps' && b !== 'coreml' && b !== 'cpu')
    throw new Error(
      `SMARTCROP_BACKEND must be apple, mps, coreml or cpu, got ${b}`
    );
  return b;
};
// Measured crossover: U2-Netp (4.6 MB) and MobileNetV2 (14 MB) are faster on
// CoreML, IS-Net (170 MB) on MPS.
const MPS_MIN_BYTES = 50e6;

/**
 * Session options. SMARTCROP_ORT_THREADS caps onnxruntime's intra-op threads
 * (default: onnxruntime's own choice, all cores); useful when several worker
 * processes share a machine.
 */
export const sessionOptions = (ep: 'coreml' | 'cpu' = 'cpu') => {
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

const ortSession = async (
  model: string | Uint8Array,
  b: 'coreml' | 'cpu'
): Promise<Session> => {
  const ort = await getOrt();
  // `as string`: create() has one overload per argument type.
  const open = (ep: 'coreml' | 'cpu') =>
    ort.InferenceSession.create(model as string, sessionOptions(ep));
  return b === 'cpu' ? open('cpu') : open('coreml').catch(() => open('cpu'));
};

/**
 * Session on the chosen backend. MPS falls back to CoreML (then CPU) for a
 * model, for good, when the worker can't start, the model doesn't convert,
 * or a run fails (onnx2torch converts some ops it can't execute; the worker
 * may die).
 */
export const createSession = async (
  model: string | Uint8Array
): Promise<Session> => {
  let b = backend();
  if (b === 'apple') {
    const size =
      typeof model === 'string' ? statSync(model).size : model.byteLength;
    b = size >= MPS_MIN_BYTES ? 'mps' : 'coreml';
  }
  if (b !== 'mps') return ortSession(model, b);
  const name = typeof model === 'string' ? model.split('/').pop() : 'model';
  const fallback = (e: unknown) => {
    console.warn(
      `smartcrop: ${name} on CoreML, not MPS (${(e as Error).message.split('\n')[0]})`
    );
    return ortSession(model, 'coreml');
  };
  let mps: Session;
  try {
    mps = await createMpsSession(model);
  } catch (e) {
    return fallback(e);
  }
  let ort: Promise<Session> | undefined;
  return {
    inputNames: mps.inputNames,
    outputNames: mps.outputNames,
    async run(feeds) {
      if (!ort) {
        try {
          return await mps.run(feeds);
        } catch (e) {
          ort ??= fallback(e);
        }
      }
      return (await ort).run(feeds);
    },
  };
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
