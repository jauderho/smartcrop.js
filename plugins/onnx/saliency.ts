// Salient-object plug-in (U2-Netp via onnxruntime-node) for smartcrop's
// `importanceMap` option. Model: bench/models/u2netp.onnx
// (`bun scripts/fetch-models.ts`).
//
//   const importanceMap = await saliencyMap(imageData);
//   smartcrop.crop(img, { ...opts, importanceMap, importanceMapWeight: 1 });
import type { ImportanceMap } from '../../src/smartcrop';
import {
  getOrt,
  getSession,
  modelDir,
  resizeRGBA,
  type RGBAImage,
} from './runtime';
import { join } from 'node:path';

const SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export async function saliencyMap(
  image: RGBAImage,
  opts: { modelPath?: string } = {}
): Promise<ImportanceMap> {
  const ort = await getOrt();
  const session = await getSession(
    opts.modelPath ?? join(modelDir, 'u2netp.onnx')
  );
  const rgba = resizeRGBA(image, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', input, [
      1,
      3,
      SIZE,
      SIZE,
    ]),
  });
  const d0 = out[session.outputNames[0]].data as Float32Array;
  let min = Infinity;
  let max = -Infinity;
  for (const v of d0) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const data = new Float32Array(plane);
  for (let i = 0; i < plane; i++) data[i] = (d0[i] - min) / range;
  return { width: SIZE, height: SIZE, data };
}
