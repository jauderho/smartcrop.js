// Learned composition scorer (src/learned.ts) on GAICD candidates, via the
// library's `candidates` option. Use with bench/gaicd-eval.ts:
//   bun bench/gaicd-eval.ts --scorer bench/learned/scorer.ts --split val
//
// Options: { with?: "faces,saliency" (cached ONNX plug-in maps, see
// bench/plugin-cache.ts), faceMinScore?: 0.6, ...smartcrop option overrides }
import { createCanvas, loadImage } from '@napi-rs/canvas';
import smartcrop from '../../src/smartcrop';
import { imageKey, parseWith, pluginOptions } from '../plugin-cache';

export async function scoreCandidates(
  imagePath: string,
  width: number,
  height: number,
  candidates: [number, number, number, number][],
  options: Record<string, unknown> = {}
): Promise<number[]> {
  const { with: withStr, faceMinScore = 0.6, ...overrides } = options;
  const img = await loadImage(imagePath);
  if (img.width !== width || img.height !== height) {
    throw new Error(
      `${imagePath}: ${img.width}x${img.height} != annotation ${width}x${height}`
    );
  }
  const use = parseWith(withStr as string | undefined);
  const po = use.size
    ? await pluginOptions(
        'gaicd',
        imageKey(imagePath),
        use,
        { width, height },
        async () => {
          const c = createCanvas(width, height);
          const ctx = c.getContext('2d');
          ctx.drawImage(img, 0, 0);
          return ctx.getImageData(0, 0, width, height);
        },
        faceMinScore as number
      )
    : {};
  const r = await smartcrop.crop(img, {
    width,
    height,
    canvasFactory: (w, h) => createCanvas(w, h),
    scorer: 'learned',
    candidates: candidates.map(([x1, y1, x2, y2]) => ({
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
    })),
    ...po,
    ...overrides,
  });
  return r.crops!.map((c) => c.score!.total);
}
