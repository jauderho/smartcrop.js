// Golden crop results: every sample image cropped with a few option sets.
// test/golden.test.ts checks the library still produces exactly these.
// Regenerate (only when a change to crop results is intended):
//   bun test/golden.ts > test/fixtures/golden-crops.json
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { Glob } from 'bun';
import { join, relative } from 'node:path';

const root = join(import.meta.dir, '..');
const canvasFactory = (w: number, h: number) => createCanvas(w, h);

export const OPTION_SETS = {
  square: { width: 256, height: 256 },
  wide: { width: 400, height: 200, ruleOfThirds: false },
  aspect: { aspect: 0.75, minScale: 0.8 },
  boosted: {
    width: 100,
    height: 100,
    boost: [{ x: 0, y: 0, width: 64, height: 64, weight: 1 }],
  },
};

export function imagePaths() {
  return [...new Glob('examples/images/**/*.jpg').scanSync(root)]
    .sort()
    .map((p) => join(root, p));
}

type Cropper = {
  crop(
    image: unknown,
    options: object
  ): Promise<{
    topCrop: {
      x: number;
      y: number;
      width: number;
      height: number;
      score?: { total: number };
    };
  }>;
};

export async function computeGolden(smartcrop: Cropper) {
  const out: Record<string, Record<string, number[]>> = {};
  for (const path of imagePaths()) {
    const img = await loadImage(path);
    const row: Record<string, number[]> = {};
    for (const [name, opts] of Object.entries(OPTION_SETS)) {
      // Golden crops pin the classic algorithm (the default before 4.0).
      const { topCrop: c } = await smartcrop.crop(img, {
        scorer: 'classic',
        ...opts,
        canvasFactory,
      });
      row[name] = [c.x, c.y, c.width, c.height, c.score?.total ?? NaN];
    }
    out[relative(root, path)] = row;
  }
  return out;
}

if (import.meta.main) {
  const lib = process.argv[2] ?? join(root, 'src/smartcrop.ts');
  const mod = await import(lib);
  const golden = await computeGolden(mod.default ?? mod);
  console.log(JSON.stringify(golden, null, 1));
}
