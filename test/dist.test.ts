// Smoke tests for the built files in dist/. Run `bun run build` first.
import { describe, expect, it } from 'bun:test';
import { createCanvas } from '@napi-rs/canvas';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';

const dist = join(import.meta.dir, '..', 'dist');
const canvasFactory = (w: number, h: number) => createCanvas(w, h);

type Lib = {
  crop(
    image: unknown,
    options: object
  ): Promise<{ topCrop: { x: number; width: number } }>;
};

async function expectCrops(lib: Lib) {
  const c = createCanvas(128, 64);
  const ctx = c.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 128, 64);
  ctx.fillStyle = 'red';
  ctx.fillRect(96, 32, 16, 16);
  const { topCrop } = await lib.crop(c, {
    width: 32,
    height: 32,
    canvasFactory,
  });
  expect(topCrop.x).toBeLessThan(96);
  expect(topCrop.x + topCrop.width).toBeGreaterThan(112);
}

describe('dist', () => {
  it('ES module default-exports smartcrop', async () => {
    const mod = await import(join(dist, 'smartcrop.js'));
    await expectCrops(mod.default);
  });

  it('CommonJS require returns smartcrop itself', async () => {
    const lib = createRequire(import.meta.url)(join(dist, 'smartcrop.cjs'));
    expect(typeof lib.crop).toBe('function');
    await expectCrops(lib);
  });

  it('browser script defines window.smartcrop and window.SmartCrop', async () => {
    const code = await Bun.file(join(dist, 'smartcrop.iife.js')).text();
    // The sandbox object becomes the script's global object (its window).
    const sandbox: { smartcrop?: Lib; SmartCrop?: Lib } = {};
    runInNewContext(code, sandbox);
    expect(sandbox.SmartCrop).toBe(sandbox.smartcrop);
    await expectCrops(sandbox.smartcrop!);
  });

  it('ships generated type definitions', async () => {
    const dts = await Bun.file(join(dist, 'smartcrop.d.ts')).text();
    expect(dts).toContain('export default');
    for (const name of ['Crop', 'CropResult', 'CropOptions', 'CropBoost']) {
      expect(dts).toMatch(new RegExp(`\\b(interface|type) ${name}\\b`));
    }
  });
});
