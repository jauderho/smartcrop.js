import { beforeAll, describe, expect, it } from 'bun:test';
import { createCanvas, loadImage, type Image } from '@napi-rs/canvas';
import smartcrop, {
  type CanvasFactory,
  type CropOptions,
  type CropResult,
} from '../src/smartcrop';

// Internal helpers are exposed on the module for testing.
const internals = smartcrop;

const KITTY = new URL('../examples/images/flickr/kitty.jpg', import.meta.url)
  .pathname;

const canvasFactory: CanvasFactory = (w, h) => createCanvas(w, h);
const crop = (image: unknown, options: CropOptions = {}) =>
  smartcrop.crop(image, { canvasFactory, ...options });

let img: Image;
beforeAll(async () => {
  img = await loadImage(KITTY);
});

function expectValidResult(result: CropResult) {
  const { x, y, width, height } = result.topCrop;
  expect(x).toBeGreaterThanOrEqual(0);
  expect(x).toBeLessThanOrEqual(img.width - width);
  expect(y).toBeGreaterThanOrEqual(0);
  expect(y).toBeLessThanOrEqual(img.height - height);
  expect(width).toBeGreaterThanOrEqual(1);
  expect(width).toBeLessThanOrEqual(img.width);
  expect(height).toBeGreaterThanOrEqual(1);
  expect(height).toBeLessThanOrEqual(img.height);
}

function expectAspectRatio(result: CropResult, expected: number) {
  const ratio = result.topCrop.width / result.topCrop.height;
  expect(Math.abs(ratio - expected)).toBeLessThanOrEqual(0.01);
}

describe('isAvailable', () => {
  it('returns true when a canvas factory is available', () => {
    expect(smartcrop.isAvailable({ canvasFactory })).toBe(true);
  });
});

describe('crop', () => {
  it('does something sane', async () => {
    const c = createCanvas(128, 64);
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 128, 64);
    ctx.fillStyle = 'red';
    ctx.fillRect(96, 32, 16, 16);
    const result = await crop(c, { width: 32, height: 32 });
    expectAspectRatio(result, 1);
    expect(result.topCrop.x).toBeLessThan(96);
    expect(result.topCrop.y).toBeLessThan(32);
    expect(result.topCrop.x + result.topCrop.width).toBeGreaterThan(112);
    expect(result.topCrop.y + result.topCrop.height).toBeGreaterThan(48);
  });

  it('takes into account ruleOfThirds', async () => {
    const c = createCanvas(256, 128);
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 256, 128);
    ctx.fillStyle = 'red';
    ctx.fillRect(128, 24, 8, 80);
    const thirds = await crop(c, {
      width: 32,
      height: 32,
      ruleOfThirds: true,
      debug: true,
    });
    const noThirds = await crop(c, {
      width: 32,
      height: 32,
      ruleOfThirds: false,
      debug: true,
    });
    expectValidResult(thirds);
    expectValidResult(noThirds);
    expectAspectRatio(noThirds, 1);
    expect(noThirds.topCrop.x + noThirds.topCrop.width / 2).toBe(128);
    expect(thirds.topCrop.x + thirds.topCrop.width / 2).toBe(144);
  });

  it('adheres to minScale', async () => {
    const result = await crop(img, { minScale: 1 });
    expectValidResult(result);
    expectAspectRatio(result, 1);
    expect(result.topCrop.y).toBe(0);
    expect(result.topCrop.height).toBe(img.height);
  });

  it('takes into account boost', async () => {
    const boost = [
      {
        x: img.width - 128,
        y: img.height - 128,
        width: 64,
        height: 64,
        weight: 1.0,
      },
    ];
    const result = await crop(img, { boost, width: 64, height: 64 });
    expectValidResult(result);
    expect(result.topCrop.y).toBe(0);
    expect(result.topCrop.x).toBe(200);
    expect(result.topCrop.height).toBe(img.height);
  });

  it('takes into account the aspect ratio', async () => {
    const wide = await crop(img, { width: 128, height: 64 });
    expectValidResult(wide);
    expectAspectRatio(wide, 2);
    const hd = await crop(img, { width: 160, height: 90 });
    expectValidResult(hd);
    expectAspectRatio(hd, 16 / 9);
  });

  it('crops the kitty', async () => {
    expectValidResult(await crop(img));
  });

  it('calls the optional callback with the result', async () => {
    let called: CropResult | undefined;
    const result = await smartcrop.crop(img, { canvasFactory }, (r) => {
      called = r;
    });
    expect(called).toBe(result);
  });
});

describe('iop.open', () => {
  it('passes input options', async () => {
    const iop = internals._canvasImageOperations(canvasFactory);
    const open = iop.open;
    let inputOptions: unknown;
    iop.open = (image, inputOptions_) => {
      inputOptions = inputOptions_;
      return open(image);
    };
    const options = { imageOperations: iop, input: { foo: 'bar' } };
    const result = await smartcrop.crop(img, options);
    expect(inputOptions).toBe(options.input);
    expectValidResult(result);
  });
});

describe('smartcrop.Promise', () => {
  class CustomPromise<T> extends Promise<T> {}

  it('is used by every image operation', async () => {
    const original = smartcrop.Promise;
    smartcrop.Promise = CustomPromise;
    try {
      const iop = internals._canvasImageOperations(canvasFactory);
      const c = createCanvas(16, 16);
      const opened = iop.open(c);
      const resampled = iop.resample(c, 8, 8);
      const data = iop.getData(c);
      expect(opened).toBeInstanceOf(CustomPromise);
      expect(resampled).toBeInstanceOf(CustomPromise);
      expect(data).toBeInstanceOf(CustomPromise);
      expect((await resampled) as { width: number }).toMatchObject({
        width: 8,
      });
      expect((await data).width).toBe(16);
    } finally {
      smartcrop.Promise = original;
    }
  });
});

describe('_downSample', () => {
  function mono2rgba(input: number[]) {
    const output = new Uint8ClampedArray(input.length * 4);
    input.forEach((v, i) => output.fill(v, i * 4, i * 4 + 4));
    return output;
  }
  const input = {
    width: 4,
    height: 4,
    data: mono2rgba([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2]),
  };

  it('keeps the image the same at a factor of one', () => {
    const output = internals._downSample(input, 1);
    expect(output.width).toBe(input.width);
    expect(output.height).toBe(input.height);
    expect(output.data).toEqual(input.data);
  });

  it('samples down an image by a factor of two', () => {
    const output = internals._downSample(input, 2);
    expect(output.width).toBe(input.width / 2);
    expect(output.height).toBe(input.height / 2);
    expect(output.data).toEqual(
      new Uint8ClampedArray([5, 4, 4, 4, 7, 6, 6, 6, 8, 7, 6, 6, 6, 5, 4, 4])
    );
  });

  it('keeps a constant value constant', () => {
    const w = 59;
    const h = 23;
    const constant = {
      width: w,
      height: h,
      data: new Uint8ClampedArray(w * h * 4).fill(119),
    };
    const output = internals._downSample(constant, 8);
    expect(output.width).toBe(Math.floor(w / 8));
    expect(output.height).toBe(Math.floor(h / 8));
    for (const v of output.data) {
      expect(v).toBeGreaterThanOrEqual(118);
      expect(v).toBeLessThanOrEqual(120);
    }
  });
});
