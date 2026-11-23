import { beforeAll, describe, expect, it } from 'bun:test';
import { createCanvas, loadImage, type Image } from '@napi-rs/canvas';
import smartcrop, {
  type CanvasFactory,
  type CropOptions,
  type CropResult,
} from '../src/smartcrop';
import { cnnContext, cnnDim, cnnFeatures, type FeatureMap } from '../src/cnn';
import { FEATURE_COUNT, packModel } from '../src/learned';

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
      scorer: 'classic',
      ruleOfThirds: true,
      debug: true,
    });
    const noThirds = await crop(c, {
      width: 32,
      height: 32,
      scorer: 'classic',
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
    const result = await crop(img, {
      scorer: 'classic',
      boost,
      width: 64,
      height: 64,
    });
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

describe('position', () => {
  it('reports the crop as object-position percentages', async () => {
    const result = await crop(img, { width: 64, height: 64 });
    const { x, y, width, height } = result.topCrop;
    const expectPercent = (offset: number, slack: number) =>
      slack > 0 ? (100 * offset) / slack : 50;
    expect(result.position!.x).toBeCloseTo(
      expectPercent(x, img.width - width),
      6
    );
    expect(result.position!.y).toBeCloseTo(
      expectPercent(y, img.height - height),
      6
    );
  });

  it('is 50% on an axis with no slack', async () => {
    const c = createCanvas(100, 50);
    const result = await crop(c, { width: 100, height: 25 });
    expect(result.topCrop.width).toBe(100);
    expect(result.position!.x).toBe(50);
  });
});

describe('importanceMap', () => {
  it('pulls the crop toward the important region', async () => {
    // A flat grey image: nothing to choose between without the map.
    const c = createCanvas(300, 100);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, 300, 100);
    const map = { width: 30, height: 10, data: new Float32Array(300) };
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 5; x++) map.data[y * 30 + x] = 1;
    }
    const plain = await crop(c, { scorer: 'classic', width: 100, height: 100 });
    const mapped = await crop(c, {
      scorer: 'classic',
      width: 100,
      height: 100,
      importanceMap: map,
      importanceMapWeight: 1,
    });
    // Without the map the image-border "detail" pulls the crop right.
    expect(plain.topCrop.x).toBe(200);
    expect(mapped.topCrop.x).toBe(0);
  });
});

describe('PRESETS', () => {
  it('classic matches the classic scorer', async () => {
    const a = await crop(img, { scorer: 'classic', width: 100, height: 100 });
    const b = await crop(img, {
      ...smartcrop.PRESETS.classic,
      width: 100,
      height: 100,
    });
    expect(b.topCrop).toEqual(a.topCrop);
  });

  it('the default is the learned scorer; next is an alias', async () => {
    expect(smartcrop.DEFAULTS.scorer).toBe('learned');
    expect(smartcrop.PRESETS.next.scorer).toBe('learned');
    expect(smartcrop.PRESETS.classic.scorer).toBe('classic');
    const a = await crop(img, { width: 100, height: 100 });
    const b = await crop(img, { scorer: 'learned', width: 100, height: 100 });
    expect(a.topCrop).toEqual(b.topCrop);
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

describe('learned scorer', () => {
  it('returns a valid crop of the requested aspect', async () => {
    const result = await crop(img, { aspect: 1, scorer: 'learned' });
    expectValidResult(result);
    expectAspectRatio(result, 1);
  });

  it('honours boost boxes and the content floor', async () => {
    const result = await crop(img, {
      width: 3,
      height: 2,
      scorer: 'learned',
      contentFloor: 0.5,
      boost: [{ x: 10, y: 10, width: 40, height: 40, weight: 1 }],
    });
    expectValidResult(result);
  });

  it('scores exactly the given candidates, one score each', async () => {
    const candidates = [
      { x: 0, y: 0, width: img.width, height: img.height },
      { x: 10, y: 20, width: 100, height: 80 },
      { x: 33.5, y: 7, width: 200, height: 120 },
    ];
    for (const scorer of ['learned', 'classic'] as const) {
      const result = await crop(img, {
        width: img.width,
        height: img.height,
        scorer,
        candidates,
      });
      expect(result.crops).toHaveLength(candidates.length);
      result.crops!.forEach((c, i) => {
        expect(c.x).toBeCloseTo(candidates[i].x, 6);
        expect(c.width).toBeCloseTo(candidates[i].width, 6);
        expect(Number.isFinite(c.score!.total)).toBe(true);
      });
    }
  });

  it('exposes one feature vector per candidate', async () => {
    const f = await internals._features(
      img,
      [
        { x: 0, y: 0, width: 100, height: 100 },
        { x: 50, y: 50, width: 120, height: 90 },
      ],
      { width: img.width, height: img.height, canvasFactory }
    );
    expect(f.length % 2).toBe(0);
    expect(f.every(Number.isFinite)).toBe(true);
  });
});

describe('cnn deep features', () => {
  const C = 4;
  const K = 3;
  const fm: FeatureMap = {
    width: 12,
    height: 8,
    channels: C,
    data: Float32Array.from(
      { length: 12 * 8 * C },
      (_, i) => Math.sin(i * 0.37) + (i % C)
    ),
  };
  const D = FEATURE_COUNT + cnnDim(K, C);
  // A fixed tiny head: D -> 4 -> 1.
  const head = {
    mean: new Array(D).fill(0),
    std: new Array(D).fill(1),
    layers: [
      {
        w: Array.from({ length: D * 4 }, (_, i) => Math.cos(i) * 0.1),
        b: [0, 0.1, 0, 0],
      },
      { w: [1, -1, 0.5, 0.25], b: [0] },
    ],
  };
  const cnnModel = {
    k: K,
    channels: C,
    inChannels: 448,
    projection: '',
    head: packModel(head, 'f32'),
  };
  const run = (options: CropOptions) =>
    crop(img, {
      width: img.width,
      height: img.height,
      scorer: 'learned',
      candidates: [
        { x: 0, y: 0, width: img.width, height: img.height },
        { x: 10, y: 20, width: 100, height: 80 },
        { x: 33.5, y: 7, width: 200, height: 120 },
      ],
      ...options,
    });

  it('changes nothing without cnn weights', async () => {
    const base = await run({});
    for (const options of [{ featureMap: fm }, { cnnModel }]) {
      const r = await run(options);
      expect(r.crops!.map((c) => c.score!.total)).toEqual(
        base.crops!.map((c) => c.score!.total)
      );
      expect(r.topCrop).toEqual(base.topCrop);
    }
  });

  it('scores each candidate with the cnn head', async () => {
    const base = await run({});
    const r = await run({ featureMap: fm, cnnModel });
    expect(r.crops).toHaveLength(3);
    for (const c of r.crops!)
      expect(Number.isFinite(c.score!.total)).toBe(true);
    expect(r.crops!.map((c) => c.score!.total)).not.toEqual(
      base.crops!.map((c) => c.score!.total)
    );
  });

  it('pools RoI, RoD and the global mean', () => {
    const cx = cnnContext(fm);
    const out = new Float32Array(cnnDim(K, C));
    // The whole image: RoI bins average the map, nothing is outside.
    cnnFeatures(cx, K, 0, 0, 120, 80, 120, 80, out, 0);
    const roiEnd = K * K * C;
    expect(out.slice(roiEnd, 2 * roiEnd).every((v) => v === 0)).toBe(true);
    const g = out.slice(2 * roiEnd);
    for (let c = 0; c < C; c++) expect(g[c]).toBeCloseTo(cx.mean[c], 5);
    let mean0 = 0;
    for (let b = 0; b < K * K; b++) mean0 += out[b * C] / (K * K);
    expect(mean0).toBeCloseTo(cx.mean[0], 4);
    // A crop in the left half: bins fully inside it have no RoD.
    cnnFeatures(cx, K, 0, 0, 60, 80, 120, 80, out, 0);
    expect(out.slice(roiEnd, 2 * roiEnd).some((v) => v !== 0)).toBe(true);
    expect(out.every(Number.isFinite)).toBe(true);
  });
});
