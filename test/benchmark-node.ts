// Run from the repo root: bun run bench
import Benchmark from 'benchmark';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import smartcrop from '../src/smartcrop';

const img = await loadImage(
  new URL('../examples/images/flickr/kitty.jpg', import.meta.url).pathname
);
const canvasFactory = (w: number, h: number) => createCanvas(w, h);

const bench = new Benchmark(
  'smartcrop.crop()',
  (deferred: { resolve(): void }) => {
    smartcrop
      .crop(img, { width: 256, height: 256, canvasFactory })
      .then(() => deferred.resolve());
  },
  {
    defer: true,
    onComplete: () => console.log(bench.toString()),
  }
);
bench.run();
