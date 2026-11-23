// Worker: holds a shard of decoded images per dataset set and evaluates
// smartcrop configs on them (box-eval FCDB/FLMS protocol, in-process).
import { createCanvas, loadImage, type Image } from '@napi-rs/canvas';
import { join } from 'node:path';
import smartcrop from '../../src/smartcrop';
import { imageKey, pluginOptions, type Plugin } from '../plugin-cache';
import { scoreBoxes, type Ann, type Box } from './common';

declare const self: Worker;
const canvasFactory = (w: number, h: number) => createCanvas(w, h);
const sets = new Map<
  string,
  { a: Ann; img: Image; po: Record<string, unknown> }[]
>();

self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  if (m.type === 'load') {
    const items: { a: Ann; img: Image; po: Record<string, unknown> }[] = [];
    const use = new Set<Plugin>(m.use ?? []);
    let skipped = 0;
    for (const a of m.anns as Ann[]) {
      try {
        const img = await loadImage(join(m.dir, a.image));
        if (img.width !== a.width || img.height !== a.height)
          throw new Error('size');
        const po = use.size
          ? await pluginOptions(
              m.dataset,
              imageKey(a.image),
              use,
              a,
              async () => {
                const c = createCanvas(img.width, img.height);
                const ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0);
                return ctx.getImageData(0, 0, img.width, img.height);
              }
            )
          : {};
        items.push({ a, img, po });
      } catch {
        skipped++;
      }
    }
    sets.set(m.set, items);
    postMessage({
      id: m.id,
      loaded: items.length,
      skipped,
      images: items.map((x) => x.a.image),
    });
  } else if (m.type === 'eval') {
    const t0 = performance.now();
    const items = sets.get(m.set) ?? [];
    const out = new Float64Array(items.length * 2);
    const options = { minScale: 0.5, maxScale: 1, ...m.options };
    for (let i = 0; i < items.length; i++) {
      const { a, img, po } = items[i];
      const cache = new Map<string, Box>();
      const preds: Box[] = [];
      for (const gt of a.boxes) {
        const key = `${gt[2]}x${gt[3]}`;
        let sc = cache.get(key);
        if (!sc) {
          const { topCrop } = await smartcrop.crop(img as never, {
            width: gt[2],
            height: gt[3],
            canvasFactory,
            ...options,
            ...(m.noPlugins ? {} : po),
          });
          sc = [topCrop.x, topCrop.y, topCrop.width, topCrop.height];
          cache.set(key, sc);
        }
        preds.push(sc);
      }
      const [i_, b_] = scoreBoxes(preds, a);
      out[i * 2] = i_;
      out[i * 2 + 1] = b_;
    }
    postMessage({ id: m.id, res: out, secs: (performance.now() - t0) / 1000 });
  }
};
