// Render contact sheets from a gallery's data.json: one row per item with the
// crops of each method side by side, for quick visual review.
//
// Usage: bun bench/contact-sheet.ts <gallery-dir> [--methods classic,next,next+fs]
//          [--rows 8] [--sort change] [--out prefix]
// --sort change orders by the box difference between the first two methods.
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { join } from 'node:path';

const args = process.argv.slice(2);
const flag = (n: string, d?: string) => {
  const i = args.indexOf(`--${n}`);
  return i > -1 ? args.splice(i, 2)[1] : d;
};
const methods = flag('methods', 'classic,next,next+fs')!.split(',');
const rows = Number(flag('rows', '8'));
const sort = flag('sort', 'change');
const dir = args[0];
const prefix = flag('out', join(dir, 'sheet'))!;

type Box = { x: number; y: number; w: number; h: number };
type Item = {
  id: string;
  img: string;
  w: number;
  h: number;
  methods: Record<string, Box>;
};
const data = (await Bun.file(join(dir, 'data.json')).json()) as {
  items: Item[];
};
let items = data.items.filter((it) => methods.every((m) => it.methods[m]));
const diff = (a: Box, b: Box) =>
  Math.abs(a.x - b.x) +
  Math.abs(a.y - b.y) +
  Math.abs(a.w - b.w) +
  Math.abs(a.h - b.h);
if (sort === 'change') {
  items = items.sort(
    (p, q) =>
      diff(q.methods[methods[0]], q.methods[methods[1]]) -
      diff(p.methods[methods[0]], p.methods[methods[1]])
  );
}

const CELL = 220;
const LABEL = 18;
const pages = Math.ceil(items.length / rows);
for (let pg = 0; pg < pages; pg++) {
  const slice = items.slice(pg * rows, (pg + 1) * rows);
  const cols = 1 + methods.length;
  const c = createCanvas(cols * CELL, LABEL + slice.length * CELL);
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = '#000';
  ctx.font = '14px sans-serif';
  ['image', ...methods].forEach((m, i) => ctx.fillText(m, i * CELL + 4, 14));
  for (const [r, it] of slice.entries()) {
    const img = await loadImage(join(dir, it.img));
    const y0 = LABEL + r * CELL;
    // full image with method boxes
    const s = Math.min((CELL - 8) / img.width, (CELL - 8) / img.height);
    ctx.drawImage(img, 4, y0 + 4, img.width * s, img.height * s);
    const colors = ['#e33', '#2a2', '#26f', '#f90'];
    methods.forEach((m, i) => {
      const b = it.methods[m];
      ctx.strokeStyle = colors[i % colors.length];
      ctx.lineWidth = 2;
      ctx.strokeRect(
        4 + b.x * img.width * s,
        y0 + 4 + b.y * img.height * s,
        b.w * img.width * s,
        b.h * img.height * s
      );
    });
    ctx.fillStyle = '#000';
    ctx.fillText(it.id.slice(0, 26), 4, y0 + CELL - 4);
    // each crop
    methods.forEach((m, i) => {
      const b = it.methods[m];
      const sx = b.x * img.width,
        sy = b.y * img.height,
        sw = b.w * img.width,
        sh = b.h * img.height;
      const k = Math.min((CELL - 8) / sw, (CELL - 8) / sh);
      ctx.drawImage(
        img,
        sx,
        sy,
        sw,
        sh,
        (i + 1) * CELL + 4,
        y0 + 4,
        sw * k,
        sh * k
      );
    });
  }
  const out = `${prefix}-${pg + 1}.jpg`;
  await Bun.write(out, await c.encode('jpeg', 80));
  console.log(out);
}
