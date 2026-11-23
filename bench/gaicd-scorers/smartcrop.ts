// Classic smartcrop score of each GAICD candidate box.
//
// Reference reimplementation of the classic scorer. smartcrop.crop() only
// scores the crops it generates itself (analyse() and score() are internal),
// so the per-pixel maps are copied here from src/smartcrop.ts (edge detect ->
// g, skin -> r, saturation -> b) and the score() loop is reproduced as is
// (classic options: no boost, saliency, centre, damping, border or map
// terms). The library's own exports are used where available:
// smartcrop.importance, smartcrop._downSample, smartcrop.DEFAULTS and
// smartcrop.ImgData. Like crop(), the image is prescaled so its short side is
// 256 px (never upscaled) and candidates are scaled by the same factor.
//
// Candidates are in the annotation width/height space, which equals the
// stored image's pixel size (the dataset's width/height are read from the
// stored jpg); the image is loaded at that size, unscaled.
//
// Options: { prescale?: boolean (default true), ...smartcrop option overrides
// such as luminance: 'rec709', detailWeight, skinWeight, ... }
import { createCanvas, loadImage } from "@napi-rs/canvas";
import smartcrop from "../../src/smartcrop";
import type { Crop, ImgDataLike, Options } from "../../src/smartcrop";

type Luma = (r: number, g: number, b: number) => number;
const cie: Luma = (r, g, b) => 0.5126 * b + 0.7152 * g + 0.0722 * r;
const rec709: Luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

const saturation = (r: number, g: number, b: number) => {
  const max = Math.max(r / 255, g / 255, b / 255);
  const min = Math.min(r / 255, g / 255, b / 255);
  if (max === min) return 0;
  const l = (max + min) / 2;
  const d = max - min;
  return l > 0.5 ? d / (2 - max - min) : d / (max + min);
};

function maps(o: Options, input: ImgDataLike): ImgDataLike {
  const luma = o.luminance === "rec709" ? rec709 : cie;
  const { width: w, height: h, data: id } = input;
  const out = new smartcrop.ImgData(w, h);
  const od = out.data;
  const L = (p: number) => luma(id[p], id[p + 1], id[p + 2]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      // detail (g): Laplacian of luma
      od[p + 1] =
        x === 0 || x >= w - 1 || y === 0 || y >= h - 1
          ? L(p)
          : L(p) * 4 - L(p - w * 4) - L(p - 4) - L(p + 4) - L(p + w * 4);
      const lightness = L(p) / 255;
      // skin (r)
      const mag = Math.sqrt(id[p] ** 2 + id[p + 1] ** 2 + id[p + 2] ** 2);
      const rd = id[p] / mag - o.skinColor[0];
      const gd = id[p + 1] / mag - o.skinColor[1];
      const bd = id[p + 2] / mag - o.skinColor[2];
      const skin = 1 - Math.sqrt(rd * rd + gd * gd + bd * bd);
      od[p] =
        skin > o.skinThreshold &&
        lightness >= o.skinBrightnessMin &&
        lightness <= o.skinBrightnessMax
          ? (skin - o.skinThreshold) * (255 / (1 - o.skinThreshold))
          : 0;
      // saturation (b)
      const sat = saturation(id[p], id[p + 1], id[p + 2]);
      od[p + 2] =
        sat > o.saturationThreshold &&
        lightness >= o.saturationBrightnessMin &&
        lightness <= o.saturationBrightnessMax
          ? (sat - o.saturationThreshold) * (255 / (1 - o.saturationThreshold))
          : 0;
    }
  }
  return out;
}

// Classic score().total for one crop (see score() in src/smartcrop.ts).
function score(o: Options, out: ImgDataLike, crop: Crop): number {
  const od = out.data;
  const ds = o.scoreDownSample;
  let detail = 0;
  let skin = 0;
  let sat = 0;
  for (let y = 0; y < out.height * ds; y += ds) {
    for (let x = 0; x < out.width * ds; x += ds) {
      const p = (~~(y / ds) * out.width + ~~(x / ds)) * 4;
      const i = smartcrop.importance(o, crop, x, y);
      const d = od[p + 1] / 255;
      skin += (od[p] / 255) * (d + o.skinBias) * i;
      detail += d * i;
      sat += (od[p + 2] / 255) * (d + o.saturationBias) * i;
    }
  }
  return (
    (detail * o.detailWeight + skin * o.skinWeight + sat * o.saturationWeight) /
    (crop.width * crop.height)
  );
}

export async function scoreCandidates(
  imagePath: string,
  width: number,
  height: number,
  candidates: [number, number, number, number][],
  options: Record<string, unknown> = {},
): Promise<number[]> {
  const { prescale: doPrescale = true, ...overrides } = options;
  const o: Options = { ...smartcrop.DEFAULTS, ...overrides };
  const img = await loadImage(imagePath);
  if (img.width !== width || img.height !== height) {
    throw new Error(
      `${imagePath}: ${img.width}x${img.height} != annotation ${width}x${height}`,
    );
  }
  const k = doPrescale ? Math.min(Math.max(256 / width, 256 / height), 1) : 1;
  const w = k < 1 ? ~~(width * k) : width;
  const h = k < 1 ? ~~(height * k) : height;
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, width, height, 0, 0, w, h);
  const id = ctx.getImageData(0, 0, w, h);
  const input = new smartcrop.ImgData(w, h, id.data);
  const scoreOut = smartcrop._downSample(maps(o, input), o.scoreDownSample);
  return candidates.map(([x1, y1, x2, y2]) =>
    score(o, scoreOut, {
      x: x1 * k,
      y: y1 * k,
      width: (x2 - x1) * k,
      height: (y2 - y1) * k,
    }),
  );
}
