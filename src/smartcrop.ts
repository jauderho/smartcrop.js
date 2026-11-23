/**
 * smartcrop.js
 * A javascript library implementing content aware image cropping
 *
 * Copyright (C) 2018 Jonas Wagner
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

import {
  FEATURE_COUNT,
  featureIndex,
  learnedContext,
  learnedFeatures,
  learnedScore,
  defaultModel,
  unpackModel,
  type LearnedMaps,
  type LearnedModel,
} from './learned';
import {
  cnnContext,
  cnnDim,
  cnnFeatures,
  type FeatureMap,
  type PackedCnn,
} from './cnn';

export type { FeatureMap, PackedCnn } from './cnn';

export interface CropBoost {
  x: number;
  y: number;
  width: number;
  height: number;
  weight: number;
}

export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CropScore {
  detail: number;
  saturation: number;
  skin: number;
  boost: number;
  /** Colour-contrast saliency (only when saliencyWeight > 0). */
  saliency?: number;
  /** Closeness of the crop centre to the image centre, 0–1 (only when centerWeight > 0). */
  center?: number;
  /** Mean importance along crop edges inside the image, 0–1 (only when borderPenalty > 0). */
  border?: number;
  /** Importance-map sum (only when an importanceMap is used). */
  map?: number;
  total: number;
}

export interface Crop {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: CropScore;
}

export interface ImgDataLike {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

// The parts of a 2D canvas context smartcrop uses. Kept minimal so canvases
// from Node libraries such as @napi-rs/canvas or node-canvas fit too.
export interface Context2DLike {
  drawImage(image: unknown, ...coordinates: number[]): void;
  getImageData(
    sx: number,
    sy: number,
    sw: number,
    sh: number
  ): { data: Uint8ClampedArray };
}

/**
 * A per-pixel importance map, e.g. from a saliency or segmentation model, in
 * the coordinate space of the input image (any resolution; it is resampled).
 * Values are 0–1 (Float32Array) or 0–255 (Uint8Array / Uint8ClampedArray).
 */
export interface ImportanceMap {
  width: number;
  height: number;
  data: Float32Array | Uint8Array | Uint8ClampedArray;
}

export interface CanvasLike {
  width: number;
  height: number;
  getContext(contextId: '2d'): Context2DLike | null;
}

export type CanvasFactory = (width: number, height: number) => CanvasLike;

export interface ImageOperations {
  // Method syntax (bivariant) so implementations may narrow the input type.
  open(
    image: unknown,
    inputOptions?: unknown
  ): Promise<{ width: number; height: number }>;
  resample(
    image: { width: number; height: number },
    width: number,
    height: number
  ): Promise<{ width: number; height: number }>;
  getData(image: { width: number; height: number }): Promise<ImgDataLike>;
}

export interface Options {
  width: number;
  height: number;
  aspect: number;
  cropWidth: number;
  cropHeight: number;
  detailWeight: number;
  skinColor: number[];
  skinBias: number;
  skinBrightnessMin: number;
  skinBrightnessMax: number;
  skinThreshold: number;
  skinWeight: number;
  saturationBrightnessMin: number;
  saturationBrightnessMax: number;
  saturationThreshold: number;
  saturationBias: number;
  saturationWeight: number;
  scoreDownSample: number;
  step: number;
  scaleStep: number;
  minScale: number;
  maxScale: number;
  edgeRadius: number;
  edgeWeight: number;
  outsideImportance: number;
  boostWeight: number;
  ruleOfThirds: boolean;
  prescale: boolean;
  imageOperations: ImageOperations | null;
  canvasFactory: CanvasFactory;
  debug: boolean;
  /**
   * Luminance weights. 'classic' keeps the historical coefficients (which
   * over-weight blue, see upstream #77); 'rec709' uses the standard ones.
   */
  luminance: 'classic' | 'rec709';
  /** Weight of the colour-contrast saliency channel (frequency-tuned, Achanta 2009). 0 = off. */
  saliencyWeight: number;
  /** Weight of a prior favouring crops near the image centre. 0 = off. */
  centerWeight: number;
  /**
   * Subtract this fraction of the local average detail from each cell, so
   * evenly textured areas (foliage, gravel, water) stop outscoring subjects.
   * 0 = off.
   */
  textureDamping: number;
  /** Penalty for crop edges that cut through important content. 0 = off. */
  borderPenalty: number;
  /** Weight of `importanceMap`. 0 = off. */
  importanceMapWeight: number;
  /** External importance map (saliency / subject mask), see ImportanceMap. */
  importanceMap?: ImportanceMap;
  boost?: CropBoost[];
  input?: unknown;
  /**
   * Crop scoring model. 'classic' is the hand-tuned heuristic. 'learned' is a
   * small model trained on human-rated crops (GAICD) over composition
   * features (see src/learned.ts); it ignores the classic weights above but
   * uses the importanceMap and boost boxes when given.
   */
  scorer: 'classic' | 'learned';
  /**
   * Learned scorer only: crops that keep less than this fraction of the
   * salient mass (importance map if given, else mean of colour-contrast
   * saliency and detail), or that cut a boost box (e.g. a face), are ranked
   * below all crops that don't. If none passes the crop with the most
   * retained mass wins. 0 = off.
   */
  contentFloor: number;
  /**
   * Learned scorer only: add this multiple of the classic score (computed
   * with the classic weights in these options) to the learned score, both
   * standardised over the image's candidates. 0 = learned only.
   */
  classicBlend: number;
  /**
   * Learned scorer only: return the softmax-weighted mean of the top windows
   * instead of the single best one (temperature in score standard deviations,
   * e.g. 0.1). 0 = plain argmax.
   */
  consensus: number;
  /**
   * Score exactly these crops (input-image coordinates) instead of generating
   * windows. result.crops then holds one scored crop per candidate, in order.
   */
  candidates?: CropBox[];
  /**
   * Learned scorer only: a deep-feature map of the whole image from the
   * optional cnn plug-in (plugins/onnx/cnn.ts). Used only together with
   * `cnnModel`; without both, scoring is unchanged.
   */
  featureMap?: FeatureMap;
  /** Weights of the cnn head (src/learned-cnn-weights.ts); see `featureMap`. */
  cnnModel?: PackedCnn;
  /** @internal Pool a K x K cnn grid into the emitted features even without a cnnModel (training dumps). */
  cnnK?: number;
  /** @internal Return the learned feature vectors (n * FEATURE_COUNT) as result.features. */
  emitFeatures?: boolean;
}

export type CropOptions = Partial<Options>;

/** Where a crop sits in the image, as CSS object-position percentages. */
export interface CropPosition {
  x: number;
  y: number;
}

export interface CropResult {
  topCrop: Crop;
  /** topCrop as CSS object-position percentages (50 when there is no slack). */
  position?: CropPosition;
  crops?: Crop[];
  debugOutput?: ImgDataLike;
  debugOptions?: Options;
  debugTopCrop?: Crop;
  /** @internal learned feature vectors, see Options.emitFeatures. */
  features?: Float32Array;
}

export class ImgData implements ImgDataLike {
  width: number;
  height: number;
  data: Uint8ClampedArray;

  constructor(width: number, height: number, data?: ArrayLike<number>) {
    this.width = width;
    this.height = height;
    if (data) {
      this.data = new Uint8ClampedArray(data);
    } else {
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  }
}

const defaultCanvasFactory: CanvasFactory = (w, h) => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

// Gets value in the range of [0, 1] where 0 is the center of the pictures
// returns weight of rule of thirds [0, 1]
const thirds = (x: number): number => {
  x = (((x - 1 / 3 + 1.0) % 2.0) * 0.5 - 0.5) * 16;
  return Math.max(1.0 - x * x, 0.0);
};

type Luma = (r: number, g: number, b: number) => number;

const cie: Luma = (r, g, b) => 0.5126 * b + 0.7152 * g + 0.0722 * r;
const rec709: Luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const lumaFor = (options: Options): Luma =>
  options.luminance === 'rec709' ? rec709 : cie;

const sample = (id: ArrayLike<number>, p: number, luma: Luma): number =>
  luma(id[p], id[p + 1], id[p + 2]);

const saturation = (r: number, g: number, b: number): number => {
  const maximum = Math.max(r / 255, g / 255, b / 255);
  const minimum = Math.min(r / 255, g / 255, b / 255);

  if (maximum === minimum) {
    return 0;
  }

  const l = (maximum + minimum) / 2;
  const d = maximum - minimum;

  return l > 0.5 ? d / (2 - maximum - minimum) : d / (maximum + minimum);
};

const edgeDetect = (i: ImgDataLike, o: ImgDataLike, luma: Luma): void => {
  const id = i.data;
  const od = o.data;
  const w = i.width;
  const h = i.height;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      let lightness;

      if (x === 0 || x >= w - 1 || y === 0 || y >= h - 1) {
        lightness = sample(id, p, luma);
      } else {
        lightness =
          sample(id, p, luma) * 4 -
          sample(id, p - w * 4, luma) -
          sample(id, p - 4, luma) -
          sample(id, p + 4, luma) -
          sample(id, p + w * 4, luma);
      }

      od[p + 1] = lightness;
    }
  }
};

const skinColor = (
  options: Options,
  r: number,
  g: number,
  b: number
): number => {
  const mag = Math.sqrt(r * r + g * g + b * b);
  const rd = r / mag - options.skinColor[0];
  const gd = g / mag - options.skinColor[1];
  const bd = b / mag - options.skinColor[2];
  const d = Math.sqrt(rd * rd + gd * gd + bd * bd);
  return 1 - d;
};

const skinDetect = (options: Options, i: ImgDataLike, o: ImgDataLike): void => {
  const id = i.data;
  const od = o.data;
  const w = i.width;
  const h = i.height;
  const luma = lumaFor(options);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      const lightness = luma(id[p], id[p + 1], id[p + 2]) / 255;
      const skin = skinColor(options, id[p], id[p + 1], id[p + 2]);
      const isSkinColor = skin > options.skinThreshold;
      const isSkinBrightness =
        lightness >= options.skinBrightnessMin &&
        lightness <= options.skinBrightnessMax;
      if (isSkinColor && isSkinBrightness) {
        od[p] =
          (skin - options.skinThreshold) * (255 / (1 - options.skinThreshold));
      } else {
        od[p] = 0;
      }
    }
  }
};

const saturationDetect = (
  options: Options,
  i: ImgDataLike,
  o: ImgDataLike
): void => {
  const id = i.data;
  const od = o.data;
  const w = i.width;
  const h = i.height;
  const luma = lumaFor(options);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;

      const lightness = luma(id[p], id[p + 1], id[p + 2]) / 255;
      const sat = saturation(id[p], id[p + 1], id[p + 2]);

      const acceptableSaturation = sat > options.saturationThreshold;
      const acceptableLightness =
        lightness >= options.saturationBrightnessMin &&
        lightness <= options.saturationBrightnessMax;
      if (acceptableLightness && acceptableSaturation) {
        od[p + 2] =
          (sat - options.saturationThreshold) *
          (255 / (1 - options.saturationThreshold));
      } else {
        od[p + 2] = 0;
      }
    }
  }
};

const applyBoost = (boost: CropBoost, output: ImgDataLike): void => {
  const od = output.data;
  const w = output.width;
  const x0 = ~~boost.x;
  const x1 = ~~(boost.x + boost.width);
  const y0 = ~~boost.y;
  const y1 = ~~(boost.y + boost.height);
  const weight = boost.weight * 255;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * w + x) * 4;
      od[i + 3] += weight;
    }
  }
};

const applyBoosts = (options: Options, output: ImgDataLike): void => {
  if (!options.boost) return;
  const od = output.data;
  for (let i = 0; i < output.width; i += 4) {
    od[i + 3] = 0;
  }
  for (let i = 0; i < options.boost.length; i++) {
    applyBoost(options.boost[i], output);
  }
};

const generateCrops = (
  options: Options,
  width: number,
  height: number
): Crop[] => {
  const results: Crop[] = [];
  const minDimension = Math.min(width, height);
  const cropWidth = options.cropWidth || minDimension;
  const cropHeight = options.cropHeight || minDimension;
  for (
    let scale = options.maxScale;
    scale >= options.minScale;
    scale -= options.scaleStep
  ) {
    for (let y = 0; y + cropHeight * scale <= height; y += options.step) {
      for (let x = 0; x + cropWidth * scale <= width; x += options.step) {
        results.push({
          x: x,
          y: y,
          width: cropWidth * scale,
          height: cropHeight * scale,
        });
      }
    }
  }
  return results;
};

const importance = (
  options: Options,
  crop: Crop,
  x: number,
  y: number
): number => {
  if (
    crop.x > x ||
    x >= crop.x + crop.width ||
    crop.y > y ||
    y >= crop.y + crop.height
  ) {
    return options.outsideImportance;
  }
  x = (x - crop.x) / crop.width;
  y = (y - crop.y) / crop.height;
  const px = Math.abs(0.5 - x) * 2;
  const py = Math.abs(0.5 - y) * 2;
  // Distance from edge
  const dx = Math.max(px - 1.0 + options.edgeRadius, 0);
  const dy = Math.max(py - 1.0 + options.edgeRadius, 0);
  const d = (dx * dx + dy * dy) * options.edgeWeight;
  let s = 1.41 - Math.sqrt(px * px + py * py);
  if (options.ruleOfThirds) {
    s += Math.max(0, s + d + 0.5) * 1.2 * (thirds(px) + thirds(py));
  }
  return s + d;
};

interface ScoreExtras {
  /** Saliency per score-grid cell, 0–1, same layout as the downsampled output. */
  saliency: Float32Array | null;
  /** Texture-damped detail per score-grid cell, 0–1, replacing the raw detail. */
  detail: Float32Array | null;
  /** Combined importance per score-grid cell, 0–1, used by the border penalty. */
  content: Float32Array | null;
  /** External importance map resampled to the score grid, 0–1. */
  map: Float32Array | null;
  imageWidth: number;
  imageHeight: number;
}

const score = (
  options: Options,
  output: ImgDataLike,
  crop: Crop,
  extras?: ScoreExtras
): CropScore => {
  const result: CropScore = {
    detail: 0,
    saturation: 0,
    skin: 0,
    boost: 0,
    total: 0,
  };
  const sal = extras?.saliency ?? null;
  const damped = extras?.detail ?? null;
  const ext = extras?.map ?? null;
  let saliency = 0;
  let mapSum = 0;

  const od = output.data;
  const downSample = options.scoreDownSample;
  const invDownSample = 1 / downSample;
  const outputHeightDownSample = output.height * downSample;
  const outputWidthDownSample = output.width * downSample;
  const outputWidth = output.width;

  for (let y = 0; y < outputHeightDownSample; y += downSample) {
    for (let x = 0; x < outputWidthDownSample; x += downSample) {
      const p =
        (~~(y * invDownSample) * outputWidth + ~~(x * invDownSample)) * 4;
      const i = importance(options, crop, x, y);
      const detail = damped ? damped[p >> 2] : od[p + 1] / 255;

      result.skin += (od[p] / 255) * (detail + options.skinBias) * i;
      result.detail += detail * i;
      result.saturation +=
        (od[p + 2] / 255) * (detail + options.saturationBias) * i;
      result.boost += (od[p + 3] / 255) * i;
      if (sal) saliency += sal[p >> 2] * i;
      if (ext) mapSum += ext[p >> 2] * i;
    }
  }

  result.total =
    (result.detail * options.detailWeight +
      result.skin * options.skinWeight +
      result.saturation * options.saturationWeight +
      result.boost * options.boostWeight) /
    (crop.width * crop.height);

  // Opt-in terms, scaled like the sums above: a cell with value 1 and
  // importance 1 contributes 1 per score-grid cell.
  const cellsPerArea =
    (output.width * output.height) / (crop.width * crop.height);
  if (sal && options.saliencyWeight) {
    result.saliency = saliency;
    result.total +=
      (saliency * options.saliencyWeight) / (crop.width * crop.height);
  }
  if (extras && options.centerWeight) {
    const dx = (crop.x + crop.width / 2) / extras.imageWidth - 0.5;
    const dy = (crop.y + crop.height / 2) / extras.imageHeight - 0.5;
    result.center = 1 - Math.sqrt(dx * dx + dy * dy) / Math.SQRT1_2;
    result.total += result.center * options.centerWeight * cellsPerArea;
  }
  if (ext && options.importanceMapWeight) {
    result.map = mapSum;
    result.total +=
      (mapSum * options.importanceMapWeight) / (crop.width * crop.height);
  }
  if (extras?.content && options.borderPenalty) {
    result.border = borderContent(extras.content, output, crop, downSample);
    result.total -= result.border * options.borderPenalty * cellsPerArea;
  }
  return result;
};

// Mean of the content grid along the crop's edges, skipping edges that lie on
// the image border (they cut nothing).
const borderContent = (
  content: Float32Array,
  grid: { width: number; height: number },
  crop: Crop,
  factor: number
): number => {
  const gw = grid.width;
  const gh = grid.height;
  const x0 = Math.floor(crop.x / factor);
  const y0 = Math.floor(crop.y / factor);
  const x1 = Math.min(gw - 1, Math.floor((crop.x + crop.width - 1) / factor));
  const y1 = Math.min(gh - 1, Math.floor((crop.y + crop.height - 1) / factor));
  let sum = 0;
  let n = 0;
  if (x0 > 0) for (let y = y0; y <= y1; y++, n++) sum += content[y * gw + x0];
  if (x1 < gw - 1)
    for (let y = y0; y <= y1; y++, n++) sum += content[y * gw + x1];
  if (y0 > 0) for (let x = x0; x <= x1; x++, n++) sum += content[y0 * gw + x];
  if (y1 < gh - 1)
    for (let x = x0; x <= x1; x++, n++) sum += content[y1 * gw + x];
  return n ? sum / n : 0;
};

// Average an external importance map into the score grid (gw x gh), 0–1.
const resampleMap = (map: ImportanceMap, gw: number, gh: number) => {
  const scale = map.data instanceof Float32Array ? 1 : 1 / 255;
  const grid = new Float32Array(gw * gh);
  for (let gy = 0; gy < gh; gy++) {
    const y0 = Math.floor((gy * map.height) / gh);
    const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * map.height) / gh));
    for (let gx = 0; gx < gw; gx++) {
      const x0 = Math.floor((gx * map.width) / gw);
      const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * map.width) / gw));
      let sum = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) sum += map.data[y * map.width + x];
      }
      grid[gy * gw + gx] = (sum * scale) / ((y1 - y0) * (x1 - x0));
    }
  }
  return grid;
};

// Per-cell maps derived from the downsampled output for the opt-in terms.
const gridMaps = (
  options: Options,
  scoreOutput: ImgDataLike,
  saliency: Float32Array | null,
  map: Float32Array | null
) => {
  const { width: gw, height: gh, data: sod } = scoreOutput;
  const n = gw * gh;
  let detail: Float32Array | null = null;
  if (options.textureDamping) {
    const raw = new Float32Array(n);
    for (let c = 0; c < n; c++) raw[c] = sod[c * 4 + 1] / 255;
    const local = boxBlur(raw, gw, gh, 2);
    detail = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      detail[c] = Math.max(0, raw[c] - options.textureDamping * local[c]);
    }
  }
  let content: Float32Array | null = null;
  if (options.borderPenalty) {
    content = new Float32Array(n);
    let max = 0;
    for (let c = 0; c < n; c++) {
      const d = detail ? detail[c] : sod[c * 4 + 1] / 255;
      const v =
        d * options.detailWeight +
        (sod[c * 4] / 255) * options.skinWeight +
        (sod[c * 4 + 2] / 255) * options.saturationWeight +
        (sod[c * 4 + 3] / 255) * options.boostWeight +
        (saliency ? saliency[c] * options.saliencyWeight : 0) +
        (map ? map[c] * options.importanceMapWeight : 0);
      content[c] = v;
      if (v > max) max = v;
    }
    if (max > 0) for (let c = 0; c < n; c++) content[c] /= max;
  }
  return { detail, content, map };
};

const downSample = (input: ImgDataLike, factor: number): ImgData => {
  const idata = input.data;
  const iwidth = input.width;
  const width = Math.floor(input.width / factor);
  const height = Math.floor(input.height / factor);
  const output = new ImgData(width, height);
  const data = output.data;
  const ifactor2 = 1 / (factor * factor);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;

      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;

      let mr = 0;
      let mg = 0;

      for (let v = 0; v < factor; v++) {
        for (let u = 0; u < factor; u++) {
          const j = ((y * factor + v) * iwidth + (x * factor + u)) * 4;
          r += idata[j];
          g += idata[j + 1];
          b += idata[j + 2];
          a += idata[j + 3];
          mr = Math.max(mr, idata[j]);
          mg = Math.max(mg, idata[j + 1]);
        }
      }
      // this is some funky magic to preserve detail a bit more for
      // skin (r) and detail (g). Saturation (b) does not get this boost.
      data[i] = r * ifactor2 * 0.5 + mr * 0.5;
      data[i + 1] = g * ifactor2 * 0.7 + mg * 0.3;
      data[i + 2] = b * ifactor2;
      data[i + 3] = a * ifactor2;
    }
  }
  return output;
};

// Frequency-tuned colour-contrast saliency (Achanta et al., CVPR 2009):
// distance of each slightly blurred pixel's Lab colour from the image's mean
// Lab colour, averaged into the score grid and normalised to 0–1.
const srgbToLinear = (c: number) => {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
};
const labF = (t: number) =>
  t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;

const boxBlur = (src: Float32Array, w: number, h: number, r: number) => {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const span = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) {
        sum += src[y * w + Math.min(w - 1, Math.max(0, x + k))];
      }
      tmp[y * w + x] = sum / span;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) {
        sum += tmp[Math.min(h - 1, Math.max(0, y + k)) * w + x];
      }
      out[y * w + x] = sum / span;
    }
  }
  return out;
};

const colorContrastSaliency = (
  input: ImgDataLike,
  factor: number
): Float32Array => {
  const { width: w, height: h, data } = input;
  const n = w * h;
  const L = new Float32Array(n);
  const A = new Float32Array(n);
  const B = new Float32Array(n);
  let mL = 0;
  let mA = 0;
  let mB = 0;
  for (let i = 0; i < n; i++) {
    const r = srgbToLinear(data[i * 4]);
    const g = srgbToLinear(data[i * 4 + 1]);
    const b = srgbToLinear(data[i * 4 + 2]);
    const fx = labF((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
    const fy = labF(0.2126 * r + 0.7152 * g + 0.0722 * b);
    const fz = labF((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
    mL += L[i];
    mA += A[i];
    mB += B[i];
  }
  mL /= n;
  mA /= n;
  mB /= n;
  const r = Math.max(1, Math.round(Math.min(w, h) / 100));
  const bL = boxBlur(L, w, h, r);
  const bA = boxBlur(A, w, h, r);
  const bB = boxBlur(B, w, h, r);

  const gw = Math.floor(w / factor);
  const gh = Math.floor(h / factor);
  const grid = new Float32Array(gw * gh);
  let max = 0;
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      let sum = 0;
      for (let v = 0; v < factor; v++) {
        for (let u = 0; u < factor; u++) {
          const i = (gy * factor + v) * w + gx * factor + u;
          const dl = bL[i] - mL;
          const da = bA[i] - mA;
          const db = bB[i] - mB;
          sum += Math.sqrt(dl * dl + da * da + db * db);
        }
      }
      grid[gy * gw + gx] = sum;
      if (sum > max) max = sum;
    }
  }
  if (max > 0) for (let i = 0; i < grid.length; i++) grid[i] /= max;
  return grid;
};

// Pixels per cell of the learned scorer's grid (fixed: the weights depend on it).
const LEARNED_GRID = 4;
const LEARNED_DAMPING = 0.5;

const learnedMaps = (
  options: Options,
  input: ImgDataLike,
  output: ImgDataLike
): LearnedMaps => {
  const factor = LEARNED_GRID;
  const sod = downSample(output, factor);
  const { width: gw, height: gh, data } = sod;
  const n = gw * gh;
  const raw = new Float32Array(n);
  for (let c = 0; c < n; c++) raw[c] = data[c * 4 + 1] / 255;
  const local = boxBlur(raw, gw, gh, 4);
  const detail = new Float32Array(n);
  const skin = new Float32Array(n);
  const sat = new Float32Array(n);
  const boost = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    detail[c] = Math.max(0, raw[c] - LEARNED_DAMPING * local[c]);
    skin[c] = data[c * 4] / 255;
    sat[c] = data[c * 4 + 2] / 255;
    boost[c] = data[c * 4 + 3] / 255;
  }
  const luma = new Float32Array(n);
  const id = input.data;
  // Luminance gradient magnitude, averaged per cell.
  const iw = input.width;
  const ih = input.height;
  const full = new Float32Array(iw * ih);
  for (let i = 0; i < full.length; i++) {
    full[i] = rec709(id[i * 4], id[i * 4 + 1], id[i * 4 + 2]);
  }
  const gradient = new Float32Array(n);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      let sum = 0;
      for (let v = 0; v < factor; v++) {
        const y = gy * factor + v;
        const ya = Math.max(0, y - 1);
        const yb = Math.min(ih - 1, y + 1);
        for (let u = 0; u < factor; u++) {
          const x = gx * factor + u;
          const dx =
            full[y * iw + Math.min(iw - 1, x + 1)] -
            full[y * iw + Math.max(0, x - 1)];
          const dy = full[yb * iw + x] - full[ya * iw + x];
          sum += Math.sqrt(dx * dx + dy * dy);
        }
      }
      gradient[gy * gw + gx] = sum / (factor * factor * 255);
    }
  }
  const inv = 1 / (factor * factor);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      let sum = 0;
      for (let v = 0; v < factor; v++) {
        for (let u = 0; u < factor; u++) {
          const p = ((gy * factor + v) * input.width + gx * factor + u) * 4;
          sum += rec709(id[p], id[p + 1], id[p + 2]);
        }
      }
      luma[gy * gw + gx] = sum * inv;
    }
  }
  return {
    gw,
    gh,
    factor,
    width: input.width,
    height: input.height,
    maps: [
      detail,
      skin,
      sat,
      colorContrastSaliency(input, factor),
      options.importanceMap ? resampleMap(options.importanceMap, gw, gh) : null,
      options.boost?.length ? boost : null,
      gradient,
    ],
    luma,
    boosts: options.boost ?? [],
  };
};

const analyse = (options: Options, input: ImgDataLike): CropResult => {
  const result = {} as CropResult;
  const output = new ImgData(input.width, input.height);

  edgeDetect(input, output, lumaFor(options));
  skinDetect(options, input, output);
  saturationDetect(options, input, output);
  applyBoosts(options, output);

  if (options.scorer === 'learned') {
    return analyseLearned(options, input, output, result);
  }

  const scoreOutput = downSample(output, options.scoreDownSample);
  const extras = classicExtras(options, input, scoreOutput);

  let topScore = -Infinity;
  let topCrop: Crop | null = null;
  const crops: Crop[] = options.candidates
    ? options.candidates.map((c) => ({ ...c }))
    : generateCrops(options, input.width, input.height);

  for (let i = 0, iLen = crops.length; i < iLen; i++) {
    const crop = crops[i];
    crop.score = score(options, scoreOutput, crop, extras);
    if (crop.score.total > topScore) {
      topCrop = crop;
      topScore = crop.score.total;
    }
  }

  // topCrop is only null if no crop fits, which crop() does not handle.
  result.topCrop = topCrop as Crop;

  if (options.candidates) result.crops = crops;
  if (options.debug && topCrop) {
    result.crops = crops;
    result.debugOutput = output;
    result.debugOptions = options;
    // Create a copy which will not be adjusted by the post scaling of smartcrop.crop
    result.debugTopCrop = { ...result.topCrop };
  }
  return result;
};

// Extra grid maps for the opt-in classic terms; undefined keeps the original
// arithmetic exact.
const classicExtras = (
  options: Options,
  input: ImgDataLike,
  scoreOutput: ImgDataLike
): ScoreExtras | undefined => {
  let extras: ScoreExtras | undefined;
  if (
    options.saliencyWeight ||
    options.centerWeight ||
    options.textureDamping ||
    options.borderPenalty ||
    (options.importanceMap && options.importanceMapWeight)
  ) {
    const saliency = options.saliencyWeight
      ? colorContrastSaliency(input, options.scoreDownSample)
      : null;
    const map =
      options.importanceMap && options.importanceMapWeight
        ? resampleMap(
            options.importanceMap,
            scoreOutput.width,
            scoreOutput.height
          )
        : null;
    extras = {
      saliency,
      ...gridMaps(options, scoreOutput, saliency, map),
      imageWidth: input.width,
      imageHeight: input.height,
    };
  }
  return extras;
};

// z-score a list in place (no-op for a constant list).
const standardize = (xs: number[]): number[] => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  return xs.map((x) => (sd > 0 ? (x - m) / sd : 0));
};

const IDX = {
  detail: featureIndex('detail.recall'),
  saliency: featureIndex('saliency.recall'),
  importance: featureIndex('importance.recall'),
  faceCut: featureIndex('faceCut'),
};
// Face/box cut tolerated by the content floor (fraction of boost-box area).
const FLOOR_FACE_CUT = 0.1;

// Push crops below the content floor behind all others; when nothing passes,
// score by retained mass instead. Returns the new top crop.
const applyContentFloor = (
  options: Options,
  hasBoost: boolean,
  hasMap: boolean,
  feats: Float32Array,
  stride: number,
  crops: Crop[]
): Crop | null => {
  const retained = crops.map((_, i) => {
    const f = i * stride;
    return hasMap
      ? feats[f + IDX.importance]
      : (feats[f + IDX.saliency] + feats[f + IDX.detail]) / 2;
  });
  const pass = crops.map(
    (_, i) =>
      retained[i] >= options.contentFloor &&
      !(hasBoost && feats[i * stride + IDX.faceCut] > FLOOR_FACE_CUT)
  );
  const any = pass.some(Boolean);
  let top: Crop | null = null;
  let topScore = -Infinity;
  crops.forEach((c, i) => {
    const s = c.score!;
    s.total = any ? s.total - (pass[i] ? 0 : 1e6) : retained[i];
    if (s.total > topScore || !top) {
      topScore = s.total;
      top = c;
    }
  });
  return top;
};

const cnnHeads = new WeakMap<PackedCnn, LearnedModel>();
const cnnHead = (p: PackedCnn): LearnedModel => {
  let m = cnnHeads.get(p);
  if (!m) {
    m = unpackModel(p.head, FEATURE_COUNT + cnnDim(p.k, p.channels));
    cnnHeads.set(p, m);
  }
  return m;
};

// Consensus crop: instead of the single highest-scoring window (a noisy
// maximum over thousands of correlated windows), return the softmax-weighted
// mean box of the best windows. temperature is in units of the score's
// standard deviation over this image's windows. All windows share the target
// aspect and lie inside the image, so their convex combination does too.
const CONSENSUS_TOP = 64;
const consensusCrop = (crops: Crop[], temperature: number, top: Crop): Crop => {
  const t = crops.map((c) => c.score!.total);
  const m = t.reduce((a, b) => a + b, 0) / t.length;
  const sd = Math.sqrt(t.reduce((a, b) => a + (b - m) ** 2, 0) / t.length) || 1;
  const best = t
    .map((_, i) => i)
    .sort((a, b) => t[b] - t[a])
    .slice(0, CONSENSUS_TOP);
  const tmax = t[best[0]];
  let wsum = 0,
    x = 0,
    y = 0,
    w = 0,
    h = 0;
  for (const i of best) {
    const wt = Math.exp((t[i] - tmax) / (temperature * sd));
    const c = crops[i];
    wsum += wt;
    x += wt * c.x;
    y += wt * c.y;
    w += wt * c.width;
    h += wt * c.height;
  }
  return {
    x: x / wsum,
    y: y / wsum,
    width: w / wsum,
    height: h / wsum,
    score: top.score,
  };
};

const analyseLearned = (
  options: Options,
  input: ImgDataLike,
  output: ImgDataLike,
  result: CropResult
): CropResult => {
  const ctx = learnedContext(learnedMaps(options, input, output));
  const crops: Crop[] = options.candidates
    ? options.candidates.map((c) => ({ ...c }))
    : generateCrops(
        {
          ...options,
          step: Math.min(options.step, 4),
          scaleStep: Math.min(options.scaleStep, 0.05),
        },
        input.width,
        input.height
      );
  // Optional deep features: appended after the hand features.
  const cnnK = options.featureMap
    ? (options.cnnModel?.k ?? options.cnnK ?? 0)
    : 0;
  const cx = cnnK ? cnnContext(options.featureMap!) : null;
  const stride =
    FEATURE_COUNT + (cx ? cnnDim(cnnK, options.featureMap!.channels) : 0);
  const feats = new Float32Array(crops.length * stride);
  const model =
    cx && options.cnnModel ? cnnHead(options.cnnModel) : defaultModel();
  let topScore = -Infinity;
  let topCrop: Crop | null = null;
  for (let i = 0; i < crops.length; i++) {
    const c = crops[i];
    learnedFeatures(ctx, c.x, c.y, c.width, c.height, feats, i * stride);
    if (cx) {
      cnnFeatures(
        cx,
        cnnK,
        c.x,
        c.y,
        c.width,
        c.height,
        ctx.width,
        ctx.height,
        feats,
        i * stride + FEATURE_COUNT
      );
    }
    const total = learnedScore(feats, i * stride, model);
    c.score = { detail: 0, saturation: 0, skin: 0, boost: 0, total };
    if (total > topScore || !topCrop) {
      topScore = total;
      topCrop = c;
    }
  }
  if (options.classicBlend > 0 && crops.length > 1) {
    // Blend in the classic score (with whatever classic weights were passed),
    // both standardised over this image's candidates.
    const scoreOutput = downSample(output, options.scoreDownSample);
    const extras = classicExtras(options, input, scoreOutput);
    const zl = standardize(crops.map((c) => c.score!.total));
    const zc = standardize(
      crops.map((c) => score(options, scoreOutput, c, extras).total)
    );
    topScore = -Infinity;
    crops.forEach((c, i) => {
      c.score!.total = zl[i] + options.classicBlend * zc[i];
      if (c.score!.total > topScore) {
        topScore = c.score!.total;
        topCrop = c;
      }
    });
  }
  if (options.contentFloor > 0) {
    topCrop = applyContentFloor(
      options,
      ctx.boosts.length > 0,
      ctx.sums[4] !== null,
      feats,
      stride,
      crops
    );
  }
  if (
    options.consensus > 0 &&
    !options.candidates &&
    crops.length > 1 &&
    topCrop
  ) {
    topCrop = consensusCrop(crops, options.consensus, topCrop);
  }
  result.topCrop = topCrop as Crop;
  if (options.candidates || options.debug) result.crops = crops;
  if (options.emitFeatures) result.features = feats;
  if (options.debug && topCrop) {
    result.debugOutput = output as ImgDataLike;
    result.debugOptions = options;
    result.debugTopCrop = { ...topCrop };
  }
  return result;
};

interface CanvasInput {
  naturalWidth?: number;
  naturalHeight?: number;
  width: number;
  height: number;
}

const canvasImageOperations = (
  canvasFactory: CanvasFactory
): ImageOperations => ({
  // Takes imageInput as argument
  // returns an object which has at least
  // {width: n, height: n}
  open: (image: CanvasInput) => {
    // Work around images scaled in css by drawing them onto a canvas
    const w = image.naturalWidth || image.width;
    const h = image.naturalHeight || image.height;
    const c = canvasFactory(w, h);
    const ctx = c.getContext('2d')!;
    if (
      image.naturalWidth &&
      (image.naturalWidth != image.width || image.naturalHeight != image.height)
    ) {
      c.width = image.naturalWidth;
      c.height = image.naturalHeight!;
    } else {
      c.width = image.width;
      c.height = image.height;
    }
    ctx.drawImage(image, 0, 0);
    return smartcrop.Promise.resolve(c);
  },
  // Takes an image (as returned by open), and changes its size by resampling
  resample: (image, width, height) =>
    smartcrop.Promise.resolve(image).then((image) => {
      const c = canvasFactory(~~width, ~~height);
      const ctx = c.getContext('2d')!;

      ctx.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        0,
        0,
        c.width,
        c.height
      );
      return smartcrop.Promise.resolve(c);
    }),
  getData: (image) =>
    smartcrop.Promise.resolve(image).then((img) => {
      const c = img as unknown as CanvasLike;
      const ctx = c.getContext('2d')!;
      const id = ctx.getImageData(0, 0, c.width, c.height);
      return new ImgData(c.width, c.height, id.data);
    }),
});

const crop = (
  inputImage: unknown,
  options_?: CropOptions,
  callback?: (result: CropResult) => void
): Promise<CropResult> => {
  const options: Options = { ...smartcrop.DEFAULTS, ...options_ };
  // The learned scorer ranks crops of any size; search 0.4-1 of the largest
  // crop unless the caller says otherwise.
  if (options.scorer === 'learned' && options_?.minScale === undefined) {
    options.minScale = 0.4;
  }

  if (options.aspect) {
    options.width = options.aspect;
    options.height = 1;
  }

  if (options.imageOperations === null) {
    options.imageOperations = canvasImageOperations(options.canvasFactory);
  }

  const iop = options.imageOperations;

  let scale = 1;
  let prescale = 1;
  let imageWidth = 0;
  let imageHeight = 0;

  // open the image
  return iop
    .open(inputImage, options.input)
    .then((image) => {
      imageWidth = image.width;
      imageHeight = image.height;
      // calculate desired crop dimensions based on the image size
      if (options.width && options.height) {
        scale = Math.min(
          image.width / options.width,
          image.height / options.height
        );
        options.cropWidth = ~~(options.width * scale);
        options.cropHeight = ~~(options.height * scale);
        // Img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
        // don't set minscale smaller than 1/scale
        // -> don't pick crops that need upscaling
        options.minScale = Math.min(
          options.maxScale,
          Math.max(1 / scale, options.minScale)
        );

        // prescale if possible
        if (options.prescale !== false) {
          prescale = Math.min(
            Math.max(256 / image.width, 256 / image.height),
            1
          );
          if (prescale < 1) {
            image = iop.resample(
              image,
              image.width * prescale,
              image.height * prescale
            ) as unknown as typeof image;
            options.cropWidth = ~~(options.cropWidth * prescale);
            options.cropHeight = ~~(options.cropHeight * prescale);
            if (options.boost) {
              options.boost = options.boost.map((boost) => ({
                x: ~~(boost.x * prescale),
                y: ~~(boost.y * prescale),
                width: ~~(boost.width * prescale),
                height: ~~(boost.height * prescale),
                weight: boost.weight,
              }));
            }
          } else {
            prescale = 1;
          }
        }
      }
      return image;
    })
    .then((image) =>
      iop.getData(image).then((data) => {
        if (options.candidates && prescale !== 1) {
          options.candidates = options.candidates.map((c) => ({
            x: c.x * prescale,
            y: c.y * prescale,
            width: c.width * prescale,
            height: c.height * prescale,
          }));
        }
        const result = analyse(options, data);

        const crops = result.crops || [result.topCrop];
        for (let i = 0, iLen = crops.length; i < iLen; i++) {
          const crop = crops[i];
          if (options.candidates) {
            crop.x /= prescale;
            crop.y /= prescale;
            crop.width /= prescale;
            crop.height /= prescale;
            continue;
          }
          crop.x = ~~(crop.x / prescale);
          crop.y = ~~(crop.y / prescale);
          crop.width = ~~(crop.width / prescale);
          crop.height = ~~(crop.height / prescale);
        }
        result.position = positionOf(result.topCrop, imageWidth, imageHeight);
        if (callback) callback(result);
        return result;
      })
    );
};

const positionOf = (
  crop: Crop,
  imageWidth: number,
  imageHeight: number
): CropPosition => {
  const percent = (offset: number, slack: number) =>
    slack > 0 ? Math.min(100, Math.max(0, (100 * offset) / slack)) : 50;
  return {
    x: percent(crop.x, imageWidth - crop.width),
    y: percent(crop.y, imageHeight - crop.height),
  };
};

// Check if all the dependencies are there
const isAvailable = (options?: { canvasFactory?: CanvasFactory }): boolean => {
  if (!smartcrop.Promise) return false;

  const canvasFactory = options ? options.canvasFactory : defaultCanvasFactory;

  if (canvasFactory === defaultCanvasFactory) {
    const c = document.createElement('canvas');
    if (!c.getContext('2d')) {
      return false;
    }
  }

  return true;
};

const DEFAULTS: Options = {
  width: 0,
  height: 0,
  aspect: 0,
  cropWidth: 0,
  cropHeight: 0,
  detailWeight: 0.2,
  skinColor: [0.78, 0.57, 0.44],
  skinBias: 0.01,
  skinBrightnessMin: 0.2,
  skinBrightnessMax: 1.0,
  skinThreshold: 0.8,
  skinWeight: 1.8,
  saturationBrightnessMin: 0.05,
  saturationBrightnessMax: 0.9,
  saturationThreshold: 0.4,
  saturationBias: 0.2,
  saturationWeight: 0.1,
  // Step * minscale rounded down to the next power of two should be good
  scoreDownSample: 8,
  step: 8,
  scaleStep: 0.1,
  minScale: 1.0,
  maxScale: 1.0,
  edgeRadius: 0.4,
  edgeWeight: -20.0,
  outsideImportance: -0.5,
  boostWeight: 100.0,
  ruleOfThirds: true,
  prescale: true,
  imageOperations: null,
  canvasFactory: defaultCanvasFactory,
  debug: false,
  luminance: 'classic',
  saliencyWeight: 0,
  centerWeight: 0,
  textureDamping: 0,
  borderPenalty: 0,
  importanceMapWeight: 0,
  scorer: 'learned',
  contentFloor: 0,
  classicBlend: 0,
  consensus: 0,
};

/**
 * Named option bundles. The learned scorer is the default since 3.0;
 * `classic` reproduces the original algorithm's crops exactly:
 * `crop(img, { ...PRESETS.classic, width, height })`. `next` is kept as an
 * alias of the default.
 */
const PRESETS: Record<'classic' | 'next', CropOptions> = {
  classic: { scorer: 'classic' },
  // Learned composition scorer (trained on GAICD human ratings). Its maps use
  // the classic luminance, which the weights were trained with.
  next: { scorer: 'learned' },
};

const smartcrop = {
  // Promise implementation to use
  Promise: Promise as PromiseConstructor,
  DEFAULTS,
  PRESETS,
  crop,
  isAvailable,
  importance,
  ImgData,
  /**
   * @internal Learned-scorer feature vectors (FEATURE_COUNT floats per crop)
   * for the given candidate crops, for training. `options` needs aspect or
   * width/height so the image is prescaled the same way crop() does.
   */
  _features: async (
    image: unknown,
    candidates: CropBox[],
    options?: CropOptions
  ): Promise<Float32Array> => {
    const r = await crop(image, {
      ...options,
      scorer: 'learned',
      candidates,
      emitFeatures: true,
    });
    return r.features!;
  },
  _downSample: downSample,
  _canvasImageOperations: canvasImageOperations,
};

export default smartcrop;
