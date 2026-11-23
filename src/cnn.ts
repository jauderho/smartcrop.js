// Deep-feature pooling for the learned scorer (optional "cnn" plug-in).
//
// A plug-in (plugins/onnx/cnn.ts) turns the image into a coarse C-channel
// feature map. Here we pool it per candidate crop, in O(C K^2) with
// summed-area tables:
//   RoI    K x K bin averages of the map inside the crop (RoIAlign),
//   RoD    K x K bin averages over the whole image with the crop masked out
//          (RoDAlign; 0 where a bin lies entirely inside the crop),
//   global mean of the map.
// Bin integrals are exact (the SAT is bilinearly interpolated at fractional
// coordinates, which integrates the piecewise-constant map exactly).
import type { PackedModel } from './learned';

/** Channels-last feature map over the whole image: data[(y * width + x) * channels + c]. */
export interface FeatureMap {
  width: number;
  height: number;
  channels: number;
  data: Float32Array;
}

/**
 * Weights of the optional cnn head (see bench/learned/train.ts --cnn): the
 * pooling grid, the fixed projection used by the plug-in, and the scoring MLP
 * over [hand features, cnn features].
 */
export interface PackedCnn {
  /** Pooling grid (K x K bins). */
  k: number;
  /** Channels of the projected map. */
  channels: number;
  /** Channels of the backbone map before projection. */
  inChannels: number;
  /** base64 float16 LE: mean (inChannels) then projection (inChannels x channels, row-major). */
  projection: string;
  /** Head over FEATURE_COUNT + cnnDim(k, channels) inputs. */
  head: PackedModel;
}

/** Number of pooled features for a K x K grid over C channels. */
export const cnnDim = (k: number, channels: number): number =>
  (2 * k * k + 1) * channels;

export interface CnnContext {
  fm: FeatureMap;
  /** Summed-area table, (width + 1) x (height + 1) x channels. */
  sat: Float64Array;
  mean: Float32Array;
}

export const cnnContext = (fm: FeatureMap): CnnContext => {
  const { width: w, height: h, channels: C, data } = fm;
  const sw = w + 1;
  const sat = new Float64Array(sw * (h + 1) * C);
  const mean = new Float32Array(C);
  const row = new Float64Array(C);
  for (let y = 0; y < h; y++) {
    row.fill(0);
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * C;
      const t = ((y + 1) * sw + x + 1) * C;
      const up = (y * sw + x + 1) * C;
      for (let c = 0; c < C; c++) {
        row[c] += data[o + c];
        sat[t + c] = sat[up + c] + row[c];
      }
    }
  }
  const tot = (h * sw + w) * C;
  for (let c = 0; c < C; c++) mean[c] = sat[tot + c] / (w * h);
  return { fm, sat, mean };
};

// Integral of channel c over [0,x] x [0,y] (grid units), bilinear in the SAT.
const satAt = (
  s: Float64Array,
  sw: number,
  C: number,
  x: number,
  y: number,
  w: number,
  h: number,
  out: Float64Array,
  sign: number,
): void => {
  x = x < 0 ? 0 : x > w ? w : x;
  y = y < 0 ? 0 : y > h ? h : y;
  const x0 = Math.min(w - 1, Math.floor(x));
  const y0 = Math.min(h - 1, Math.floor(y));
  const fx = x - x0;
  const fy = y - y0;
  const a = (y0 * sw + x0) * C;
  const b = a + C;
  const c0 = a + sw * C;
  const d = c0 + C;
  const w00 = (1 - fx) * (1 - fy) * sign;
  const w10 = fx * (1 - fy) * sign;
  const w01 = (1 - fx) * fy * sign;
  const w11 = fx * fy * sign;
  for (let c = 0; c < C; c++)
    out[c] += w00 * s[a + c] + w10 * s[b + c] + w01 * s[c0 + c] + w11 * s[d + c];
};

// out[c] = integral of the map over the rectangle (grid units).
const rectSum = (
  cx: CnnContext,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  out: Float64Array,
): void => {
  out.fill(0);
  const { width: w, height: h, channels: C } = cx.fm;
  const sw = w + 1;
  satAt(cx.sat, sw, C, x1, y1, w, h, out, 1);
  satAt(cx.sat, sw, C, x0, y1, w, h, out, -1);
  satAt(cx.sat, sw, C, x1, y0, w, h, out, -1);
  satAt(cx.sat, sw, C, x0, y0, w, h, out, 1);
};

/**
 * Write cnnDim(k, C) pooled features of the crop (x, y, w, h) (pixels of an
 * image of size W x H) into out[off ..): RoI bins, RoD bins, global mean.
 */
export const cnnFeatures = (
  cx: CnnContext,
  k: number,
  x: number,
  y: number,
  w: number,
  h: number,
  W: number,
  H: number,
  out: Float32Array,
  off: number,
): void => {
  const { width: gw, height: gh, channels: C } = cx.fm;
  const sx = gw / W;
  const sy = gh / H;
  const cx0 = Math.max(0, x * sx);
  const cy0 = Math.max(0, y * sy);
  const cx1 = Math.min(gw, (x + w) * sx);
  const cy1 = Math.min(gh, (y + h) * sy);
  const bw = (cx1 - cx0) / k;
  const bh = (cy1 - cy0) / k;
  const tmp = new Float64Array(C);
  let o = off;
  // RoI
  const roiArea = bw * bh;
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < k; i++) {
      if (roiArea > 1e-9) {
        rectSum(cx, cx0 + i * bw, cy0 + j * bh, cx0 + (i + 1) * bw, cy0 + (j + 1) * bh, tmp);
        for (let c = 0; c < C; c++) out[o + c] = tmp[c] / roiArea;
      } else for (let c = 0; c < C; c++) out[o + c] = cx.mean[c];
      o += C;
    }
  }
  // RoD: bins of the whole image minus the crop.
  const dw = gw / k;
  const dh = gh / k;
  const inter = new Float64Array(C);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < k; i++) {
      const bx0 = i * dw;
      const by0 = j * dh;
      const bx1 = bx0 + dw;
      const by1 = by0 + dh;
      const ix0 = Math.max(bx0, cx0);
      const iy0 = Math.max(by0, cy0);
      const ix1 = Math.min(bx1, cx1);
      const iy1 = Math.min(by1, cy1);
      const ia = ix1 > ix0 && iy1 > iy0 ? (ix1 - ix0) * (iy1 - iy0) : 0;
      const area = dw * dh - ia;
      if (area < 1e-6 * dw * dh) {
        for (let c = 0; c < C; c++) out[o + c] = 0;
      } else {
        rectSum(cx, bx0, by0, bx1, by1, tmp);
        if (ia > 0) {
          rectSum(cx, ix0, iy0, ix1, iy1, inter);
          for (let c = 0; c < C; c++) tmp[c] -= inter[c];
        }
        for (let c = 0; c < C; c++) out[o + c] = tmp[c] / area;
      }
      o += C;
    }
  }
  for (let c = 0; c < C; c++) out[o + c] = cx.mean[c];
};
