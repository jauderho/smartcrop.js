export interface Crop {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: CropScore;
}
export interface CropScore {
  detail: number;
  saturation: number;
  skin: number;
  boost: number;
  total: number;
}
export interface CropResult {
  topCrop: Crop;
  crops?: Crop[];
}
export interface CropBoost {
  x: number;
  y: number;
  width: number;
  height: number;
  weight: number;
}
export interface ImageData {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}
export interface ImageOperations<T = unknown> {
  open(image: unknown, inputOptions?: unknown): Promise<T>;
  resample(image: T, width: number, height: number): Promise<T>;
  getData(image: T): Promise<ImageData>;
}
export interface CanvasLike {
  width: number;
  height: number;
  getContext(contextId: '2d'): unknown;
}
export type CanvasFactory = (width: number, height: number) => CanvasLike;
export interface CropOptions {
  width?: number;
  height?: number;
  aspect?: number;
  minScale?: number;
  maxScale?: number;
  boost?: CropBoost[];
  ruleOfThirds?: boolean;
  prescale?: boolean;
  canvasFactory?: CanvasFactory;
  imageOperations?: ImageOperations | null;
  input?: unknown;
  debug?: boolean;
}
declare const smartcrop: {
  DEFAULTS: Required<Omit<CropOptions, 'aspect' | 'boost' | 'input'>>;
  crop(
    image: CanvasImageSource | unknown,
    options?: CropOptions,
    callback?: (result: CropResult) => void
  ): Promise<CropResult>;
  isAvailable(options?: Pick<CropOptions, 'canvasFactory'>): boolean;
};
export default smartcrop;
