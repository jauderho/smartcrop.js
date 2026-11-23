// Baseline: seeded random scores (deterministic per image path and seed).
// Options: { seed?: number }
const hash = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++)
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
};

// mulberry32
const rng = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

export async function scoreCandidates(
  imagePath: string,
  _width: number,
  _height: number,
  candidates: [number, number, number, number][],
  options: { seed?: number } = {},
): Promise<number[]> {
  const next = rng(hash(imagePath) ^ (options.seed ?? 1));
  return candidates.map(() => next());
}
