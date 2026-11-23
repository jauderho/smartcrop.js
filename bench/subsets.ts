// Content subsets for per-subset benchmark scoring. This is the ONLY place the
// subset rules live; tags are raw classifier probabilities written to
// bench/out/tags/<dataset>.json as { "<annotation image path>": {people, face,
// landscape, sky} }. Images without tags belong to no subset (only to "all").
import { join } from 'node:path';

export type Tag = {
  people: number;
  face: number;
  landscape: number;
  sky: number;
};
export const SUBSETS = [
  'people',
  'faces',
  'landscape',
  'sky',
  'other',
] as const;
export type Subset = (typeof SUBSETS)[number];
// Subsets averaged (macro) by the balanced tuning objective.
export const BALANCED_SUBSETS: Subset[] = ['people', 'landscape', 'other'];

const T = 0.7;
export function classify(t: Tag): Subset[] {
  const out: Subset[] = [];
  const people = t.people >= T;
  const faces = t.face >= T;
  const landscape = t.landscape >= T && t.people < 0.3;
  if (people) out.push('people');
  if (faces) out.push('faces');
  if (landscape) out.push('landscape');
  if (t.sky >= T) out.push('sky');
  if (!people && !faces && !landscape) out.push('other');
  return out;
}

const loaded = new Map<string, Record<string, Tag>>();
export async function loadTags(dataset: string): Promise<Record<string, Tag>> {
  let t = loaded.get(dataset);
  if (!t) {
    const f = Bun.file(join(import.meta.dir, 'out/tags', `${dataset}.json`));
    t = (await f.exists()) ? await f.json() : {};
    loaded.set(dataset, t!);
  }
  return t!;
}

/** Subsets of an image; call loadTags(dataset) first. [] when untagged. */
export function subsetsOf(dataset: string, image: string): Subset[] {
  const t = loaded.get(dataset)?.[image];
  return t ? classify(t) : [];
}

/** Indices (into `images`) per subset. */
export function indexBySubset(
  dataset: string,
  images: string[]
): Record<Subset, number[]> {
  const out = Object.fromEntries(
    SUBSETS.map((s) => [s, [] as number[]])
  ) as Record<Subset, number[]>;
  images.forEach((im, i) => {
    for (const s of subsetsOf(dataset, im)) out[s].push(i);
  });
  return out;
}
