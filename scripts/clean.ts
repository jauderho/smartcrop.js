// Reclaim disk space used by benchmark data when you're not working on crop
// quality. Shows what would be removed (with sizes) unless --yes is given.
//
// Usage: bun scripts/clean.ts [--level light|datasets|full] [--results] [--yes]
//
//   light (default)  computed caches only (plug-in outputs, learned-scorer
//                    features, debug images, dist). Rebuilt offline and exactly
//                    by `bun bench/precompute.ts <set>` and the bench scripts.
//   datasets         also the extracted datasets. `bun scripts/fetch-datasets.ts`
//                    restores FLMS and GAICD offline from the archives in
//                    bench/datasets/_dl; FCDB is re-fetched from Flickr (the
//                    Drive archive differs slightly from the Flickr originals, so
//                    it is only a fallback for photos that have gone).
//   full             also the downloaded archives and model weights. Restoring
//                    needs the network: `bun scripts/fetch-datasets.ts`,
//                    `bun scripts/fetch-models.ts`.
//   --results        also remove results: tuning runs, galleries, stage tables
//                    and content tags (tags cost Jev credit to recreate).
//
// Never touched: bench/private (hand-made labels for private photos), source,
// and anything outside bench/ and dist/.
import { existsSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = join(import.meta.dir, '..');
const args = process.argv.slice(2);
const yes = args.includes('--yes');
const results = args.includes('--results');
const li = args.indexOf('--level');
const level = li > -1 ? args[li + 1] : 'light';
if (!['light', 'datasets', 'full'].includes(level)) {
  console.error('--level must be light, datasets or full');
  process.exit(2);
}

const p = (...s: string[]) => join(root, ...s);
const targets: [string, string][] = [
  [p('bench/out/cache'), 'ONNX plug-in caches (faces, saliency)'],
  [p('bench/out/learned'), 'learned-scorer feature caches'],
  [p('bench/out/plugin-check'), 'plug-in debug images'],
  [p('dist'), 'build output (bun run build)'],
];
if (level !== 'light') {
  for (const d of ['fcdb', 'flms', 'gaicd']) {
    targets.push([p('bench/datasets', d), `extracted ${d.toUpperCase()}`]);
  }
}
if (level === 'full') {
  targets.push(
    [p('bench/datasets/_dl'), 'downloaded dataset archives'],
    [p('bench/models'), 'downloaded model weights']
  );
}
if (results) {
  for (const name of existsSync(p('bench/out'))
    ? readdirSync(p('bench/out'))
    : []) {
    const full = p('bench/out', name);
    if (!targets.some(([t]) => t === full)) targets.push([full, 'results']);
  }
}

function size(path: string): number {
  const st = statSync(path, { throwIfNoEntry: false });
  if (!st) return 0;
  if (!st.isDirectory()) return st.size;
  return readdirSync(path).reduce((s, n) => s + size(join(path, n)), 0);
}
const fmt = (b: number) =>
  b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`;

// Don't pull data out from under a running benchmark or tuning job.
const busy = Bun.spawnSync(['pgrep', '-fl', 'bench/|fetch-(datasets|models)'])
  .stdout.toString()
  .split('\n')
  .filter((l) => /^\d+ bun /.test(l) && !l.includes('scripts/clean.ts'))
  .map((l) => l.slice(0, 120));
if (busy.length) {
  console.error(`Benchmark jobs are running:\n  ${busy.join('\n  ')}`);
  if (yes) {
    console.error('Not removing anything; stop them first.');
    process.exit(1);
  }
}

let total = 0;
for (const [path, what] of targets) {
  if (!existsSync(path)) continue;
  if (path.startsWith(p('bench/private'))) continue;
  const b = size(path);
  total += b;
  console.log(`${fmt(b).padStart(10)}  ${relative(root, path)}  (${what})`);
  if (yes) rmSync(path, { recursive: true, force: true });
}
console.log(
  `${fmt(total).padStart(10)}  total ${yes ? 'removed' : 'would be removed — re-run with --yes'} (level ${level}${results ? ', with results' : ''})`
);
