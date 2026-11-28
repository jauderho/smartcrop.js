// Fingerprint of everything that determines the trained scorer weights, and the
// freshness check that compares it with the one recorded beside the shipped
// weights (src/weights-manifest.json).
//
// The fingerprint is a SHA-256 over
//   - the feature and training code: src/smartcrop.ts, src/learned.ts,
//     src/cnn.ts, plugins/onnx/*.ts, bench/plugin-cache.ts, bench/precompute.ts,
//     bench/learned/*.ts (the shipped weight files themselves are outputs and
//     are excluded);
//   - the dataset identities: scripts/fetch-datasets.ts and scripts/mat5.ts
//     (pinned URLs, ids, split logic);
//   - the pinned ONNX model sha256 values in scripts/fetch-models.ts;
//   - the versions of onnxruntime-node and @napi-rs/canvas in bun.lock (they
//     produce the plug-in features);
//   - for the cnn head without config.refitPca, the PCA projection stored in the
//     shipped cnn weights (it is the feature input of the new head);
//   - the training configuration (flags and seed), canonicalised.
// Nothing else (docs, tests, eval scripts, the CLI, the classic scorer, ...)
// is hashed, so unrelated edits do not make weights stale.
//
// Usage:
//   bun scripts/weights-fingerprint.ts --config '{"cnn":3,"epochs":30,...}'
//        [--verbose] [--root DIR]        print the fingerprint
//   bun scripts/weights-fingerprint.ts --check [--manifest FILE] [--root DIR]
//        freshness check: exit 1 when a recorded fingerprint is stale
//   bun scripts/weights-fingerprint.ts --write --config '{...}'
//        [--weights FILE] [--manifest FILE] [--metrics '{...}']
//        record the fingerprint (and the sha256 of the weights file) for the
//        config's head (cnn when config.cnn > 0, otherwise hand)
//
// --root points at another checkout (or a temp copy) of the repository.
import { existsSync } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';

/** Bump when the hashing scheme itself changes. */
const SCHEME = 1;

export type TrainConfig = Record<string, string | number | boolean>;

const sha = (data: string | Uint8Array) =>
  new Bun.CryptoHasher('sha256').update(data).digest('hex');

/** Hash text with line endings normalised (stable across checkouts). */
const fileHash = async (path: string) =>
  sha((await readFile(path, 'utf8')).replace(/\r\n/g, '\n'));

const listTs = async (root: string, dir: string, skip: RegExp) =>
  (await readdir(join(root, dir)))
    .filter((f) => f.endsWith('.ts') && !skip.test(f))
    .sort()
    .map((f) => `${dir}/${f}`);

/** Repo-relative files hashed as feature/training code. */
export async function codeFiles(root: string): Promise<string[]> {
  return [
    'src/smartcrop.ts',
    'src/learned.ts',
    'src/cnn.ts',
    ...(await listTs(root, 'plugins/onnx', /^$/)),
    'bench/plugin-cache.ts',
    'bench/precompute.ts',
    // scratch copies (_*.ts) are not part of the pipeline
    ...(await listTs(root, 'bench/learned', /^_/)),
    'scripts/fetch-datasets.ts',
    'scripts/mat5.ts',
  ];
}

/** Pinned ONNX model hashes, read from the source of scripts/fetch-models.ts. */
export async function modelHashes(root: string): Promise<string[]> {
  const src = await readFile(join(root, 'scripts/fetch-models.ts'), 'utf8');
  return [...src.matchAll(/sha256:\s*'([0-9a-f]{64})'/g)]
    .map((m) => m[1])
    .sort();
}

/** Versions of the packages that produce the plug-in features, from bun.lock. */
export async function packageVersions(root: string): Promise<string[]> {
  const lock = await readFile(join(root, 'bun.lock'), 'utf8');
  return ['onnxruntime-node', 'onnxruntime-common', '@napi-rs/canvas']
    .map((name) => {
      const m = lock.match(
        new RegExp(`"${name.replace('/', '\\/')}":\\s*\\["([^"]+)"`)
      );
      return `${name}=${m ? m[1] : 'missing'}`;
    })
    .sort();
}

/** Canonical form of a config: sorted keys, numeric strings as numbers. */
export function canonicalConfig(c: TrainConfig): string {
  const out: Record<string, string | number | boolean> = {};
  for (const k of Object.keys(c).sort()) {
    const v = c[k];
    out[k] =
      typeof v === 'string' && v.trim() !== '' && !isNaN(Number(v))
        ? Number(v)
        : v;
  }
  return JSON.stringify(out);
}

export const headOf = (c: TrainConfig): 'cnn' | 'hand' =>
  Number(c.cnn) > 0 ? 'cnn' : 'hand';

export interface Fingerprint {
  fingerprint: string;
  parts: Record<string, string>;
}

/**
 * Fingerprint of the weights trained by `config` with the code in `root`.
 * `parts` lists the hash of every input (for --verbose and debugging).
 */
export async function fingerprint(
  config: TrainConfig,
  root = resolve(import.meta.dir, '..')
): Promise<Fingerprint> {
  const parts: Record<string, string> = { scheme: String(SCHEME) };
  for (const f of await codeFiles(root)) {
    const p = join(root, f);
    parts[f] = existsSync(p) ? await fileHash(p) : 'missing';
  }
  parts['models'] = sha((await modelHashes(root)).join('\n'));
  parts['packages'] = sha((await packageVersions(root)).join('\n'));
  parts['config'] = sha(canonicalConfig(config));
  if (headOf(config) === 'cnn' && !config.refitPca) {
    // Without a PCA refit the projection of the shipped cnn weights is an input.
    const w = join(root, weightsFileOf('cnn'));
    const m = existsSync(w)
      ? // JSON as written by shrink.ts, or TS after bun fmt
        (await readFile(w, 'utf8')).match(
          /["']?projection["']?:\s*["']([^"']*)["']/
        )
      : null;
    parts['projection'] = m ? sha(m[1]) : 'missing';
  }
  const fp = sha(
    Object.keys(parts)
      .sort()
      .map((k) => `${k} ${parts[k]}`)
      .join('\n')
  );
  return { fingerprint: fp, parts };
}

/**
 * Key of a feature dump (bench/learned/dump.ts): only what changes the
 * features, not the training code. `extra` carries the dump arguments and the
 * hashes of its input files (annotations, projection).
 */
export async function featureKey(
  extra: Record<string, string | number | boolean>,
  root = resolve(import.meta.dir, '..')
): Promise<string> {
  const files = [
    'src/smartcrop.ts',
    'src/learned.ts',
    'src/cnn.ts',
    ...(await listTs(root, 'plugins/onnx', /^$/)),
    'bench/plugin-cache.ts',
    'bench/learned/dump.ts',
  ];
  const parts: string[] = [`scheme ${SCHEME}`];
  for (const f of files) parts.push(`${f} ${await fileHash(join(root, f))}`);
  parts.push(`models ${(await modelHashes(root)).join(',')}`);
  parts.push(`packages ${(await packageVersions(root)).join(',')}`);
  parts.push(`extra ${canonicalConfig(extra)}`);
  return sha(parts.join('\n'));
}

// ---- manifest ----------------------------------------------------------------
export interface HeadRecord {
  fingerprint: string;
  config: TrainConfig;
  /** sha256 of the weights source file the fingerprint was recorded for. */
  weightsSha256: string;
  weightsFile: string;
  recordedAt: string;
  metrics?: Record<string, unknown>;
}
export interface Manifest {
  scheme: number;
  heads: Record<string, HeadRecord>;
}

export const weightsFileOf = (head: string) =>
  head === 'cnn' ? 'src/learned-cnn-weights.ts' : 'src/learned-weights.ts';

const MANIFEST = 'src/weights-manifest.json';

export type CheckResult = { ok: boolean; messages: string[] };

/** Compare the recorded fingerprints with the current code. */
export async function check(
  root = resolve(import.meta.dir, '..'),
  manifestPath = join(root, MANIFEST)
): Promise<CheckResult> {
  const msgs: string[] = [];
  if (!existsSync(manifestPath)) {
    return {
      ok: false,
      messages: [
        `${relative(root, manifestPath)} is missing: no training fingerprint is recorded for the shipped weights.`,
        'Run the "Train" workflow and commit its weights together with the manifest it produces,',
        "or record the current weights with: bun scripts/weights-fingerprint.ts --write --config '<json>'",
      ],
    };
  }
  const man = JSON.parse(await readFile(manifestPath, 'utf8')) as Manifest;
  if (man.scheme !== SCHEME || !man.heads || !Object.keys(man.heads).length) {
    return {
      ok: false,
      messages: [
        `${MANIFEST} has an unknown scheme or no heads; re-record it.`,
      ],
    };
  }
  let ok = true;
  for (const [head, rec] of Object.entries(man.heads)) {
    const now = (await fingerprint(rec.config, root)).fingerprint;
    const wf = join(root, weightsFileOf(head));
    const wsha = existsSync(wf) ? await fileHash(wf) : 'missing';
    if (now !== rec.fingerprint) {
      ok = false;
      msgs.push(
        `${head}: STALE. The training code, dataset/model pins or config changed since the shipped weights were trained.`,
        `  recorded ${rec.fingerprint.slice(0, 16)}  current ${now.slice(0, 16)}`,
        '  Run the "Train" workflow (Actions > Train > Run workflow) and commit the weights and manifest it produces.'
      );
    } else if (wsha !== rec.weightsSha256) {
      ok = false;
      msgs.push(
        `${head}: ${weightsFileOf(head)} does not match the weights the manifest was recorded for (edited or replaced by hand).`,
        '  Commit the weights and src/weights-manifest.json from the same training run.'
      );
    } else {
      msgs.push(`${head}: fresh (${now.slice(0, 16)})`);
    }
  }
  return { ok, messages: msgs };
}

/** Record (or replace) the manifest entry of the config's head. */
export async function record(
  config: TrainConfig,
  opts: {
    root?: string;
    manifestPath?: string;
    weightsPath?: string;
    metrics?: Record<string, unknown>;
  } = {}
): Promise<{ head: string; record: HeadRecord }> {
  const root = opts.root ?? resolve(import.meta.dir, '..');
  const manifestPath = opts.manifestPath ?? join(root, MANIFEST);
  const head = headOf(config);
  const weightsPath = opts.weightsPath ?? join(root, weightsFileOf(head));
  const rec: HeadRecord = {
    fingerprint: (await fingerprint(config, root)).fingerprint,
    config: JSON.parse(canonicalConfig(config)),
    weightsSha256: await fileHash(weightsPath),
    weightsFile: weightsFileOf(head),
    recordedAt: new Date().toISOString(),
    ...(opts.metrics ? { metrics: opts.metrics } : {}),
  };
  const man: Manifest = existsSync(manifestPath)
    ? JSON.parse(await readFile(manifestPath, 'utf8'))
    : { scheme: SCHEME, heads: {} };
  man.scheme = SCHEME;
  man.heads[head] = rec;
  await Bun.write(manifestPath, JSON.stringify(man, null, 2) + '\n');
  return { head, record: rec };
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const opt = (n: string) => {
    const i = args.indexOf(`--${n}`);
    return i > -1 ? args[i + 1] : undefined;
  };
  const root = opt('root') ? resolve(opt('root')!) : undefined;
  const config = (
    opt('config') ? JSON.parse(opt('config')!) : {}
  ) as TrainConfig;
  if (args.includes('--check')) {
    const r = await check(root, opt('manifest'));
    for (const m of r.messages) console.log(m);
    if (!r.ok) {
      console.error('Weights freshness check FAILED.');
      process.exit(1);
    }
    console.log('Weights freshness check passed.');
  } else if (args.includes('--write')) {
    const { head, record: rec } = await record(config, {
      root,
      manifestPath: opt('manifest'),
      weightsPath: opt('weights'),
      metrics: opt('metrics') ? JSON.parse(opt('metrics')!) : undefined,
    });
    console.log(`recorded ${head}: ${rec.fingerprint}`);
  } else {
    const r = await fingerprint(config, root);
    if (args.includes('--verbose'))
      for (const [k, v] of Object.entries(r.parts))
        console.log(`${v.slice(0, 12)}  ${k}`);
    console.log(r.fingerprint);
  }
}
