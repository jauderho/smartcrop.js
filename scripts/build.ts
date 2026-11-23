import { rm, mkdir, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dir, '..');
const dist = resolve(root, 'dist');
const srcDir = resolve(root, 'src');

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

const common = {
  outdir: dist,
  target: 'browser',
  minify: false,
  sourcemap: 'linked',
} as const;

const builds = [
  { entry: 'smartcrop.ts', format: 'esm', naming: 'smartcrop.js' },
  { entry: 'entry-cjs.ts', format: 'cjs', naming: 'smartcrop.cjs' },
  { entry: 'entry-browser.ts', format: 'iife', naming: 'smartcrop.iife.js' },
] as const;

for (const b of builds) {
  const result = await Bun.build({
    ...common,
    entrypoints: [resolve(srcDir, b.entry)],
    format: b.format,
    naming: b.naming,
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    process.exit(1);
  }
}

const tsc = Bun.spawnSync(
  [
    'bunx',
    'tsc',
    resolve(srcDir, 'smartcrop.ts'),
    '--ignoreConfig',
    '--declaration',
    '--emitDeclarationOnly',
    '--outDir',
    dist,
    '--rootDir',
    srcDir,
    '--skipLibCheck',
    '--strict',
    '--lib',
    'ES2022,DOM',
    '--target',
    'ES2020',
    '--module',
    'ESNext',
    '--moduleResolution',
    'bundler',
    '--types',
    '',
  ],
  { cwd: root, stdout: 'inherit', stderr: 'inherit' }
);
if (tsc.exitCode !== 0) process.exit(tsc.exitCode ?? 1);

for (const f of [
  'smartcrop.js',
  'smartcrop.cjs',
  'smartcrop.iife.js',
  'smartcrop.d.ts',
]) {
  const { size } = await stat(resolve(dist, f));
  console.log(`dist/${f}`.padEnd(28), `${(size / 1024).toFixed(2)} KB`);
}
