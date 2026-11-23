// Static dev server for the examples and browser benchmark.
// Usage: bun run start [--port 8000]
import { join, normalize } from 'node:path';

const root = normalize(join(import.meta.dir, '..'));
const portArg = process.argv.indexOf('--port');
const port = portArg > -1 ? Number(process.argv[portArg + 1]) : 8000;

const server = Bun.serve({
  hostname: 'localhost',
  port,
  async fetch(req) {
    let path = decodeURIComponent(new URL(req.url).pathname);
    if (path.endsWith('/')) path += 'index.html';
    const file = normalize(join(root, path));
    if (!file.startsWith(root))
      return new Response('Forbidden', { status: 403 });
    const f = Bun.file(file);
    return (await f.exists())
      ? new Response(f)
      : new Response('Not found', { status: 404 });
  },
});

const base = server.url.href.replace(/\/$/, '');
console.log(`Serving ${root}`);
console.log(`  Test suite: ${base}/examples/testsuite.html`);
console.log(`  Testbed:    ${base}/examples/testbed.html`);
console.log(`  Slideshow:  ${base}/examples/slideshow.html`);
console.log(`  Benchmark:  ${base}/test/benchmark.html`);
