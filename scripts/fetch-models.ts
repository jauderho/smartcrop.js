// Download the ONNX models used by the optional plug-ins in plugins/onnx.
// Files land in bench/models/ (gitignored); size + sha256 go to models.json.
// Existing files are skipped. Models are data only; nothing is executed.
//
// Usage: bun scripts/fetch-models.ts
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const MODELS = [
  {
    name: 'face_detection_yunet_2023mar.onnx',
    url: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    license: 'MIT (OpenCV Zoo)',
    minBytes: 100_000,
  },
  {
    name: 'u2netp.onnx',
    url: 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
    license: 'Apache-2.0 (U2-Net)',
    minBytes: 1_000_000,
  },
];

const dir = join(import.meta.dir, '../bench/models');
await mkdir(dir, { recursive: true });
const manifestFile = Bun.file(join(dir, 'models.json'));
const manifest: Record<string, unknown> = (await manifestFile.exists())
  ? await manifestFile.json()
  : {};

for (const m of MODELS) {
  const path = join(dir, m.name);
  if (!(await Bun.file(path).exists())) {
    console.log(`downloading ${m.name}`);
    const res = await fetch(m.url);
    if (!res.ok) throw new Error(`${m.url}: HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    // Git LFS pointers are tiny text files starting with "version https://git-lfs".
    const head = new TextDecoder().decode(buf.subarray(0, 40));
    if (buf.length < m.minBytes || head.startsWith('version https://git-lfs')) {
      throw new Error(`${m.name}: not a real model (${buf.length} bytes)`);
    }
    await Bun.write(path, buf);
  } else {
    console.log(`have ${m.name}`);
  }
  const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
  manifest[m.name] = {
    url: m.url,
    license: m.license,
    size: bytes.length,
    sha256: new Bun.CryptoHasher('sha256').update(bytes).digest('hex'),
  };
}
await Bun.write(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
