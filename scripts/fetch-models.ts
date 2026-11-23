// Download the ONNX models used by the optional plug-ins in plugins/onnx.
// Files land in bench/models/ (gitignored) and must match the pinned sha256;
// size + sha256 go to models.json.
// Existing files are skipped. Models are data only; nothing is executed.
//
// Usage: bun scripts/fetch-models.ts
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const MODELS = [
  {
    name: 'face_detection_yunet_2023mar.onnx',
    sha256: '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    url: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    license: 'MIT (OpenCV Zoo)',
    minBytes: 100_000,
  },
  {
    name: 'u2netp.onnx',
    sha256: '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8',
    url: 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
    license: 'Apache-2.0 (U2-Net)',
    minBytes: 1_000_000,
  },
  {
    // ImageNet MobileNetV2 (MXNet Gluon) from the ONNX model zoo; used as a
    // frozen feature backbone by plugins/onnx/cnn.ts.
    name: 'mobilenetv2-12.onnx',
    sha256: 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5',
    url: 'https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
    license: 'Apache-2.0 (ONNX Model Zoo)',
    minBytes: 10_000_000,
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
  const sha256 = new Bun.CryptoHasher('sha256').update(bytes).digest('hex');
  if (sha256 !== m.sha256) {
    throw new Error(`${m.name}: sha256 ${sha256} does not match pinned ${m.sha256}`);
  }
  manifest[m.name] = {
    url: m.url,
    license: m.license,
    size: bytes.length,
    sha256,
  };
}
await Bun.write(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
