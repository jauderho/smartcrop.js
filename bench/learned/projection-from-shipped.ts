// Rebuild bench/out/cnn/projection.bin from the projection already shipped in
// src/learned-cnn-weights.ts (float16, exactly what cnn-pca.ts writes).
//
// Retraining the cnn head with the shipped projection keeps the new head
// comparable with the shipped one (same features) and spares the PCA fit on
// GAICD-train. Use cnn-pca.ts instead to refit the projection.
//
// Usage: bun bench/learned/projection-from-shipped.ts [--out path]
import { dirname } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { decodeF16 } from '../../src/learned';
import { LEARNED_CNN_PACKED } from '../../src/learned-cnn-weights';
import { PROJECTION_FILE } from '../plugin-cache';

const i = process.argv.indexOf('--out');
const out = i > -1 ? process.argv[i + 1] : PROJECTION_FILE;
const { channels, projection } = LEARNED_CNN_PACKED;
const f = decodeF16(projection); // mean[448] then proj[448 x channels]
const buf = new Uint8Array(4 + f.byteLength);
new Uint32Array(buf.buffer, 0, 1)[0] = channels;
buf.set(new Uint8Array(f.buffer, f.byteOffset, f.byteLength), 4);
await mkdir(dirname(out), { recursive: true });
await Bun.write(out, buf);
console.log(`wrote ${out}: ${channels} channels, ${f.length} values`);
