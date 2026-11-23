// Minimal MATLAB level-5 .mat reader (numeric, char, cell, struct only).
// Parses untrusted bytes in memory; never executes anything.
import { inflateSync } from 'node:zlib';

export type MatValue =
  | { cls: 'num'; dims: number[]; data: number[] }
  | { cls: 'char'; dims: number[]; text: string }
  | { cls: 'cell'; dims: number[]; items: MatValue[] }
  | {
      cls: 'struct';
      dims: number[];
      fields: string[];
      items: Record<string, MatValue>[];
    };

const SIZES: Record<number, number> = {
  1: 1,
  2: 1,
  3: 2,
  4: 2,
  5: 4,
  6: 4,
  7: 4,
  9: 8,
  12: 8,
  13: 8,
};

function readNumbers(type: number, b: Buffer): number[] {
  const n = b.length / SIZES[type];
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * SIZES[type];
    switch (type) {
      case 1:
        out.push(b.readInt8(o));
        break;
      case 2:
        out.push(b.readUInt8(o));
        break;
      case 3:
        out.push(b.readInt16LE(o));
        break;
      case 4:
        out.push(b.readUInt16LE(o));
        break;
      case 5:
        out.push(b.readInt32LE(o));
        break;
      case 6:
        out.push(b.readUInt32LE(o));
        break;
      case 7:
        out.push(b.readFloatLE(o));
        break;
      case 9:
        out.push(b.readDoubleLE(o));
        break;
      default:
        throw new Error(`unsupported mat numeric type ${type}`);
    }
  }
  return out;
}

// Returns [type, payload, nextOffset].
function element(buf: Buffer, off: number): [number, Buffer, number] {
  const w = buf.readUInt32LE(off);
  if (w >> 16) {
    // small data element: size in upper 16 bits, 4 bytes of payload
    return [w & 0xffff, buf.subarray(off + 4, off + 4 + (w >> 16)), off + 8];
  }
  const size = buf.readUInt32LE(off + 4);
  const end = off + 8 + size;
  return [w, buf.subarray(off + 8, end), end + ((8 - (size % 8)) % 8)];
}

function matrix(payload: Buffer): { name: string; value: MatValue } {
  let off = 0;
  let t: number, p: Buffer;
  [t, p, off] = element(payload, off);
  const cls = p[0] & 0xff;
  [t, p, off] = element(payload, off);
  const dims = readNumbers(t, p);
  [t, p, off] = element(payload, off);
  const name = p.toString('latin1');
  if (cls === 1) {
    const items: MatValue[] = [];
    while (off < payload.length) {
      [t, p, off] = element(payload, off);
      items.push(matrix(p).value);
    }
    return { name, value: { cls: 'cell', dims, items } };
  }
  if (cls === 2) {
    [t, p, off] = element(payload, off);
    const flen = readNumbers(t, p)[0];
    [t, p, off] = element(payload, off);
    const fields: string[] = [];
    for (let i = 0; i < p.length / flen; i++) {
      fields.push(
        p
          .subarray(i * flen, (i + 1) * flen)
          .toString('latin1')
          .replace(/\0+$/, '')
      );
    }
    const count = dims.reduce((a, b) => a * b, 1);
    const items: Record<string, MatValue>[] = [];
    for (let i = 0; i < count; i++) {
      const rec: Record<string, MatValue> = {};
      for (const f of fields) {
        [t, p, off] = element(payload, off);
        rec[f] = matrix(p).value;
      }
      items.push(rec);
    }
    return { name, value: { cls: 'struct', dims, fields, items } };
  }
  [t, p, off] = element(payload, off);
  if (cls === 4) {
    const text =
      t === 4 || t === 2
        ? String.fromCharCode(...readNumbers(t, p))
        : p.toString('latin1');
    return { name, value: { cls: 'char', dims, text } };
  }
  return { name, value: { cls: 'num', dims, data: readNumbers(t, p) } };
}

export function readMat(file: Buffer): Record<string, MatValue> {
  const out: Record<string, MatValue> = {};
  let off = 128;
  while (off < file.length) {
    let [t, p, next] = element(file, off);
    if (t === 15) [t, p] = element(inflateSync(p), 0);
    if (t === 14) {
      const m = matrix(p);
      out[m.name] = m.value;
    }
    off = next;
  }
  return out;
}
