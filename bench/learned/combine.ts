// Combine independently trained ensemble members into one model file.
//
// train.ts --ensemble K trains seeds s, s+1, ... s+K-1 one after another and
// averages them. Each member is trained and selected independently, so the
// members can be trained in parallel processes (train.ts --seed N --write
// --out member-N.json) and merged here; the result is the same model file
// train.ts --ensemble would have written.
//
// Usage: bun bench/learned/combine.ts <out.json> <member1.json> [member2.json ...]
// The first member supplies the standardisation and `cnn` field; the rest are
// added to `model.members`.
import { readFile } from 'node:fs/promises';
import type { LearnedModel } from '../../src/learned';

const [out, first, ...rest] = process.argv.slice(2);
if (!out || !first) {
  console.error(
    'usage: combine.ts <out.json> <member1.json> [member2.json ...]'
  );
  process.exit(2);
}
const read = async (p: string) =>
  JSON.parse(await readFile(p, 'utf8')) as {
    note: string;
    model: LearnedModel;
    cnn?: unknown;
  };
const head = await read(first);
const sameArch = (a: LearnedModel, b: LearnedModel) =>
  a.layers.length === b.layers.length &&
  a.layers.every(
    (l, i) =>
      l.w.length === b.layers[i].w.length && l.b.length === b.layers[i].b.length
  ) &&
  a.mean.length === b.mean.length;
const members: LearnedModel['layers'][] = [];
for (const p of rest) {
  const m = await read(p);
  if (!sameArch(head.model, m.model))
    throw new Error(`${p}: architecture differs`);
  if (JSON.stringify(m.model.mean) !== JSON.stringify(head.model.mean))
    throw new Error(`${p}: standardisation differs (different training sets?)`);
  members.push(m.model.layers);
}
const model: LearnedModel = { ...head.model };
if (members.length) model.members = members;
const note = `ensemble of ${1 + members.length} (combine.ts); member 1: ${head.note}`;
await Bun.write(
  out,
  JSON.stringify({ note, model, ...(head.cnn ? { cnn: head.cnn } : {}) })
);
console.log(`wrote ${out}: ${1 + members.length} members`);
