// Minimal ONNX (protobuf) rewriting, so a stock classification model can be
// used as a feature backbone without shipping a modified file:
//   - make the input's spatial dims symbolic (any H x W), and
//   - expose intermediate tensors as extra graph outputs.
// Only the fields on that path are decoded; everything else is copied as-is.

type Field = { f: number; w: number; s: number; e: number; v?: number };

function* fields(b: Uint8Array, s = 0, e = b.length): Generator<Field> {
  let i = s;
  const varint = () => {
    let r = 0;
    let m = 1;
    let x: number;
    do {
      x = b[i++];
      r += (x & 0x7f) * m;
      m *= 128;
    } while (x & 0x80);
    return r;
  };
  while (i < e) {
    const start = i;
    const key = varint();
    const f = key >>> 3;
    const w = key & 7;
    if (w === 0) {
      const v = varint();
      yield { f, w, s: start, e: i, v };
    } else if (w === 2) {
      const n = varint();
      i += n;
      yield { f, w, s: start, e: i };
    } else if (w === 5) {
      i += 4;
      yield { f, w, s: start, e: i };
    } else if (w === 1) {
      i += 8;
      yield { f, w, s: start, e: i };
    } else {
      throw new Error(`unsupported protobuf wire type ${w}`);
    }
  }
}

/** Payload range of a length-delimited field (after key and length). */
const payload = (b: Uint8Array, x: Field): [number, number] => {
  let i = x.s;
  while (b[i++] & 0x80); // key
  while (b[i++] & 0x80); // length
  return [i, x.e];
};

const varintBytes = (n: number): number[] => {
  const out: number[] = [];
  do {
    let byte = n % 128;
    n = Math.floor(n / 128);
    if (n > 0) byte |= 0x80;
    out.push(byte);
  } while (n > 0);
  return out;
};
const ld = (f: number, body: Uint8Array): Uint8Array => {
  const head = [...varintBytes((f << 3) | 2), ...varintBytes(body.length)];
  const out = new Uint8Array(head.length + body.length);
  out.set(head);
  out.set(body, head.length);
  return out;
};
const concat = (parts: Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};
const text = (s: string) => new TextEncoder().encode(s);

/** Rebuild a message, replacing chosen length-delimited fields' payloads. */
const rewrite = (
  b: Uint8Array,
  s: number,
  e: number,
  edit: (f: number, body: Uint8Array, index: number) => Uint8Array | null,
  append: Uint8Array[] = []
): Uint8Array => {
  const parts: Uint8Array[] = [];
  const seen = new Map<number, number>();
  for (const x of fields(b, s, e)) {
    const k = seen.get(x.f) ?? 0;
    seen.set(x.f, k + 1);
    if (x.w === 2) {
      const [ps, pe] = payload(b, x);
      const out = edit(x.f, b.subarray(ps, pe), k);
      if (out) {
        parts.push(ld(x.f, out));
        continue;
      }
    }
    parts.push(b.subarray(x.s, x.e));
  }
  return concat([...parts, ...append]);
};

// TensorShapeProto: dim = 1 (Dimension: dim_value = 1, dim_param = 2).
const symbolicShape = (shape: Uint8Array, names: (string | null)[]) =>
  rewrite(shape, 0, shape.length, (f, body, i) =>
    f === 1 && names[i] ? ld(2, text(names[i]!)) : null
  );

/**
 * Return a patched model: input spatial dims made symbolic and the named
 * intermediate tensors added as outputs (type info is left for the runtime to
 * infer).
 */
export function patchOnnx(
  model: Uint8Array,
  extraOutputs: string[]
): Uint8Array {
  // ModelProto.graph = 7; GraphProto.input = 11, output = 12;
  // ValueInfoProto.type = 2; TypeProto.tensor_type = 1; Tensor.shape = 2.
  return rewrite(model, 0, model.length, (f, graph) => {
    if (f !== 7) return null;
    const outputs = extraOutputs.map((name) => ld(12, ld(1, text(name))));
    return rewrite(
      graph,
      0,
      graph.length,
      (gf, body, i) => {
        if (gf !== 11 || i !== 0) return null;
        return rewrite(body, 0, body.length, (vf, type) => {
          if (vf !== 2) return null;
          return rewrite(type, 0, type.length, (tf, tensor) => {
            if (tf !== 1) return null;
            return rewrite(tensor, 0, tensor.length, (sf, shape) =>
              sf === 2
                ? symbolicShape(shape, [null, null, 'height', 'width'])
                : null
            );
          });
        });
      },
      outputs
    );
  });
}
