// Client for mps_worker.py: runs ONNX models with PyTorch on the Apple GPU
// (MPS). One worker process per JS process, started on first use; requests
// are answered in order, so a FIFO of pending replies is enough.
import { spawn, type ChildProcess } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

export interface Tensor {
  data: unknown;
  dims: readonly number[];
}
/** The part of onnxruntime's InferenceSession the plug-ins use. */
export interface Session {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
}

type Reply = { header: any; payload: Buffer };

const WORKER = join(import.meta.dir, 'mps_worker.py');
// Python with torch + onnx2torch: SMARTCROP_MPS_PYTHON, else an ephemeral uv
// environment (downloaded once into uv's cache).
const command = (): string[] => {
  const py = process.env.SMARTCROP_MPS_PYTHON;
  return py
    ? [py, '-I', WORKER]
    : [
        'uv',
        'run',
        '--no-project',
        '--quiet',
        '--with',
        'torch==2.15.0',
        '--with',
        'onnx2torch==1.5.15',
        '--with',
        'onnx',
        '--with',
        'numpy',
        'python',
        '-I',
        WORKER,
      ];
};

class Worker {
  private buf = Buffer.alloc(0);
  private pending: ((r: Reply) => void)[] = [];
  private dead: Error | undefined;
  readonly ready: Promise<void>;

  constructor(private proc: ChildProcess) {
    // Don't keep the JS process alive for the worker; it exits on stdin EOF.
    proc.unref();
    (proc.stdout as any)?.unref?.();
    (proc.stdin as any)?.unref?.();
    proc.stdout!.on('data', (c: Buffer) => this.onData(c));
    proc.on('error', (e) => this.fail(e));
    proc.on('exit', (code) =>
      this.fail(new Error(`mps worker exited (${code})`))
    );
    this.ready = this.next().then(({ header }) => {
      if (!header.ready) throw new Error(header.error ?? 'mps worker failed');
    });
  }

  private fail(e: Error) {
    this.dead ??= e;
    for (const p of this.pending.splice(0))
      p({ header: { error: e.message }, payload: Buffer.alloc(0) });
  }

  private next(): Promise<Reply> {
    if (this.dead)
      return Promise.resolve({
        header: { error: this.dead.message },
        payload: Buffer.alloc(0),
      });
    return new Promise((r) => this.pending.push(r));
  }

  // Chunks are collected and joined only once a whole frame has arrived:
  // re-joining on every 64 KB chunk is quadratic for 10+ MB outputs.
  private chunks: Buffer[] = [];
  private queued = 0;
  private need = 4;

  private onData(chunk: Buffer) {
    this.chunks.push(chunk);
    this.queued += chunk.length;
    if (this.queued < this.need) return;
    this.buf = Buffer.concat([this.buf, ...this.chunks]);
    this.chunks = [];
    this.queued = 0;
    for (;;) {
      // Bytes of the next frame known so far: length word, header, payload.
      let size = 4;
      let header: any;
      if (this.buf.length >= 4) {
        const hl = this.buf.readUInt32LE(0);
        size = 4 + hl;
        if (this.buf.length >= size) {
          header = JSON.parse(this.buf.subarray(4, size).toString());
          size += header.bytes ?? 0;
        }
      }
      this.need = size - this.buf.length;
      if (this.need > 0 || !header) return;
      const payload = this.buf.subarray(size - (header.bytes ?? 0), size);
      this.buf = this.buf.subarray(size);
      this.pending.shift()?.({ header, payload });
    }
  }

  async request(
    header: object,
    payload: Uint8Array = new Uint8Array(0)
  ): Promise<Reply> {
    const h = Buffer.from(
      JSON.stringify({ ...header, bytes: payload.byteLength })
    );
    const len = Buffer.alloc(4);
    len.writeUInt32LE(h.length);
    const reply = this.next();
    this.proc.stdin!.write(Buffer.concat([len, h, payload]));
    const r = await reply;
    if (r.header.error) throw new Error(`mps worker: ${r.header.error}`);
    return r;
  }
}

let worker: Promise<Worker> | undefined;
const getWorker = (): Promise<Worker> =>
  (worker ??= (async () => {
    const [cmd, ...args] = command();
    const w = new Worker(
      spawn(cmd, args, { stdio: ['pipe', 'pipe', 'inherit'] })
    );
    await w.ready;
    return w;
  })());

/** Load an ONNX model (path or bytes) into the worker. Rejects if the worker
 *  can't start or the model doesn't convert; callers fall back to onnxruntime. */
export async function createMpsSession(
  model: string | Uint8Array
): Promise<Session> {
  const w = await getWorker();
  const bytes = typeof model === 'string' ? await readFile(model) : model;
  const { header } = await w.request({ op: 'load' }, bytes);
  const id = header.id as number;
  return {
    inputNames: header.inputs,
    outputNames: header.outputs,
    async run(feeds) {
      const inputs = Object.entries(feeds).map(([name, t]) => ({
        name,
        dims: [...t.dims],
        data: t.data as Float32Array,
      }));
      const body = Buffer.concat(
        inputs.map((t) =>
          Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength)
        )
      );
      const r = await w.request(
        {
          op: 'run',
          id,
          inputs: inputs.map(({ name, dims }) => ({ name, dims })),
        },
        body
      );
      // Copy into an aligned buffer: the payload may start at any byte offset.
      const all = new Float32Array(new Uint8Array(r.payload).buffer);
      const out: Record<string, Tensor> = {};
      let off = 0;
      for (const o of r.header.outputs as { name: string; dims: number[] }[]) {
        const n = o.dims.reduce((a, b) => a * b, 1);
        out[o.name] = { data: all.subarray(off, off + n), dims: o.dims };
        off += n;
      }
      return out;
    },
  };
}
