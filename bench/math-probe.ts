// Hash of the libm-backed Math functions the trainer uses (exp, log, log1p,
// log2, cos, sqrt) over a fixed input grid. Different hashes on two machines
// mean their training results can differ in the last bits even for identical
// feature dumps. Usage: bun bench/math-probe.ts
const fns: Record<string, (x: number) => number> = {
  exp: Math.exp,
  log: Math.log,
  log1p: Math.log1p,
  log2: Math.log2,
  cos: Math.cos,
  sqrt: Math.sqrt,
};
const buf = new Float64Array(1);
for (const [name, f] of Object.entries(fns)) {
  const h = new Bun.CryptoHasher('sha256');
  let n = 0;
  for (let i = 1; i <= 400000; i++) {
    const x = (i * 0.000731) % 40;
    buf[0] = f(x + 1e-9);
    h.update(new Uint8Array(buf.buffer));
    buf[0] = f(-x * 0.5 + 0.37);
    h.update(new Uint8Array(buf.buffer));
    n += 2;
  }
  console.log(name, h.digest('hex').slice(0, 16), n);
}
console.log(process.platform, process.arch, Bun.version);
