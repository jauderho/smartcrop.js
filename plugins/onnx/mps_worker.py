# PyTorch-MPS backend for plugins/onnx on Apple silicon (see runtime.ts).
# Converts the pinned ONNX models with onnx2torch and runs them in float32 on
# the GPU (fp16 is not used: it corrupts IS-Net's output).
#
# Frames on stdin/stdout: uint32 LE header length, JSON header, payload bytes.
#   ready      -> {"ready": true, "torch": version}        (sent at start)
#   {"op": "load", "bytes": n} + ONNX model
#              -> {"id": k, "inputs": [...], "outputs": [...]} | {"error": msg}
#   {"op": "run", "id": k, "inputs": [{"name", "dims"}], "bytes": n} + float32 inputs
#              -> {"outputs": [{"name", "dims"}], "bytes": n} + float32 outputs | {"error": msg}
import json
import struct
import sys
import warnings

# onnx2torch's Slice uses list indexing that torch warns about on every run.
warnings.filterwarnings('ignore', category=UserWarning, module='onnx2torch')

out = sys.stdout.buffer
sys.stdout = sys.stderr  # stray prints must not corrupt the frame stream
inp = sys.stdin.buffer


def send(header, payload=b''):
    h = json.dumps(header).encode()
    out.write(struct.pack('<I', len(h)) + h + payload)
    out.flush()


def read_exact(n):
    b = inp.read(n)
    if len(b) < n:
        sys.exit(0)  # parent closed the pipe
    return b


try:
    import numpy as np
    import onnx
    import torch
    from onnx2torch import convert

    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available')
except Exception as e:  # noqa: BLE001 - report any import/setup failure
    send({'error': f'{type(e).__name__}: {e}'})
    sys.exit(1)

send({'ready': True, 'torch': torch.__version__})
models = []


def patch(proto):
    """Work around onnx2torch bugs in place, keeping the model's maths."""
    for i, n in enumerate(proto.graph.node):
        # A Split into one output becomes torch.split's 1-tuple, which
        # onnx2torch passes on instead of the tensor (breaks UNISAL's
        # Squeeze). Splitting into one part is a copy.
        if n.op_type == 'Split' and len(n.output) == 1:
            proto.graph.node[i].CopyFrom(
                onnx.helper.make_node('Identity', [n.input[0]], [n.output[0]], name=n.name))
    return proto

while True:
    (hl,) = struct.unpack('<I', read_exact(4))
    req = json.loads(read_exact(hl))
    payload = read_exact(req.get('bytes', 0))
    try:
        if req['op'] == 'load':
            proto = onnx.load_from_string(payload)
            init = {i.name for i in proto.graph.initializer}
            names_in = [i.name for i in proto.graph.input if i.name not in init]
            names_out = [o.name for o in proto.graph.output]
            net = convert(patch(proto)).eval().to('mps')
            models.append((net, names_in, names_out))
            send({'id': len(models) - 1, 'inputs': names_in, 'outputs': names_out})
        elif req['op'] == 'run':
            net, names_in, names_out = models[req['id']]
            flat = np.frombuffer(payload, dtype=np.float32)
            feeds, off = {}, 0
            for t in req['inputs']:
                n = int(np.prod(t['dims']))
                feeds[t['name']] = torch.from_numpy(flat[off:off + n].reshape(t['dims']).copy())
                off += n
            with torch.inference_mode():
                res = net(*[feeds[n].to('mps') for n in names_in])
            res = res if isinstance(res, (tuple, list)) else (res,)
            arrs = [r.float().cpu().numpy() for r in res]
            body = b''.join(np.ascontiguousarray(a).tobytes() for a in arrs)
            send({'outputs': [{'name': n, 'dims': list(a.shape)} for n, a in zip(names_out, arrs)], 'bytes': len(body)}, body)
        else:
            send({'error': f'unknown op {req["op"]}'})
    except Exception as e:  # noqa: BLE001 - errors go back to the caller
        send({'error': f'{type(e).__name__}: {e}'})
