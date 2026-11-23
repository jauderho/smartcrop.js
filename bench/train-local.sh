#!/bin/bash
# Run the Train workflow's pipeline on this machine with the same settings
# (4 inference processes x 1 ONNX thread, 4 dump processes, workflow default
# config, seed 1), timing each stage, for laptop-vs-runner comparisons.
# Datasets must already be in bench/datasets. Plug-in outputs go to a fresh
# cache dir so nothing cached is reused.
#
# Usage: bench/train-local.sh <outdir>
set -eu
cd "$(dirname "$0")/.."
out=${1:?usage: bench/train-local.sh <outdir>}
mkdir -p "$out"
export SMARTCROP_CACHE_DIR="$out/cache"
log="$out/timings.txt"
: > "$log"
stage() { # name, command...
  local name=$1
  shift
  local t0=$SECONDS
  "$@"
  echo "$name $((SECONDS - t0))s" | tee -a "$log"
}
infer() {
  for job in "gaicd train" "gaicd val" "fcdb train"; do
    set -- $job
    for k in 0 1 2 3; do
      SMARTCROP_ORT_THREADS=1 bun bench/precompute.ts "$1" --split "$2" --shard "$k/4" > /dev/null &
    done
    wait
  done
}
dumps() {
  for job in "gaicd train" "gaicd val" "fcdb train"; do
    set -- $job
    bun bench/learned/dump.ts --dataset "$1" --split "$2" --with faces,saliency,cnn --cnnk 3 --procs 4 --force
  done
}
train() {
  bun bench/learned/train.ts --cnn 3 --hidden 32,32 --l2 3e-3 --lr 2e-3 --epochs 30 \
    --fcdb 0.5 --ppi 64 --lambda 0 --ema 0 --seed 1 --write --out "$out/model-full.json" > "$out/train.log" 2>&1
  grep '^hidden=' "$out/train.log"
}
gate() {
  bun bench/learned/gate.ts "$out/model-full.json" --cnn 3 --fcdb 0.5 --margin 0 \
    --report "$out/gate.json" --markdown "$out/gate.md" || true
}
stage pca bun bench/learned/projection-from-shipped.ts
stage inference infer
stage dumps dumps
stage train train
stage gate gate
echo "total $SECONDS s" | tee -a "$log"
