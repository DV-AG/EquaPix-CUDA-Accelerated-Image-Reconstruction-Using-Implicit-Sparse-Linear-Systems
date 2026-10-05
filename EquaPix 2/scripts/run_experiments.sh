#!/usr/bin/env bash
# Run inside a CUDA lab. Never synthesizes or substitutes benchmark results.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
exec > >(tee results/lab_session_log.txt) 2>&1
printf 'EquaPix CUDA lab session started: '
date -u
command -v nvcc
nvcc --version
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
make
make test
make gpu-test
python3 scripts/generate_dataset.py --count 200 --width 256 --height 256 --seed 42
for level in 5 10 20 30; do
  ./equapix --input data/generated --output "results/random_${level}" --mask random --missing-percent "$level" --iterations 500 --seed 42 --cpu-benchmark-count 10
done
./equapix --input data/sample --output results/block_64 --mask block --block-size 64 --iterations 1500 --cpu-benchmark-count 8
if [[ "${1:-}" == "--large" ]]; then
  python3 scripts/generate_dataset.py --output data/large --count 20 --width 1024 --height 1024 --seed 42
  ./equapix --input data/large --output results/large --mask random --missing-percent 20 --iterations 500 --cpu-benchmark-count 2
fi
python3 scripts/check_execution_proof.py
printf 'EquaPix CUDA lab session completed: '
date -u
