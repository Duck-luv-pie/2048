#!/bin/bash
# Train the 12-tuple (m12cov) network from scratch on one GPU, exactly as for the released weights:
# optimistic init (320k), then TD(0) with alpha 0.1 -> 0.01 -> 0.001 (100 + 30 + 20 minutes on an A100).
# Usage: scripts/train_m12cov.sh [output.bin]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-weights/m12cov_1stage.bin}
mkdir -p weights logs
make bin/train bin/play gpu

./bin/train --net m12cov --stages 1 --init 320000 --games 0 --threads 1 --out weights/m12cov_init.bin

S0="--starts data/init_boards.bin --fresh-prob 1 --restart-prob 0 --push-prob 0"
./bin/train_gpu --load weights/m12cov_init.bin --out weights/m12cov_a.bin $S0 --alpha 0.1   --run-secs 6000 | tee logs/a.log
./bin/train_gpu --load weights/m12cov_a.bin    --out weights/m12cov_b.bin $S0 --alpha 0.01  --run-secs 1800 | tee logs/b.log
./bin/train_gpu --load weights/m12cov_b.bin    --out "$OUT"               $S0 --alpha 0.001 --run-secs 1200 | tee logs/c.log
rm -f weights/m12cov_init.bin weights/m12cov_a.bin weights/m12cov_b.bin

# Frozen evaluation: 20,000 games, no lookahead
./bin/play --weights "$OUT" --games 20000 --depth 0 --threads "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
