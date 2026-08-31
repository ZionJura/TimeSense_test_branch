#!/usr/bin/env bash
set -e
python -m timesense.train --config configs/stage2.yaml --model results/stage1 --data data/train/chron_gen_train.jsonl --output results/stage2 --export-inference "$@"
