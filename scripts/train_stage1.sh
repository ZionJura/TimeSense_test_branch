#!/usr/bin/env bash
set -e
python -m timesense.train --config configs/stage1.yaml --model models/ChatTS-14B-train --data data/train/chron_gen_train.jsonl --output results/stage1 "$@"
