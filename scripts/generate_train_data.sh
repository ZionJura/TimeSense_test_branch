#!/usr/bin/env bash
set -e
python -m chron_gen.generate train --format "$@"
