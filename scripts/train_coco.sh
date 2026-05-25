#!/usr/bin/env bash
set -euo pipefail

python -m hfcma.train \
  --config configs/coco_vitl14.yaml \
  --output_dir outputs/coco_vitl14

