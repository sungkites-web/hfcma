#!/usr/bin/env bash
set -euo pipefail

python -m hfcma.evaluate \
  --config configs/coco_vitl14.yaml \
  --checkpoint outputs/coco_vitl14/best.pt \
  --output_dir outputs/coco_vitl14 \
  --evaluate_only

