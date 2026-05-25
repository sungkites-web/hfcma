#!/usr/bin/env bash
set -euo pipefail

python -m hfcma.evaluate \
  --config configs/flickr30k_vitl14.yaml \
  --checkpoint outputs/flickr30k_vitl14/best.pt \
  --output_dir outputs/flickr30k_vitl14 \
  --evaluate_only

