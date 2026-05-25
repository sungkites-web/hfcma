#!/usr/bin/env bash
set -euo pipefail

python -m hfcma.train \
  --config configs/flickr30k_vitl14.yaml \
  --output_dir outputs/flickr30k_vitl14

