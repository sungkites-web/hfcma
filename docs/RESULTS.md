# Expected Manuscript Results

The values below help users verify that their local setup is consistent with the manuscript protocol. Minor variation is expected across hardware, CUDA versions, annotation conversions, and random seeds.

## Main Retrieval Results

| Dataset/protocol | Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | RSUM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flickr30K-1K | Full HFCMA | 86.60 | 98.30 | 99.40 | 67.62 | 89.28 | 94.58 | 535.78 |
| Flickr30K-1K | HFCMA-Fusion-Calib | 87.00 | 98.00 | 99.40 | 72.20 | 91.14 | 95.18 | 542.92 |
| MS-COCO 1K | Full HFCMA | 81.38 | 96.58 | 98.50 | 62.56 | 88.04 | 95.89 | 522.95 |
| MS-COCO 1K | HFCMA-Fusion-Calib | 81.72 | 97.50 | 98.82 | 66.18 | 91.52 | 96.88 | 532.62 |
| MS-COCO 5K | Full HFCMA | 64.78 | 85.76 | 91.98 | 46.93 | 70.72 | 79.42 | 439.59 |
| MS-COCO 5K | HFCMA-Fusion-Calib | 64.70 | 85.52 | 90.94 | 47.86 | 72.37 | 80.88 | 442.27 |

`Full HFCMA` corresponds to the core model implemented in this repository. `HFCMA-Fusion-Calib` is the manuscript's calibrated inference setting and is listed only as a reference target.
