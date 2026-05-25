# Reproducibility Notes

Report the following settings when reproducing HFCMA experiments.

## Model

- CLIP backbone and pretrained weights, for example `ViT-L-14/openai`.
- Whether the CLIP backbone is frozen.
- Visual slot number, text slot number, and semantic prototype number.
- OT entropy coefficient and Sinkhorn iteration count.
- Fine-grained score weight and candidate reranking size.

## Data

- Dataset name and protocol, for example Flickr30K-1K, MS-COCO 1K five-fold, or MS-COCO 5K.
- Annotation conversion version and number of images/captions in each split.
- Whether the evaluation uses the full test set or fold averaging.

## Training

- Random seed.
- Batch size.
- Number of epochs and steps per epoch.
- Learning rate, weight decay, and gradient clipping.
- Loss weights for dynamic granularity regularization, background debiasing, prototype consistency, and global-local consistency.

## Evaluation

- Image-to-text and text-to-image Recall@1, Recall@5, Recall@10.
- RSUM, defined as the sum of the six recall metrics.
- Candidate pool size used for HFCMA reranking.
- Whether fusion or calibration is used. If used, report the fusion parameters separately.

## Repository Contents

The GitHub repository should contain source code, config files, data format documentation, training/evaluation scripts, README, and citation information.

The GitHub repository should not contain dataset images, restricted captions, checkpoints unless explicitly licensed, feature caches, logs, temporary experiment queues, SSH commands, passwords, or machine-specific paths.
