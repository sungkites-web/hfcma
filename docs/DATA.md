# Data Preparation

This repository does not redistribute Flickr30K, MS-COCO, Karpathy split files, image files, extracted feature caches, or checkpoints. Download the datasets from their official sources and convert annotations into the JSON format below.

## JSON Schema

Each item corresponds to one image and one or more captions:

```json
{
  "image": "relative/path/to/image.jpg",
  "captions": ["caption 1", "caption 2"],
  "image_id": 123,
  "split": "train"
}
```

The loader also accepts `image_path`, `file_name`, or `filename` as aliases of `image`.

Required fields:

- `image`: relative path under the dataset image root. `image_path`, `file_name`, and `filename` are also supported.
- `captions`: list of captions. A single `caption` string can be converted into a one-element list.
- `image_id`: stable image identifier used by retrieval metrics.
- `split`: one of `train`, `val`, `test`, or the split names used in your Karpathy-style conversion.

## Recommended Layout

```text
data/
|-- flickr30k/
|   |-- images/
|   `-- cache/flickr30k_all.json
`-- coco/
    |-- images/
    `-- cache/coco_all.json
```

The example configs expect the following files by default:

- `data/flickr30k/cache/flickr30k_all.json`
- `data/coco/cache/coco_all.json`

Use `--data_root` or the `DATA_ROOT` environment variable to point the scripts to another location.

## Evaluation Protocols

- Flickr30K uses the 1K test protocol.
- MS-COCO is evaluated under the 5K full test setting and the 1K five-fold setting.
- Each image is assumed to have five captions in the standard Flickr30K/MS-COCO retrieval protocol.

## Do Not Commit

Keep the following local-only artifacts outside Git:

- dataset images and downloaded captions if redistribution is restricted
- `.pt`, `.pth`, `.ckpt`, `.pkl`, `.npy`, `.npz` checkpoints or caches
- logs, temporary job scripts, debug outputs, and machine-specific paths
