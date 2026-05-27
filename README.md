# HFCMA

Implementation of **Hierarchical Fine-grained Cross-modal Alignment via Dynamic Granularity Discovery and Optimal Transport**.

HFCMA is a fine-grained image-text retrieval framework built on a CLIP-style dual encoder. It adaptively groups image patches and text tokens into visual regions and textual phrases, then performs structured region-phrase matching with entropy-regularized optimal transport.

## Abstract

Fine-grained image-text retrieval requires a model to determine not only whether an image and a text are globally consistent, but also whether local semantic elements such as objects, attributes, quantities, and relations are correctly aligned. Existing methods often use fixed image patches, detected regions, or text tokens as basic matching units, and then conduct local interaction through max pooling, cross-attention, or similarity aggregation. Such strategies still have limitations in semantic unit construction and globally consistent matching. This paper proposes a hierarchical fine-grained cross-modal alignment framework, termed HFCMA. The proposed method first introduces dynamic granularity discovery (DGD) to adaptively aggregate raw patch/token sequences into visual regions and textual phrases. It then formulates region-phrase matching as an entropy-regularized optimal transport (OT) problem, which produces a structured alignment matrix under global marginal constraints. To reduce background noise and local semantic drift, HFCMA further incorporates background debiasing, text importance weighting, and prototype consistency constraints. Experiments on Flickr30K and MS-COCO image-text retrieval benchmarks show that HFCMA effectively exploits the complementarity between dynamic semantic unit construction and OT-based structured matching within a dual-encoder retrieval framework, improving fine-grained discrimination and providing structurally constrained local alignment information for candidate reranking.

## 摘要

细粒度图文检索要求模型不仅判断图像与文本在整体语义上是否一致，还要识别目标、属性、数量和关系等局部语义之间的对应关系。现有方法多以固定图像块、检测区域或文本词元作为基本匹配单元，再通过最大池化、交叉注意力或相似度累积完成局部交互，这类策略在语义单元划分和全局匹配一致性上仍存在不足。本文提出一种层次化细粒度跨模态对齐框架 HFCMA。该方法首先通过动态粒度发现将原始 patch/token 自适应聚合为视觉区域和文本短语，再将区域—短语匹配形式化为熵正则最优传输问题，从而在全局边缘约束下求解结构化对齐矩阵。为减少背景噪声和局部语义漂移，模型进一步引入背景去偏、文本重要性加权和原型一致性约束。在 Flickr30K 和 MS-COCO 图文检索基准上的实验表明，HFCMA 能够在双编码器检索框架下有效利用动态语义单元和 OT 结构化匹配之间的互补性，增强模型对局部语义差异的判别能力，并在候选重排序阶段提供更具结构约束的细粒度对齐信息。

## Repository Structure

```text
hfcma-public/
|-- configs/                 # dataset and hyperparameter examples
|-- docs/                    # data format and reproducibility notes
|-- scripts/                 # training and evaluation commands
|-- src/hfcma/
|   |-- datasets/            # JSON retrieval dataset loader
|   |-- evaluation/          # retrieval metrics
|   |-- models/              # HFCMA modules and OpenCLIP wrapper
|   |-- losses.py            # retrieval and auxiliary training losses
|   |-- rerank.py            # top-K fine-grained reranking utilities
|   |-- train.py             # training and validation entry point
|   `-- evaluate.py          # evaluation-only entry point
|-- tests/                   # lightweight correctness checks
|-- tools/                   # annotation conversion utilities
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Installation

```bash
conda create -n hfcma python=3.10 -y
conda activate hfcma
pip install -r requirements.txt
pip install -e .
```

The default configuration uses OpenAI CLIP ViT-L/14 through OpenCLIP. A CUDA GPU is recommended for training and reranking.

## Data Format

Prepare each dataset as a JSON list. Each entry corresponds to one image and its captions:

```json
{
  "image": "train2014/COCO_train2014_000000000009.jpg",
  "captions": [
    "A group of people standing near a bus.",
    "People are waiting beside a bus."
  ],
  "image_id": 9,
  "split": "train"
}
```

Recommended local layout:

```text
data/
|-- flickr30k/
|   |-- images/
|   `-- cache/flickr30k_all.json
`-- coco/
    |-- images/
    `-- cache/coco_all.json
```

See `docs/DATA.md` for supported fields and evaluation protocol notes.

If your annotations follow the Karpathy-style JSON format, convert them with:

```bash
python tools/convert_retrieval_json.py \
  --input path/to/dataset_karpathy.json \
  --output data/flickr30k/cache/flickr30k_all.json
```

## Training

Flickr30K:

```bash
bash scripts/train_flickr30k.sh
```

MS-COCO:

```bash
bash scripts/train_coco.sh
```

The scripts train the HFCMA head with the CLIP backbone frozen.

## Evaluation

Flickr30K:

```bash
bash scripts/eval_flickr30k.sh
```

MS-COCO:

```bash
bash scripts/eval_coco.sh
```

Evaluation computes global CLIP similarities and reranks top-ranked candidates with the HFCMA fine-grained OT score.

## Sanity Checks

Run the lightweight tests before launching a full experiment:

```bash
pytest tests
```

These tests verify retrieval-metric computation and the marginal constraints of the Sinkhorn transport plan.

## Main Configuration

| Dataset | Visual/text slots | Prototypes | Sinkhorn steps | OT epsilon | Rerank top-K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flickr30K | 8 | 32 | 10 | 0.10 | 128 |
| MS-COCO | 12 | 48 | 15 | 0.07 | 128 |

## Expected Results

The manuscript reports the following main RSUM values. Minor variation is expected across hardware, CUDA version, annotation conversion, and random seed.

| Dataset/protocol | Full HFCMA | HFCMA-Fusion-Calib |
| --- | ---: | ---: |
| Flickr30K-1K | 535.78 | 542.92 |
| MS-COCO 1K | 522.95 | 532.62 |
| MS-COCO 5K | 439.59 | 442.27 |

## Reproducibility

See `docs/REPRODUCIBILITY.md` for settings that should be reported when reproducing the experiments, including backbone, slot number, OT settings, candidate pool size, random seed, and dataset split.

## Citation

```bibtex
@article{hfcma2026,
  title   = {Hierarchical Fine-grained Cross-modal Alignment via Dynamic Granularity Discovery and Optimal Transport},
  author  = {Pengfeng Song},
  journal = {To appear},
  year    = {2026}
}
```

## License

The license will be updated before the final release.
