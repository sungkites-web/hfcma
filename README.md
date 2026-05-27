# HFCMA

Implementation of **Hierarchical Fine-grained Cross-modal Alignment via Dynamic Granularity Discovery and Optimal Transport**.

HFCMA is a fine-grained image-text retrieval framework built on a CLIP-style dual encoder. It progressively organizes low-level patch/token representations into adaptive visual regions and textual phrases, then performs structured region-phrase matching with entropy-regularized optimal transport.

## Abstract

Fine-grained image-text retrieval requires a model not only to assess global semantic consistency between an image and a text, but also to model visual evidence for local textual semantics, including objects, attributes, quantities, actions, and relations, so as to distinguish semantically similar candidates. Existing methods commonly use fixed image patches, detected regions, or text tokens as the basic matching units and then aggregate local interactions through max pooling, cross attention, or similarity accumulation. Such strategies remain limited in semantic unit construction and global matching consistency. This paper proposes a hierarchical fine-grained cross-modal alignment framework, termed HFCMA. The framework progressively transforms low-level patch and token representations into adaptive visual regions and textual phrases, and then performs structured region-phrase matching within a global retrieval scoring process. Specifically, HFCMA first adopts dynamic granularity discovery (DGD) to aggregate raw patches and tokens into semantically compact visual regions and textual phrases. It then formulates region-phrase matching as an entropy-regularized optimal transport (OT) problem, thereby estimating a structured alignment matrix under global marginal constraints. To reduce background noise and local semantic drift, HFCMA further introduces background debiasing, textual importance weighting, and prototype consistency regularization. Experiments on Flickr30K and MS-COCO show that HFCMA effectively combines dynamic semantic unit construction with OT-based structured matching under a dual-encoder retrieval framework. The resulting fine-grained score provides structured local alignment information for candidate ranking and improves the discrimination of subtle semantic differences.

## 摘要

细粒度图文检索要求模型不仅评估图像与文本的全局语义一致性，还需要建模局部文本语义对应的视觉证据，包括目标、属性、数量、动作和关系等，以区分语义相近的候选样本。现有方法通常将固定图像块、检测区域或文本词元作为基本匹配单元，再通过最大池化、交叉注意力或相似度累积聚合局部交互，这类策略在语义单元构建和全局匹配一致性方面仍存在不足。本文提出层次化细粒度跨模态对齐框架 HFCMA。该框架将低层 patch/token 表示逐步转换为自适应视觉区域和文本短语，并在全局检索评分过程中执行结构化区域—短语匹配。具体而言，HFCMA 首先采用动态粒度发现将原始 patch 和 token 聚合为语义紧凑的视觉区域与文本短语，随后将区域—短语匹配形式化为熵正则最优传输问题，在全局边缘约束下估计结构化对齐矩阵。为减少背景噪声和局部语义漂移，HFCMA 进一步引入背景去偏、文本重要性加权和原型一致性正则化。在 Flickr30K 和 MS-COCO 上的实验表明，HFCMA 能够在双编码器检索框架下有效结合动态语义单元构建与 OT 结构化匹配，所得细粒度分数为候选排序提供结构化局部对齐信息，并提升模型对细微语义差异的区分能力。

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
