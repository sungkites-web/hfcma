from typing import Literal

import torch

from hfcma.models import HFCMA


@torch.no_grad()
def rerank_topk(
    model: HFCMA,
    global_sim: torch.Tensor,
    image_regions: torch.Tensor,
    text_phrases: torch.Tensor,
    topk: int = 128,
    direction: Literal["both", "i2t", "t2i"] = "both",
    chunk_size: int = 128,
) -> torch.Tensor:
    """Rerank top-K candidates with the HFCMA fine-grained OT score.

    ``global_sim`` is an image-by-text similarity matrix produced by the frozen
    CLIP encoder. Only the top-K candidates in each retrieval direction are
    refined, which preserves the scalable dual-encoder retrieval pipeline.
    """

    refined = global_sim.clone()
    topk_i2t = min(topk, global_sim.size(1))
    topk_t2i = min(topk, global_sim.size(0))

    if direction in ("both", "i2t"):
        for i in range(global_sim.size(0)):
            idx = global_sim[i].topk(topk_i2t).indices
            fine = model.allpairs_fine_score(
                image_regions[i : i + 1], text_phrases[idx], chunk_size=1
            ).squeeze(0)
            refined[i, idx] = global_sim[i, idx] + model.beta * fine

    if direction in ("both", "t2i"):
        for j in range(global_sim.size(1)):
            idx = global_sim[:, j].topk(topk_t2i).indices
            fine = model.allpairs_fine_score(
                image_regions[idx], text_phrases[j : j + 1], chunk_size=chunk_size
            ).squeeze(1)
            refined[idx, j] = global_sim[idx, j] + model.beta * fine

    return refined
