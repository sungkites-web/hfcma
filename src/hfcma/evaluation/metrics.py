from typing import Dict, Iterable

import torch


def recall_at_k(sim: torch.Tensor, positives: list[list[int]], ks: Iterable[int]) -> Dict[int, float]:
    out = {}
    for k in ks:
        hits = 0
        effective_k = min(k, sim.size(1))
        topk = sim.topk(effective_k, dim=1).indices.cpu().tolist()
        for row_idx, retrieved in enumerate(topk):
            if set(retrieved).intersection(positives[row_idx]):
                hits += 1
        out[k] = 100.0 * hits / max(1, len(positives))
    return out


def retrieval_metrics(
    sim_i2t: torch.Tensor,
    image_ids: list,
    caption_image_ids: list,
    ks: tuple[int, int, int] = (1, 5, 10),
) -> Dict[str, float]:
    image_to_text_pos = []
    for image_id in image_ids:
        image_to_text_pos.append(
            [idx for idx, cap_image_id in enumerate(caption_image_ids) if cap_image_id == image_id]
        )
    text_to_image_pos = []
    image_id_to_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
    for cap_image_id in caption_image_ids:
        text_to_image_pos.append([image_id_to_index[cap_image_id]])

    i2t = recall_at_k(sim_i2t, image_to_text_pos, ks)
    t2i = recall_at_k(sim_i2t.t(), text_to_image_pos, ks)
    metrics = {f"I2T_R@{k}": i2t[k] for k in ks}
    metrics.update({f"T2I_R@{k}": t2i[k] for k in ks})
    metrics["RSUM"] = sum(metrics.values())
    return metrics

