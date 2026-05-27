import torch

from hfcma.evaluation import retrieval_metrics


def test_retrieval_metrics_perfect_ranking():
    sim = torch.eye(3)
    image_ids = [10, 11, 12]
    caption_image_ids = [10, 11, 12]
    metrics = retrieval_metrics(sim, image_ids, caption_image_ids)
    assert metrics["I2T_R@1"] == 100.0
    assert metrics["T2I_R@1"] == 100.0
    assert metrics["RSUM"] == 600.0
