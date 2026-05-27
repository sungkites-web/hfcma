import torch

from hfcma.models import sinkhorn_transport


def test_sinkhorn_plan_has_balanced_marginals():
    cost = torch.rand(2, 4, 5)
    plan = sinkhorn_transport(cost, epsilon=0.1, iters=50)
    row_sum = plan.sum(dim=2)
    col_sum = plan.sum(dim=1)
    assert torch.allclose(row_sum, torch.full_like(row_sum, 1 / 4), atol=1e-3)
    assert torch.allclose(col_sum, torch.full_like(col_sum, 1 / 5), atol=1e-3)
