import torch
import torch.nn.functional as F


def bidirectional_contrastive_loss(score_matrix: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric image-text contrastive loss over a paired mini-batch."""

    labels = torch.arange(score_matrix.size(0), device=score_matrix.device)
    logits = score_matrix / tau
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def weighted_hfcma_loss(
    retrieval_loss: torch.Tensor,
    l_gran: torch.Tensor,
    l_debias: torch.Tensor,
    l_proto: torch.Tensor,
    l_cons: torch.Tensor,
    lambda_gran: float,
    lambda_debias: float,
    lambda_proto: float,
    lambda_cons: float,
) -> torch.Tensor:
    """Training objective used by the HFCMA head."""

    return (
        retrieval_loss
        + lambda_gran * l_gran
        + lambda_debias * l_debias
        + lambda_proto * l_proto
        + lambda_cons * l_cons
    )
