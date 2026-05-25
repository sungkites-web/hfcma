import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HFCMAOutput:
    global_score: torch.Tensor
    fine_score: torch.Tensor
    final_score: torch.Tensor
    l_gran: torch.Tensor
    l_debias: torch.Tensor
    l_proto: torch.Tensor
    l_cons: torch.Tensor
    region_feats: torch.Tensor
    phrase_feats: torch.Tensor
    transport_plan: torch.Tensor


class SlotGrouping(nn.Module):
    """Dynamic granularity discovery with slot-based soft assignment."""

    def __init__(
        self,
        dim: int,
        num_slots: int,
        assign_temp: float = 0.5,
        topk_assign: Optional[int] = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.assign_temp = assign_temp
        self.topk_assign = topk_assign

        self.slots = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_s = nn.Linear(dim, dim)
        self.gate_fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, _, dim = x.shape
        x_proj = self.proj_x(x)
        s_proj = self.proj_s(self.slots)
        logits = torch.einsum("bnd,sd->bns", x_proj, s_proj) / (
            math.sqrt(dim) * self.assign_temp
        )
        assign = F.softmax(logits, dim=2)

        if self.topk_assign is not None and self.topk_assign < self.num_slots:
            _, topk_idx = torch.topk(assign, k=self.topk_assign, dim=2)
            mask = torch.zeros_like(assign).scatter_(2, topk_idx, 1.0)
            assign = assign * mask
            assign = assign / (assign.sum(dim=2, keepdim=True) + 1e-8)

        assign_sum = assign.sum(dim=1, keepdim=True) + 1e-6
        grouped = torch.einsum("bns,bnd->bsd", assign, x)
        grouped = grouped / assign_sum.squeeze(1).unsqueeze(-1)

        gate = torch.sigmoid(self.gate_fc(grouped)).squeeze(-1)
        grouped = grouped * gate.unsqueeze(-1)
        return {"grouped": grouped, "gate": gate, "assign": assign}


def sinkhorn_transport(
    cost: torch.Tensor,
    epsilon: float = 0.1,
    iters: int = 10,
) -> torch.Tensor:
    """Entropy-regularized optimal transport solved by log-domain Sinkhorn."""

    bsz, num_v, num_t = cost.shape
    device = cost.device
    log_a = torch.full((bsz, num_v), -math.log(num_v), device=device)
    log_b = torch.full((bsz, num_t), -math.log(num_t), device=device)
    kernel = -cost / epsilon

    log_u = torch.zeros(bsz, num_v, device=device)
    log_v = torch.zeros(bsz, num_t, device=device)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)

    log_plan = log_u.unsqueeze(2) + kernel + log_v.unsqueeze(1)
    return torch.exp(log_plan)


class PrototypeHead(nn.Module):
    """Prototype consistency head with balanced soft targets."""

    def __init__(self, dim: int, num_proto: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_proto, dim) * 0.02)

    @torch.no_grad()
    def _sinkhorn_target(self, scores: torch.Tensor, iters: int = 3) -> torch.Tensor:
        q = torch.exp(scores / 0.05)
        q /= q.sum()
        for _ in range(iters):
            q /= q.sum(dim=0, keepdim=True)
            q /= q.sum(dim=1, keepdim=True)
        return q.clamp(min=1e-8)

    def forward(self, x: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        bsz, num_slots, dim = x.shape
        x_flat = F.normalize(x.reshape(bsz * num_slots, dim), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        scores = x_flat @ proto.t()
        with torch.no_grad():
            target = self._sinkhorn_target(scores.detach())
        log_pred = F.log_softmax(scores / tau, dim=-1)
        return -(target * log_pred).sum(dim=-1).mean()


class HFCMA(nn.Module):
    """Hierarchical fine-grained cross-modal alignment head."""

    def __init__(
        self,
        dim: int = 768,
        num_visual_slots: int = 12,
        num_text_slots: int = 12,
        epsilon: float = 0.07,
        sinkhorn_iters: int = 15,
        beta: float = 0.1,
        debias_margin: float = 0.05,
        num_proto: int = 48,
        tau_proto: float = 0.07,
        assign_temp: float = 0.5,
        topk_assign: Optional[int] = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_visual_slots = num_visual_slots
        self.num_text_slots = num_text_slots
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.beta = beta
        self.debias_margin = debias_margin
        self.tau_proto = tau_proto

        self.visual_group = SlotGrouping(
            dim, num_visual_slots, assign_temp=assign_temp, topk_assign=topk_assign
        )
        self.text_group = SlotGrouping(
            dim, num_text_slots, assign_temp=assign_temp, topk_assign=topk_assign
        )
        self.proto_head = PrototypeHead(dim, num_proto)

    def fine_grained_score(
        self,
        regions: torch.Tensor,
        phrases: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regions = F.normalize(regions, dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        sim = torch.einsum("bkd,bld->bkl", phrases, regions)
        cost = 1.0 - sim
        plan = sinkhorn_transport(cost, epsilon=self.epsilon, iters=self.sinkhorn_iters)
        fine_score = -(plan * cost).sum(dim=(1, 2))
        return fine_score, plan, cost

    def allpairs_fine_score(
        self,
        regions: torch.Tensor,
        phrases: torch.Tensor,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        num_images, num_regions, _ = regions.shape
        num_texts, num_phrases, _ = phrases.shape
        r = F.normalize(regions, dim=-1)
        p = F.normalize(phrases, dim=-1)
        scores = torch.zeros(num_images, num_texts, device=regions.device)

        for start in range(0, num_images, chunk_size):
            end = min(start + chunk_size, num_images)
            r_chunk = r[start:end]
            chunk = r_chunk.size(0)
            sim = torch.einsum("jkd,ild->ijkl", p, r_chunk)
            cost = 1.0 - sim
            plan = sinkhorn_transport(
                cost.reshape(chunk * num_texts, num_phrases, num_regions),
                self.epsilon,
                self.sinkhorn_iters,
            ).reshape(chunk, num_texts, num_phrases, num_regions)
            scores[start:end] = -(plan * cost).sum(dim=(2, 3))
        return scores

    def granularity_loss(
        self,
        x: torch.Tensor,
        grouped: torch.Tensor,
        assign: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        dist = ((x.unsqueeze(2) - grouped.unsqueeze(1)) ** 2).sum(dim=-1)
        compact = (assign * dist).mean()
        sparse = gate.mean()
        mean_assign = assign.mean(dim=1)
        diversity = -torch.mean(
            torch.sum(mean_assign * torch.log(mean_assign + 1e-8), dim=-1)
        )
        return compact + 0.1 * sparse + 0.1 * diversity

    def debias_loss(
        self,
        regions: torch.Tensor,
        gate_v: torch.Tensor,
        text_global: torch.Tensor,
    ) -> torch.Tensor:
        regions_n = F.normalize(regions, dim=-1)
        text_global_n = F.normalize(text_global, dim=-1)
        gate_sg = gate_v.detach()
        low_mask = (gate_sg < gate_sg.mean(dim=1, keepdim=True)).float()
        if low_mask.sum() < 1:
            return torch.tensor(0.0, device=regions.device)
        bg_region = (regions_n * low_mask.unsqueeze(-1)).sum(dim=1)
        bg_region = F.normalize(bg_region / (low_mask.sum(dim=1, keepdim=True) + 1e-6), dim=-1)
        bg_score = (bg_region * text_global_n).sum(dim=-1)
        return F.relu(bg_score - self.debias_margin).mean()

    def consistency_loss(
        self,
        regions: torch.Tensor,
        phrases: torch.Tensor,
        global_pair: torch.Tensor,
    ) -> torch.Tensor:
        local_pair = F.normalize((regions.mean(dim=1) + phrases.mean(dim=1)) / 2.0, dim=-1)
        global_pair = F.normalize(global_pair, dim=-1)
        return ((local_pair - global_pair) ** 2).sum(dim=-1).mean()

    def forward(
        self,
        patch_feats: torch.Tensor,
        token_feats: torch.Tensor,
        image_global: torch.Tensor,
        text_global: torch.Tensor,
    ) -> HFCMAOutput:
        v_out = self.visual_group(patch_feats)
        t_out = self.text_group(token_feats)
        regions = v_out["grouped"]
        phrases = t_out["grouped"]

        image_global_n = F.normalize(image_global, dim=-1)
        text_global_n = F.normalize(text_global, dim=-1)
        global_score = (image_global_n * text_global_n).sum(dim=-1)
        fine_score, plan, _ = self.fine_grained_score(regions, phrases)
        final_score = global_score + self.beta * fine_score

        l_gran = self.granularity_loss(
            patch_feats, regions, v_out["assign"], v_out["gate"]
        ) + self.granularity_loss(
            token_feats, phrases, t_out["assign"], t_out["gate"]
        )
        l_debias = self.debias_loss(regions, v_out["gate"], text_global)
        l_proto = self.proto_head(regions, self.tau_proto) + self.proto_head(
            phrases, self.tau_proto
        )
        l_cons = self.consistency_loss(regions, phrases, (image_global + text_global) / 2.0)

        return HFCMAOutput(
            global_score=global_score,
            fine_score=fine_score,
            final_score=final_score,
            l_gran=l_gran,
            l_debias=l_debias,
            l_proto=l_proto,
            l_cons=l_cons,
            region_feats=regions,
            phrase_feats=phrases,
            transport_plan=plan,
        )

