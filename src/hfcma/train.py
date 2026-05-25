import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from hfcma.datasets import RetrievalJsonDataset, collate_retrieval_batch
from hfcma.evaluation import retrieval_metrics
from hfcma.models import HFCMA
from hfcma.models.clip_feature_extractor import OpenCLIPFeatureExtractor


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(score_matrix: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    labels = torch.arange(score_matrix.size(0), device=score_matrix.device)
    logits = score_matrix / tau
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def build_hfcma(cfg: Dict[str, Any], device: torch.device) -> HFCMA:
    model_cfg = cfg["model"]
    return HFCMA(
        dim=model_cfg["dim"],
        num_visual_slots=model_cfg["num_visual_slots"],
        num_text_slots=model_cfg["num_text_slots"],
        epsilon=model_cfg["epsilon"],
        sinkhorn_iters=model_cfg["sinkhorn_iters"],
        beta=model_cfg["beta"],
        debias_margin=model_cfg["debias_margin"],
        num_proto=model_cfg["num_proto"],
        tau_proto=model_cfg["tau_proto"],
        assign_temp=model_cfg["assign_temp"],
        topk_assign=model_cfg["topk_assign"],
    ).to(device)


def train_one_epoch(
    model: HFCMA,
    extractor: OpenCLIPFeatureExtractor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    extractor.eval()
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    meters = {"loss": 0.0, "retrieval": 0.0, "steps": 0}

    for step, batch in enumerate(tqdm(loader, desc=f"train epoch {epoch}")):
        if step >= train_cfg["steps_per_epoch"]:
            break
        if len(batch["images"]) < 2:
            continue

        images = extractor.preprocess_images(batch["images"])
        tokens = extractor.tokenize(batch["captions"])
        with torch.no_grad():
            image_global = extractor.encode_image_global(images)
            text_global = extractor.encode_text_global(tokens)
            patch_feats = extractor.encode_image_patches(images)
            token_feats = extractor.encode_text_tokens(tokens)

        v_out = model.visual_group(patch_feats)
        t_out = model.text_group(token_feats)
        regions = v_out["grouped"]
        phrases = t_out["grouped"]

        global_scores = F.normalize(image_global, dim=-1) @ F.normalize(text_global, dim=-1).t()
        fine_scores = model.allpairs_fine_score(
            regions, phrases, chunk_size=train_cfg["allpairs_chunk"]
        )
        score_matrix = global_scores + model.beta * fine_scores
        retrieval = contrastive_loss(score_matrix, tau=train_cfg["tau_contrastive"])

        matched_out = model(patch_feats, token_feats, image_global, text_global)
        loss = (
            retrieval
            + loss_cfg["lambda_gran"] * matched_out.l_gran
            + loss_cfg["lambda_debias"] * matched_out.l_debias
            + loss_cfg["lambda_proto"] * matched_out.l_proto
            + loss_cfg["lambda_cons"] * matched_out.l_cons
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()

        meters["loss"] += float(loss.item())
        meters["retrieval"] += float(retrieval.item())
        meters["steps"] += 1

    steps = max(1, meters.pop("steps"))
    return {key: value / steps for key, value in meters.items()}


@torch.no_grad()
def evaluate(
    model: HFCMA,
    extractor: OpenCLIPFeatureExtractor,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    dataset = RetrievalJsonDataset(
        data_cfg["annotation_file"],
        data_cfg["image_root"],
        split=data_cfg.get("test_split", "test"),
        return_all_captions=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_cfg["image_batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        collate_fn=collate_retrieval_batch,
    )

    image_globals, region_feats, image_ids = [], [], []
    all_captions, caption_image_ids = [], []
    for batch in tqdm(loader, desc="encode images"):
        images = extractor.preprocess_images(batch["images"])
        image_global = extractor.encode_image_global(images)
        patches = extractor.encode_image_patches(images)
        regions = model.visual_group(patches)["grouped"]
        image_globals.append(F.normalize(image_global, dim=-1).cpu())
        region_feats.append(regions.cpu())
        image_ids.extend(batch["image_ids"])
        for image_id, captions in zip(batch["image_ids"], batch["all_captions"]):
            for caption in captions:
                all_captions.append(caption)
                caption_image_ids.append(image_id)

    image_globals_t = torch.cat(image_globals, dim=0).to(device)
    region_feats_t = torch.cat(region_feats, dim=0).to(device)

    text_globals, phrase_feats = [], []
    for start in tqdm(range(0, len(all_captions), eval_cfg["text_batch_size"]), desc="encode text"):
        caps = all_captions[start : start + eval_cfg["text_batch_size"]]
        tokens = extractor.tokenize(caps)
        text_global = extractor.encode_text_global(tokens)
        token_feats = extractor.encode_text_tokens(tokens)
        phrases = model.text_group(token_feats)["grouped"]
        text_globals.append(F.normalize(text_global, dim=-1).cpu())
        phrase_feats.append(phrases.cpu())
    text_globals_t = torch.cat(text_globals, dim=0).to(device)
    phrase_feats_t = torch.cat(phrase_feats, dim=0).to(device)

    sim = image_globals_t @ text_globals_t.t()
    topk = eval_cfg["rerank_topk"]
    refined = sim.clone()
    for i in tqdm(range(sim.size(0)), desc="rerank I2T"):
        idx = sim[i].topk(min(topk, sim.size(1))).indices
        fine = model.allpairs_fine_score(
            region_feats_t[i : i + 1], phrase_feats_t[idx], chunk_size=1
        ).squeeze(0)
        refined[i, idx] = sim[i, idx] + model.beta * fine
    for j in tqdm(range(sim.size(1)), desc="rerank T2I"):
        idx = sim[:, j].topk(min(topk, sim.size(0))).indices
        fine = model.allpairs_fine_score(
            region_feats_t[idx], phrase_feats_t[j : j + 1], chunk_size=eval_cfg["rerank_chunk"]
        ).squeeze(1)
        refined[idx, j] = sim[idx, j] + model.beta * fine

    return retrieval_metrics(refined.cpu(), image_ids, caption_image_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--evaluate_only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2025))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = OpenCLIPFeatureExtractor(
        cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device=device
    )
    model = build_hfcma(cfg, device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)

    if args.evaluate_only:
        metrics = evaluate(model, extractor, cfg, device)
        print(json.dumps(metrics, indent=2))
        return

    train_dataset = RetrievalJsonDataset(
        cfg["data"]["annotation_file"],
        cfg["data"]["image_root"],
        split=cfg["data"].get("train_split", "train"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_retrieval_batch,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = []
    best_rsum = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(model, extractor, train_loader, optimizer, cfg, device, epoch)
        metrics = evaluate(model, extractor, cfg, device)
        record = {"epoch": epoch, **train_stats, **metrics}
        history.append(record)
        (output_dir / "train_log.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if metrics["RSUM"] > best_rsum:
            best_rsum = metrics["RSUM"]
            torch.save({"model": model.state_dict(), "config": cfg}, output_dir / "best.pt")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

