import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


def _normalize_captions(item: Dict[str, Any]) -> List[str]:
    captions = item.get("captions")
    if captions is None and "caption" in item:
        captions = [item["caption"]]
    if isinstance(captions, str):
        captions = [captions]
    if not captions:
        raise ValueError("Each retrieval item must contain `captions` or `caption`.")
    return [str(c).strip() for c in captions if str(c).strip()]


def _image_field(item: Dict[str, Any]) -> str:
    for key in ("image", "image_path", "file_name", "filename"):
        if key in item and item[key]:
            return str(item[key])
    raise ValueError("Each retrieval item must contain an image path field.")


class RetrievalJsonDataset(Dataset):
    """JSON-backed image-text retrieval dataset.

    The expected annotation format is documented in ``docs/DATA.md``. Each item
    describes one image and one or more captions. During training one caption is
    sampled per image. During evaluation, ``return_all_captions=True`` keeps all
    captions so standard multi-caption retrieval metrics can be computed.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        split: Optional[str] = None,
        return_all_captions: bool = False,
    ) -> None:
        import json

        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.return_all_captions = return_all_captions
        with self.annotation_file.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = records.get("images", records.get("annotations", []))
        if split is not None:
            records = [r for r in records if str(r.get("split", split)) == split]
        if not records:
            raise ValueError(f"No records found in {self.annotation_file} for split={split}.")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        image_path = self.image_root / _image_field(item)
        image = Image.open(image_path).convert("RGB")
        captions = _normalize_captions(item)
        image_id = item.get("image_id", item.get("id", idx))
        out = {
            "image": image,
            "image_path": str(image_path),
            "image_id": image_id,
            "all_captions": captions,
        }
        out["caption"] = captions[0] if self.return_all_captions else random.choice(captions)
        return out


def collate_retrieval_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "all_captions": [item["all_captions"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }
