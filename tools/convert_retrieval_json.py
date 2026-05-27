"""Convert common retrieval annotations into the HFCMA JSON format.

This helper supports two simple inputs:

1. A Karpathy-style JSON file with an ``images`` list and ``sentences`` fields.
2. A JSON list that already contains image/caption fields but needs field
   normalization.

The script does not download or redistribute any dataset files.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    image = item.get("filename") or item.get("file_name") or item.get("image") or item.get("image_path")
    if image is None:
        raise ValueError(f"Missing image path in item {idx}.")

    if "sentences" in item:
        captions = [s["raw"] if isinstance(s, dict) and "raw" in s else str(s) for s in item["sentences"]]
    else:
        captions = item.get("captions", item.get("caption", []))
        if isinstance(captions, str):
            captions = [captions]

    return {
        "image": image,
        "captions": captions,
        "image_id": item.get("image_id", item.get("imgid", item.get("id", idx))),
        "split": item.get("split", "train"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON annotation file.")
    parser.add_argument("--output", required=True, help="Output HFCMA-format JSON file.")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[Dict[str, Any]] = data["images"] if isinstance(data, dict) and "images" in data else data
    converted = [convert_item(item, idx) for idx, item in enumerate(records)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(converted)} items to {out}")


if __name__ == "__main__":
    main()
