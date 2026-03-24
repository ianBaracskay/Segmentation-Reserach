from pathlib import Path

import numpy as np
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry


def ensure_checkpoint(path: str, url: str) -> None:
    ckpt = Path(path)
    if ckpt.exists():
        return

    print(f"SAM checkpoint not found: {ckpt}. Downloading from {url} ...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with ckpt.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded checkpoint to {ckpt.resolve()}")


def resolve_device(device_setting: str) -> str:
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def build_sam_predictor(config, image: np.ndarray) -> tuple[SamPredictor, str]:
    device = resolve_device(config.sam_device)
    ensure_checkpoint(config.sam_checkpoint, config.sam_checkpoint_url)

    print(f"[INFO] Loading SAM model '{config.sam_model_type}' on device '{device}'")
    sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    return predictor, device


def generate_sam_masks_from_detections(predictor: SamPredictor, detections: list[dict]) -> list[dict]:
    masks: list[dict] = []
    print("[INFO] Running SAM refinement on DINO boxes")

    for i, rec in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in rec["box"]]
        phrase = rec["phrase"]
        prompt_group = rec["prompt_group"]

        try:
            sam_masks, scores, _ = predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False,
            )
            masks.append(
                {
                    "segmentation": sam_masks[0],
                    "predicted_iou": float(scores[0]),
                    "stability_score": float(scores[0]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "dino_phrase": phrase,
                    "dino_prompt_group": prompt_group,
                    "dino_score": float(rec["score"]),
                }
            )
        except Exception as exc:
            print(f"[WARN] SAM failed on box {i}: {exc}")

    print(f"[INFO] Generated {len(masks)} masks from DINO + SAM")
    return masks
