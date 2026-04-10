from pathlib import Path

import numpy as np
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


def _iter_tile_coords(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)
    coords: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            coords.append((y0, y1, x0, x1))
    return coords


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


def _expand_box_xyxy(
    box_xyxy: list[int] | np.ndarray,
    img_w: int,
    img_h: int,
    factor: float,
) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if factor <= 1.0:
        return [x1, y1, x2, y2]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    new_w = w * float(factor)
    new_h = h * float(factor)

    nx1 = int(round(cx - 0.5 * new_w))
    ny1 = int(round(cy - 0.5 * new_h))
    nx2 = int(round(cx + 0.5 * new_w))
    ny2 = int(round(cy + 0.5 * new_h))

    nx1 = max(0, min(img_w - 1, nx1))
    nx2 = max(0, min(img_w - 1, nx2))
    ny1 = max(0, min(img_h - 1, ny1))
    ny2 = max(0, min(img_h - 1, ny2))
    return [nx1, ny1, nx2, ny2]


def generate_sam_masks_from_detections(
    predictor: SamPredictor,
    detections: list[dict],
    config=None,
) -> list[dict]:
    masks: list[dict] = []
    print("[INFO] Running SAM refinement on DINO boxes")

    img_h, img_w = predictor.original_size

    # Optional prompt-aware seed expansion before SAM refinement.
    expand_factor_by_prompt = {}
    if config is not None:
        expand_factor_by_prompt = getattr(config, "sam_prompt_box_expand_factors", {}) or {}

    for i, rec in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in rec["box"]]
        prompt_group = rec.get("prompt_group", "")
        expand_factor = float(expand_factor_by_prompt.get(prompt_group, 1.0))
        if expand_factor > 1.0:
            x1, y1, x2, y2 = _expand_box_xyxy([x1, y1, x2, y2], img_w, img_h, expand_factor)

        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] SAM skipping invalid box {i}: {[x1, y1, x2, y2]}")
            continue

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

        if (i + 1) == 1 or (i + 1) % 25 == 0 or (i + 1) == len(detections):
            print(
                f"[PROGRESS] SAM refined {i + 1}/{len(detections)} boxes, "
                f"masks so far: {len(masks)}"
            )

    print(f"[INFO] Generated {len(masks)} masks from DINO + SAM")
    return masks


def generate_sam_masks_automatic(sam_model, config, image: np.ndarray) -> list[dict]:
    """Generate SAM masks automatically without DINO detections using point-prompt grid."""
    print("[INFO] Running SAM automatic mask generation (no DINO)")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=config.sam_points_per_side,
        pred_iou_thresh=config.sam_pred_iou_thresh,
        stability_score_thresh=config.sam_stability_score_thresh,
    )
    
    auto_masks = mask_generator.generate(image)
    
    masks: list[dict] = []
    for i, mask_dict in enumerate(auto_masks):
        masks.append(
            {
                "segmentation": mask_dict["segmentation"],
                "predicted_iou": float(mask_dict.get("predicted_iou", 0.0)),
                "stability_score": float(mask_dict.get("stability_score", 0.0)),
                "bbox": mask_dict["bbox"],
                "dino_phrase": "auto-generated",
                "dino_prompt_group": "auto",
                "dino_score": 1.0,  # No DINO score for auto-generated masks
                "area": mask_dict.get("area", 0),
            }
        )
    
    print(f"[INFO] Generated {len(masks)} masks from automatic SAM")
    return masks


def generate_sam_masks_automatic_tiled(sam_model, config, image: np.ndarray) -> list[dict]:
    """Run automatic SAM per tile to avoid large-memory failures on huge rasters."""
    h, w = image.shape[:2]
    tile_size = int(getattr(config, "sam_auto_tile_size_px", 1400))
    tile_overlap = int(getattr(config, "sam_auto_tile_overlap_px", 160))
    max_total_masks = int(getattr(config, "sam_auto_max_total_masks", 3000))

    base_points = int(getattr(config, "sam_points_per_side", 32))
    capped_points = int(getattr(config, "sam_auto_max_points_per_side", 24))
    points_per_side = max(8, min(base_points, capped_points))

    tile_coords = _iter_tile_coords(h, w, tile_size, tile_overlap)
    print(
        "[INFO] Running tiled SAM automatic mask generation "
        f"({len(tile_coords)} tiles, tile={tile_size}, overlap={tile_overlap}, points_per_side={points_per_side})"
    )

    masks: list[dict] = []
    for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
        tile_img = image[y0:y1, x0:x1]
        tile_h, tile_w = tile_img.shape[:2]

        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            points_per_batch=32,
            pred_iou_thresh=float(getattr(config, "sam_pred_iou_thresh", 0.88)),
            stability_score_thresh=float(getattr(config, "sam_stability_score_thresh", 0.95)),
            crop_n_layers=0,
        )

        try:
            auto_masks = mask_generator.generate(tile_img)
        except RuntimeError as exc:
            print(f"[WARN] SAM auto tile {tile_idx}/{len(tile_coords)} failed: {exc}")
            continue

        for mask_dict in auto_masks:
            seg = mask_dict["segmentation"]
            if not np.any(seg):
                continue

            bx, by, bw, bh = mask_dict.get("bbox", [0, 0, tile_w, tile_h])
            masks.append(
                {
                    "segmentation": seg,
                    "predicted_iou": float(mask_dict.get("predicted_iou", 0.0)),
                    "stability_score": float(mask_dict.get("stability_score", 0.0)),
                    "bbox": [int(bx + x0), int(by + y0), int(bw), int(bh)],
                    "dino_phrase": "auto-generated",
                    "dino_prompt_group": "auto",
                    "dino_score": 1.0,
                    "area": int(mask_dict.get("area", int(seg.sum()))),
                    "tile_bounds": (y0, y1, x0, x1),
                    "full_shape": (h, w),
                }
            )

        if tile_idx == 1 or tile_idx % 5 == 0 or tile_idx == len(tile_coords):
            print(
                f"[PROGRESS] SAM auto tiles {tile_idx}/{len(tile_coords)} complete, "
                f"masks so far: {len(masks)}"
            )

        if len(masks) >= max_total_masks:
            print(
                f"[INFO] Stopping tiled auto SAM early at tile {tile_idx}/{len(tile_coords)} "
                f"after reaching {len(masks)} masks"
            )
            break

    print(f"[INFO] Generated {len(masks)} masks from tiled automatic SAM")
    return masks
