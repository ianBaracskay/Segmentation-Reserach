from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from matplotlib.colors import rgb_to_hsv
from PIL import Image

import config as cfg
from dino_processing import (
    build_dino_model_and_transform,
    run_dino_prompts,
    save_dino_debug_image,
    save_dino_detection_viz,
)
from image_processing import (
    build_amenity_heatmap,
    get_loaded_extent_meters,
    load_rgb_image,
    log_stage,
    normalize_to_uint8,
    report_geotiff_spatial_info,
)
from sam_processing import build_sam_predictor, generate_sam_masks_from_detections


script_start = perf_counter()
log_stage("Starting satellite sidewalk segmentation pipeline")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.results_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saving output figures to: {cfg.results_dir.resolve()}")
saved_figure_paths: list[Path] = []

# Function for consistent figure saving with logging and tracking of saved paths.
def save_current_figure(filename: str, category: str) -> None:
    category_dir = cfg.results_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_figure_paths.append(output_path)
    print(f"[INFO] Saved figure: {output_path}")

# Function used for combining masks that may be from tiled SAM runs back into a full-size image mask for visualization and heatmap building.
def expand_mask_to_full_image(mask_dict: dict) -> np.ndarray:
    if "tile_bounds" not in mask_dict:
        return mask_dict["segmentation"]
    ty0, ty1, tx0, tx1 = mask_dict["tile_bounds"]
    full_h, full_w = mask_dict["full_shape"]
    full_seg = np.zeros((full_h, full_w), dtype=bool)
    full_seg[ty0:ty1, tx0:tx1] = mask_dict["segmentation"]
    return full_seg


report_geotiff_spatial_info(cfg.tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)

img = load_rgb_image(cfg.tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)
loaded_extent_m = get_loaded_extent_meters(cfg.tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)

if loaded_extent_m is None:
    print("[WARN] Could not estimate loaded image extent in meters; amenity heatmap will be skipped.")
else:
    print(
        "[INFO] Loaded image real size (approx meters): "
        f"{loaded_extent_m[0]:.2f} m x {loaded_extent_m[1]:.2f} m"
    )

if img.size == 0:
    raise RuntimeError("Cropped image is empty. Check bbox values and raster coverage.")

img = normalize_to_uint8(img)
print(f"[INFO] Cropped image shape: {img.shape[0]}x{img.shape[1]} ({img.shape[2]} channels)")

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Input Aerial Image")
input_image_filename = f"{Path(cfg.tif_file).stem}_input_image.png"
save_current_figure(input_image_filename, "input_images")

print("[INFO] Running Grounding DINO detection")
dino_model, dino_transform, dino_device = build_dino_model_and_transform(cfg)
save_dino_debug_image(img, run_id, save_current_figure)
dino_records = run_dino_prompts(img, cfg, dino_model, dino_transform, dino_device)

if not dino_records:
    raise RuntimeError("Grounding DINO produced no detections across prompts.")

print(f"[INFO] Grounding DINO total kept detections for SAM: {len(dino_records)}")
save_dino_detection_viz(img, dino_records, run_id, save_current_figure)

stage_start = perf_counter()
predictor, sam_device = build_sam_predictor(cfg, img)
log_stage("SAM model ready", stage_start)
masks = generate_sam_masks_from_detections(predictor, dino_records)

if not masks:
    raise RuntimeError("No masks were produced from DINO detections. Check prompt thresholds.")

clip_device = sam_device if sam_device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")

stage_start = perf_counter()
print("[INFO] Loading CLIP model and preprocessing")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
clip_model = clip_model.to(clip_device)
clip_model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")
log_stage("CLIP model ready", stage_start)


def build_text_features(target_prompt: str):
    with torch.no_grad():
        pos_templates = [
            f"aerial image of {target_prompt}",
            f"overhead view of {target_prompt}",
            f"satellite photo of {target_prompt}",
            f"{target_prompt} in an urban area",
        ]
        neg_templates = [
            f"aerial image of {cfg.negative_prompt}",
            f"overhead view of {cfg.negative_prompt}",
        ]

        pos_tokens = tokenizer(pos_templates).to(clip_device)
        neg_tokens = tokenizer(neg_templates).to(clip_device)

        pos_features = clip_model.encode_text(pos_tokens)
        neg_features = clip_model.encode_text(neg_tokens)

        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        pos_text_feature = pos_features.mean(dim=0, keepdim=True)
        neg_text_feature = neg_features.mean(dim=0, keepdim=True)
        pos_text_feature = pos_text_feature / pos_text_feature.norm(dim=-1, keepdim=True)
        neg_text_feature = neg_text_feature / neg_text_feature.norm(dim=-1, keepdim=True)

    return pos_text_feature, neg_text_feature


stage_start = perf_counter()
print("[INFO] Encoding sidewalk and sitting-area prompts with CLIP")
sidewalk_pos_feature, sidewalk_neg_feature = build_text_features(cfg.sidewalk_prompt)
sitting_pos_feature, sitting_neg_feature = build_text_features(cfg.sitting_prompt)
log_stage("Text prompt encoding complete", stage_start)


def score_mask(mask_dict: dict, pos_text_feature: torch.Tensor, neg_text_feature: torch.Tensor) -> float:
    tile_seg = mask_dict["segmentation"]
    tile_bounds = mask_dict.get("tile_bounds")

    ys, xs = np.where(tile_seg)
    if ys.size == 0 or xs.size == 0:
        return -1.0

    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1

    if tile_bounds is not None:
        ty0, _, tx0, _ = tile_bounds
        full_h, full_w = mask_dict["full_shape"]
        iy0, iy1 = ly0 + ty0, ly1 + ty0
        ix0, ix1 = lx0 + tx0, lx1 + tx0
    else:
        full_h, full_w = tile_seg.shape[:2]
        iy0, iy1, ix0, ix1 = ly0, ly1, lx0, lx1

    area_ratio = float(tile_seg.sum()) / float(full_h * full_w)
    if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
        return -1.0

    crop = img[iy0:iy1, ix0:ix1]
    crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
    object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
    pil_img = Image.fromarray(object_only)

    with torch.no_grad():
        image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(clip_device)
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pos_sim = float((image_features @ pos_text_feature.T).item())
        neg_sim = float((image_features @ neg_text_feature.T).item())
        score = pos_sim - neg_sim

        iou = float(mask_dict.get("predicted_iou", 0.0))
        stability = float(mask_dict.get("stability_score", 0.0))
        score += 0.05 * iou + 0.05 * stability

    h = max(1, iy1 - iy0)
    w = max(1, ix1 - ix0)
    aspect = max(h, w) / float(min(h, w))
    bbox_fill = float(tile_seg.sum()) / float(h * w)

    elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
    sparse_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)
    score += 0.12 * float(elongation_score)
    score += 0.06 * float(sparse_fill_score)

    mask_pixels = img[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
    if mask_pixels.size > 0:
        pixels_norm = mask_pixels.astype(np.float32) / 255.0
        hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
        sat = float(hsv[:, 1].mean())
        val = float(hsv[:, 2].mean())

        low_sat_score = np.clip(
            (cfg.max_saturation_for_sidewalk - sat) / cfg.max_saturation_for_sidewalk,
            0.0,
            1.0,
        )
        bright_enough_score = np.clip(
            (val - cfg.min_value_for_sidewalk) / (1.0 - cfg.min_value_for_sidewalk),
            0.0,
            1.0,
        )

        score += 0.05 * float(low_sat_score)
        score += 0.04 * float(bright_enough_score)

    return score


def select_masks_for_prompt(
    masks_input: list,
    score_key: str,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor,
    prompt_name: str,
) -> tuple[list, float]:
    stage_start = perf_counter()
    print(f"[INFO] Scoring {len(masks_input)} masks for '{prompt_name}'")

    for idx, m in enumerate(masks_input, start=1):
        m[score_key] = score_mask(m, pos_text_feature, neg_text_feature)
        if idx == 1 or idx % 25 == 0 or idx == len(masks_input):
            elapsed = perf_counter() - stage_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(masks_input) - idx) / max(rate, 1e-6)
            print(
                f"[PROGRESS] [{prompt_name}] Scored {idx}/{len(masks_input)} masks "
                f"({rate:.2f} masks/s, ETA {eta:.1f}s)"
            )

    log_stage(f"Mask scoring complete for '{prompt_name}'", stage_start)

    filtered = [m for m in masks_input if m[score_key] >= 0.0]
    filtered.sort(key=lambda m: m[score_key], reverse=True)

    if not filtered:
        print(
            f"[WARN] No non-negative CLIP scores for '{prompt_name}'. "
            "Falling back to top-scoring masks regardless of score."
        )
        filtered = sorted(masks_input, key=lambda m: m[score_key], reverse=True)

    if filtered:
        best_score = float(filtered[0][score_key])
        dynamic_threshold = max(cfg.score_threshold, best_score - cfg.relative_score_margin)
    else:
        dynamic_threshold = cfg.score_threshold

    selected = [m for m in filtered if m[score_key] >= dynamic_threshold][: cfg.top_k]

    if len(selected) < 10:
        selected = filtered[: min(cfg.top_k, max(10, len(filtered)))]

    if not selected:
        selected = filtered[: min(cfg.top_k, 5)]

    if not selected:
        raise RuntimeError(f"No valid masks were generated/scored for prompt '{prompt_name}'.")

    print(f"Prompt: {prompt_name}")
    print(f"Negative prompt: {cfg.negative_prompt}")
    print(f"Selection threshold used: {dynamic_threshold:.3f}")
    print(f"Selected {len(selected)} / {len(masks_input)} masks by CLIP score")
    print("Top scores:", ", ".join(f"{m[score_key]:.3f}" for m in selected[:5]))

    return selected, dynamic_threshold


selected_sidewalk_masks, _ = select_masks_for_prompt(
    masks,
    "clip_score_sidewalk",
    sidewalk_pos_feature,
    sidewalk_neg_feature,
    cfg.sidewalk_prompt,
)

selected_sitting_masks, _ = select_masks_for_prompt(
    masks,
    "clip_score_sitting",
    sitting_pos_feature,
    sitting_neg_feature,
    cfg.sitting_prompt,
)

sidewalk_mask_combined = np.zeros(img.shape[:2], dtype=bool)
for m in selected_sidewalk_masks:
    sidewalk_mask_combined |= expand_mask_to_full_image(m)

sitting_mask_combined = np.zeros(img.shape[:2], dtype=bool)
for m in selected_sitting_masks:
    sitting_mask_combined |= expand_mask_to_full_image(m)

overlap_mask = sidewalk_mask_combined & sitting_mask_combined

viz_stride = max(1, int(np.ceil(max(img.shape[:2]) / 2000)))
if viz_stride > 1:
    print(f"[INFO] Downsampling visualization by stride={viz_stride} to reduce memory use")

viz_img = img[::viz_stride, ::viz_stride]
viz_sidewalk_mask = sidewalk_mask_combined[::viz_stride, ::viz_stride]
viz_sitting_mask = sitting_mask_combined[::viz_stride, ::viz_stride]
viz_overlap_mask = overlap_mask[::viz_stride, ::viz_stride]

viz_sidewalk_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
viz_sidewalk_overlay[viz_sidewalk_mask, :3] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
viz_sidewalk_overlay[viz_sidewalk_mask, 3] = 0.35

viz_sitting_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
viz_sitting_overlay[viz_sitting_mask, :3] = np.array([1.0, 0.65, 0.1], dtype=np.float32)
viz_sitting_overlay[viz_sitting_mask, 3] = 0.35

viz_combined_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
viz_combined_overlay[viz_sidewalk_mask, :3] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
viz_combined_overlay[viz_sidewalk_mask, 3] = 0.32
viz_combined_overlay[viz_sitting_mask, :3] = np.array([1.0, 0.65, 0.1], dtype=np.float32)
viz_combined_overlay[viz_sitting_mask, 3] = 0.32
viz_combined_overlay[viz_overlap_mask, :3] = np.array([0.8, 0.3, 0.95], dtype=np.float32)
viz_combined_overlay[viz_overlap_mask, 3] = 0.45

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_sidewalk_overlay)
plt.axis("off")
plt.title(f"SAM + CLIP Sidewalk Masks: '{cfg.sidewalk_prompt}'")
save_current_figure(f"{run_id}_sidewalk_masks.png", "individual_masks")

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_sitting_overlay)
plt.axis("off")
plt.title(f"SAM + CLIP Sitting-Area Masks: '{cfg.sitting_prompt}'")
save_current_figure(f"{run_id}_sitting_area_masks.png", "individual_masks")

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_combined_overlay)
plt.axis("off")
plt.title("Combined Sidewalk + Sitting-Area Masks")
save_current_figure(f"{run_id}_combined_masks.png", "combined_masks")

amenity_mask_union = sidewalk_mask_combined | sitting_mask_combined

if loaded_extent_m is not None:
    stage_start = perf_counter()
    print(
        "[INFO] Building amenity heatmap grid from mask union "
        f"(cell area: {cfg.amenity_grid_cell_area_m2:.2f} m^2)"
    )
    amenity_heatmap, cell_px_w, cell_px_h, cell_side_m = build_amenity_heatmap(
        amenity_mask_union,
        img.shape,
        loaded_extent_m,
        cfg.amenity_grid_cell_area_m2,
    )
    log_stage("Amenity heatmap complete", stage_start)

    alpha_map = np.where(amenity_heatmap > 0, np.clip(amenity_heatmap, 0.10, 0.90), 0.0)
    print(
        "[INFO] Heatmap grid cell size (approx): "
        f"{cell_side_m:.2f}m x {cell_side_m:.2f}m "
        f"(~{cell_px_w} x {cell_px_h} px)"
    )

    plt.figure(figsize=(10, 10))
    viz_amenity_heatmap = amenity_heatmap[::viz_stride, ::viz_stride]
    viz_alpha_map = alpha_map[::viz_stride, ::viz_stride]
    plt.imshow(viz_img)
    plt.imshow(viz_amenity_heatmap, cmap="Reds", alpha=viz_alpha_map, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Amenity Heatmap (Red = Higher Amenity Coverage)")
    save_current_figure(f"{run_id}_amenity_heatmap.png", "heatmaps")
else:
    print("[WARN] Skipping amenity heatmap because real-world extent is unavailable.")

print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
for p in saved_figure_paths:
    print(f"[INFO] -> {p}")

log_stage("Pipeline complete", script_start)
