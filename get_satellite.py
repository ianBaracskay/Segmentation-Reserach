from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
import warnings

# Use non-interactive Agg backend to avoid pixmap allocation errors on large images
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from segment_anything import sam_model_registry

import config as cfg

# Optional, targeted suppression for known low-risk third-party warnings.
if getattr(cfg, "dino_suppress_low_risk_warnings", False):
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The `device` argument is deprecated and will be removed in v5 of Transformers\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"None of the inputs have requires_grad=True\. Gradients will be None",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
        category=FutureWarning,
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

from dino_processing import (
    build_dino_model_and_transform,
    run_dino_prompts,
    save_dino_detection_viz,
    save_dino_detection_viz_pil,
)
from dino_visualizations import (
    save_per_prompt_breakdown,
    save_filtering_stage_comparison,
    save_detection_heatmap,
    save_box_size_distribution,
)
from image_processing import (
    build_amenity_heatmap,
    get_loaded_extent_meters,
    load_rgb_image,
    log_stage,
    normalize_to_uint8,
    normalize_to_uint8_robust,
    report_geotiff_spatial_info,
    save_figure_high_resolution,
)
from sam_processing import (
    build_sam_predictor,
    generate_sam_masks_from_detections,
    generate_sam_masks_automatic,
    generate_sam_masks_automatic_tiled,
    ensure_checkpoint,
)


script_start = perf_counter()
log_stage("Starting satellite sidewalk segmentation pipeline")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.results_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saving output figures to: {cfg.results_dir.resolve()}")
saved_figure_paths: list[Path] = []

# Save figures at native raster pixel resolution when possible.
DEFAULT_SAVE_DPI = int(getattr(cfg, "output_dpi", 100))
DINO_SAVE_DPI = int(getattr(cfg, "dino_visualization_dpi", 75))

# Function for consistent figure saving with logging and tracking of saved paths.
def save_current_figure(filename: str, category: str, dpi: int | None = None, pil_img=None) -> None:
    category_dir = cfg.results_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / filename
   
    if pil_img is not None:
        # Save PIL image directly
        pil_img.save(str(output_path), "PNG", compress_level=6)
        saved_figure_paths.append(output_path)
        print(f"[INFO] Saved figure (PIL): {output_path}")
    else:
        # Save matplotlib figure
        use_dpi = dpi if dpi is not None else DEFAULT_SAVE_DPI
        save_figure_high_resolution(output_path, dpi=use_dpi, close_figure=True)
        saved_figure_paths.append(output_path)
        print(f"[INFO] Saved figure: {output_path}")


def finish_pipeline_early(reason: str) -> None:
    print(f"[INFO] {reason}")
    print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
    for p in saved_figure_paths:
        print(f"[INFO] -> {p}")
    log_stage("Pipeline complete", script_start)
    sys.exit(0)

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

img_raw = load_rgb_image(cfg.tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)
loaded_extent_m = get_loaded_extent_meters(cfg.tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)

if loaded_extent_m is None:
    print("[WARN] Could not estimate loaded image extent in meters; amenity heatmap will be skipped.")
else:
    print(
        "[INFO] Loaded image real size (approx meters): "
        f"{loaded_extent_m[0]:.2f} m x {loaded_extent_m[1]:.2f} m"
    )

if img_raw.size == 0:
    raise RuntimeError("Cropped image is empty. Check bbox values and raster coverage.")

# Keep model input separate from display image so DINO/SAM can run at full resolution.
if img_raw.dtype == np.uint8:
    img_model = img_raw
else:
    if bool(getattr(cfg, "model_input_use_robust_uint8", True)):
        img_model = normalize_to_uint8_robust(
            img_raw,
            low_percentile=float(getattr(cfg, "model_input_percentile_low", 1.0)),
            high_percentile=float(getattr(cfg, "model_input_percentile_high", 99.0)),
        )
    else:
        img_model = normalize_to_uint8(img_raw)

img_display = normalize_to_uint8(img_raw.copy()) if img_raw.dtype != np.uint8 else img_raw.copy()
print(
    f"[INFO] Model image shape: {img_model.shape[0]}x{img_model.shape[1]} "
    f"({img_model.shape[2]} channels), dtype={img_model.dtype}"
)

# Save input image using PIL to avoid matplotlib memory issues with large arrays
from PIL import Image
input_image_filename = f"{Path(cfg.tif_file).stem}_input_image.png"
input_image_out = cfg.results_dir / "input_images" / input_image_filename
input_image_out.parent.mkdir(parents=True, exist_ok=True)
img_pil = Image.fromarray(img_display, mode="RGB")
img_pil.save(str(input_image_out))
saved_figure_paths.append(input_image_out)
print(f"[INFO] Saved figure: {input_image_out}")

# Conditional DINO + SAM or automatic SAM
stage_start = perf_counter()
if cfg.dino_only and not cfg.use_dino:
    raise RuntimeError("dino_only=True requires use_dino=True")

if cfg.use_dino:
    print("[INFO] Running Grounding DINO detection")
    dino_model, dino_transform, dino_device = build_dino_model_and_transform(cfg)
    dino_run_id = f"{run_id}_dino_only" if cfg.dino_only else run_id
    dino_title_suffix = " (DINO-Only)" if cfg.dino_only else ""
    if cfg.dino_only:
        print("[INFO] DINO-only mode active: DINO outputs will be labeled with '_dino_only'.")
    
    # Calculate pixels-to-meters conversion for real-world area checking
    pixels_per_meter_sq = None
    if loaded_extent_m is not None:
        img_h, img_w = img_model.shape[:2]
        extent_w_m, extent_h_m = loaded_extent_m
        pixels_per_meter_sq = (img_w / extent_w_m) * (img_h / extent_h_m)
        print(f"[INFO] Pixel-to-meters: {pixels_per_meter_sq:.6f} px²/m²")
    
    print("[DEBUG] About to call run_dino_prompts", flush=True)
    try:
        dino_records, dino_unfiltered_records, dino_filtered_records = run_dino_prompts(
            img_model,
            cfg,
            dino_model,
            dino_transform,
            dino_device,
            pixels_per_meter_sq,
            return_unfiltered=True,
        )
        print(f"[DEBUG] run_dino_prompts returned: {len(dino_records) if dino_records else 0} kept, {len(dino_unfiltered_records) if dino_unfiltered_records else 0} unfiltered", flush=True)
    except Exception as e:
        print(f"[ERROR] run_dino_prompts failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if dino_records:
        print(f"[INFO] Grounding DINO total kept detections: {len(dino_records)}")
        # Choose visualization backend based on config
        if cfg.dino_visualization_backend == "pil":
            save_dino_detection_viz_pil(
                img_display,
                dino_records,
                dino_run_id,
                save_current_figure,
                title_suffix=dino_title_suffix,
                filtered_records=dino_filtered_records,
            )
        else:
            save_dino_detection_viz(
                img_display,
                dino_records,
                dino_run_id,
                save_current_figure,
                title_suffix=dino_title_suffix,
                filtered_records=dino_filtered_records,
                dpi=DINO_SAVE_DPI,
            )
    else:
        print("[WARN] Grounding DINO produced no valid detections after filtering.")
        if dino_unfiltered_records:
            print(
                f"[INFO] DINO produced {len(dino_unfiltered_records)} unfiltered candidates; "
                "saving debug visualization before filtering."
            )
            if cfg.dino_visualization_backend == "pil":
                save_dino_detection_viz_pil(
                    img_display,
                    dino_unfiltered_records,
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                )
            else:
                save_dino_detection_viz(
                    img_display,
                    dino_unfiltered_records,
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                    dpi=DINO_SAVE_DPI,
                )
        else:
            if cfg.dino_visualization_backend == "pil":
                save_dino_detection_viz_pil(
                    img_display,
                    [],
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                )
            else:
                save_dino_detection_viz(
                    img_display,
                    [],
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                    dpi=DINO_SAVE_DPI,
                )

    # Generate diagnostic visualizations for DINO detections
    if dino_records or dino_unfiltered_records:
        print("[INFO] Generating DINO diagnostic visualizations...")
        
        # 1. Per-prompt breakdown - separate visualization for each detector
        if dino_records:
            try:
                save_per_prompt_breakdown(img_display, dino_records, dino_run_id, config=cfg)
            except Exception as e:
                print(f"[ERROR] Per-prompt breakdown visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. Filtering stage comparison - show stages from unfiltered to passed to filtered-out
        if dino_unfiltered_records:
            try:
                save_filtering_stage_comparison(
                    img_display, 
                    dino_unfiltered_records, 
                    dino_records,
                    dino_filtered_records, 
                    dino_run_id, 
                    config=cfg
                )
            except Exception as e:
                print(f"[ERROR] Filtering stage comparison visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. Detection heatmap - confidence density map
        if dino_records:
            try:
                save_detection_heatmap(img_display, dino_records, dino_run_id, config=cfg)
            except Exception as e:
                print(f"[ERROR] Detection heatmap visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. Box size distribution - analysis of detected object sizes
        if dino_records:
            try:
                save_box_size_distribution(img_display, dino_records, dino_run_id, config=cfg)
            except Exception as e:
                print(f"[ERROR] Box size distribution visualization failed: {e}")
                import traceback
                traceback.print_exc()

    if cfg.dino_only:
        finish_pipeline_early("DINO-only mode enabled; skipping SAM and CLIP stages.")

    if not dino_records:
        print("[WARN] Grounding DINO produced no valid detections after filtering. Running tiled automatic SAM fallback.")
        from sam_processing import resolve_device

        device = resolve_device(cfg.sam_device)
        ensure_checkpoint(cfg.sam_checkpoint, cfg.sam_checkpoint_url)
        print(f"[INFO] Loading SAM model '{cfg.sam_model_type}' on device '{device}'")
        sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
        sam.to(device=device)

        log_stage("SAM model ready", stage_start)
        masks = generate_sam_masks_automatic_tiled(sam, cfg, img_model)
        sam_device = device
    else:
        print(f"[INFO] Grounding DINO total kept detections for SAM: {len(dino_records)}")

        predictor, sam_device = build_sam_predictor(cfg, img_model)
        log_stage("SAM model ready", stage_start)
        masks = generate_sam_masks_from_detections(predictor, dino_records, cfg)
else:
    print("[INFO] Skipping DINO and using automatic SAM mask generation")
    from sam_processing import resolve_device
    
    device = resolve_device(cfg.sam_device)
    ensure_checkpoint(cfg.sam_checkpoint, cfg.sam_checkpoint_url)
    print(f"[INFO] Loading SAM model '{cfg.sam_model_type}' on device '{device}'")
    sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
    sam.to(device=device)
    
    log_stage("SAM model ready", stage_start)
    masks = generate_sam_masks_automatic_tiled(sam, cfg, img_model)
    sam_device = device

if not masks:
    print("[WARN] No masks were produced for selected prompts. Continuing with empty overlays and heatmap.")

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


def build_text_features(target_prompt: str, negative_prompts: list[str] | None = None):
    with torch.no_grad():
        pos_templates = [
            f"aerial image of {target_prompt}",
            f"overhead view of {target_prompt}",
            f"satellite photo of {target_prompt}",
            f"{target_prompt} in an urban area",
        ]

        pos_tokens = tokenizer(pos_templates).to(clip_device)
        pos_features = clip_model.encode_text(pos_tokens)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        pos_text_feature = pos_features.mean(dim=0, keepdim=True)
        pos_text_feature = pos_text_feature / pos_text_feature.norm(dim=-1, keepdim=True)

        neg_text_feature = None
        if negative_prompts:
            neg_templates: list[str] = []
            for neg in negative_prompts:
                neg_templates.extend(
                    [
                        f"aerial image of {neg}",
                        f"overhead view of {neg}",
                        f"satellite photo of {neg}",
                    ]
                )
            neg_tokens = tokenizer(neg_templates).to(clip_device)
            neg_features = clip_model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            neg_text_feature = neg_features.mean(dim=0, keepdim=True)
            neg_text_feature = neg_text_feature / neg_text_feature.norm(dim=-1, keepdim=True)

    return pos_text_feature, neg_text_feature


stage_start = perf_counter()
print("[INFO] Encoding CLIP prompts for active amenities")
# Build text features for all active prompts using their captions
clip_features = {}
for prompt_config in cfg.dino_prompt_configs:
    prompt_name = prompt_config["name"]
    caption = prompt_config["caption"]
    neg_captions = prompt_config.get("negative_captions", [])
    pos_feature, neg_feature = build_text_features(caption, neg_captions)
    clip_features[prompt_name] = {
        "pos": pos_feature,
        "neg": neg_feature,
        "text": caption,
        "config": prompt_config,
    }
log_stage("Text prompt encoding complete", stage_start)


def score_mask(
    mask_dict: dict,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor | None,
    prompt_config: dict,
) -> float:
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
    min_area = prompt_config.get("clip_min_area_ratio", 0.00015)
    max_area = prompt_config.get("clip_max_area_ratio", 0.25)
    if area_ratio < min_area or area_ratio > max_area:
        return -1.0

    crop = img_model[iy0:iy1, ix0:ix1]
    crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
    object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
    pil_img = Image.fromarray(object_only)

    with torch.no_grad():
        image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(clip_device)
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pos_sim = float((image_features @ pos_text_feature.T).item())
        score = pos_sim

        if neg_text_feature is not None:
            neg_sim = float((image_features @ neg_text_feature.T).item())
            neg_weight = float(prompt_config.get("clip_negative_weight", 0.35))
            score -= neg_weight * max(0.0, neg_sim)

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

    mask_pixels = img_model[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
    if mask_pixels.size > 0:
        pixels_norm = mask_pixels.astype(np.float32) / 255.0
        hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
        sat = float(hsv[:, 1].mean())
        val = float(hsv[:, 2].mean())

        max_sat = prompt_config.get("max_saturation", 0.28)
        min_val = prompt_config.get("min_value", 0.35)

        low_sat_score = np.clip(
            (max_sat - sat) / max_sat,
            0.0,
            1.0,
        )
        bright_enough_score = np.clip(
            (val - min_val) / (1.0 - min_val),
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
    neg_text_feature: torch.Tensor | None,
    prompt_name: str,
    prompt_config: dict,
) -> tuple[list, float]:
    stage_start = perf_counter()
    print(f"[INFO] Scoring {len(masks_input)} masks for '{prompt_name}'")

    for idx, m in enumerate(masks_input, start=1):
        m[score_key] = score_mask(m, pos_text_feature, neg_text_feature, prompt_config)
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
        print(f"[WARN] No non-negative CLIP scores for '{prompt_name}'. Selecting 0 masks.")
        return [], float(prompt_config.get("clip_score_threshold", -0.03))

    top_k = prompt_config.get("clip_top_k", 45)
    score_threshold = prompt_config.get("clip_score_threshold", -0.03)
    relative_margin = prompt_config.get("clip_relative_score_margin", 0.10)

    if filtered:
        best_score = float(filtered[0][score_key])
        dynamic_threshold = max(score_threshold, best_score - relative_margin)
    else:
        dynamic_threshold = score_threshold

    selected = [m for m in filtered if m[score_key] >= dynamic_threshold][:top_k]

    if not selected and filtered:
        selected = filtered[:1]

    print(f"Prompt: {prompt_name}")
    print(f"Selection threshold used: {dynamic_threshold:.3f}")
    print(f"Selected {len(selected)} / {len(masks_input)} masks by CLIP score")
    print("Top scores:", ", ".join(f"{m[score_key]:.3f}" for m in selected[:5]))

    return selected, dynamic_threshold


selected_masks_by_prompt = {}
is_auto_mask_run = all(m.get("dino_prompt_group") in (None, "auto") for m in masks)

if is_auto_mask_run:
    print(
        "[INFO] SAM-only conceptual per-prompt mode: cloning auto-SAM masks into "
        "independent prompt candidate pools before CLIP scoring"
    )

for prompt_name, features in clip_features.items():
    if is_auto_mask_run:
        # In SAM-only mode, each prompt evaluates an independent copy of the auto-SAM pool.
        # This keeps prompt scoring isolated even though mask geometry came from one SAM run.
        prompt_masks = [{**m, "dino_prompt_group": prompt_name} for m in masks]
        print(
            f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} independent SAM-only candidates"
        )
    else:
        # In DINO-guided mode, keep strict prompt isolation from DINO group labels.
        prompt_masks = [m for m in masks if m.get("dino_prompt_group") == prompt_name]
        if not prompt_masks:
            print(
                f"[WARN] No SAM masks linked to prompt '{prompt_name}'; selecting 0 masks for this prompt."
            )
            selected_masks_by_prompt[prompt_name] = []
            continue
        print(
            f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} prompt-linked SAM masks"
        )

    selected, _ = select_masks_for_prompt(
        prompt_masks,
        f"clip_score_{prompt_name}",
        features["pos"],
        features["neg"],
        prompt_name,
        features["config"],
    )
    selected_masks_by_prompt[prompt_name] = selected

# Compute combined masks for each prompt
combined_masks = {}
for prompt_name, selected_masks in selected_masks_by_prompt.items():
    combined = np.zeros(img_model.shape[:2], dtype=bool)
    for m in selected_masks:
        combined |= expand_mask_to_full_image(m)
    combined_masks[prompt_name] = combined

# Apply prompt priority: earlier prompts in ACTIVE_PROMPTS take precedence
# (so sports_court masks remove overlapping building_roof, etc.)
# Build ordered dict to access prompts in order
prompt_order = list(cfg.dino_prompt_configs) if hasattr(cfg, 'dino_prompt_configs') else []
prompt_names_ordered = [p['name'] for p in prompt_order]

combined_masks_with_priority = {}
accumulated_mask = np.zeros(img_model.shape[:2], dtype=bool)

for prompt_name in prompt_names_ordered:
    if prompt_name in combined_masks:
        # Remove areas already claimed by higher-priority prompts
        original_mask = combined_masks[prompt_name]
        priority_mask = original_mask.copy()
        priority_mask &= ~accumulated_mask  # Subtract accumulated mask from earlier prompts
        combined_masks_with_priority[prompt_name] = priority_mask
        
        # DEBUG: Log mask pixel counts
        original_pixels = original_mask.sum()
        remaining_pixels = priority_mask.sum()
        removed_pixels = original_pixels - remaining_pixels
        print(f"[DEBUG] {prompt_name}: {original_pixels:,} pixels → {remaining_pixels:,} pixels (removed {removed_pixels:,} due to priority masking)")
        
        accumulated_mask |= priority_mask
    else:
        combined_masks_with_priority[prompt_name] = np.zeros(img_model.shape[:2], dtype=bool)
        print(f"[DEBUG] {prompt_name}: NOT FOUND in combined_masks - creating empty mask")

# Use priority-ordered masks instead
combined_masks = combined_masks_with_priority

# Compute overlap mask (union of all masks)
union_of_all = np.zeros(img_model.shape[:2], dtype=bool)
for combined in combined_masks.values():
    union_of_all |= combined

# Visualize each prompt separately and create combined overlay
viz_stride = max(1, int(np.ceil(max(img_model.shape[:2]) / 2000)))
if viz_stride > 1:
    print(f"[INFO] Downsampling visualization by stride={viz_stride} to reduce memory use")

# DEBUG: Show what's in combined_masks before visualization
print(f"[DEBUG] combined_masks keys: {list(combined_masks.keys())}")
for pname, pmask in combined_masks.items():
    print(f"[DEBUG]   {pname}: {pmask.sum():,} pixels ({pmask.sum() / pmask.size * 100:.2f}% coverage)")

viz_img = img_display[::viz_stride, ::viz_stride]

# Colors for different prompts (explicit and high-contrast for reliable legend matching)
prompt_colors = {
    "sports_court": [1.00, 0.84, 0.00],   # gold
    "park": [0.00, 0.65, 1.00],           # vivid blue
    "building_roof": [1.00, 0.20, 0.20],  # bold red
    "outdoor_seating": [0.10, 0.90, 0.90],
    "standing_gathering": [1.00, 0.65, 0.10],
    "furniture": [0.90, 0.20, 0.20],
    "seated_dining": [0.85, 0.30, 0.85],
}


def get_prompt_color(prompt_name: str) -> np.ndarray:
    # Magenta fallback makes missing color mappings immediately obvious.
    return np.array(prompt_colors.get(prompt_name, [1.0, 0.0, 1.0]), dtype=np.float32)

# Save individual visualizations for each prompt
for prompt_name, combined_mask in combined_masks.items():
    viz_mask = combined_mask[::viz_stride, ::viz_stride]
    color = get_prompt_color(prompt_name)
    
    viz_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
    viz_overlay[viz_mask, :3] = color
    viz_overlay[viz_mask, 3] = 0.50  # Increased alpha from 0.35 to 0.50 for better visibility

    plt.figure(figsize=(10, 10))
    plt.imshow(viz_img)
    plt.imshow(viz_overlay)
    plt.axis("off")
    plt.title(f"SAM + CLIP {prompt_name.replace('_', ' ').title()} Masks: '{clip_features[prompt_name]['text']}'")
    save_current_figure(f"{run_id}_{prompt_name}_masks.png", "individual_masks")

# Create combined visualization with all prompts overlaid
viz_combined_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
for prompt_name, combined_mask in combined_masks.items():
    viz_mask = combined_mask[::viz_stride, ::viz_stride]
    color = get_prompt_color(prompt_name)
    viz_combined_overlay[viz_mask, :3] = color
    viz_combined_overlay[viz_mask, 3] = 0.45  # Increased alpha from 0.32 to 0.45 for better visibility

# Create figure with extra space for legend on the right
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(viz_img)
ax.imshow(viz_combined_overlay)
ax.axis("off")

# Create legend with custom patches for each prompt
from matplotlib.patches import Patch
legend_elements = []
for prompt_name in sorted(combined_masks.keys()):
    color_rgb = get_prompt_color(prompt_name).tolist()
    label = prompt_name.replace('_', ' ').title()
    legend_elements.append(Patch(facecolor=color_rgb, edgecolor='black', label=label))

# Add legend outside the plot area on the right side
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
         fontsize=11, frameon=True, fancybox=True, shadow=True)

fig.suptitle(f"Combined Masks: {', '.join(p.replace('_', ' ').title() for p in combined_masks.keys())}", 
            fontsize=14, weight='bold')
plt.tight_layout()
save_current_figure(f"{run_id}_combined_masks.png", "combined_masks")

heatmap_excluded_prompts = set(getattr(cfg, "amenity_heatmap_excluded_prompts", ["building_roof"]))
heatmap_mask_union = np.zeros(img_model.shape[:2], dtype=bool)
heatmap_prompt_names: list[str] = []
for prompt_name, combined_mask in combined_masks.items():
    if prompt_name in heatmap_excluded_prompts:
        continue
    heatmap_mask_union |= combined_mask
    heatmap_prompt_names.append(prompt_name)

if not heatmap_prompt_names:
    print("[WARN] Heatmap excluded all prompts; falling back to union of all prompt masks.")
    amenity_mask_union = union_of_all
else:
    print(
        "[INFO] Heatmap amenities from prompts: "
        f"{', '.join(heatmap_prompt_names)} (excluded: {', '.join(sorted(heatmap_excluded_prompts))})"
    )
    amenity_mask_union = heatmap_mask_union

if loaded_extent_m is not None:
    stage_start = perf_counter()
    print(
        "[INFO] Building amenity heatmap grid from mask union "
        f"(cell area: {cfg.amenity_grid_cell_area_m2:.2f} m^2)"
    )
    amenity_heatmap, cell_px_w, cell_px_h, cell_side_m = build_amenity_heatmap(
        amenity_mask_union,
        img_model.shape,
        loaded_extent_m,
        cfg.amenity_grid_cell_area_m2,
        taper_sigma_cells=float(getattr(cfg, "amenity_heatmap_taper_sigma_cells", 0.90)),
        taper_blend=float(getattr(cfg, "amenity_heatmap_taper_blend", 0.75)),
    )
    log_stage("Amenity heatmap complete", stage_start)

    print(
        "[INFO] Heatmap grid cell size (approx): "
        f"{cell_side_m:.2f}m x {cell_side_m:.2f}m "
        f"(~{cell_px_w} x {cell_px_h} px)"
    )

    plt.figure(figsize=(10, 10))
    viz_amenity_heatmap = amenity_heatmap[::viz_stride, ::viz_stride]
    viz_alpha_map = np.where(
        viz_amenity_heatmap > 0,
        np.clip(viz_amenity_heatmap, 0.10, 0.90),
        0.0,
    )
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
