from pathlib import Path

# Input image settings
tif_file = "Maps/NYC/NYC(small).tif"
use_bbox_crop = False
bbox_lonlat = (-84.4010, 33.7720, -84.3950, 33.7760)

# Output settings
results_dir = Path("results")
output_dpi = 100  # Effective native-pixel export when combined with figure size matching
dino_visualization_dpi = 100  # Lower DPI for DINO detections to reduce memory usage during render (still ~high quality)
dino_visualization_backend = "pil"  # "matplotlib" or "pil" - PIL is faster, smaller files, more efficient

# Heatmap settings
amenity_grid_cell_area_m2 = 256.0
amenity_heatmap_excluded_prompts = ["building_roof"]
amenity_heatmap_taper_sigma_cells = 0.90
amenity_heatmap_taper_blend = 0.75
dino_heatmap_mode = "average"  # "average" or "sum" - average shows per-pixel detection confidence, sum shows detection density

# DINO settings - Set use_dino=False to skip DINO and use SAM's automatic mask generation instead
use_dino = True
dino_only = False  # Temporary debug mode: run only DINO and skip SAM + CLIP stages
dino_suppress_low_risk_warnings = True
dino_full_resolution = False  # If True, skip global DINO resize and run at native pixel dimensions; False uses resize (faster, avoids OOM)
dino_resize_short_side = 1200
dino_resize_max_size = 2000
dino_device = "cpu"  # "auto", "cpu", "cuda"
dino_enable_tiled_fallback = True
dino_tile_size_px = 6000
dino_tile_overlap_px = 600
#dino_tiled_max_detections_per_prompt = 24
dino_enable_area_split = False  # Keep original DINO boxes; do not subdivide into smaller boxes
dino_validate_split_boxes = False
dino_validate_split_max_candidates = 120
dino_nms_iou_threshold = 0.55
dino_negative_overlap_iou_threshold = 0.35
dino_max_boxes_per_prompt_for_sam = 200
dino_refine_bounds = True  # Tighten boxes by rerunning DINO inside candidate boxes
dino_refine_bounds_max_depth = 1  # Keep refinement shallow to avoid excessive compute
dino_refine_bounds_min_area_ratio = 0.80  # Accept refinements only when they shrink meaningfully

# DINO prompt selection - see prompts.py for all available prompts
# Each prompt gets its own DINO run, then all detections are merged before SAM.
from prompts import AVAILABLE_PROMPTS

ACTIVE_PROMPTS = ["sports_court", "park", "building_roof"]  # ORDERED: sports_court first, then park, then building_roof

# Auto-build dino_prompt_configs from selected prompts
dino_prompt_configs = [
    {"name": name, **AVAILABLE_PROMPTS[name]}
    for name in ACTIVE_PROMPTS
    if name in AVAILABLE_PROMPTS
]

# SAM settings
sam_model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
sam_device = "auto"  # "auto", "cpu", "cuda"
sam_full_resolution = True  # Keep SAM input at full image resolution
runtime_profile = "fast"  # options: "fast", "balanced", "quality"
sam_prompt_box_expand_factors = {
    "park": 1.7,  # Expand DINO park seed boxes before SAM to capture broader park context
}

# When use_dino=False, SAM will auto-generate masks. These settings control the auto-generation:
sam_points_per_side = 32  # Grid density for automatic mask generation
sam_pred_iou_thresh = 0.88  # Prediction IoU threshold for filtering masks
sam_stability_score_thresh = 0.95  # Stability score threshold for filtering masks

# Low-memory automatic SAM fallback settings for large rasters
sam_auto_tile_size_px = 1600
sam_auto_tile_overlap_px = 320
sam_auto_max_points_per_side = 24
sam_auto_max_total_masks = 3000

# Model input conversion settings (applied when source image is not uint8)
model_input_use_robust_uint8 = True
model_input_percentile_low = 1.0
model_input_percentile_high = 99.0

# CLIP settings - all per-prompt settings are now in prompts.py
