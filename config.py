from pathlib import Path

# Input image settings
tif_file = "LowGTCampus.tif"
use_bbox_crop = False
bbox_lonlat = (-84.4010, 33.7720, -84.3950, 33.7760)

# Output settings
results_dir = Path("results")

# Heatmap settings
amenity_grid_cell_area_m2 = 400.0

# DINO settings
dino_resize_short_side = 1200
dino_resize_max_size = 2000
dino_device = "auto"  # "auto", "cpu", "cuda"
dino_enable_tiled_fallback = True
dino_tile_size_px = 2200
dino_tile_overlap_px = 200
dino_tiled_max_detections_per_prompt = 24

# Prompt-by-prompt DINO configs
# Each prompt gets its own DINO run, then all detections are merged before SAM.
dino_prompt_configs = [
    {
        "name": "sidewalk",
        "caption": "sidewalk . pedestrian walkway . pavement . footpath",
        "box_threshold": 0.22,
        "text_threshold": 0.20,
        "keywords": ("sidewalk", "walkway", "pavement", "footpath", "pedestrian"),
        "min_area_ratio": 0.00015,
        "max_area_ratio": 0.25,
        "min_aspect_ratio": 1.05,
        "enable_tiled_fallback": True,
    },
    {
        "name": "sitting",
        "caption": "bench . park bench . outdoor table . plaza seating",
        "box_threshold": 0.18,
        "text_threshold": 0.18,
        "keywords": ("bench", "table", "seating", "plaza"),
        "min_area_ratio": 0.00002,
        "max_area_ratio": 0.10,
        "min_aspect_ratio": 1.0,
        "enable_tiled_fallback": False,
    },
]

# SAM settings
sam_model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
sam_device = "auto"  # "auto", "cpu", "cuda"
runtime_profile = "fast"  # options: "fast", "balanced", "quality"

# CLIP settings
sidewalk_prompt = "aerial overhead sidewalk"
sitting_prompt = "aerial overhead outdoor sitting area"
negative_prompt = "building roof, parking lot, trees, grass, water, shadow"

top_k = 45
score_threshold = -0.03
relative_score_margin = 0.10
min_area_ratio = 0.00015
max_area_ratio = 0.35
max_saturation_for_sidewalk = 0.28
min_value_for_sidewalk = 0.35
