"""
Prompt definitions for DINO detection, SAM refinement, and CLIP evaluation.
Each prompt includes all necessary settings for the detection and evaluation pipeline.
Add new prompts here as you discover them.
"""

# All available prompt definitions
AVAILABLE_PROMPTS = {
    "sports_court": {
        # Sports courts and athletic areas - HIGHEST PRIORITY (most specific, very reliable)
        "caption": "basketball court . tennis court . sports court . hard court . fenced court",
        "negative_dino_caption": "building . roof . parking lot . road",
        "negative_box_threshold": 0.35,
        "negative_text_threshold": 0.30,
        # DINO detection settings
        "box_threshold": 0.17,
        "text_threshold": 0.12,
        "keywords": ("court", "sports", "basketball", "tennis", "athletic", "playground"),
        "negative_keywords": ("building", "roof", "parking", "road", "car", "grass", "tree"),
        "min_aspect_ratio": 0.6,  # Courts are somewhat rectangular
        "max_aspect_ratio": 3.0,   # But not too elongated
        "min_box_side_px": 30,     # Courts are substantial size
        "max_split_boxes_per_detection": 64,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 50000,
        "negative_captions": ["building", "roof", "parking lot", "road"],
        "clip_negative_weight": 0.38,
        # SAM/CLIP evaluation settings
        "clip_top_k": 20,
        "clip_score_threshold": 0.12,
        "clip_relative_score_margin": 0.08,
        "clip_min_area_ratio": 0.0005,
        "clip_max_area_ratio": 0.20,
        "max_saturation": 1.0,
        "min_value": 0.20,
    },
    "outdoor_seating": {
        # Broad but intentional gathering-area detector
        "caption": "outdoor seating . patio . terrace . plaza . courtyard . bench . picnic table . outdoor dining area . pedestrian area . sidewalk seating",
        "negative_dino_caption": "building . roof . parking lot . road . car . highway . construction site . grass field . empty field",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.30,
        # DINO detection settings
        "box_threshold": 0.30,
        "text_threshold": 0.26,
        "keywords": ("bench", "table", "seating", "patio", "plaza", "courtyard", "dining"),
        "negative_keywords": ("building", "roof", "parking", "road", "car", "highway", "construction", "grass", "field"),
        "min_aspect_ratio": 0.8,
        "max_aspect_ratio": 8.0,  # Gathering areas are more rectangular, not elongated
        "min_box_side_px": 25,
        "max_split_boxes_per_detection": 64,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 50000,  # Courtyards/plazas can be large
        "negative_captions": [
            "building",
            "roof",
            "parking lot",
            "road",
            "car",
            "highway",
            "construction site",
            "grass field",
            "empty field",
        ],
        "clip_negative_weight": 0.45,  # Strong rejection of empty/barren areas
        # SAM/CLIP evaluation settings
        "clip_top_k": 50,
        "clip_score_threshold": 0.05,  # Gathering areas should have positive signals
        "clip_relative_score_margin": 0.15,
        "clip_min_area_ratio": 0.0002,
        "clip_max_area_ratio": 0.40,
        "max_saturation": 1.0,  # Varied saturation (furniture, people have colors)
        "min_value": 0.25,  # Can be in partial shade
    },
    "seated_dining": {
        # Actual prompts given to DINO/SAM - specific to outdoor restaurants/cafes
        "caption": "outdoor cafe . restaurant patio . dining tables . outdoor seating with tables . alfresco dining",
        # DINO detection settings
        "box_threshold": 0.26,
        "text_threshold": 0.24,
        "keywords": ("dining", "cafe", "restaurant", "table", "seating", "patio"),
        "negative_keywords": ("building", "roof", "parking", "road", "car", "highway", "grass", "field"),
        "min_aspect_ratio": 0.9,
        "max_aspect_ratio": 7.0,
        "min_box_side_px": 20,
        "max_split_boxes_per_detection": 48,
        "enable_area_split": False,
        "enable_tiled_fallback": False,
        "max_area_meters_sq": 20000,  # Cafe areas are typically smaller than plazas
        "negative_captions": [
            "parking lot",
            "highway",
            "empty street",
            "grass field",
            "roof",
        ],
        "clip_negative_weight": 0.40,
        # SAM/CLIP evaluation settings
        "clip_top_k": 45,
        "clip_score_threshold": 0.10,  # Dining areas should have strong positive signal
        "clip_relative_score_margin": 0.12,
        "clip_min_area_ratio": 0.0003,
        "clip_max_area_ratio": 0.30,
        "max_saturation": 1.0,
        "min_value": 0.30,
    },
    "standing_gathering": {
        # Actual prompts given to DINO/SAM - plazas, courtyards, open gathering spaces
        "caption": "plaza . courtyard . outdoor gathering area . public square . terrace . open congregation area",
        "negative_dino_caption": "building . roof . parking lot . road . car . highway . construction site . grass field . empty field",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.30,
        # DINO detection settings
        "box_threshold": 0.29,
        "text_threshold": 0.25,
        "keywords": ("plaza", "courtyard", "gathering", "square", "terrace"),
        "negative_keywords": ("building", "roof", "parking", "road", "car", "highway", "construction", "grass", "field"),
        "min_aspect_ratio": 0.7,
        "max_aspect_ratio": 9.0,
        "min_box_side_px": 30,
        "max_split_boxes_per_detection": 72,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 60000,  # Plazas can be quite large
        "negative_captions": [
            "building roof",
            "parking lot",
            "sports field",
            "tree canopy",
            "grass lawn",
        ],
        "clip_negative_weight": 0.42,
        # SAM/CLIP evaluation settings
        "clip_top_k": 50,
        "clip_score_threshold": 0.02,
        "clip_relative_score_margin": 0.14,
        "clip_min_area_ratio": 0.0004,
        "clip_max_area_ratio": 0.50,
        "max_saturation": 1.0,
        "min_value": 0.20,  # Plazas may have shadows
    },
    "park": {
        # Parks and green public open spaces
        "caption": "park . city park . public park . urban park . green space . playground . parkland . tree-lined park . grassy park . lawn",
        "negative_dino_caption": "building . roof . parking lot . road . highway . car . tree canopy . deep shadow",
        "negative_box_threshold": 0.35,
        "negative_text_threshold": 0.28,
        # DINO detection settings
        "box_threshold": 0.10,
        "text_threshold": 0.12,
        "keywords": ("park", "green", "lawn", "playground", "trees", "garden", "field"),
        "negative_keywords": ("building", "roof", "parking", "road", "car", "highway", "shadow", "canopy"),
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 12.0,
        "min_box_side_px": 25,
        "max_split_boxes_per_detection": 96,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "refine_bounds": False,
        "prefer_smaller_boxes": False,
        "max_area_meters_sq": 150000,
        "negative_captions": [
            "building",
            "roof",
            "parking lot",
            "road",
            "highway",
            "tree canopy",
            "deep shadow",
        ],
        "clip_negative_weight": 0.35,
        # SAM/CLIP evaluation settings
        "clip_top_k": 50,
        "clip_score_threshold": 0.00,
        "clip_relative_score_margin": 0.10,
        "clip_min_area_ratio": 0.0002,
        "clip_max_area_ratio": 0.60,
        "max_saturation": 1.0,
        "min_value": 0.18,
    },
    "transit_hub": {
        # Transit-adjacent gathering points.
        "caption": "bus stop . subway entrance . metro entrance . train station entrance . bike rack . transit shelter",
        "negative_dino_caption": "building roof . parking lot . road lane . highway",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.28,
        "box_threshold": 0.18,
        "text_threshold": 0.14,
        "keywords": ("bus", "subway", "metro", "station", "bike", "rack", "transit", "shelter", "entrance"),
        "negative_keywords": ("roof", "building", "highway", "parking"),
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 12.0,
        "min_box_side_px": 14,
        "max_split_boxes_per_detection": 96,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 35000,
        "negative_captions": [
            "building roof",
            "parking lot",
            "highway",
            "empty grass field",
        ],
        "clip_negative_weight": 0.40,
        "clip_top_k": 45,
        "clip_score_threshold": 0.02,
        "clip_relative_score_margin": 0.10,
        "clip_min_area_ratio": 0.00005,
        "clip_max_area_ratio": 0.25,
        "max_saturation": 1.0,
        "min_value": 0.16,
    },
    "pedestrian_features": {
        # Pedestrian-centric public features often correlated with gathering.
        "caption": "crosswalk . playground . gazebo . amphitheater . pedestrian plaza . public square",
        "negative_dino_caption": "building roof . highway . parking lot . industrial lot",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.28,
        "box_threshold": 0.16,
        "text_threshold": 0.13,
        "keywords": ("crosswalk", "playground", "gazebo", "amphitheater", "pedestrian", "plaza", "square"),
        "negative_keywords": ("roof", "building", "highway", "industrial", "warehouse"),
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 14.0,
        "min_box_side_px": 16,
        "max_split_boxes_per_detection": 96,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 90000,
        "negative_captions": [
            "building roof",
            "industrial roof",
            "highway",
            "parking lot",
        ],
        "clip_negative_weight": 0.42,
        "clip_top_k": 50,
        "clip_score_threshold": 0.00,
        "clip_relative_score_margin": 0.12,
        "clip_min_area_ratio": 0.00008,
        "clip_max_area_ratio": 0.40,
        "max_saturation": 1.0,
        "min_value": 0.18,
    },
    "furniture": {
        # Optional prompt kept for future high-resolution imagery; usually too small at campus-scale satellite GSD.
        "caption": "table . chair . bench . umbrella . outdoor furniture",
        # DINO detection settings
        "box_threshold": 0.22,
        "text_threshold": 0.20,
        "keywords": ("table", "chair", "bench", "umbrella", "furniture", "picnic"),
        "negative_keywords": ("building", "roof", "parking", "road", "highway", "grass", "field"),
        "min_aspect_ratio": 0.6,
        "max_aspect_ratio": 6.0,
        "min_box_side_px": 15,
        "max_split_boxes_per_detection": 128,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 5000,  # Individual furniture items
        "negative_captions": [
            "building roof",
            "asphalt",
            "street",
            "parking",
        ],
        "clip_negative_weight": 0.35,
        # SAM/CLIP evaluation settings
        "clip_top_k": 40,
        "clip_score_threshold": 0.0,
        "clip_relative_score_margin": 0.10,
        "clip_min_area_ratio": 0.00001,
        "clip_max_area_ratio": 0.08,
        "max_saturation": 1.0,
        "min_value": 0.15,
    },
    "building_roof": {
        # Building roof detection - easy to identify for model testing
        "caption": "building . roof . apartment building . brownstone . skyscraper . office building . residential building",
        # DINO detection settings - RELAXED for testing
        "box_threshold": 0.08,  # Low threshold to catch more roof candidates
        "text_threshold": 0.05,  # Low text threshold for better recall
        "keywords": ("roof", "building", "top"),
        "negative_keywords": ("car", "truck", "person", "tree", "grass"),
        "min_aspect_ratio": 0.3,  # Relaxed to catch angled buildings
        "max_aspect_ratio": 25.0,  # Increased significantly for rotated buildings appearing as elongated boxes
        "min_box_side_px": 8,  # Reduced to catch smaller buildings
        "max_split_boxes_per_detection": 256,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 1000000,  # Buildings can be large
        "negative_captions": [
            "parking lot",
            "road",
            "grass",
        ],
        "clip_negative_weight": 0.30,
        # SAM/CLIP evaluation settings
        "clip_top_k": 50,
        "clip_score_threshold": -0.15,  # More aggressive - lower threshold to catch more buildings
        "clip_relative_score_margin": 0.10,
        "clip_min_area_ratio": 0.0001,
        "clip_max_area_ratio": 0.90,  # Can be large
        "max_saturation": 1.0,
        "min_value": 0.10,  # Roofs can be dark
    },
    "warehouse_roof": {
        # Large industrial/commercial roofs, kept separate for shape-specific tuning.
        "caption": "warehouse . warehouse roof . industrial building . factory roof . distribution center . logistics building",
        "box_threshold": 0.10,
        "text_threshold": 0.06,
        "keywords": ("warehouse", "industrial", "factory", "distribution", "logistics", "roof", "building"),
        "negative_keywords": ("car", "truck", "person", "tree", "grass"),
        "min_aspect_ratio": 0.15,
        "max_aspect_ratio": 40.0,
        "min_box_side_px": 16,
        "max_split_boxes_per_detection": 256,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 1500000,
        "negative_captions": [
            "parking lot",
            "road",
            "grass",
            "sports court",
        ],
        "clip_negative_weight": 0.32,
        "clip_top_k": 50,
        "clip_score_threshold": -0.10,
        "clip_relative_score_margin": 0.10,
        "clip_min_area_ratio": 0.0002,
        "clip_max_area_ratio": 0.95,
        "max_saturation": 1.0,
        "min_value": 0.08,
    },
    "sidewalk_surface": {
        # Sidewalks/walkways with explicit paved-surface semantics.
        "caption": "sidewalk . pedestrian sidewalk . concrete sidewalk . paved walkway . footpath . concrete path . asphalt walkway",
        "negative_dino_caption": "building roof . parking lot . highway lane . roadway",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.28,
        # DINO detection settings
        "box_threshold": 0.18,
        "text_threshold": 0.14,
        "keywords": ("sidewalk", "pedestrian", "walkway", "footpath", "pavement", "concrete", "asphalt"),
        "negative_keywords": ("building", "roof", "car", "highway", "parking", "grass", "field", "roadway"),
        "min_aspect_ratio": 0.6,
        "max_aspect_ratio": 20.0,  # Paths can be very elongated
        "min_box_side_px": 18,
        "max_split_boxes_per_detection": 96,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 120000,
        "negative_captions": [
            "building roof",
            "parking lot",
            "highway",
            "road lane",
            "tree canopy",
        ],
        "clip_negative_weight": 0.42,
        # SAM/CLIP evaluation settings
        "clip_top_k": 50,
        "clip_score_threshold": 0.0,
        "clip_relative_score_margin": 0.12,
        "clip_min_area_ratio": 0.00015,
        "clip_max_area_ratio": 0.45,
        "max_saturation": 1.0,
        "min_value": 0.18,
    },
    "road_surface": {
        # Roads/streets with explicit asphalt-vs-concrete phrasing for CLIP confirmation.
        "caption": "road . asphalt road . concrete road . street . roadway . traffic lane . paved road",
        "negative_dino_caption": "building roof . parking lot . sidewalk . pedestrian walkway",
        "negative_box_threshold": 0.34,
        "negative_text_threshold": 0.28,
        # DINO detection settings
        "box_threshold": 0.16,
        "text_threshold": 0.12,
        "keywords": ("road", "street", "lane", "roadway", "asphalt", "concrete", "paved"),
        "negative_keywords": ("building", "roof", "parking", "grass", "field", "sidewalk", "walkway"),
        "min_aspect_ratio": 0.35,
        "max_aspect_ratio": 40.0,
        "min_box_side_px": 20,
        "max_split_boxes_per_detection": 120,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 300000,
        "negative_captions": [
            "building roof",
            "parking lot",
            "sidewalk",
            "pedestrian path",
            "tree canopy",
        ],
        "clip_negative_weight": 0.44,
        # SAM/CLIP evaluation settings
        "clip_top_k": 60,
        "clip_score_threshold": -0.02,
        "clip_relative_score_margin": 0.12,
        "clip_min_area_ratio": 0.0002,
        "clip_max_area_ratio": 0.80,
        "max_saturation": 1.0,
        "min_value": 0.16,
    },
    "sidewalks": {
        # Backward-compatible alias settings (kept for old configs/scripts).
        "caption": "sidewalk . pedestrian path . walkway . footpath . pavement . concrete path . pedestrian sidewalk",
        "box_threshold": 0.20,
        "text_threshold": 0.15,
        "keywords": ("sidewalk", "pedestrian", "walkway", "footpath", "pavement", "path"),
        "negative_keywords": ("building", "roof", "car", "road", "highway", "parking", "grass", "field"),
        "min_aspect_ratio": 0.6,
        "max_aspect_ratio": 15.0,
        "min_box_side_px": 20,
        "max_split_boxes_per_detection": 96,
        "enable_area_split": False,
        "enable_tiled_fallback": True,
        "max_area_meters_sq": 100000,
        "negative_captions": ["parking lot", "highway", "grass field", "tree canopy"],
        "clip_negative_weight": 0.40,
        "clip_top_k": 50,
        "clip_score_threshold": 0.0,
        "clip_relative_score_margin": 0.12,
        "clip_min_area_ratio": 0.0002,
        "clip_max_area_ratio": 0.35,
        "max_saturation": 1.0,
        "min_value": 0.20,
    },
}
