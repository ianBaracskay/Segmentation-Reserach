from pathlib import Path

import groundingdino
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from groundingdino.util.inference import load_model, predict
from matplotlib.patches import Rectangle
from PIL import Image


def ensure_download(path: str, urls: list[str]) -> None:
    target = Path(path)
    if target.exists():
        return

    print(f"[INFO] Downloading model checkpoint: {target.name}")
    last_error: Exception | None = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with target.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"[INFO] Downloaded checkpoint to: {target.resolve()}")
            return
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download checkpoint: {target.name}") from last_error


def iter_tile_coords(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)
    coords: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            coords.append((y0, y1, x0, x1))
    return coords


def cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor, width: int, height: int) -> np.ndarray:
    boxes_scaled = boxes_cxcywh * torch.tensor([width, height, width, height])
    boxes_xyxy_local = torch.zeros_like(boxes_scaled)
    boxes_xyxy_local[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2.0
    boxes_xyxy_local[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2.0
    boxes_xyxy_local[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2.0
    boxes_xyxy_local[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2.0
    return boxes_xyxy_local.cpu().numpy().astype(int)


def _is_box_geometry_valid(box_xyxy: np.ndarray, image_shape: tuple[int, int, int], cfg: dict) -> bool:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    area_ratio = float(bw * bh) / float(max(1, w * h))
    min_area_ratio = float(cfg.get("min_area_ratio", 0.0))
    max_area_ratio = float(cfg.get("max_area_ratio", 1.0))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    aspect_ratio = max(bw, bh) / float(min(bw, bh))
    min_aspect_ratio = float(cfg.get("min_aspect_ratio", 1.0))
    if aspect_ratio < min_aspect_ratio:
        return False

    return True


def build_dino_model_and_transform(config) -> tuple[object, object, str]:
    dino_pkg_dir = Path(groundingdino.__file__).resolve().parent
    dino_config_path = dino_pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
    dino_checkpoint_path = Path("groundingdino_swint_ogc.pth")
    dino_checkpoint_urls = [
        "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/alpha/groundingdino_swint_ogc.pth",
    ]
    ensure_download(str(dino_checkpoint_path), dino_checkpoint_urls)

    dino_model = load_model(str(dino_config_path), str(dino_checkpoint_path))

    if config.dino_device == "auto":
        dino_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dino_device = config.dino_device

    dino_transform = T.Compose(
        [
            T.RandomResize([config.dino_resize_short_side], max_size=config.dino_resize_max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return dino_model, dino_transform, dino_device


def _run_dino_inference(
    image_rgb: np.ndarray,
    caption: str,
    box_th: float,
    text_th: float,
    dino_model,
    dino_transform,
    dino_device: str,
):
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = dino_transform(image_pil, None)
    boxes_local, logits_local, phrases_local = predict(
        model=dino_model,
        image=image_tensor,
        caption=caption,
        box_threshold=box_th,
        text_threshold=text_th,
        device=dino_device,
    )
    logits_np_local = (
        logits_local.detach().cpu().numpy()
        if hasattr(logits_local, "detach")
        else np.array(logits_local)
    )
    return boxes_local, logits_np_local, list(phrases_local)


def run_dino_prompts(img: np.ndarray, config, dino_model, dino_transform, dino_device: str) -> list[dict]:
    print("[INFO] Running prompt-by-prompt DINO detection")
    all_records: list[dict] = []

    for cfg in config.dino_prompt_configs:
        boxes_local, logits_local, phrases_local = _run_dino_inference(
            img,
            cfg["caption"],
            cfg["box_threshold"],
            cfg["text_threshold"],
            dino_model,
            dino_transform,
            dino_device,
        )
        boxes_xyxy_local = cxcywh_to_xyxy(boxes_local, img.shape[1], img.shape[0])
        print(f"[INFO] DINO [{cfg['name']}] full-image detections: {len(boxes_xyxy_local)}")

        use_tiled_fallback = bool(cfg.get("enable_tiled_fallback", True))
        prompt_records: list[dict] = [
            {
                "box": box_xyxy,
                "phrase": phrase,
                "score": float(score),
                "prompt_group": cfg["name"],
            }
            for box_xyxy, phrase, score in zip(boxes_xyxy_local, phrases_local, logits_local)
        ]

        if (
            config.dino_enable_tiled_fallback
            and use_tiled_fallback
            and len(prompt_records) == 0
        ):
            tile_records: list[dict] = []
            tile_coords = iter_tile_coords(
                img.shape[0],
                img.shape[1],
                config.dino_tile_size_px,
                config.dino_tile_overlap_px,
            )
            print(
                f"[INFO] DINO [{cfg['name']}] using tiled fallback "
                f"({len(tile_coords)} tiles, tile={config.dino_tile_size_px}, "
                f"overlap={config.dino_tile_overlap_px})"
            )

            for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
                tile_img = img[y0:y1, x0:x1]
                t_boxes, t_logits, t_phrases = _run_dino_inference(
                    tile_img,
                    cfg["caption"],
                    cfg["box_threshold"],
                    cfg["text_threshold"],
                    dino_model,
                    dino_transform,
                    dino_device,
                )
                t_xyxy = cxcywh_to_xyxy(t_boxes, tile_img.shape[1], tile_img.shape[0])

                for box_xyxy, phrase, score in zip(t_xyxy, t_phrases, t_logits):
                    gx1, gy1, gx2, gy2 = box_xyxy
                    tile_records.append(
                        {
                            "box": np.array([gx1 + x0, gy1 + y0, gx2 + x0, gy2 + y0], dtype=int),
                            "phrase": phrase,
                            "score": float(score),
                            "prompt_group": cfg["name"],
                        }
                    )

                if len(tile_records) >= config.dino_tiled_max_detections_per_prompt:
                    print(
                        f"[INFO] DINO [{cfg['name']}] stopping tiled fallback early at "
                        f"tile {tile_idx}/{len(tile_coords)} with {len(tile_records)} detections"
                    )
                    break

            if tile_records:
                print(f"[INFO] DINO [{cfg['name']}] tiled detections: {len(tile_records)}")
                prompt_records = tile_records

        keyword_filtered = [
            rec for rec in prompt_records if any(k in rec["phrase"].lower() for k in cfg["keywords"])
        ]
        if keyword_filtered:
            prompt_records = keyword_filtered

        geom_filtered = [
            rec for rec in prompt_records if _is_box_geometry_valid(rec["box"], img.shape, cfg)
        ]
        if geom_filtered:
            prompt_records = geom_filtered

        print(f"[INFO] DINO [{cfg['name']}] kept for SAM: {len(prompt_records)}")
        all_records.extend(prompt_records)

    return all_records


def save_dino_debug_image(img: np.ndarray, run_id: str, save_figure_fn) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("DINO Input Debug Image")
    save_figure_fn(f"{run_id}_dino_input_debug.png", "dino_detections")


def save_dino_detection_viz(img: np.ndarray, records: list[dict], run_id: str, save_figure_fn) -> None:
    dino_boxes_xyxy = np.stack([rec["box"] for rec in records]).astype(int)
    dino_phrases = [rec["phrase"] for rec in records]
    dino_logits = np.array([rec["score"] for rec in records], dtype=float)
    dino_prompt_groups = [rec["prompt_group"] for rec in records]

    dino_viz_stride = max(1, int(np.ceil(max(img.shape[:2]) / 2000)))
    dino_viz_img = img[::dino_viz_stride, ::dino_viz_stride]
    dino_group_colors = {"sidewalk": "lime", "sitting": "orange"}

    plt.figure(figsize=(10, 10))
    plt.imshow(dino_viz_img)
    ax = plt.gca()

    for box, phrase, score, group in zip(dino_boxes_xyxy, dino_phrases, dino_logits, dino_prompt_groups):
        x1, y1, x2, y2 = box.astype(float) / float(dino_viz_stride)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        color = dino_group_colors.get(group, "lime")

        ax.add_patch(
            Rectangle((x1, y1), box_w, box_h, fill=False, edgecolor=color, linewidth=1.3)
        )
        ax.text(
            x1,
            max(0.0, y1 - 3.0),
            f"[{group}] {phrase} {float(score):.2f}",
            color="white",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
        )

    plt.axis("off")
    plt.title("Grounding DINO Detections (Before SAM)")
    save_figure_fn(f"{run_id}_dino_detections.png", "dino_detections")
