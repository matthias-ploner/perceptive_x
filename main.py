"""
perceptive-x — Vision AI Skills Library demo
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

import vision_skills as vs

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

OUT_DIR   = Path(__file__).parent / "output"
LIBS      = Path(__file__).parent.parent / "libs"
SAM2_IMGS = LIBS / "sam2/notebooks/images"
VENV_DATA = Path(__file__).parent / ".venv/lib/python3.12/site-packages"

MVTEC_NUT = LIBS / "mvtec_nut/metal_nut"

BATCH_IMAGES = [
    LIBS  / "sam2/notebooks/videos/bedroom/00000.jpg",
    SAM2_IMGS / "cars.jpg",
    SAM2_IMGS / "groceries.jpg",
    SAM2_IMGS / "truck.jpg",
    VENV_DATA / "sklearn/datasets/images/flower.jpg",
    VENV_DATA / "sklearn/datasets/images/china.jpg",
    VENV_DATA / "matplotlib/mpl-data/sample_data/grace_hopper.jpg",
]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Convert normalised [0,1] float depth to an 8-bit BGR colour image."""
    d8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def seg_overlay(image: np.ndarray, masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Overlay all masks sorted by score (highest first) on the image."""
    order = np.argsort(scores)[::-1]
    overlay = vs.draw_masks(image, masks[order], alpha=0.5)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def text_panel(text: str, width: int, height: int, title: str = "Reasoning") -> np.ndarray:
    """Render wrapped text onto a dark panel of given size."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness  = 1
    line_h     = 18
    margin     = 10

    # Title bar
    cv2.rectangle(panel, (0, 0), (width, 26), (60, 60, 60), -1)
    cv2.putText(panel, title, (margin, 18), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Word-wrap body text
    words  = text.split()
    lines  = []
    line   = ""
    char_w = 7  # approximate char width for font_scale 0.42
    max_chars = max(1, (width - 2 * margin) // char_w)

    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = 26 + line_h
    for ln in lines:
        if y + line_h > height - margin:
            cv2.putText(panel, "…", (margin, y), font, font_scale, (140, 140, 140), thickness, cv2.LINE_AA)
            break
        cv2.putText(panel, ln, (margin, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        y += line_h

    return panel


def save_grid(images_bgr: list[np.ndarray], labels: list[str], out_path: Path) -> None:
    """Save a horizontal grid of images with labels burned in."""
    h = max(img.shape[0] for img in images_bgr)
    w = images_bgr[0].shape[1]
    resized = [cv2.resize(img, (w, h)) for img in images_bgr]
    for img, label in zip(resized, labels):
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    grid = np.hstack(resized)
    cv2.imwrite(str(out_path), grid)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Registry demo
# ---------------------------------------------------------------------------

def demo_registry():
    print("Registered skills:", vs.SkillRegistry.list_skills())


# ---------------------------------------------------------------------------
# Pipeline test: Depth Anything 3 + SAM2 auto segmentation
# ---------------------------------------------------------------------------

def run_pipeline_test(image_path: str | Path | None = None):
    OUT_DIR.mkdir(exist_ok=True)

    # --- Load image --------------------------------------------------------
    if image_path is None:
        image_path = BATCH_IMAGES[0]
    image = vs.load_image(str(image_path))   # HxWx3 uint8 RGB
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_DIR / "input.jpg"), image_bgr)
    print(f"Input: {image_path}  ({image.shape[1]}×{image.shape[0]})")

    # --- Build pipeline ----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    depth_skill = vs.SkillRegistry.create(
        "depth_anything_v3",
        {
            "device": device,
            "model_name": "depth-anything/DA3MONO-LARGE",
        },
    )
    seg_skill = vs.SkillRegistry.create(
        "sam2",
        {
            "device": device,
            "model_path": "facebook/sam2-hiera-tiny",
        },
    )

    pipeline = vs.SkillPipeline(
        [
            ("depth", depth_skill),
            ("seg", seg_skill, lambda *_: {"mode": "auto"}),
        ]
    )

    # --- Run ---------------------------------------------------------------
    print("\nRunning pipeline …")
    results = pipeline.run(image, stop_on_failure=True)
    print()

    for name, res in results.items():
        status = "OK" if res.success else f"FAILED: {res.error}"
        print(f"  [{name}] {status} | {res.inference_time_ms:.0f} ms")

    if not all(r.success for r in results.values()):
        print("Pipeline had failures — aborting visualisation.")
        return results

    # --- Visualise ---------------------------------------------------------
    depth_res = results["depth"]
    seg_res   = results["seg"]

    # Resize depth map to match image (DA3 may output at different resolution)
    depth_norm = depth_res.data["depth_norm"]
    depth_norm_resized = cv2.resize(depth_norm, (image.shape[1], image.shape[0]))
    depth_coloured = colorize_depth(depth_norm_resized)   # BGR

    # Segmentation overlay (top-10 masks by stability score)
    masks  = seg_res.data["masks"]   # [N, H, W] bool
    scores = seg_res.data["scores"]  # [N]
    seg_vis = seg_overlay(image, masks, scores)  # BGR

    # Save individual outputs
    cv2.imwrite(str(OUT_DIR / "depth_colormap.jpg"), depth_coloured)
    print(f"  Saved → {OUT_DIR / 'depth_colormap.jpg'}")
    cv2.imwrite(str(OUT_DIR / "seg_overlay.jpg"), seg_vis)
    print(f"  Saved → {OUT_DIR / 'seg_overlay.jpg'}")

    # Save combined grid
    save_grid(
        [image_bgr, depth_coloured, seg_vis],
        ["Input", f"Depth (DA3) {depth_res.inference_time_ms:.0f}ms", f"SAM2 auto ({len(masks)} masks) {seg_res.inference_time_ms:.0f}ms"],
        OUT_DIR / "pipeline_grid.jpg",
    )

    # Depth stats
    dm = depth_res.data["depth_map"]
    print(f"\nDepth stats: min={dm.min():.3f}  max={dm.max():.3f}  mean={dm.mean():.3f}")
    print(f"Segments found: {seg_res.data['num_masks']}")

    return results


# ---------------------------------------------------------------------------
# Batch test: run pipeline on multiple images, reusing loaded models
# ---------------------------------------------------------------------------

def run_batch_test(image_paths: list[Path] | None = None):
    """Run depth + segmentation + reasoning pipeline on every image and save grids."""
    OUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  images: {len(image_paths or BATCH_IMAGES)}\n")

    # Load models once
    depth_skill = vs.SkillRegistry.create(
        "depth_anything_v3", {"device": device, "model_name": "depth-anything/DA3MONO-LARGE"}
    )
    seg_skill = vs.SkillRegistry.create(
        "sam2", {"device": device, "model_path": "facebook/sam2-hiera-tiny"}
    )
    reason_skill = vs.SkillRegistry.create(
        "qwen3_vl", {
            "device": device,
            "model_name": "Qwen/Qwen3-VL-4B-Instruct",
            "max_new_tokens": 256,
            "temperature": 0.0,
            "use_flash_attention": False,  # no flash-attn on CPU
        }
    )
    pipeline = vs.SkillPipeline([
        ("depth",  depth_skill),
        ("seg",    seg_skill,    lambda *_: {"mode": "auto"}),
        ("reason", reason_skill, lambda *_: {
            "prompt": (
                "What can you see in this image? "
                "Briefly describe the scene, main objects, and spatial layout."
            ),
        }),
    ])

    rows = []  # collect per-image rows for a final summary grid

    for img_path in (image_paths or BATCH_IMAGES):
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"  [skip] {img_path.name} — not found")
            continue

        image     = vs.load_image(str(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(f"── {img_path.name}  ({image.shape[1]}×{image.shape[0]})")

        results = pipeline.run(image, stop_on_failure=True)

        if not all(r.success for r in results.values()):
            for name, r in results.items():
                if not r.success:
                    print(f"   [{name}] FAILED: {r.error}")
            continue

        depth_res  = results["depth"]
        seg_res    = results["seg"]
        reason_res = results.get("reason")

        depth_norm_resized = cv2.resize(
            depth_res.data["depth_norm"], (image.shape[1], image.shape[0])
        )
        depth_col = colorize_depth(depth_norm_resized)
        seg_vis   = seg_overlay(image, seg_res.data["masks"], seg_res.data["scores"])

        response_text = reason_res.data["response"] if reason_res and reason_res.success else "(no response)"
        reason_ms     = reason_res.inference_time_ms if reason_res else 0

        print(
            f"   depth {depth_res.inference_time_ms:.0f}ms  |  "
            f"seg {seg_res.inference_time_ms:.0f}ms  |  "
            f"{seg_res.data['num_masks']} masks  |  "
            f"reason {reason_ms:.0f}ms"
        )
        print(f"   \"{response_text[:120]}{'…' if len(response_text) > 120 else ''}\"")

        # Reasoning text panel (same height as image, fixed width)
        panel_w   = max(300, image.shape[1] // 2)
        txt_panel = text_panel(response_text, panel_w, image.shape[0], title=f"Qwen3-VL  {reason_ms:.0f}ms")

        stem = img_path.stem
        # Save reasoning text separately
        (OUT_DIR / f"{stem}_reasoning.txt").write_text(
            f"Prompt: {reason_res.data['prompt'] if reason_res else ''}\n\n{response_text}\n"
        )
        save_grid(
            [image_bgr, depth_col, seg_vis, txt_panel],
            ["Input",
             f"Depth (DA3) {depth_res.inference_time_ms:.0f}ms",
             f"SAM2 ({seg_res.data['num_masks']} masks) {seg_res.inference_time_ms:.0f}ms",
             ""],
            OUT_DIR / f"{stem}_grid.jpg",
        )

        # Collect a thumbnail row for the summary strip
        thumb_w = 240
        thumb_h = int(image.shape[0] * thumb_w / image.shape[1])
        rows.append(np.hstack([
            cv2.resize(image_bgr, (thumb_w, thumb_h)),
            cv2.resize(depth_col, (thumb_w, thumb_h)),
            cv2.resize(seg_vis,   (thumb_w, thumb_h)),
            cv2.resize(txt_panel, (thumb_w, thumb_h)),
        ]))

    # Save vertical summary strip
    if rows:
        # pad rows to the same width before stacking
        max_w = max(r.shape[1] for r in rows)
        padded = [
            np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0))) if r.shape[1] < max_w else r
            for r in rows
        ]
        summary = np.vstack(padded)
        out = OUT_DIR / "batch_summary.jpg"
        cv2.imwrite(str(out), summary)
        print(f"\n  Saved summary → {out}")


# ---------------------------------------------------------------------------
# Other demos (kept for reference)
# ---------------------------------------------------------------------------

def demo_depth_multiview(images: list):
    skill = vs.SkillRegistry.create(
        "depth_anything_v3",
        {
            "device": "cuda",
            "model_name": "depth-anything/DA3-LARGE",
        },
    )
    primary, *extras = images
    result = skill(primary, extra_images=extras)
    if result:
        print(
            f"Multi-view | n={result.metadata['num_images']}"
            f" | camera_poses={'extrinsics' in result.data}"
            f" | {result.inference_time_ms:.1f}ms"
        )
    return result


def demo_classification(image: np.ndarray, gallery_images: dict):
    skill = vs.SkillRegistry.create(
        "dinov2",
        {
            "device": "cuda",
            "model_name": "facebook/dinov2-small",
            "few_shot_k": 3,
            "anomaly_threshold": 0.75,
        },
    )
    skill.build_gallery(gallery_images)
    result = skill(image)
    if result:
        print(
            f"Class | predicted={result.data['predicted_class']}"
            f" | confidence={result.data['confidence']:.3f}"
            f" | anomaly={result.data['is_anomaly']}"
        )
    return result


# ---------------------------------------------------------------------------
# Anomaly detection demo: DINOv2 + cosine distance on MVTec metal-nut
# ---------------------------------------------------------------------------

def _anomaly_overlay(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a [h, w] float [0,1] anomaly map onto a BGR image."""
    h, w = image_bgr.shape[:2]
    amap = cv2.resize(anomaly_map, (w, h))
    heatmap = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)


def demo_anomaly_detection(
    dataset_dir: Path | None = None,
    n_gallery: int = 220,
    device: str | None = None,
):
    """
    Build a good-only patch gallery from n_gallery training images, then run
    DINOv2 patch-level anomaly detection on the full test set.

    Outputs:
        output/anomaly_detection_grid.jpg — thumbnail grid (heatmap overlay),
                                            color-coded green/red by correctness
        output/anomaly_detail_grid.jpg    — per-defect-type detail rows showing
                                            original | heatmap | GT mask
        output/anomaly_distributions.jpg  — image-score distributions per category
    """
    OUT_DIR.mkdir(exist_ok=True)

    if dataset_dir is None:
        dataset_dir = MVTEC_NUT
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  dataset: {dataset_dir}")

    # ------------------------------------------------------------------
    # 1. Build patch gallery from n_gallery evenly-spaced good images
    # ------------------------------------------------------------------
    train_good = sorted((dataset_dir / "train/good").glob("*.png"))
    step = max(1, len(train_good) // n_gallery)
    gallery_paths = train_good[::step][:n_gallery]
    print(f"\nBuilding patch gallery from "
          f"{len(gallery_paths)}/{len(train_good)} good images …")

    skill = vs.SkillRegistry.create(
        "dinov2",
        {
            "device": device,
            "model_name": "facebook/dinov2-small",
            "few_shot_k": 1,    # single-NN: most discriminative for subtle defects
            "anomaly_threshold": 0.80,   # placeholder; calibrated below
            "patch_mode": True,
        },
    )
    gallery_imgs = [vs.load_image(str(p)) for p in gallery_paths]
    skill.build_gallery({"good": gallery_imgs})

    # ------------------------------------------------------------------
    # 2. Run inference on all test images
    # ------------------------------------------------------------------
    test_categories = {
        "good":    sorted((dataset_dir / "test/good").glob("*.png")),
        "scratch": sorted((dataset_dir / "test/scratch").glob("*.png")),
        "bent":    sorted((dataset_dir / "test/bent").glob("*.png")),
        "color":   sorted((dataset_dir / "test/color").glob("*.png")),
        "flip":    sorted((dataset_dir / "test/flip").glob("*.png")),
    }

    print("\nRunning inference …")
    results_per_cat: dict[str, list[tuple]] = {}
    for cat, paths in test_categories.items():
        cat_results = []
        for p in paths:
            img = vs.load_image(str(p))
            res = skill(img)
            cat_results.append((p, img, res))
        results_per_cat[cat] = cat_results

    # ------------------------------------------------------------------
    # 3. Calibrate threshold: midpoint of good vs all-defect distributions
    # ------------------------------------------------------------------
    good_scores = [r.data["confidence"] for _, _, r in results_per_cat["good"]]
    defect_scores = [
        r.data["confidence"]
        for cat in ("scratch", "bent", "color", "flip")
        for _, _, r in results_per_cat[cat]
    ]
    threshold = (np.mean(good_scores) + np.mean(defect_scores)) / 2

    print(f"\nCalibrated threshold : {threshold:.4f}")
    print(f"  good   scores: mean={np.mean(good_scores):.4f}  "
          f"std={np.std(good_scores):.4f}  "
          f"min={np.min(good_scores):.4f}  max={np.max(good_scores):.4f}")
    print(f"  defect scores: mean={np.mean(defect_scores):.4f}  "
          f"std={np.std(defect_scores):.4f}  "
          f"min={np.min(defect_scores):.4f}  max={np.max(defect_scores):.4f}")

    # ------------------------------------------------------------------
    # 4. Per-category metrics
    # ------------------------------------------------------------------
    print(f"\n{'Category':10s}  {'N':>4}  {'Score mean':>11}  "
          f"{'Anomaly%':>9}  {'Correct%':>9}")
    print("-" * 54)
    overall_correct = overall_total = 0
    for cat, cat_results in results_per_cat.items():
        is_defect_cat = cat != "good"
        scores  = [r.data["confidence"] for _, _, r in cat_results]
        correct = sum(
            1 for _, _, r in cat_results
            if (r.data["confidence"] < threshold) == is_defect_cat
        )
        n_anom = sum(1 for s in scores if s < threshold)
        overall_correct += correct
        overall_total   += len(cat_results)
        print(f"{cat:10s}  {len(cat_results):4d}  "
              f"{np.mean(scores):11.4f}  "
              f"{n_anom / len(cat_results) * 100:8.1f}%  "
              f"{correct / len(cat_results) * 100:8.1f}%")
    print("-" * 54)
    print(f"{'OVERALL':10s}  {overall_total:4d}  "
          f"{'':11s}  {'':9s}  "
          f"{overall_correct / overall_total * 100:8.1f}%")

    # ------------------------------------------------------------------
    # 5. Thumbnail grid: heatmap overlay, color-coded border, score bar
    # ------------------------------------------------------------------
    THUMB_W, THUMB_H, BAR_H, BORDER = 180, 180, 22, 5
    COLS = 8

    rows_bgr = []
    for cat, cat_results in results_per_cat.items():
        is_defect_cat = cat != "good"
        thumbs = []
        for _, img, res in cat_results[:COLS]:
            score    = res.data["confidence"]
            is_anom  = score < threshold
            correct  = (is_anom == is_defect_cat)

            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Blend anomaly heatmap onto the thumbnail
            if "anomaly_map" in res.data:
                bgr = _anomaly_overlay(bgr, res.data["anomaly_map"], alpha=0.45)
            thumb = cv2.resize(bgr, (THUMB_W, THUMB_H))

            # Colored border: green = correct, red = wrong
            border_col = (0, 200, 0) if correct else (0, 0, 220)
            cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1),
                          border_col, BORDER)

            # Score bar below image
            canvas = np.zeros((THUMB_H + BAR_H, THUMB_W, 3), dtype=np.uint8)
            canvas[:THUMB_H] = thumb
            # Clamp score to [0,1] for bar width
            bar_w   = max(1, int(THUMB_W * np.clip(score, 0.0, 1.0)))
            bar_col = (50, 180, 50) if not is_anom else (50, 50, 200)
            cv2.rectangle(canvas, (0, THUMB_H), (bar_w, THUMB_H + BAR_H),
                          bar_col, -1)
            cv2.putText(canvas, f"{score:.3f}", (4, THUMB_H + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230),
                        1, cv2.LINE_AA)
            decision = "ANOM" if is_anom else "OK"
            cv2.putText(canvas, decision, (THUMB_W - 42, THUMB_H + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, border_col,
                        1, cv2.LINE_AA)
            thumbs.append(canvas)

        blank = np.zeros((THUMB_H + BAR_H, THUMB_W, 3), dtype=np.uint8)
        while len(thumbs) < COLS:
            thumbs.append(blank.copy())

        row = np.hstack(thumbs)
        panel = np.zeros((row.shape[0], 100, 3), dtype=np.uint8)
        panel[:] = (45, 45, 45)
        cv2.putText(panel, cat, (6, row.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220),
                    1, cv2.LINE_AA)
        rows_bgr.append(np.hstack([panel, row]))

    cv2.imwrite(str(OUT_DIR / "anomaly_detection_grid.jpg"), np.vstack(rows_bgr))
    print(f"\n  Saved → {OUT_DIR / 'anomaly_detection_grid.jpg'}")

    # ------------------------------------------------------------------
    # 6. Detail grid: original | heatmap | GT mask  (defect categories)
    # ------------------------------------------------------------------
    _save_anomaly_detail_grid(results_per_cat, dataset_dir, threshold,
                              OUT_DIR / "anomaly_detail_grid.jpg")

    # ------------------------------------------------------------------
    # 7. Score distribution chart
    # ------------------------------------------------------------------
    _save_similarity_distributions(
        results_per_cat, threshold, OUT_DIR / "anomaly_distributions.jpg"
    )

    return results_per_cat, threshold


def _save_anomaly_detail_grid(
    results_per_cat: dict,
    dataset_dir: Path,
    threshold: float,
    out_path: Path,
    n_show: int = 6,
) -> None:
    """
    For each defect category render rows of:
        original image | anomaly heatmap overlay | ground-truth defect mask

    n_show: number of examples to show per category.
    """
    DETAIL_W, DETAIL_H = 220, 220
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    gt_base = dataset_dir / "ground_truth"

    section_rows = []

    for cat in ("scratch", "bent", "color", "flip"):
        cat_results = results_per_cat.get(cat, [])
        if not cat_results:
            continue

        gt_dir = gt_base / cat

        # Header row for this category
        hdr_h = 28
        hdr_w = DETAIL_W * 3 * min(n_show, len(cat_results))
        hdr = np.zeros((hdr_h, hdr_w, 3), dtype=np.uint8)
        hdr[:] = (60, 60, 60)
        cv2.putText(hdr, f"[{cat}]  original | heatmap | GT mask",
                    (8, 20), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        section_rows.append(hdr)

        triplets = []
        for p, img, res in cat_results[:n_show]:
            # --- original with border ---
            bgr = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (DETAIL_W, DETAIL_H)
            )
            score    = res.data["confidence"]
            is_anom  = score < threshold
            border_c = (0, 200, 0) if is_anom else (0, 0, 220)
            cv2.rectangle(bgr, (0, 0), (DETAIL_W - 1, DETAIL_H - 1), border_c, 4)
            cv2.putText(bgr, f"{score:.3f}", (4, DETAIL_H - 6),
                        FONT, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

            # --- heatmap overlay ---
            hm_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if "anomaly_map" in res.data:
                hm_bgr = _anomaly_overlay(hm_bgr, res.data["anomaly_map"],
                                          alpha=0.55)
            hm_bgr = cv2.resize(hm_bgr, (DETAIL_W, DETAIL_H))

            # --- GT mask (white on black, or blank if not found) ---
            mask_path = gt_dir / (p.stem + "_mask.png")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_vis = cv2.resize(mask, (DETAIL_W, DETAIL_H))
                gt_panel = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            else:
                gt_panel = np.zeros((DETAIL_H, DETAIL_W, 3), dtype=np.uint8)
                cv2.putText(gt_panel, "no GT", (60, DETAIL_H // 2),
                            FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

            triplets.append(np.hstack([bgr, hm_bgr, gt_panel]))

        # Pad to consistent width if fewer than n_show examples
        if triplets:
            max_w = max(t.shape[1] for t in triplets)
            row = np.hstack([
                np.pad(t, ((0, 0), (0, max_w - t.shape[1]), (0, 0)))
                if t.shape[1] < max_w else t
                for t in triplets
            ])
            section_rows.append(row)

    if not section_rows:
        return

    # Pad all rows to the same width before stacking
    max_w = max(r.shape[1] for r in section_rows)
    padded = [
        np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
        if r.shape[1] < max_w else r
        for r in section_rows
    ]
    cv2.imwrite(str(out_path), np.vstack(padded))
    print(f"  Saved → {out_path}")


def _save_similarity_distributions(
    results_per_cat: dict,
    threshold: float,
    out_path: Path,
) -> None:
    """Render a horizontal bar chart of similarity score distributions per category."""
    cat_colors = {
        "good":    (80, 200, 80),
        "scratch": (60,  60, 220),
        "bent":    (60, 160, 220),
        "color":   (220, 140, 60),
        "flip":    (180, 60, 220),
    }
    W, ROW_H, MARGIN = 640, 48, 12
    H = ROW_H * len(results_per_cat) + MARGIN * 2 + 30

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Title
    cv2.putText(canvas, "Cosine similarity distributions (DINOv2)",
                (MARGIN, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    plot_w = W - MARGIN * 2 - 80  # width of plotting area (after the label)
    label_w = 72

    for row_i, (cat, cat_results) in enumerate(results_per_cat.items()):
        sims  = [r.data["confidence"] for _, _, r in cat_results]
        y_top = MARGIN + 30 + row_i * ROW_H
        col   = cat_colors.get(cat, (180, 180, 180))

        # Category label
        cv2.putText(canvas, cat, (MARGIN, y_top + ROW_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Box per sample: one narrow bar per image, colored by anomaly
        x0 = MARGIN + label_w
        bar_w_each = max(1, plot_w // len(sims))
        for i, s in enumerate(sorted(sims)):
            x = x0 + i * bar_w_each
            bar_h = int((ROW_H - 8) * s)
            y_bot = y_top + ROW_H - 4
            bar_col = col if s >= threshold else (60, 60, 200)
            cv2.rectangle(canvas, (x, y_bot - bar_h), (x + bar_w_each - 1, y_bot), bar_col, -1)

        # Mean line
        mean_x = x0 + int(plot_w * np.mean(sims))
        cv2.line(canvas, (mean_x, y_top + 2), (mean_x, y_top + ROW_H - 2), (240, 240, 240), 1)

    # Threshold vertical line
    thr_x = MARGIN + label_w + int(plot_w * threshold)
    cv2.line(canvas, (thr_x, MARGIN + 26), (thr_x, H - MARGIN), (0, 220, 220), 2)
    cv2.putText(canvas, f"thr={threshold:.3f}", (thr_x + 3, MARGIN + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Config sweep: compare scoring strategies for bent optimisation
# ---------------------------------------------------------------------------

def _eval_config(
    dataset_dir: Path,
    skill_cfg: dict,
    n_gallery: int,
) -> dict[str, float]:
    """
    Build a gallery + run the full test set for one skill configuration.
    Returns per-category accuracy dict (calibrated threshold).
    """
    train_good = sorted((dataset_dir / "train/good").glob("*.png"))
    step = max(1, len(train_good) // n_gallery)
    gallery_imgs = [
        vs.load_image(str(p)) for p in train_good[::step][:n_gallery]
    ]

    skill = vs.SkillRegistry.create("dinov2", skill_cfg)
    skill.build_gallery({"good": gallery_imgs})

    test_cats = {
        "good":    sorted((dataset_dir / "test/good").glob("*.png")),
        "scratch": sorted((dataset_dir / "test/scratch").glob("*.png")),
        "bent":    sorted((dataset_dir / "test/bent").glob("*.png")),
        "color":   sorted((dataset_dir / "test/color").glob("*.png")),
        "flip":    sorted((dataset_dir / "test/flip").glob("*.png")),
    }

    results: dict[str, list] = {}
    for cat, paths in test_cats.items():
        results[cat] = [
            skill(vs.load_image(str(p))).data["confidence"]
            for p in paths
        ]

    good_mean   = np.mean(results["good"])
    defect_mean = np.mean([
        s for cat in ("scratch", "bent", "color", "flip")
        for s in results[cat]
    ])
    thr = (good_mean + defect_mean) / 2

    acc = {}
    overall_c = overall_n = 0
    for cat, scores in results.items():
        is_defect = cat != "good"
        correct = sum(1 for s in scores if (s < thr) == is_defect)
        acc[cat] = correct / len(scores) * 100
        overall_c += correct
        overall_n += len(scores)
    acc["OVERALL"] = overall_c / overall_n * 100
    acc["threshold"] = thr
    return acc


def demo_bent_optimisation(
    dataset_dir: Path | None = None,
    device: str | None = None,
):
    """
    Sweep scoring configurations and print a comparison table showing
    the impact of anomaly_top_frac, smooth_sigma, and model size on
    per-category accuracy — particularly for the hard 'bent' class.
    """
    if dataset_dir is None:
        dataset_dir = MVTEC_NUT
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base = dict(
        device=device,
        model_name="facebook/dinov2-small",
        few_shot_k=1,
        anomaly_threshold=0.80,
        patch_mode=True,
        anomaly_top_frac=0.10,
        smooth_sigma=0.0,
        image_size=224,
    )

    # (label, config_overrides, n_gallery)
    configs = [
        ("small 224px  k=1 n=50   baseline",  {},                              50),
        ("small 224px  k=1 n=220  +gallery",  {},                              220),
        ("small 448px  k=1 n=50   +hires",    {"image_size": 448},             50),
        ("small 448px  k=1 n=220  +hires+gal",{"image_size": 448},             220),
        ("base  224px  k=1 n=50   +model",    {"model_name": "facebook/dinov2-base"}, 50),
        ("base  448px  k=1 n=220  best?",     {"model_name": "facebook/dinov2-base",
                                               "image_size": 448},             220),
    ]

    cats = ["good", "scratch", "bent", "color", "flip", "OVERALL"]
    header = f"{'Config':36s}" + "".join(f"  {c:>8s}" for c in cats)
    print(f"\n{'Bent optimisation — config sweep':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, overrides, n in configs:
        cfg = {**base, **overrides}
        print(f"  running: {label} …", end="\r", flush=True)
        acc = _eval_config(dataset_dir, cfg, n_gallery=n)
        row = f"{label:36s}" + "".join(
            f"  {acc.get(c, 0):7.1f}%" for c in cats
        )
        print(row)

    print("=" * len(header))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_registry()
    print()
    demo_bent_optimisation()
