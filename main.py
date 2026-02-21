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

OUT_DIR = Path(__file__).parent / "output"
LIBS = Path(__file__).parent.parent / "libs"
SAM2_IMGS = LIBS / "sam2/notebooks/images"
VENV_DATA = Path(__file__).parent / ".venv/lib/python3.12/site-packages"

MVTEC_NUT = LIBS / "mvtec_nut/metal_nut"

BATCH_IMAGES = [
    LIBS / "sam2/notebooks/videos/bedroom/00000.jpg",
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


def text_panel(
    text: str, width: int, height: int, title: str = "Reasoning"
) -> np.ndarray:
    """Render wrapped text onto a dark panel of given size."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_h = 18
    margin = 10

    # Title bar
    cv2.rectangle(panel, (0, 0), (width, 26), (60, 60, 60), -1)
    cv2.putText(panel, title, (margin, 18), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Word-wrap body text
    words = text.split()
    lines = []
    line = ""
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
            cv2.putText(
                panel,
                "…",
                (margin, y),
                font,
                font_scale,
                (140, 140, 140),
                thickness,
                cv2.LINE_AA,
            )
            break
        cv2.putText(
            panel,
            ln,
            (margin, y),
            font,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    return panel


def save_grid(images_bgr: list[np.ndarray], labels: list[str], out_path: Path) -> None:
    """Save a horizontal grid of images with labels burned in."""
    h = max(img.shape[0] for img in images_bgr)
    w = images_bgr[0].shape[1]
    resized = [cv2.resize(img, (w, h)) for img in images_bgr]
    for img, label in zip(resized, labels):
        cv2.putText(
            img,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
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
    image = vs.load_image(str(image_path))  # HxWx3 uint8 RGB
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
    seg_res = results["seg"]

    # Resize depth map to match image (DA3 may output at different resolution)
    depth_norm = depth_res.data["depth_norm"]
    depth_norm_resized = cv2.resize(depth_norm, (image.shape[1], image.shape[0]))
    depth_coloured = colorize_depth(depth_norm_resized)  # BGR

    # Segmentation overlay (top-10 masks by stability score)
    masks = seg_res.data["masks"]  # [N, H, W] bool
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
        [
            "Input",
            f"Depth (DA3) {depth_res.inference_time_ms:.0f}ms",
            f"SAM2 auto ({len(masks)} masks) {seg_res.inference_time_ms:.0f}ms",
        ],
        OUT_DIR / "pipeline_grid.jpg",
    )

    # Depth stats
    dm = depth_res.data["depth_map"]
    print(
        f"\nDepth stats: min={dm.min():.3f}  max={dm.max():.3f}  mean={dm.mean():.3f}"
    )
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
        "depth_anything_v3",
        {"device": device, "model_name": "depth-anything/DA3MONO-LARGE"},
    )
    seg_skill = vs.SkillRegistry.create(
        "sam2", {"device": device, "model_path": "facebook/sam2-hiera-tiny"}
    )
    reason_skill = vs.SkillRegistry.create(
        "qwen3_vl",
        {
            "device": device,
            "model_name": "Qwen/Qwen3-VL-4B-Instruct",
            "max_new_tokens": 256,
            "temperature": 0.0,
            "use_flash_attention": False,  # no flash-attn on CPU
        },
    )
    pipeline = vs.SkillPipeline(
        [
            ("depth", depth_skill),
            ("seg", seg_skill, lambda *_: {"mode": "auto"}),
            (
                "reason",
                reason_skill,
                lambda *_: {
                    "prompt": (
                        "What can you see in this image? "
                        "Briefly describe the scene, main objects, and spatial layout."
                    ),
                },
            ),
        ]
    )

    rows = []  # collect per-image rows for a final summary grid

    for img_path in image_paths or BATCH_IMAGES:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"  [skip] {img_path.name} — not found")
            continue

        image = vs.load_image(str(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(f"── {img_path.name}  ({image.shape[1]}×{image.shape[0]})")

        results = pipeline.run(image, stop_on_failure=True)

        if not all(r.success for r in results.values()):
            for name, r in results.items():
                if not r.success:
                    print(f"   [{name}] FAILED: {r.error}")
            continue

        depth_res = results["depth"]
        seg_res = results["seg"]
        reason_res = results.get("reason")

        depth_norm_resized = cv2.resize(
            depth_res.data["depth_norm"], (image.shape[1], image.shape[0])
        )
        depth_col = colorize_depth(depth_norm_resized)
        seg_vis = seg_overlay(image, seg_res.data["masks"], seg_res.data["scores"])

        response_text = (
            reason_res.data["response"]
            if reason_res and reason_res.success
            else "(no response)"
        )
        reason_ms = reason_res.inference_time_ms if reason_res else 0

        print(
            f"   depth {depth_res.inference_time_ms:.0f}ms  |  "
            f"seg {seg_res.inference_time_ms:.0f}ms  |  "
            f"{seg_res.data['num_masks']} masks  |  "
            f"reason {reason_ms:.0f}ms"
        )
        print(f"   \"{response_text[:120]}{'…' if len(response_text) > 120 else ''}\"")

        # Reasoning text panel (same height as image, fixed width)
        panel_w = max(300, image.shape[1] // 2)
        txt_panel = text_panel(
            response_text, panel_w, image.shape[0], title=f"Qwen3-VL  {reason_ms:.0f}ms"
        )

        stem = img_path.stem
        # Save reasoning text separately
        (OUT_DIR / f"{stem}_reasoning.txt").write_text(
            f"Prompt: {reason_res.data['prompt'] if reason_res else ''}\n\n{response_text}\n"
        )
        save_grid(
            [image_bgr, depth_col, seg_vis, txt_panel],
            [
                "Input",
                f"Depth (DA3) {depth_res.inference_time_ms:.0f}ms",
                f"SAM2 ({seg_res.data['num_masks']} masks) {seg_res.inference_time_ms:.0f}ms",
                "",
            ],
            OUT_DIR / f"{stem}_grid.jpg",
        )

        # Collect a thumbnail row for the summary strip
        thumb_w = 240
        thumb_h = int(image.shape[0] * thumb_w / image.shape[1])
        rows.append(
            np.hstack(
                [
                    cv2.resize(image_bgr, (thumb_w, thumb_h)),
                    cv2.resize(depth_col, (thumb_w, thumb_h)),
                    cv2.resize(seg_vis, (thumb_w, thumb_h)),
                    cv2.resize(txt_panel, (thumb_w, thumb_h)),
                ]
            )
        )

    # Save vertical summary strip
    if rows:
        # pad rows to the same width before stacking
        max_w = max(r.shape[1] for r in rows)
        padded = [
            (
                np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
                if r.shape[1] < max_w
                else r
            )
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
    print(
        f"\nBuilding patch gallery from "
        f"{len(gallery_paths)}/{len(train_good)} good images …"
    )

    skill = vs.SkillRegistry.create(
        "dinov2",
        {
            "device": device,
            "model_name": "facebook/dinov2-small",
            "few_shot_k": 1,  # single-NN: most discriminative for subtle defects
            "anomaly_threshold": 0.80,  # placeholder; calibrated below
            "patch_mode": True,
        },
    )
    gallery_imgs = [vs.load_image(str(p)) for p in gallery_paths]
    skill.build_gallery({"good": gallery_imgs})

    # ------------------------------------------------------------------
    # 2. Run inference on all test images
    # ------------------------------------------------------------------
    test_categories = {
        "good": sorted((dataset_dir / "test/good").glob("*.png")),
        "scratch": sorted((dataset_dir / "test/scratch").glob("*.png")),
        "bent": sorted((dataset_dir / "test/bent").glob("*.png")),
        "color": sorted((dataset_dir / "test/color").glob("*.png")),
        "flip": sorted((dataset_dir / "test/flip").glob("*.png")),
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
    print(
        f"  good   scores: mean={np.mean(good_scores):.4f}  "
        f"std={np.std(good_scores):.4f}  "
        f"min={np.min(good_scores):.4f}  max={np.max(good_scores):.4f}"
    )
    print(
        f"  defect scores: mean={np.mean(defect_scores):.4f}  "
        f"std={np.std(defect_scores):.4f}  "
        f"min={np.min(defect_scores):.4f}  max={np.max(defect_scores):.4f}"
    )

    # ------------------------------------------------------------------
    # 4. Per-category metrics
    # ------------------------------------------------------------------
    print(
        f"\n{'Category':10s}  {'N':>4}  {'Score mean':>11}  "
        f"{'Anomaly%':>9}  {'Correct%':>9}"
    )
    print("-" * 54)
    overall_correct = overall_total = 0
    for cat, cat_results in results_per_cat.items():
        is_defect_cat = cat != "good"
        scores = [r.data["confidence"] for _, _, r in cat_results]
        correct = sum(
            1
            for _, _, r in cat_results
            if (r.data["confidence"] < threshold) == is_defect_cat
        )
        n_anom = sum(1 for s in scores if s < threshold)
        overall_correct += correct
        overall_total += len(cat_results)
        print(
            f"{cat:10s}  {len(cat_results):4d}  "
            f"{np.mean(scores):11.4f}  "
            f"{n_anom / len(cat_results) * 100:8.1f}%  "
            f"{correct / len(cat_results) * 100:8.1f}%"
        )
    print("-" * 54)
    print(
        f"{'OVERALL':10s}  {overall_total:4d}  "
        f"{'':11s}  {'':9s}  "
        f"{overall_correct / overall_total * 100:8.1f}%"
    )

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
            score = res.data["confidence"]
            is_anom = score < threshold
            correct = is_anom == is_defect_cat

            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Blend anomaly heatmap onto the thumbnail
            if "anomaly_map" in res.data:
                bgr = _anomaly_overlay(bgr, res.data["anomaly_map"], alpha=0.45)
            thumb = cv2.resize(bgr, (THUMB_W, THUMB_H))

            # Colored border: green = correct, red = wrong
            border_col = (0, 200, 0) if correct else (0, 0, 220)
            cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), border_col, BORDER)

            # Score bar below image
            canvas = np.zeros((THUMB_H + BAR_H, THUMB_W, 3), dtype=np.uint8)
            canvas[:THUMB_H] = thumb
            # Clamp score to [0,1] for bar width
            bar_w = max(1, int(THUMB_W * np.clip(score, 0.0, 1.0)))
            bar_col = (50, 180, 50) if not is_anom else (50, 50, 200)
            cv2.rectangle(canvas, (0, THUMB_H), (bar_w, THUMB_H + BAR_H), bar_col, -1)
            cv2.putText(
                canvas,
                f"{score:.3f}",
                (4, THUMB_H + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            decision = "ANOM" if is_anom else "OK"
            cv2.putText(
                canvas,
                decision,
                (THUMB_W - 42, THUMB_H + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                border_col,
                1,
                cv2.LINE_AA,
            )
            thumbs.append(canvas)

        blank = np.zeros((THUMB_H + BAR_H, THUMB_W, 3), dtype=np.uint8)
        while len(thumbs) < COLS:
            thumbs.append(blank.copy())

        row = np.hstack(thumbs)
        panel = np.zeros((row.shape[0], 100, 3), dtype=np.uint8)
        panel[:] = (45, 45, 45)
        cv2.putText(
            panel,
            cat,
            (6, row.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        rows_bgr.append(np.hstack([panel, row]))

    cv2.imwrite(str(OUT_DIR / "anomaly_detection_grid.jpg"), np.vstack(rows_bgr))
    print(f"\n  Saved → {OUT_DIR / 'anomaly_detection_grid.jpg'}")

    # ------------------------------------------------------------------
    # 6. Detail grid: original | heatmap | GT mask  (defect categories)
    # ------------------------------------------------------------------
    _save_anomaly_detail_grid(
        results_per_cat, dataset_dir, threshold, OUT_DIR / "anomaly_detail_grid.jpg"
    )

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
        cv2.putText(
            hdr,
            f"[{cat}]  original | heatmap | GT mask",
            (8, 20),
            FONT,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        section_rows.append(hdr)

        triplets = []
        for p, img, res in cat_results[:n_show]:
            # --- original with border ---
            bgr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (DETAIL_W, DETAIL_H))
            score = res.data["confidence"]
            is_anom = score < threshold
            border_c = (0, 200, 0) if is_anom else (0, 0, 220)
            cv2.rectangle(bgr, (0, 0), (DETAIL_W - 1, DETAIL_H - 1), border_c, 4)
            cv2.putText(
                bgr,
                f"{score:.3f}",
                (4, DETAIL_H - 6),
                FONT,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            # --- heatmap overlay ---
            hm_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if "anomaly_map" in res.data:
                hm_bgr = _anomaly_overlay(hm_bgr, res.data["anomaly_map"], alpha=0.55)
            hm_bgr = cv2.resize(hm_bgr, (DETAIL_W, DETAIL_H))

            # --- GT mask (white on black, or blank if not found) ---
            mask_path = gt_dir / (p.stem + "_mask.png")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_vis = cv2.resize(mask, (DETAIL_W, DETAIL_H))
                gt_panel = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            else:
                gt_panel = np.zeros((DETAIL_H, DETAIL_W, 3), dtype=np.uint8)
                cv2.putText(
                    gt_panel,
                    "no GT",
                    (60, DETAIL_H // 2),
                    FONT,
                    0.5,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA,
                )

            triplets.append(np.hstack([bgr, hm_bgr, gt_panel]))

        # Pad to consistent width if fewer than n_show examples
        if triplets:
            max_w = max(t.shape[1] for t in triplets)
            row = np.hstack(
                [
                    (
                        np.pad(t, ((0, 0), (0, max_w - t.shape[1]), (0, 0)))
                        if t.shape[1] < max_w
                        else t
                    )
                    for t in triplets
                ]
            )
            section_rows.append(row)

    if not section_rows:
        return

    # Pad all rows to the same width before stacking
    max_w = max(r.shape[1] for r in section_rows)
    padded = [
        (
            np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
            if r.shape[1] < max_w
            else r
        )
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
        "good": (80, 200, 80),
        "scratch": (60, 60, 220),
        "bent": (60, 160, 220),
        "color": (220, 140, 60),
        "flip": (180, 60, 220),
    }
    W, ROW_H, MARGIN = 640, 48, 12
    H = ROW_H * len(results_per_cat) + MARGIN * 2 + 30

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Title
    cv2.putText(
        canvas,
        "Cosine similarity distributions (DINOv2)",
        (MARGIN, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    plot_w = W - MARGIN * 2 - 80  # width of plotting area (after the label)
    label_w = 72

    for row_i, (cat, cat_results) in enumerate(results_per_cat.items()):
        sims = [r.data["confidence"] for _, _, r in cat_results]
        y_top = MARGIN + 30 + row_i * ROW_H
        col = cat_colors.get(cat, (180, 180, 180))

        # Category label
        cv2.putText(
            canvas,
            cat,
            (MARGIN, y_top + ROW_H // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

        # Box per sample: one narrow bar per image, colored by anomaly
        x0 = MARGIN + label_w
        bar_w_each = max(1, plot_w // len(sims))
        for i, s in enumerate(sorted(sims)):
            x = x0 + i * bar_w_each
            bar_h = int((ROW_H - 8) * s)
            y_bot = y_top + ROW_H - 4
            bar_col = col if s >= threshold else (60, 60, 200)
            cv2.rectangle(
                canvas, (x, y_bot - bar_h), (x + bar_w_each - 1, y_bot), bar_col, -1
            )

        # Mean line
        mean_x = x0 + int(plot_w * np.mean(sims))
        cv2.line(
            canvas, (mean_x, y_top + 2), (mean_x, y_top + ROW_H - 2), (240, 240, 240), 1
        )

    # Threshold vertical line
    thr_x = MARGIN + label_w + int(plot_w * threshold)
    cv2.line(canvas, (thr_x, MARGIN + 26), (thr_x, H - MARGIN), (0, 220, 220), 2)
    cv2.putText(
        canvas,
        f"thr={threshold:.3f}",
        (thr_x + 3, MARGIN + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 220, 220),
        1,
        cv2.LINE_AA,
    )

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
    gallery_imgs = [vs.load_image(str(p)) for p in train_good[::step][:n_gallery]]

    skill = vs.SkillRegistry.create("dinov2", skill_cfg)
    skill.build_gallery({"good": gallery_imgs})

    test_cats = {
        "good": sorted((dataset_dir / "test/good").glob("*.png")),
        "scratch": sorted((dataset_dir / "test/scratch").glob("*.png")),
        "bent": sorted((dataset_dir / "test/bent").glob("*.png")),
        "color": sorted((dataset_dir / "test/color").glob("*.png")),
        "flip": sorted((dataset_dir / "test/flip").glob("*.png")),
    }

    results: dict[str, list] = {}
    for cat, paths in test_cats.items():
        results[cat] = [skill(vs.load_image(str(p))).data["confidence"] for p in paths]

    good_mean = np.mean(results["good"])
    defect_mean = np.mean(
        [s for cat in ("scratch", "bent", "color", "flip") for s in results[cat]]
    )
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
        ("small 224px  k=1 n=50   baseline", {}, 50),
        ("small 224px  k=1 n=220  +gallery", {}, 220),
        ("small 448px  k=1 n=50   +hires", {"image_size": 448}, 50),
        ("small 448px  k=1 n=220  +hires+gal", {"image_size": 448}, 220),
        ("base  224px  k=1 n=50   +model", {"model_name": "facebook/dinov2-base"}, 50),
        (
            "base  448px  k=1 n=220  best?",
            {"model_name": "facebook/dinov2-base", "image_size": 448},
            220,
        ),
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
        row = f"{label:36s}" + "".join(f"  {acc.get(c, 0):7.1f}%" for c in cats)
        print(row)

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Visualisation helper for bounding boxes
# ---------------------------------------------------------------------------


def draw_detections(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    color: tuple = (0, 200, 50),
    thickness: int = 2,
) -> np.ndarray:
    """Draw labelled bounding boxes on a BGR image."""
    vis = image_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis,
            txt,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
    return vis


# ---------------------------------------------------------------------------
# Grounding DINO demo: text-prompted detection
# ---------------------------------------------------------------------------


def demo_grounding_dino(
    image_path: str | Path | None = None,
    text_prompt: str | None = None,
    device: str | None = None,
):
    """
    Industrial inspection demo: detect machine components in an image.

    Default: truck image → detect tire, wheel, headlight, bumper, mirror.
    Pass image_path + text_prompt to use your own image.

    Output: output/grounding_dino_detections.jpg
    """
    OUT_DIR.mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Default: truck (vehicle/industrial inspection task)
    if image_path is None:
        image_path = SAM2_IMGS / "truck.jpg"
        text_prompt = text_prompt or (
            "tire . wheel . headlight . bumper . side mirror . axle"
        )
    else:
        image_path = Path(image_path)
        text_prompt = text_prompt or "object"

    if not Path(image_path).exists():
        print(f"  [skip] Image not found: {image_path}")
        return

    print(f"\n── Grounding DINO  (vehicle inspection)  |  device={device}")
    print(f"   image  : {Path(image_path).name}")
    print(f"   prompt : '{text_prompt}'")

    image = vs.load_image(str(image_path))
    skill = vs.SkillRegistry.create(
        "grounding_dino",
        {
            "device": device,
            "box_threshold": 0.30,
            "text_threshold": 0.25,
        },
    )
    result = skill(image, text_prompt=text_prompt)

    if not result.success:
        print(f"   FAILED: {result.error}")
        return result

    n = result.data["n_detections"]
    print(f"\n   Detections: {n}  ({result.inference_time_ms:.0f} ms)")
    for box, score, label in zip(
        result.data["boxes"], result.data["scores"], result.data["labels"]
    ):
        x1, y1, x2, y2 = box.astype(int)
        print(f"     {label:22s}  score={score:.3f}  " f"box=[{x1},{y1},{x2},{y2}]")

    # --- visualise --------------------------------------------------------
    vis = draw_detections(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        result.data["boxes"],
        result.data["scores"],
        result.data["labels"],
    )
    out_path = OUT_DIR / "grounding_dino_detections.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\n   Saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Assembly verification demo
# ---------------------------------------------------------------------------


def _build_nut_tray(
    nut_dir: Path,
    n_cols: int = 4,
    n_rows: int = 2,
    cell_px: int = 256,
    n_missing: int = 0,
) -> np.ndarray | None:
    """
    Tile individual MVTec metal-nut images into a parts-tray grid (BGR).

    n_missing: leave this many trailing slots empty (dark) to simulate
               missing parts for the assembly verification demo.
    Returns None if nut_dir does not exist.
    """
    if not nut_dir.exists():
        return None
    total = n_cols * n_rows
    paths = sorted(nut_dir.glob("*.png"))[: total - n_missing]
    if not paths:
        return None
    cells = []
    for p in paths:
        img = cv2.imread(str(p))
        cells.append(cv2.resize(img, (cell_px, cell_px)))
    # Empty slot: dark grey rectangle with a dashed border to signal absence
    blank = np.full((cell_px, cell_px, 3), 30, dtype=np.uint8)
    cv2.rectangle(blank, (8, 8), (cell_px - 9, cell_px - 9), (80, 80, 80), 2)
    cv2.putText(
        blank,
        "MISSING",
        (cell_px // 2 - 38, cell_px // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    while len(cells) < total:
        cells.append(blank.copy())
    rows = [np.hstack(cells[r * n_cols : (r + 1) * n_cols]) for r in range(n_rows)]
    return np.vstack(rows)


def demo_assembly_verification(
    image_path: str | Path | None = None,
    checklist: dict | None = None,
    device: str | None = None,
):
    """
    Industrial parts-tray verification: count metal nuts on an inspection tray.

    Default: builds a 4×2 tray from MVTec training nuts and verifies 8 nuts
    are present.  Introduce a defect by removing one cell to see a FAIL.

    Pass image_path + checklist to use your own image/part list.

    Output: output/assembly_verification.jpg
             output/assembly_tray.jpg  (the generated tray, for reference)
    """
    OUT_DIR.mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tray_bgr = None
    if image_path is None:
        # Build a 4×2 tray of good metal nuts from MVTec
        nut_good_dir = MVTEC_NUT / "train/good"
        # 4×2 tray with 2 slots deliberately empty — should FAIL
        tray_bgr = _build_nut_tray(
            nut_good_dir, n_cols=4, n_rows=2, cell_px=256, n_missing=2
        )
        if tray_bgr is not None:
            tray_path = OUT_DIR / "assembly_tray.jpg"
            cv2.imwrite(str(tray_path), tray_bgr)
            print(f"   Tray image saved → {tray_path}")
            # Convert BGR tray to RGB for vs pipeline
            image = cv2.cvtColor(tray_bgr, cv2.COLOR_BGR2RGB)
            checklist = checklist or {"nut": 8}  # full tray expects 8
        else:
            # Fallback: truck image as assembly scene
            image_path = SAM2_IMGS / "truck.jpg"
            checklist = checklist or {"headlight": 2, "tire": 2, "mirror": 1}
    else:
        image_path = Path(image_path)
        checklist = checklist or {}

    if image_path is not None:
        if not Path(image_path).exists():
            print(f"  [skip] Image not found: {image_path}")
            return
        image = vs.load_image(str(image_path))

    print(f"\n── Assembly Verification  (industrial parts tray)  |  device={device}")
    if tray_bgr is not None:
        print(f"   image     : generated 4×2 metal-nut tray")
    else:
        print(f"   image     : {Path(image_path).name}")
    print(f"   checklist : {checklist}")

    skill = vs.SkillRegistry.create(
        "assembly_verification",
        {
            "device": device,
            "component_checklist": checklist,
            "box_threshold": 0.35,
            "text_threshold": 0.25,
        },
    )
    result = skill(image)

    if not result.success:
        print(f"   FAILED: {result.error}")
        return result

    d = result.data
    status = "PASS ✓" if d["all_present"] else "FAIL ✗"
    print(f"\n   Status  : {status}  ({result.inference_time_ms:.0f} ms)")
    print(f"   Detected: {d['component_counts']}")
    print(f"   Expected: {d['expected_counts']}")
    if d["missing"]:
        for m in d["missing"]:
            print(f"     MISSING  → {m}")
    if d["extra"]:
        for e in d["extra"]:
            print(f"     EXTRA    → {e}")

    # Per-detection breakdown — helps spot false positives / duplicate boxes
    print(f"\n   Per-detection boxes ({len(d['boxes'])} total):")
    cell_px = 256  # matches _build_nut_tray cell_px
    for i, (box, score, label) in enumerate(zip(d["boxes"], d["scores"], d["labels"])):
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        col, row = cx // cell_px, cy // cell_px
        print(
            f"     [{i}] {label:6s}  score={score:.3f}"
            f"  box=[{x1},{y1},{x2},{y2}]"
            f"  centre=({cx},{cy})  → cell(col={col},row={row})"
        )

    # --- visualise --------------------------------------------------------
    vis = draw_detections(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        d["boxes"],
        d["scores"],
        d["labels"],
        color=(0, 200, 50) if d["all_present"] else (0, 0, 220),
    )
    # Overall status banner
    banner_col = (0, 200, 50) if d["all_present"] else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), banner_col, -1)
    cv2.putText(
        vis,
        f"Assembly: {status}  |  {d['component_counts']}",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    out_path = OUT_DIR / "assembly_verification.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\n   Saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# GigaPose demo: template-based 6-DoF pose estimation
# ---------------------------------------------------------------------------


def demo_gigapose(
    image_path: str | Path | None = None,
    template_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    bbox: list | None = None,
    obj_id: int = 1,
    device: str | None = None,
):
    """
    6-DoF pose estimation with GigaPose.

    Checks prerequisites and runs inference if everything is in place.
    Prints clear instructions for any missing component.

    Prerequisites
    -------------
    1.  git clone https://github.com/nv-nguyen/gigapose ~/libs/gigapose
    2.  pip install -e ~/libs/gigapose
    3.  python ~/libs/gigapose/src/scripts/download_gigapose.py
    4.  python ~/libs/gigapose/src/scripts/render_custom_templates.py
        (requires Panda3D and a CAD model)

    Output: output/gigapose_pose.jpg  (query image with pose overlay)
    """
    OUT_DIR.mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gigapose_root = LIBS / "gigapose"

    print(f"\n── GigaPose  (6-DoF pose)  |  device={device}")

    # ── Prerequisite checks (fail fast with clear messages) ──────────
    ok = True

    if not gigapose_root.exists():
        print("  [MISSING] GigaPose repo not found.")
        print(
            f"    Clone with:  git clone "
            "https://github.com/nv-nguyen/gigapose "
            f"{gigapose_root}"
        )
        ok = False

    if ok:
        try:
            import sys as _sys

            _sys.path.insert(0, str(gigapose_root))
            import src.models.gigaPose  # noqa: F401
        except ImportError as _exc:
            print(f"  [MISSING] GigaPose import failed: {_exc}")
            print("    Install missing packages, e.g.:")
            print("      pip install pytorch_lightning hydra-core " "omegaconf einops")
            ok = False

    default_ckpt = gigapose_root / "pretrained" / "gigaPose_v1.ckpt"
    ckpt_path = Path(checkpoint).expanduser() if checkpoint else default_ckpt
    if ok and not ckpt_path.exists():
        print(f"  [MISSING] Checkpoint not found: {ckpt_path}")
        print("    Download:")
        print("      python ~/libs/gigapose/src/scripts/download_gigapose.py")
        ok = False

    # template_dir is the PARENT directory (contains 000001/ sub-dirs and
    # object_poses/ — BOP template format).  Default: LMO dataset templates.
    _gigapose_data = Path(__file__).parent / "gigaPose_datasets"
    if template_dir is None:
        template_dir = _gigapose_data / "datasets" / "templates" / "lmo"
    template_dir = Path(template_dir)
    if ok and not template_dir.exists():
        print(f"  [MISSING] Template directory not found: {template_dir}")
        print("    Download pre-rendered BOP templates (1.4 GB):")
        print(
            f"      mkdir -p {_gigapose_data}/datasets/tmp\n"
            f"      wget -O {_gigapose_data}/datasets/tmp/templates.zip \\\n"
            "        https://huggingface.co/datasets/nv-nguyen/gigaPose"
            "/resolve/main/templates.zip\n"
            f"      unzip {_gigapose_data}/datasets/tmp/templates.zip"
            f" -d {_gigapose_data}/datasets/"
        )
        print("    (or render custom objects with render_custom_templates.py)")
        ok = False

    if not ok:
        print("\n  Resolve the above and re-run to test GigaPose inference.\n")
        return None

    # ── All prerequisites met — run inference ────────────────────────
    # Use the first template render (000001/000000.png) as a smoke-test image
    # when no real query image is provided.  This self-referential query is
    # not meaningful pose-wise but verifies the full inference pipeline.
    if image_path is None:
        render_png = template_dir / f"{obj_id:06d}" / "000000.png"
        if render_png.exists():
            import PIL.Image as _PILImage
            _rgba = np.array(_PILImage.open(render_png).convert("RGBA"))
            image_path = OUT_DIR / "gigapose_template_query.png"
            _PILImage.fromarray(_rgba[:, :, :3]).save(image_path)
        else:
            # Fall back to MVTec nut images if available
            paths = sorted((MVTEC_NUT / "test/good").glob("*.png"))
            if paths:
                image_path = paths[0]

    image_path = Path(image_path) if image_path else None
    if not image_path or not image_path.exists():
        print(f"  [skip] No test image found (checked {template_dir}/{obj_id:06d}/000000.png)")
        return None

    print(f"   image      : {image_path.name}")
    print(f"   templates  : {template_dir}")
    print(f"   checkpoint : {ckpt_path.name}")

    image = vs.load_image(str(image_path))

    if bbox is None:
        h, w = image.shape[:2]
        bbox = [0, 0, w, h]  # use full image if no detection provided

    skill = vs.SkillRegistry.create(
        "gigapose",
        {
            "device": device,
            "gigapose_dir": str(gigapose_root),
            "checkpoint": str(ckpt_path),
            "template_dir": str(template_dir),
        },
    )
    result = skill(image, bbox=bbox, obj_id=obj_id)

    if not result.success:
        print(f"   FAILED: {result.error}")
        return result

    d = result.data
    R, t = d["rotation"], d["translation"]
    print(f"\n   Score : {d['score']:.4f}  ({result.inference_time_ms:.0f} ms)")
    print(f"   R     :\n{R}")
    print(f"   t     : {t}")

    # Draw pose axes on the image (3D→2D projection)
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if skill.cfg.intrinsics:
        fx, fy, cx, cy = skill.cfg.intrinsics
        K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        axis_len = 0.05  # 5 cm
        axes_3d = np.float32(
            [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]
        )
        rvec, _ = cv2.Rodrigues(R)
        pts, _ = cv2.projectPoints(axes_3d, rvec, t, K_mat, None)
        pts = pts.astype(int)
        origin = tuple(pts[0].ravel())
        for pt, col in zip(pts[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.arrowedLine(vis, origin, tuple(pt.ravel()), col, 2)

    out_path = OUT_DIR / "gigapose_pose.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\n   Saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Skill router demo: LLM-planned multi-skill pipeline
# ---------------------------------------------------------------------------


def demo_skill_router(
    image_path: str | Path | None = None,
    task: str | None = None,
    device: str | None = None,
):
    """
    Let the LLM planner choose which skills to run for a given task.

    Requires ANTHROPIC_API_KEY in the environment.

    Output: output/skill_router_plan.txt — the JSON execution plan
    """
    import os

    OUT_DIR.mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "\n── Skill Router  |  SKIPPED\n"
            "   Set ANTHROPIC_API_KEY to run this demo."
        )
        return

    if image_path is None:
        image_path = SAM2_IMGS / "groceries.jpg"
    if task is None:
        task = (
            "Detect all food items in the image and describe "
            "the overall scene content."
        )

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"  [skip] Image not found: {image_path}")
        return

    print(f"\n── Skill Router  |  device={device}")
    print(f"   image : {image_path.name}")
    print(f"   task  : {task}")

    image = vs.load_image(str(image_path))
    router = vs.SkillRegistry.create(
        "skill_router", {"model": "claude-haiku-4-5-20251001"}
    )
    result = router(image, task=task)

    if not result.success:
        print(f"   FAILED: {result.error}")
        return result

    print(f"\n   Plan summary : {result.data['summary']}")
    print(f"   Steps        : {len(result.data['plan'])}")
    for step in result.data["results"]:
        status = "OK" if step["success"] else "FAIL"
        print(
            f"     Step {step['step']:d}: {step['skill']:22s}  "
            f"{status}  {step['time_ms']:.0f}ms  — {step['reason']}"
        )

    # Save the plan for inspection
    plan_path = OUT_DIR / "skill_router_plan.txt"
    import json

    plan_path.write_text(
        json.dumps(
            {"plan": result.data["plan"], "summary": result.data["summary"]},
            indent=2,
        )
    )
    print(f"\n   Saved plan → {plan_path}")
    return result


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_registry()
    print()
    # demo_grounding_dino()
    # demo_assembly_verification()
    demo_gigapose()
    # demo_skill_router()
    # demo_bent_optimisation()
