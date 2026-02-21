import numpy as np
import cv2
from typing import Tuple, Optional
import open3d as o3d


def load_image(path: str, mode: str = "rgb") -> np.ndarray:
    """Load image from disk. Returns HxWx3 uint8 array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if mode == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_keep_aspect(
    image: np.ndarray, max_size: int = 1024
) -> Tuple[np.ndarray, float]:
    """Resize so the longest side equals max_size; returns (image, scale)."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
    return image, scale


def normalize_image(
    image: np.ndarray,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """ImageNet-normalize a uint8 HxWx3 image to float32."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,  # 3x3 camera matrix
    rgb: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """
    Convert a metric depth map + camera intrinsics to an Open3D PointCloud.

    Args:
        depth:      HxW float32 depth map in metres.
        intrinsics: 3x3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        rgb:        Optional HxWx3 uint8 RGB image for coloring the cloud.

    Returns:
        open3d.geometry.PointCloud
    """
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    mask = z.flatten() > 0
    pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if rgb is not None:
        colors = rgb.reshape(-1, 3)[mask].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def draw_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay boolean segmentation masks on an image with random colours.

    Args:
        image: HxWx3 uint8 RGB image.
        masks: NxHxW bool array of segmentation masks.
        alpha: Blending factor (0 = original, 1 = solid colour).

    Returns:
        HxWx3 uint8 image with masks blended in.
    """
    overlay = image.copy()
    rng = np.random.default_rng(42)
    for mask in masks:
        color = rng.integers(50, 255, 3).tolist()
        overlay[mask.astype(bool)] = (
            np.array(overlay[mask.astype(bool)]) * (1 - alpha)
            + np.array(color) * alpha
        ).astype(np.uint8)
    return overlay
