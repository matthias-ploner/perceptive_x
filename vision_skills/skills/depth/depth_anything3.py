import numpy as np
from typing import List, Optional
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import DepthAnythingV3Config
from .base_depth import BaseDepthSkill


@register_skill("depth_anything_v3")
class DepthAnything3Skill(BaseDepthSkill):
    """
    Depth estimation with Depth Anything 3 (DA3).

    Supports two modes depending on the model variant chosen:

        Monocular (default):
            model_name = "depth-anything/DA3MONO-LARGE"  (Apache 2.0)
            skill(image)
            → result.data["depth_map"]   HxW float32
            → result.data["confidence"]  HxW float32

        Multi-view (camera pose + geometry):
            model_name = "depth-anything/DA3-LARGE"      (Apache 2.0)
                      or "depth-anything/DA3NESTED-GIANT-LARGE"  (CC BY-NC 4.0)
            skill(image, extra_images=[img2, img3, ...])
            → result.data["depth_map"]    HxW float32 for the primary image
            → result.data["depth_maps"]   list[HxW] for all N images
            → result.data["extrinsics"]   Nx3x4 camera poses  (world-to-camera)
            → result.data["intrinsics"]   Nx3x3 camera intrinsic matrices
            → result.data["confidence"]   HxW float32 for the primary image

    Installation (required before use):
        pip install git+https://github.com/ByteDance-Seed/depth-anything-3

    Available model variants:
        depth-anything/DA3MONO-LARGE        monocular, Apache 2.0  ← default
        depth-anything/DA3-BASE             multi-view, Apache 2.0
        depth-anything/DA3-LARGE            multi-view, Apache 2.0
        depth-anything/DA3NESTED-GIANT-LARGE  all capabilities, CC BY-NC 4.0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = DepthAnythingV3Config(**config)

    def load(self) -> None:
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "depth-anything-3 is not installed. Install it with:\n"
                "  pip install git+https://github.com/ByteDance-Seed/depth-anything-3"
            ) from e

        import torch

        self.model = DepthAnything3.from_pretrained(self.cfg.model_name)
        self.model = self.model.to(device=self.cfg.device)
        self.model.eval()
        self.logger.info(f"Depth Anything 3 loaded: {self.cfg.model_name}")

    def preprocess(
        self,
        image: np.ndarray,
        extra_images: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Collect all images into a list; DA3 handles its own internal preprocessing."""
        images = [image]
        if extra_images:
            images.extend(extra_images)
        return images

    def infer(self, images: List[np.ndarray], **kwargs):
        return self.model.inference(images)

    def postprocess(self, prediction, **kwargs) -> SkillResult:
        # prediction.depth: [N, H, W] float32
        depth_all = prediction.depth  # numpy array

        primary_depth = depth_all[0]
        depth_min, depth_max = float(primary_depth.min()), float(primary_depth.max())
        depth_norm = (primary_depth - depth_min) / (depth_max - depth_min + 1e-8)

        data: dict = {
            "depth_map": primary_depth,       # HxW — primary image, matches V2 interface
            "depth_norm": depth_norm,          # HxW [0, 1]
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_maps": list(depth_all),     # all N depth maps
        }

        # Confidence (always present)
        if hasattr(prediction, "conf") and prediction.conf is not None:
            data["confidence"] = prediction.conf[0]      # HxW primary
            data["confidences"] = list(prediction.conf)  # all N

        # Camera geometry (multi-view models only)
        if hasattr(prediction, "extrinsics") and prediction.extrinsics is not None:
            data["extrinsics"] = prediction.extrinsics   # [N, 3, 4]
        if hasattr(prediction, "intrinsics") and prediction.intrinsics is not None:
            data["intrinsics"] = prediction.intrinsics   # [N, 3, 3]

        return SkillResult(
            skill_name="depth_anything_v3",
            success=True,
            data=data,
            metadata={
                "model": self.cfg.model_name,
                "num_images": len(data["depth_maps"]),
                "shape": primary_depth.shape,
            },
        )
