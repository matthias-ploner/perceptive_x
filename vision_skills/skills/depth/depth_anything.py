import numpy as np
import torch
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import DepthConfig
from .base_depth import BaseDepthSkill


@register_skill("depth_anything_v2")
class DepthAnythingV2Skill(BaseDepthSkill):
    """
    Monocular depth estimation with Depth Anything V2.

    Loads via HuggingFace transformers (AutoModelForDepthEstimation).
    Returns relative metric depth; pair with camera intrinsics for 3-D reconstruction.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = DepthConfig(**config)

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dtype = torch.float16 if self.cfg.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.cfg.model_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.cfg.model_path,
            torch_dtype=dtype,
        ).to(self.cfg.device)
        self.model.eval()
        self.logger.info(f"DepthAnythingV2 loaded from {self.cfg.model_path}")

    def preprocess(self, image: np.ndarray, **kwargs) -> dict:
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.cfg.device) for k, v in inputs.items()}

    def infer(self, processed: dict, **kwargs):
        with torch.no_grad():
            return self.model(**processed)

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        depth = raw_output.predicted_depth.squeeze().float().cpu().numpy()
        depth_min, depth_max = depth.min(), depth.max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        return SkillResult(
            skill_name="depth_anything_v2",
            success=True,
            data={
                "depth_map": depth,        # raw float (relative metric)
                "depth_norm": depth_norm,  # normalised [0, 1]
                "depth_min": float(depth_min),
                "depth_max": float(depth_max),
            },
            metadata={"shape": depth.shape, "model": self.cfg.model_path},
        )
