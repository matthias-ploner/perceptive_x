import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import GigaPoseConfig
from .base_pose import BasePoseSkill


@register_skill("gigapose")
class GigaPoseSkill(BasePoseSkill):
    """
    Template-based 6-DoF pose estimation using GigaPose.

    GigaPose matches a query image against pre-rendered templates of the
    target object and refines the initial estimate, returning a full SE(3)
    pose without needing depth or CAD-at-inference-time.

    Prerequisites:
        git clone https://github.com/nv-nguyen/gigapose ~/libs/gigapose
        pip install -e ~/libs/gigapose

    Usage:
        skill = SkillRegistry.create("gigapose", {
            "template_dir": "/path/to/object/templates",
            "intrinsics": [fx, fy, cx, cy],   # camera intrinsics
            "device": "cuda",
        })
        result = skill(
            image,                         # HxWx3 uint8 RGB
            mask=mask_uint8,               # HxW uint8 object mask (optional)
        )
        # result.data["pose_matrix"]   np.ndarray 4x4  SE(3) camera←object
        # result.data["rotation"]      np.ndarray 3x3
        # result.data["translation"]   np.ndarray (3,)  metres
        # result.data["score"]         float  template-matching confidence
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = GigaPoseConfig(**config)
        self._predictor = None

    def load(self) -> None:
        gigapose_path = Path(self.cfg.gigapose_dir).expanduser().resolve()
        if not gigapose_path.exists():
            raise RuntimeError(
                f"GigaPose source not found at {gigapose_path}.\n"
                "Install with:\n"
                "  git clone https://github.com/nv-nguyen/gigapose "
                "~/libs/gigapose\n"
                "  pip install -e ~/libs/gigapose"
            )
        if str(gigapose_path) not in sys.path:
            sys.path.insert(0, str(gigapose_path))

        try:
            from gigapose.lib.poses.multiview import MultiViewPoseEstimator
            from gigapose.lib.utils.config import DictConfig
        except ImportError as exc:
            raise ImportError(
                "GigaPose python package not importable. "
                f"Source found at {gigapose_path} but import failed: {exc}\n"
                "Run:  pip install -e ~/libs/gigapose"
            ) from exc

        cfg = DictConfig({
            "n_templates": self.cfg.n_templates,
            "device":      self.cfg.device,
        })
        self._predictor = MultiViewPoseEstimator(cfg)
        self.logger.info(
            f"GigaPose loaded from {gigapose_path} "
            f"(n_templates={self.cfg.n_templates})"
        )

    def preprocess(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple:
        if self.cfg.template_dir is None:
            raise ValueError(
                "template_dir must be set in config. "
                "Point it to the pre-rendered template directory for the "
                "target object."
            )
        template_dir = Path(self.cfg.template_dir).expanduser().resolve()
        if not template_dir.exists():
            raise FileNotFoundError(
                f"template_dir not found: {template_dir}"
            )

        # Build camera intrinsics matrix
        if self.cfg.intrinsics is not None:
            fx, fy, cx, cy = self.cfg.intrinsics
            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1],
            ], dtype=np.float32)
        else:
            # Rough estimate from image shape (use only for qualitative tests)
            h, w = image.shape[:2]
            f = max(h, w)
            K = np.array([
                [f,  0, w / 2],
                [0,  f, h / 2],
                [0,  0,     1],
            ], dtype=np.float32)
            self.logger.warning(
                "No intrinsics provided — using rough focal estimate. "
                "Pass intrinsics=[fx,fy,cx,cy] for metric accuracy."
            )

        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        return image, mask, K, str(template_dir)

    def infer(self, processed: Tuple, **kwargs):
        image, mask, K, template_dir = processed
        pose, score = self._predictor.estimate(
            query_image=image,
            query_mask=mask,
            K=K,
            template_dir=template_dir,
        )
        return pose, score  # pose: 4x4 SE(3), score: float

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        pose_matrix, score = raw_output
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3,  3]

        return SkillResult(
            skill_name="gigapose",
            success=True,
            data={
                "pose_matrix": pose_matrix,  # [4,4] SE(3)
                "rotation":    R,            # [3,3]
                "translation": t,            # [3,]  in metres
                "score":       float(score),
            },
            metadata={
                "n_templates":   self.cfg.n_templates,
                "template_dir":  self.cfg.template_dir,
            },
        )
