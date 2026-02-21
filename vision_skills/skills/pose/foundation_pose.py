import numpy as np
from typing import Optional
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import PoseConfig
from .base_pose import BasePoseSkill


@register_skill("foundation_pose")
class FoundationPoseSkill(BasePoseSkill):
    """
    6-DoF object pose estimation using FoundationPose (NVlabs).

    Requires:
        - RGB image
        - Metric depth map (aligned with RGB)
        - Binary segmentation mask for the target object
        - Camera intrinsic matrix (3x3)
        - Object CAD mesh (OBJ/PLY/STL) — set via config mesh_path or per-call

    Installation:
        Follow https://github.com/NVlabs/FoundationPose for CUDA environment setup.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = PoseConfig(**config)
        self.mesh = None

    def load(self) -> None:
        from foundationpose.estimator import FoundationPoseEstimator
        import trimesh

        self.estimator = FoundationPoseEstimator(self.cfg.model_path)
        if self.cfg.mesh_path:
            self.mesh = trimesh.load(self.cfg.mesh_path)
            self.logger.info(f"Loaded mesh from {self.cfg.mesh_path}")
        self.logger.info("FoundationPose estimator ready.")

    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image  # raw passing; depth/mask provided as kwargs in infer

    def infer(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        mesh=None,
        **kwargs,
    ):
        _mesh = mesh or self.mesh
        if _mesh is None:
            raise ValueError(
                "An object mesh is required. Provide mesh_path in config or pass mesh= at inference time."
            )
        pose = self.estimator.estimate(
            rgb=image,
            depth=depth,
            mask=mask,
            K=intrinsics,
            mesh=_mesh,
        )
        return pose

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        pose_matrix = np.array(raw_output, dtype=np.float64)
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]

        return SkillResult(
            skill_name="foundation_pose",
            success=True,
            data={
                "pose_matrix": pose_matrix,  # 4x4 SE(3)
                "rotation": R,               # 3x3
                "translation": t,            # [x, y, z] metres
            },
            metadata={"mesh": str(self.cfg.mesh_path)},
        )
