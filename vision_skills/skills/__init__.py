from .depth import DepthAnythingV2Skill, DepthAnything3Skill
from .segmentation import SAM2Skill
from .pose import FoundationPoseSkill
from .reasoning import QwenVLSkill, Qwen3VLSkill
from .classification import DINOv2Skill

__all__ = [
    "DepthAnythingV2Skill",
    "DepthAnything3Skill",
    "SAM2Skill",
    "FoundationPoseSkill",
    "QwenVLSkill",
    "Qwen3VLSkill",
    "DINOv2Skill",
]
