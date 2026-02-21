"""
vision_skills — Modular Vision AI Skills Library
=================================================

Quick start:

    import vision_skills as vs

    # Create a skill via the registry
    depth = vs.SkillRegistry.create("depth_anything_v2", {"device": "cuda"})
    image = vs.load_image("scene.jpg")
    result = depth(image)
    pcd = vs.depth_to_pointcloud(result.data["depth_map"], intrinsics=K, rgb=image)

    # Compose skills into a pipeline
    pipeline = vs.SkillPipeline([
        ("depth", depth),
        ("seg",   seg,   lambda img, ctx: {"mode": "auto"}),
    ])
    results = pipeline.run(image)

Available skills (registry keys):
    depth_anything_v2     — monocular depth estimation (Depth Anything V2)
    depth_anything_v3     — mono + multi-view depth + camera poses (DA3)
    sam2                  — instance segmentation (SAM2)
    foundation_pose       — 6-DoF pose estimation (FoundationPose)
    gigapose              — 6-DoF pose from RGB templates (GigaPose)
    qwen_vl               — visual reasoning / VQA (Qwen2.5-VL)
    qwen3_vl              — visual reasoning / VQA (Qwen3-VL)
    dinov2                — few-shot classification + anomaly detection
    grounding_dino        — zero-shot text-prompted object detection
    assembly_verification — component count verification
    skill_router          — LLM-planned multi-skill pipeline (Anthropic)
"""

from .core.registry import SkillRegistry
from .core.pipeline import SkillPipeline
from .core.base_skill import BaseSkill, SkillResult
from .core.image_utils import (
    load_image,
    resize_keep_aspect,
    normalize_image,
    depth_to_pointcloud,
    draw_masks,
)
from .config import (
    DepthConfig,
    DepthAnythingV3Config,
    SegmentationConfig,
    PoseConfig,
    ReasoningConfig,
    ReasoningV3Config,
    ClassificationConfig,
    DetectionConfig,
    AssemblyConfig,
    GigaPoseConfig,
    SkillRouterConfig,
)

# Auto-register all built-in skills
from .skills.depth.depth_anything import DepthAnythingV2Skill
from .skills.depth.depth_anything3 import DepthAnything3Skill
from .skills.segmentation.sam2 import SAM2Skill
from .skills.pose.foundation_pose import FoundationPoseSkill
from .skills.pose.gigapose import GigaPoseSkill
from .skills.reasoning.qwen_vl import QwenVLSkill
from .skills.reasoning.qwen3_vl import Qwen3VLSkill
from .skills.classification.dinov2 import DINOv2Skill
from .skills.detection.grounding_dino import GroundingDINOSkill
from .skills.assembly.assembly_verification import AssemblyVerificationSkill
from .skills.routing.skill_router import SkillRouterSkill

__all__ = [
    # Core
    "SkillRegistry",
    "SkillPipeline",
    "BaseSkill",
    "SkillResult",
    # Image utilities
    "load_image",
    "resize_keep_aspect",
    "normalize_image",
    "depth_to_pointcloud",
    "draw_masks",
    # Configs
    "DepthConfig",
    "DepthAnythingV3Config",
    "SegmentationConfig",
    "PoseConfig",
    "ReasoningConfig",
    "ReasoningV3Config",
    "ClassificationConfig",
    "DetectionConfig",
    "AssemblyConfig",
    "GigaPoseConfig",
    "SkillRouterConfig",
    # Skill classes (for direct subclassing / type hints)
    "DepthAnythingV2Skill",
    "DepthAnything3Skill",
    "SAM2Skill",
    "FoundationPoseSkill",
    "GigaPoseSkill",
    "QwenVLSkill",
    "Qwen3VLSkill",
    "DINOv2Skill",
    "GroundingDINOSkill",
    "AssemblyVerificationSkill",
    "SkillRouterSkill",
]
