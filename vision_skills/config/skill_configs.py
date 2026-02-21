from pydantic import BaseModel
from typing import Dict, Optional, List, Literal


class DepthConfig(BaseModel):
    model_path: str = "depth-anything/Depth-Anything-V2-Large"
    device: str = "cuda"
    input_size: int = 518
    encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl"


class DepthAnythingV3Config(BaseModel):
    # Monocular variants  (Apache 2.0):  DA3MONO-LARGE
    # Multi-view variants (Apache 2.0):  DA3-BASE, DA3-LARGE
    # Full model (CC BY-NC 4.0):         DA3NESTED-GIANT-LARGE
    model_name: str = "depth-anything/DA3MONO-LARGE"
    device: str = "cuda"


class SegmentationConfig(BaseModel):
    model_path: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    multimask_output: bool = True
    points_per_side: int = 32  # for automatic mask generation mode


class PoseConfig(BaseModel):
    model_path: str = "foundationpose"
    device: str = "cuda"
    mesh_path: Optional[str] = None  # path to object CAD mesh (OBJ/PLY/STL)
    # axes of symmetry for objects with rotational ambiguity
    symmetry_axes: Optional[List[int]] = None


class ReasoningConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.1
    use_flash_attention: bool = True


class ReasoningV3Config(BaseModel):
    # Qwen3-VL variants:
    #   Qwen/Qwen3-VL-4B-Instruct    — compact, CPU-feasible
    #   Qwen/Qwen3-VL-8B-Instruct    — best quality / speed balance
    #   Qwen/Qwen3-VL-32B-Instruct   — highest quality, GPU required
    #   Qwen/Qwen3-VL-30B-A3B-Instruct — MoE variant
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.1
    use_flash_attention: bool = True


class ClassificationConfig(BaseModel):
    model_name: str = "facebook/dinov2-large"
    device: str = "cuda"
    few_shot_k: int = 5           # k nearest neighbours for voting
    anomaly_threshold: float = 0.85  # cosine similarity below this → anomaly
    embedding_dim: int = 1024
    index_path: Optional[str] = None  # pre-built FAISS index path
    # True → patch-level anomaly scoring + spatial anomaly map output
    patch_mode: bool = False
    # Input resolution fed to the ViT processor (pixels, square).
    # 224 → 16×16 = 256 patches (default).
    # 448 → 32×32 = 1024 patches (4× spatial detail, better for fine defects).
    image_size: int = 224
    # Fraction of most-anomalous patches used for the image-level score.
    # Lower → focuses on extreme local defects (good for scratches).
    # Higher → captures distributed deformations (good for bent/dented).
    anomaly_top_frac: float = 0.10
    # Gaussian sigma applied to the 2-D patch-similarity grid before scoring.
    # 0 = disabled.  sigma≈1 amplifies spatially-coherent anomaly regions
    # (e.g. bent edges) and suppresses isolated noisy patches.
    smooth_sigma: float = 0.0


class DetectionConfig(BaseModel):
    # IDEA-Research/grounding-dino-tiny  — fast, lower accuracy
    # IDEA-Research/grounding-dino-base  — best quality/speed balance
    model_name: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    box_threshold: float = 0.30   # min score for a box to be kept
    text_threshold: float = 0.25  # min per-token score for label matching
    # IoU threshold for greedy NMS — suppresses duplicate boxes for the same
    # object.  Set to 1.0 to disable.
    nms_iou_threshold: float = 0.50


class AssemblyConfig(BaseModel):
    detection_model: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.50
    # Expected component counts: {"bolt": 4, "washer": 4, "nut": 1}
    component_checklist: Dict[str, int] = {}
    # Allowed deviation per component (0 = exact match required)
    count_tolerance: int = 0


class GigaPoseConfig(BaseModel):
    # Path to cloned gigapose source (added to sys.path at runtime).
    # setup.cfg declares package_dir = src/, so the importable roots are
    # models.*, megapose.*, lib3d.*, utils.*, etc. (all under src/).
    gigapose_dir: str = "~/libs/gigapose"
    device: str = "cuda"
    # Number of rendered templates per object (default from gigapose)
    n_templates: int = 162
    # Path to the BOP-format pre-rendered template directory for the object.
    # Each object has its own sub-directory with rgb/mask/poses rendered from
    # 162 viewpoints.  Generate with:
    #   python ~/libs/gigapose/src/scripts/render_custom_templates.py
    template_dir: Optional[str] = None
    # Camera intrinsics [fx, fy, cx, cy] – required for metric pose
    intrinsics: Optional[List[float]] = None
    # Path to the downloaded gigaPose_v1.ckpt checkpoint.  Download with:
    #   python ~/libs/gigapose/src/scripts/download_gigapose.py
    checkpoint: Optional[str] = None


class SkillRouterConfig(BaseModel):
    # Anthropic model used to plan the skill execution sequence
    # claude-haiku-4-5-20251001 is fast and cheap for structured planning
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024
    temperature: float = 0.0
    # Maximum number of skills the router may chain per request
    max_steps: int = 5
