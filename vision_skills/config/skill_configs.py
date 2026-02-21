from pydantic import BaseModel
from typing import Optional, List, Literal


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
