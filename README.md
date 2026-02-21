# perceptive-x

Modular Vision AI Skills Library — plug-and-play computer-vision primitives
with a unified interface, composable pipelines, and an optional LLM planner.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Skills overview](#skills-overview)
  - [Depth Anything V2](#depth-anything-v2)
  - [Depth Anything 3](#depth-anything-3)
  - [SAM 2](#sam-2)
  - [Grounding DINO](#grounding-dino)
  - [Assembly Verification](#assembly-verification)
  - [DINOv2 (classification / anomaly)](#dinov2)
  - [Qwen2.5-VL (reasoning)](#qwen25-vl)
  - [Qwen3-VL (reasoning)](#qwen3-vl)
  - [FoundationPose](#foundationpose)
  - [GigaPose](#gigapose)
  - [Skill Router (LLM planner)](#skill-router)
- [Pipeline composition](#pipeline-composition)
- [Adding a new skill](#adding-a-new-skill)
- [Demo scripts](#demo-scripts)

---

## Installation

Requires Python ≥ 3.12 and PyTorch ≥ 2.3.

```bash
git clone <this-repo> perceptive_x
cd perceptive_x
pip install -e .
```

**Optional extras**

| Extra | What it adds |
|---|---|
| `pip install -e ".[gpu]"` | FAISS-GPU (CUDA 12) for DINOv2 gallery search |
| `pip install -e ".[flash]"` | Flash Attention 2 for Qwen2.5/3-VL |
| `pip install -e ".[triton]"` | Triton Inference Server backend |

**Source-only dependencies** (install separately)

```bash
# Depth Anything 3
pip install git+https://github.com/ByteDance-Seed/depth-anything-3

# SAM 2
pip install git+https://github.com/facebookresearch/sam2.git

# GigaPose
git clone https://github.com/nv-nguyen/gigapose ~/libs/gigapose
pip install -e ~/libs/gigapose
pip install pytorch_lightning hydra-core omegaconf einops
python ~/libs/gigapose/src/scripts/download_gigapose.py

# FoundationPose
# See https://github.com/NVlabs/FoundationPose
```

---

## Quick start

```python
import vision_skills as vs

# Load any image → HxWx3 uint8 RGB
image = vs.load_image("scene.jpg")

# Create a skill — all follow the same factory pattern
skill = vs.SkillRegistry.create("depth_anything_v3", {"device": "cuda"})

# Run inference — returns a SkillResult
result = skill(image)

print(result.success)              # True
print(result.inference_time_ms)    # e.g. 42.3
depth = result.data["depth_norm"]  # numpy array [H, W] in [0, 1]

# Discover all registered skills
print(vs.SkillRegistry.list_skills())
```

**`SkillResult` fields**

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether inference succeeded |
| `data` | `dict` | Skill-specific outputs (see per-skill docs below) |
| `metadata` | `dict` | Config echo, model name, etc. |
| `inference_time_ms` | `float` | Wall-clock time |
| `error` | `str \| None` | Error message on failure |

---

## Skills overview

### Depth Anything V2

Monocular depth estimation (ViT backbone, relative metric depth).

```python
skill = vs.SkillRegistry.create("depth_anything_v2", {
    "device": "cuda",
    "model_path": "depth-anything/Depth-Anything-V2-Large",  # default
    "encoder": "vitl",   # vits | vitb | vitl | vitg
})
result = skill(image)
```

| `result.data` key | Type | Description |
|---|---|---|
| `depth_map` | `[H, W]` float32 | Raw relative-metric depth |
| `depth_norm` | `[H, W]` float32 | Normalised to [0, 1] |
| `depth_min` | float | Min of depth_map |
| `depth_max` | float | Max of depth_map |

---

### Depth Anything 3

Upgraded depth model with monocular and multi-view modes.

```python
# Monocular (Apache 2.0)
skill = vs.SkillRegistry.create("depth_anything_v3", {
    "device": "cuda",
    "model_name": "depth-anything/DA3MONO-LARGE",  # default
})
result = skill(image)

# Multi-view: pass extra views for camera pose estimation
result = skill(primary_image, extra_images=[view2, view3])
```

| `result.data` key | Available in | Description |
|---|---|---|
| `depth_map` | both | Primary depth [H, W] |
| `depth_norm` | both | Normalised [0, 1] |
| `depth_maps` | multi-view | All N depth maps |
| `extrinsics` | multi-view | [N, 3, 4] world-to-camera |
| `intrinsics` | multi-view | [N, 3, 3] K matrices |
| `confidence` | multi-view | [H, W] float32 |

Demo: `demo_pipeline_test()` in [main.py](main.py)

---

### SAM 2

Instance segmentation — automatic, point-prompted, or box-prompted.

```python
skill = vs.SkillRegistry.create("sam2", {
    "device": "cuda",
    "model_path": "facebook/sam2-hiera-large",  # default
})

# Automatic mask generation
result = skill(image, mode="auto")

# Point prompt  (foreground=1 / background=0)
result = skill(image, mode="point",
               points=[[320, 240]], point_labels=[1])

# Box prompt
result = skill(image, mode="box",
               boxes=[[100, 80, 400, 350]])
```

| `result.data` key | Type | Description |
|---|---|---|
| `masks` | `[N, H, W]` bool | Segmentation masks |
| `scores` | `[N]` float | Confidence / stability |
| `bboxes` | `[N, 4]` int | xywh bounding boxes |
| `num_masks` | int | Count |

Demo: `run_pipeline_test()` in [main.py](main.py)

---

### Grounding DINO

Zero-shot text-prompted object detection.

```python
skill = vs.SkillRegistry.create("grounding_dino", {
    "device": "cuda",
    "model_name": "IDEA-Research/grounding-dino-base",  # or grounding-dino-tiny
    "box_threshold": 0.30,
    "text_threshold": 0.25,
})
result = skill(image, text_prompt="bolt . washer . nut")
```

Text prompt format: period-separated class names.
A single class can be written as `"bolt"` (normalised to `"bolt."` automatically).

| `result.data` key | Type | Description |
|---|---|---|
| `boxes` | `[N, 4]` float | x1, y1, x2, y2 in pixels |
| `scores` | `[N]` float | Confidence in [0, 1] |
| `labels` | `list[str]` | One label per box |
| `n_detections` | int | Count |

NMS deduplication (including large "group" boxes) is applied automatically
via `nms_iou_threshold` (default 0.50, uses max(IoU, containment-ratio)).

Demo: `demo_grounding_dino()` in [main.py](main.py)

---

### Assembly Verification

Count components and verify an assembly is complete.
Internally uses Grounding DINO for detection.

```python
skill = vs.SkillRegistry.create("assembly_verification", {
    "device": "cuda",
    "component_checklist": {"bolt": 4, "washer": 4, "nut": 1},
    "count_tolerance": 0,      # exact match required
    "box_threshold": 0.35,
})
result = skill(image)

if result.data["all_present"]:
    print("PASS — all components detected")
else:
    print("FAIL")
    for m in result.data["missing"]:
        print(" missing:", m)
```

| `result.data` key | Type | Description |
|---|---|---|
| `all_present` | bool | True if assembly is complete |
| `component_counts` | dict | Detected count per component |
| `expected_counts` | dict | Required count per component |
| `missing` | `list[str]` | Components below required count |
| `extra` | `list[str]` | Components above required count |
| `boxes` / `scores` / `labels` | arrays | Raw detections |

Demo: `demo_assembly_verification()` in [main.py](main.py)

---

### DINOv2

Few-shot classification and patch-level anomaly detection.
Build a gallery from reference images, then classify or score test images.

```python
skill = vs.SkillRegistry.create("dinov2", {
    "device": "cuda",
    "model_name": "facebook/dinov2-large",
    "few_shot_k": 5,
    "anomaly_threshold": 0.85,
    # Patch-level anomaly map (slower but localises defects):
    "patch_mode": True,
    "image_size": 448,          # 4× spatial detail vs default 224
    "anomaly_top_frac": 0.10,   # use worst 10 % of patches for scoring
    "smooth_sigma": 1.0,        # Gaussian blur for spatial coherence
})

# Build gallery once
skill.build_gallery({
    "good": [img1, img2, ...],
    "defect": [img3, img4, ...],
})

# Classify / detect anomalies
result = skill(test_image)
print(result.data["predicted_class"])  # "good" | "defect"
print(result.data["is_anomaly"])       # bool
```

| `result.data` key | Mode | Description |
|---|---|---|
| `predicted_class` | CLS | Nearest-neighbour voted class |
| `confidence` | both | Mean cosine similarity score |
| `is_anomaly` | both | Below anomaly_threshold |
| `anomaly_map` | patch | [H, W] float normalised heatmap |
| `image_score` | patch | Scalar (lower = more anomalous) |

Demo: `demo_anomaly_detection()` in [main.py](main.py)

---

### Qwen2.5-VL

Visual reasoning and VQA with Qwen2.5-VL.

```python
skill = vs.SkillRegistry.create("qwen_vl", {
    "device": "cuda",
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "use_flash_attention": True,  # requires flash-attn extra
})
result = skill(image, prompt="List all visible objects as JSON.")

print(result.data["response"])    # raw text
print(result.data["structured"])  # auto-parsed dict if JSON present
```

---

### Qwen3-VL

Next-generation VLM with 256K context, stronger OCR, and structured output.
Drop-in replacement for `qwen_vl` with better accuracy.

```python
skill = vs.SkillRegistry.create("qwen3_vl", {
    "device": "cuda",
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",  # 4B | 8B | 32B | 30B-A3B
})
result = skill(image, prompt="Describe any defects you see.")
```

Demo: `run_batch_test()` in [main.py](main.py)

---

### FoundationPose

6-DoF object pose estimation from RGB-D + object mask + CAD model.

```python
skill = vs.SkillRegistry.create("foundation_pose", {
    "device": "cuda",
    "mesh_path": "path/to/object.ply",
})
result = skill(image, depth=depth_m, mask=mask, intrinsics=K)

pose = result.data["pose_matrix"]   # [4, 4] SE(3)
R    = result.data["rotation"]      # [3, 3]
t    = result.data["translation"]   # [3] in metres
```

Requires: see https://github.com/NVlabs/FoundationPose for environment setup.

---

### GigaPose

Template-based 6-DoF pose from a single RGB view (no depth required).

```python
skill = vs.SkillRegistry.create("gigapose", {
    "device": "cuda",
    "gigapose_dir":  "~/libs/gigapose",
    "checkpoint":    "~/libs/gigapose/pretrained/gigaPose_v1.ckpt",
    "template_dir":  "/path/to/bop_templates/obj_000001",
    "intrinsics":    [fx, fy, cx, cy],  # optional but needed for metric t
})
result = skill(image,
               bbox=[x1, y1, x2, y2],  # 2-D detection bounding box
               obj_id=1)               # BOP object ID (1-indexed)

pose  = result.data["pose_matrix"]   # [4, 4] SE(3)
score = result.data["score"]         # template-match confidence
```

Setup steps:

```bash
git clone https://github.com/nv-nguyen/gigapose ~/libs/gigapose
pip install -e ~/libs/gigapose
pip install pytorch_lightning hydra-core omegaconf einops
python ~/libs/gigapose/src/scripts/download_gigapose.py          # checkpoint
python ~/libs/gigapose/src/scripts/render_custom_templates.py    # needs Panda3D
```

Demo: `demo_gigapose()` in [main.py](main.py) — checks prerequisites and prints
instructions for anything that is missing.

---

### Skill Router

LLM-powered planner that composes vision skills from a free-text task description.
Requires `ANTHROPIC_API_KEY` in the environment.

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "..."

router = vs.SkillRegistry.create("skill_router", {
    "model": "claude-haiku-4-5-20251001",  # planner model
    "max_steps": 5,
})
result = router(image, task="Detect all bolts and count how many are present.")

print(result.data["summary"])    # LLM plan summary
for step in result.data["results"]:
    print(step["skill"], step["success"], step["time_ms"])
```

Demo: `demo_skill_router()` in [main.py](main.py)

---

## Pipeline composition

`SkillPipeline` chains skills sequentially.
Each step receives the original image and a context dict of all previous results.

```python
pipeline = vs.SkillPipeline([
    ("depth", depth_skill),
    ("seg",   seg_skill,   lambda img, ctx: {"mode": "auto"}),
    ("pose",  pose_skill,  lambda img, ctx: {
        "depth":      ctx["depth"].data["depth_map"],
        "mask":       ctx["seg"].data["masks"][0],
        "intrinsics": K,
    }),
])

results = pipeline.run(image, stop_on_failure=True)
for name, res in results.items():
    print(name, "OK" if res.success else res.error)
```

---

## Adding a new skill

1. Create `vision_skills/skills/<category>/my_skill.py`
2. Subclass `BaseSkill` and implement the four lifecycle methods:

```python
from ...core.registry import register_skill
from ...core.base_skill import BaseSkill, SkillResult

@register_skill("my_skill")
class MySkill(BaseSkill):
    def load(self) -> None:
        # load model weights once
        ...

    def preprocess(self, image, **kwargs):
        # normalise / tensorise
        ...
        return processed

    def infer(self, processed, **kwargs):
        # model forward pass
        ...
        return raw_output

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        return SkillResult(
            skill_name="my_skill",
            success=True,
            data={"result": ...},
        )
```

3. Import the class in `vision_skills/skills/<category>/__init__.py`
   and ensure the category package is imported in `vision_skills/skills/__init__.py`.
4. Add a Pydantic config class to `vision_skills/config/skill_configs.py` (optional).

---

## Demo scripts

All demos are standalone functions in [main.py](main.py):

| Function | What it runs |
|---|---|
| `demo_registry()` | Print all registered skill names |
| `demo_grounding_dino()` | Text-prompted detection on a vehicle image |
| `demo_assembly_verification()` | Parts-tray nut count against MVTec dataset |
| `demo_anomaly_detection()` | DINOv2 patch anomaly on MVTec metal-nut test set |
| `demo_bent_optimisation()` | Config sweep: anomaly scoring strategy comparison |
| `demo_gigapose()` | GigaPose prerequisite check + inference if ready |
| `demo_skill_router()` | LLM-planned multi-skill pipeline (needs API key) |
| `run_pipeline_test()` | Depth Anything 3 + SAM 2 pipeline |
| `run_batch_test()` | Depth + SAM 2 + Qwen3-VL on a folder of images |

Run all default demos:

```bash
python main.py
```
