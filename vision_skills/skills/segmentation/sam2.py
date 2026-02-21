import numpy as np
from typing import Optional
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import SegmentationConfig
from .base_segmentation import BaseSegmentationSkill


@register_skill("sam2")
class SAM2Skill(BaseSegmentationSkill):
    """
    SAM2 instance segmentation with three prompt modes:

        'auto'  — automatic mask generation (no prompt required)
        'point' — point prompt: pass points=[[x,y],...] and point_labels=[1,0,...]
        'box'   — bounding-box prompt: pass boxes=[[x1,y1,x2,y2],...]

    Note: SAM2 must be installed from the official Meta repository:
        pip install git+https://github.com/facebookresearch/sam2.git
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = SegmentationConfig(**config)
        self.predictor = None
        self.auto_generator = None

    def load(self) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.predictor = SAM2ImagePredictor.from_pretrained(
            self.cfg.model_path, device=self.cfg.device
        )
        self.auto_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            self.cfg.model_path,
            points_per_side=self.cfg.points_per_side,
            device=self.cfg.device,
        )
        self.logger.info(f"SAM2 loaded from {self.cfg.model_path}")

    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image  # SAM2 handles internal preprocessing

    def infer(
        self,
        image: np.ndarray,
        mode: str = "auto",
        points: Optional[np.ndarray] = None,       # [[x, y], ...]
        point_labels: Optional[np.ndarray] = None,  # [1, 0, ...]
        boxes: Optional[np.ndarray] = None,         # [[x1, y1, x2, y2], ...]
        **kwargs,
    ) -> dict:
        if mode == "auto":
            return {"masks_data": self.auto_generator.generate(image)}

        self.predictor.set_image(image)

        if mode == "point":
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=self.cfg.multimask_output,
            )
        elif mode == "box":
            masks, scores, logits = self.predictor.predict(
                box=boxes,
                multimask_output=False,
            )
        else:
            raise ValueError(f"Unknown SAM2 mode: '{mode}'. Use 'auto', 'point', or 'box'.")

        return {"masks": masks, "scores": scores, "logits": logits}

    def postprocess(self, raw_output: dict, **kwargs) -> SkillResult:
        if "masks_data" in raw_output:
            anns = raw_output["masks_data"]
            masks = np.stack([a["segmentation"] for a in anns])
            scores = np.array([a["stability_score"] for a in anns])
            bboxes = np.array([a["bbox"] for a in anns])  # xywh
        else:
            masks = raw_output["masks"]
            scores = raw_output["scores"]
            bboxes = None

        return SkillResult(
            skill_name="sam2",
            success=True,
            data={
                "masks": masks,             # [N, H, W] bool
                "scores": scores,           # [N] float
                "bboxes": bboxes,           # [N, 4] xywh or None
                "num_masks": len(masks),
            },
        )
