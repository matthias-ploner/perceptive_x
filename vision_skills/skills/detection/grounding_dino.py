import numpy as np
from PIL import Image
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import DetectionConfig
from .base_detection import BaseDetectionSkill


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy NMS using max(IoU, containment-ratio) as the overlap metric.

    Standard IoU catches same-scale duplicate boxes.  Containment ratio
    catches a large 'group' box that wraps multiple individual objects:
    such a box has low standard IoU (~0.09) with each kept box but high
    containment (~0.9), so it is correctly suppressed.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (ix2 - ix1).clip(min=0) * (iy2 - iy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        containment = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-6)
        overlap = np.maximum(iou, containment)
        order = order[1:][overlap <= iou_threshold]
    return np.array(keep, dtype=np.int64)


@register_skill("grounding_dino")
class GroundingDINOSkill(BaseDetectionSkill):
    """
    Zero-shot text-prompted object detection using Grounding DINO.

    Text prompt format: period-separated class names.
    Single class:   "bolt"          → normalised to "bolt."
    Multi-class:    "bolt . washer . nut"

    Usage:
        skill = SkillRegistry.create("grounding_dino", {"device": "cuda"})
        result = skill(image, text_prompt="bolt . washer . nut")
        # result.data["boxes"]         np.ndarray [N,4]  x1,y1,x2,y2 pixels
        # result.data["scores"]        np.ndarray [N]    confidence in [0,1]
        # result.data["labels"]        list[str]         one label per box
        # result.data["n_detections"]  int
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = DetectionConfig(**config)

    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.model_name
        ).to(self.cfg.device)
        self.model.eval()
        self.logger.info(f"Grounding DINO loaded: {self.cfg.model_name}")

    def preprocess(
        self,
        image: np.ndarray,
        text_prompt: str = "object",
        **kwargs,
    ):
        pil = Image.fromarray(image)
        # Grounding DINO expects a period at the end of the text prompt
        prompt = text_prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        inputs = self.processor(
            images=pil,
            text=prompt,
            return_tensors="pt",
        ).to(self.cfg.device)
        return inputs, pil.size  # (inputs, (W, H))

    def infer(self, processed, **kwargs):
        import torch
        inputs, img_size = processed
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs, inputs, img_size

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        import inspect
        outputs, inputs, img_size = raw_output
        w, h = img_size

        # `box_threshold` was renamed to `threshold` in transformers 5.x —
        # detect the right kwarg name at runtime for cross-version compat.
        _fn = self.processor.post_process_grounded_object_detection
        _params = inspect.signature(_fn).parameters
        score_kwarg = (
            "threshold" if "threshold" in _params else "box_threshold"
        )
        results = _fn(
            outputs,
            inputs.input_ids,
            **{score_kwarg: self.cfg.box_threshold},
            text_threshold=self.cfg.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = results["boxes"].cpu().numpy()    # [N, 4]
        scores = results["scores"].cpu().numpy()  # [N]
        # `text_labels` (str) preferred; `labels` becomes int IDs in future
        labels = results.get("text_labels", results["labels"])  # list[str]

        # Suppress duplicate boxes for the same object (transformer decoders
        # can fire multiple predictions at slightly different coordinates).
        keep = _nms(boxes, scores, self.cfg.nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep]

        return SkillResult(
            skill_name="grounding_dino",
            success=True,
            data={
                "boxes":        boxes,
                "scores":       scores,
                "labels":       labels,
                "n_detections": len(boxes),
            },
            metadata={
                "model":          self.cfg.model_name,
                "box_threshold":  self.cfg.box_threshold,
                "text_threshold": self.cfg.text_threshold,
            },
        )
