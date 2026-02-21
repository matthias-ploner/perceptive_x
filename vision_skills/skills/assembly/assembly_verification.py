import numpy as np
from typing import Dict
from ...core.registry import register_skill, SkillRegistry
from ...core.base_skill import BaseSkill, SkillResult
from ...config.skill_configs import AssemblyConfig


@register_skill("assembly_verification")
class AssemblyVerificationSkill(BaseSkill):
    """
    Assembly completeness verification via Grounding DINO detection.

    Detects the components listed in component_checklist and compares
    observed counts against required counts.

    Usage:
        skill = SkillRegistry.create("assembly_verification", {
            "component_checklist": {"bolt": 4, "washer": 4, "nut": 1},
        })
        result = skill(image)
        # result.data["all_present"]       bool  — True if assembly is complete
        # result.data["component_counts"]  dict  — detected count per component
        # result.data["missing"]    list  — components below required count
        # result.data["extra"]      list  — components above required count
        # result.data["boxes"]             np.ndarray [N,4]
        # result.data["scores"]            np.ndarray [N]
        # result.data["labels"]            list[str]
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = AssemblyConfig(**config)
        self._detector = None

    def load(self) -> None:
        self._detector = SkillRegistry.create("grounding_dino", {
            "model_name":        self.cfg.detection_model,
            "device":            self.cfg.device,
            "box_threshold":     self.cfg.box_threshold,
            "text_threshold":    self.cfg.text_threshold,
            "nms_iou_threshold": self.cfg.nms_iou_threshold,
        })
        self.logger.info(
            "AssemblyVerification ready — checklist: "
            + str(self.cfg.component_checklist)
        )

    def preprocess(self, image: np.ndarray, **kwargs):
        if not self.cfg.component_checklist:
            raise ValueError(
                "component_checklist is empty. "
                "Pass e.g. {\"bolt\": 4, \"washer\": 4}."
            )
        text_prompt = " . ".join(self.cfg.component_checklist.keys())
        det_result = self._detector(image, text_prompt=text_prompt)
        return det_result

    def infer(self, processed, **kwargs):
        return processed  # detection result passes straight through

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        det = raw_output
        if not det.success:
            return SkillResult(
                skill_name="assembly_verification",
                success=False,
                error=f"Detection failed: {det.error}",
            )
        labels = det.data["labels"]
        boxes = det.data["boxes"]
        scores = det.data["scores"]

        detected_counts: Dict[str, int] = {}
        for lbl in labels:
            detected_counts[lbl] = detected_counts.get(lbl, 0) + 1

        checklist = self.cfg.component_checklist
        missing: list = []
        extra: list = []
        for component, required in checklist.items():
            found = detected_counts.get(component, 0)
            if found < required - self.cfg.count_tolerance:
                missing.append(
                    f"{component}: need {required}, found {found}"
                )
            elif found > required + self.cfg.count_tolerance:
                extra.append(
                    f"{component}: need {required}, found {found}"
                )

        all_present = len(missing) == 0 and len(extra) == 0

        return SkillResult(
            skill_name="assembly_verification",
            success=True,
            data={
                "all_present":      all_present,
                "component_counts": detected_counts,
                "expected_counts":  dict(checklist),
                "missing":          missing,
                "extra":            extra,
                "boxes":            boxes,
                "scores":           scores,
                "labels":           labels,
            },
            metadata={
                "count_tolerance": self.cfg.count_tolerance,
            },
        )
