from abc import abstractmethod
from typing import Any
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BaseSegmentationSkill(BaseSkill):
    """
    Base class for segmentation skills.

    Expected postprocess result data keys:
        - masks  (np.ndarray NxHxW bool): binary masks per instance
        - scores (np.ndarray N float32): confidence / stability scores
        - bboxes (np.ndarray Nx4 or None): bounding boxes in xywh
        - num_masks (int)
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
