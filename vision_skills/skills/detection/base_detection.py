from abc import abstractmethod
from typing import Any
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BaseDetectionSkill(BaseSkill):
    """
    Base class for object detection skills.

    Expected postprocess result data keys:
        - boxes         (np.ndarray [N,4]): bounding boxes [x1,y1,x2,y2] pixels
        - scores        (np.ndarray [N]):   confidence scores in [0, 1]
        - labels        (list[str]):        class label for each detection
        - n_detections  (int):              number of detections returned
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
