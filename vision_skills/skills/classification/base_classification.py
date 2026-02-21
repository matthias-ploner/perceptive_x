from abc import abstractmethod
from typing import Any, Dict, List
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BaseClassificationSkill(BaseSkill):
    """
    Base class for few-shot classification and anomaly detection skills.

    Workflow:
        1. build_gallery(images_by_class) — register reference images per class
        2. __call__(image)                — classify or detect anomaly

    Expected postprocess result data keys:
        - predicted_class (str): winning class label or "anomaly"
        - confidence      (float): mean top-k cosine similarity
        - is_anomaly      (bool): True when confidence < threshold
        - top_k_labels    (list[str])
        - top_k_similarities (list[float])
        - votes           (dict[str, int])
    """

    @abstractmethod
    def build_gallery(self, images_by_class: Dict[str, List[np.ndarray]]) -> None:
        """Register reference images. Must be called before inference."""
        ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
