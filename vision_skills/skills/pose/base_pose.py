from abc import abstractmethod
from typing import Any
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BasePoseSkill(BaseSkill):
    """
    Base class for 6-DoF object pose estimation skills.

    Expected postprocess result data keys:
        - pose_matrix  (np.ndarray 4x4): SE(3) transformation (camera ← object)
        - rotation     (np.ndarray 3x3): rotation component
        - translation  (np.ndarray 3,):  translation in metres [x, y, z]
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
