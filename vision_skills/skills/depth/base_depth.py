from abc import abstractmethod
from typing import Any, Dict
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BaseDepthSkill(BaseSkill):
    """
    Base class for monocular depth estimation skills.

    Subclasses must implement load/preprocess/infer/postprocess.
    The postprocess result data dict is expected to contain:
        - depth_map  (np.ndarray HxW float32): raw depth values
        - depth_norm (np.ndarray HxW float32): depth normalised to [0, 1]
        - depth_min  (float)
        - depth_max  (float)
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
