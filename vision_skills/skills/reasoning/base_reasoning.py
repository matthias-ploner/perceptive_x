from abc import abstractmethod
from typing import Any
import numpy as np
from ...core.base_skill import BaseSkill, SkillResult


class BaseReasoningSkill(BaseSkill):
    """
    Base class for visual language model reasoning skills.

    Expected postprocess result data keys:
        - response   (str): free-text model response
        - structured (dict | None): parsed JSON if the response contained it
        - prompt     (str): the prompt that was used
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any: ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult: ...
