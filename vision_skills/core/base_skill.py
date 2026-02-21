from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Unified result container for all skills."""

    skill_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    error: Optional[str] = None

    def __bool__(self):
        return self.success


class BaseSkill(ABC):
    """
    Abstract base for all vision AI skills.
    Enforces: load → preprocess → infer → postprocess → SkillResult
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
        self.model = None
        self._loaded = False
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize."""
        ...

    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> Any:
        """Normalize/resize/tensorize input."""
        ...

    @abstractmethod
    def infer(self, processed: Any, **kwargs) -> Any:
        """Run model forward pass."""
        ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> SkillResult:
        """Convert raw output to SkillResult."""
        ...

    def __call__(self, image: np.ndarray, **kwargs) -> SkillResult:
        if not self._loaded:
            self.load()
            self._loaded = True

        t0 = time.perf_counter()
        try:
            processed = self.preprocess(image, **kwargs)
            raw = self.infer(processed, **kwargs)
            result = self.postprocess(raw, **kwargs)
            result.inference_time_ms = (time.perf_counter() - t0) * 1000
            self.logger.debug(
                f"{self.__class__.__name__} | {result.inference_time_ms:.1f}ms | success={result.success}"
            )
            return result
        except Exception as e:
            self.logger.exception(f"Skill inference failed: {e}")
            return SkillResult(
                skill_name=self.__class__.__name__,
                success=False,
                error=str(e),
                inference_time_ms=(time.perf_counter() - t0) * 1000,
            )

    def unload(self) -> None:
        """Release model from memory."""
        self.model = None
        self._loaded = False

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device}, loaded={self._loaded})"
