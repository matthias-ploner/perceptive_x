from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseBackend(ABC):
    """
    Abstract inference backend.

    Backends decouple skill logic from the execution runtime (PyTorch,
    TensorRT, ONNX Runtime, Triton Inference Server, etc.).

    A skill's infer() method can delegate to a backend like:
        output = self.backend.run({"input": tensor})
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load / compile the model for this backend."""
        ...

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference. Returns a dict of named outputs."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release resources held by the backend."""
        ...
