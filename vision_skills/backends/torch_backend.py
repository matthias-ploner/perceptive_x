import torch
from typing import Any, Dict
from .base_backend import BaseBackend


class TorchBackend(BaseBackend):
    """
    PyTorch backend for TorchScript or standard nn.Module models.

    Config keys:
        device   (str):  "cuda" | "cpu"
        dtype    (str):  "float32" | "float16" | "bfloat16"
        compile  (bool): enable torch.compile (requires PyTorch ≥ 2.0)
    """

    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get("device", "cuda")
        self.dtype = self._DTYPE_MAP.get(config.get("dtype", "float32"), torch.float32)
        self.use_compile = config.get("compile", False)
        self.model = None

    def load(self, model_path: str) -> None:
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        if self.use_compile:
            self.model = torch.compile(self.model)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tensors = {
            k: v.to(device=self.device, dtype=self.dtype) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            output = self.model(**tensors)
        return {"output": output}

    def unload(self) -> None:
        del self.model
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
