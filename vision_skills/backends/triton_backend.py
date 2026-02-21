import numpy as np
from typing import Any, Dict
from .base_backend import BaseBackend


class TritonBackend(BaseBackend):
    """
    NVIDIA Triton Inference Server backend via tritonclient.

    Config keys:
        url          (str):  Triton server URL, e.g. "localhost:8001"
        model_name   (str):  Model name as registered in Triton
        model_version(str):  Model version string, default "1"
        protocol     (str):  "grpc" | "http"  (default "grpc")

    Install:
        pip install tritonclient[grpc]   # or [http]
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config["url"]
        self.model_name = config["model_name"]
        self.model_version = config.get("model_version", "1")
        self.protocol = config.get("protocol", "grpc")
        self.client = None

    def load(self, model_path: str = "") -> None:
        if self.protocol == "grpc":
            import tritonclient.grpc as grpcclient
            self.client = grpcclient.InferenceServerClient(url=self.url)
            self._infer_input_cls = grpcclient.InferInput
            self._infer_output_cls = grpcclient.InferRequestedOutput
        else:
            import tritonclient.http as httpclient
            self.client = httpclient.InferenceServerClient(url=self.url)
            self._infer_input_cls = httpclient.InferInput
            self._infer_output_cls = httpclient.InferRequestedOutput

        if not self.client.is_model_ready(self.model_name):
            raise RuntimeError(
                f"Model '{self.model_name}' is not ready on Triton server at {self.url}"
            )

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        triton_inputs = []
        for name, array in inputs.items():
            t = self._infer_input_cls(name, array.shape, _np_to_triton_dtype(array.dtype))
            t.set_data_from_numpy(array)
            triton_inputs.append(t)

        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=triton_inputs,
        )
        return {
            out.name: response.as_numpy(out.name)
            for out in response.get_output()
        }

    def unload(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def _np_to_triton_dtype(dtype: np.dtype) -> str:
    _MAP = {
        np.float32: "FP32",
        np.float16: "FP16",
        np.int32:   "INT32",
        np.int64:   "INT64",
        np.uint8:   "UINT8",
    }
    return _MAP.get(dtype.type, "FP32")
