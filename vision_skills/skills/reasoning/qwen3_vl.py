import json
import numpy as np
from typing import Optional
from PIL import Image
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import ReasoningV3Config
from .base_reasoning import BaseReasoningSkill


@register_skill("qwen3_vl")
class Qwen3VLSkill(BaseReasoningSkill):
    """
    Visual reasoning with Qwen3-VL.

    Improvements over Qwen2.5-VL:
        - Interleaved-MRoPE: better spatial / temporal understanding
        - 256K native context (expandable to 1M)
        - 32-language OCR, robust in challenging conditions
        - Stronger structured JSON and agent-style outputs

    Variants (all on HuggingFace):
        Qwen/Qwen3-VL-4B-Instruct     — compact, runs on CPU
        Qwen/Qwen3-VL-8B-Instruct     — best quality/speed balance  ← default
        Qwen/Qwen3-VL-32B-Instruct    — highest quality, GPU required
        Qwen/Qwen3-VL-30B-A3B-Instruct — MoE, efficient at large scale

    Usage:
        result = skill(image, prompt="What can you see in this image?")
        result = skill(image,
                       prompt="List objects as JSON: {objects: [{name, depth_hint}]}",
                       system_prompt="You are a robotic perception system.")
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = ReasoningV3Config(**config)

    def load(self) -> None:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        attn = "flash_attention_2" if self.cfg.use_flash_attention else "eager"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            device_map=self.cfg.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name)
        self.logger.info(f"Qwen3-VL loaded: {self.cfg.model_name}")

    def preprocess(
        self,
        image: np.ndarray,
        prompt: str = "What can you see in this image? Describe the scene concisely.",
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        pil_image = Image.fromarray(image)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt},
            ],
        })

        # Qwen3-VL: apply_chat_template returns tokenised inputs directly
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.cfg.device)
        return inputs, prompt

    def infer(self, processed, **kwargs):
        import torch
        inputs, prompt = processed
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=self.cfg.temperature > 0,
            )
        # Trim the input prefix from each output sequence
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        return trimmed, prompt

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        trimmed, prompt = raw_output
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Attempt to extract structured JSON from the response
        structured = None
        if "{" in response:
            try:
                clean = response.replace("```json", "").replace("```", "").strip()
                start = clean.index("{")
                end = clean.rindex("}") + 1
                structured = json.loads(clean[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        return SkillResult(
            skill_name="qwen3_vl",
            success=True,
            data={
                "response": response,
                "structured": structured,
                "prompt": prompt,
            },
        )
