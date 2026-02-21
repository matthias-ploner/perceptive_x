import json
import numpy as np
from typing import Optional
from PIL import Image
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import ReasoningConfig
from .base_reasoning import BaseReasoningSkill


@register_skill("qwen_vl")
class QwenVLSkill(BaseReasoningSkill):
    """
    Visual reasoning with Qwen2.5-VL.

    Capabilities:
        - Visual question answering (VQA)
        - Object grounding / referring expressions
        - Scene understanding & description
        - Structured JSON output (auto-detected from model response)

    Usage:
        result = skill(image, prompt="What defects are visible?")
        result = skill(image, prompt="Return JSON with keys: objects, count",
                       system_prompt="You are a quality inspector.")
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = ReasoningConfig(**config)

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        attn = "flash_attention_2" if self.cfg.use_flash_attention else "eager"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            device_map=self.cfg.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name)
        self.logger.info(f"Qwen2.5-VL loaded: {self.cfg.model_name}")

    def preprocess(
        self,
        image: np.ndarray,
        prompt: str = "Describe the scene.",
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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[pil_image],
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
        return output_ids, inputs["input_ids"].shape[1], prompt

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        output_ids, input_len, prompt = raw_output
        generated = output_ids[:, input_len:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

        # Attempt to extract structured JSON from the response
        structured = None
        if "{" in response:
            try:
                clean = response.replace("```json", "").replace("```", "").strip()
                # Find first '{' and last '}'
                start = clean.index("{")
                end = clean.rindex("}") + 1
                structured = json.loads(clean[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        return SkillResult(
            skill_name="qwen_vl",
            success=True,
            data={
                "response": response,
                "structured": structured,
                "prompt": prompt,
            },
        )
