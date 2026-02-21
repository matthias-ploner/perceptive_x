import json
import time
import numpy as np
from typing import Any, Dict, List
from ...core.registry import register_skill, SkillRegistry
from ...core.base_skill import BaseSkill, SkillResult
from ...config.skill_configs import SkillRouterConfig


_SKILL_CATALOGUE = """
Available vision skills (registry key → description):

  depth_anything_v3   Monocular depth estimation. kwargs: none required.
  sam2                Instance segmentation. kwargs: mode="auto" or
                      mode="points" with points=[[x,y],...], labels=[1,...].
  grounding_dino      Zero-shot text-prompted detection. kwargs:
                      text_prompt="class1 . class2 . class3"
  dinov2              Few-shot classification / anomaly detection.
                      Requires a pre-built gallery; use for anomaly tasks.
  qwen3_vl            Visual reasoning / VQA. kwargs: prompt="<question>",
                      system_prompt="<optional system message>".
  assembly_verification  Assembly completeness check. config:
                      component_checklist={"bolt":4,"washer":4}.
  gigapose            6-DoF pose estimation from RGB. config:
                      template_dir="<path>", intrinsics=[fx,fy,cx,cy].
"""

_SYSTEM_PROMPT = (
    "You are a vision pipeline planner for an industrial robot system.\n"
    "Given an image (described via a task) and a task description, output a "
    "minimal JSON execution plan — no explanation outside the JSON block.\n\n"
    + _SKILL_CATALOGUE
    + """
Output exactly this JSON schema:
{
  "plan": [
    {
      "step": 1,
      "skill": "<registry_key>",
      "config": {},
      "kwargs": {},
      "reason": "<one sentence>"
    }
  ],
  "summary": "<overall approach in one sentence>"
}

Rules:
- Use only skills listed above.
- config keys go into SkillRegistry.create(..., config); kwargs go into skill(image, **kwargs).
- Keep the plan to the minimum number of steps needed.
- If a single skill is enough, use one step.
"""
)


@register_skill("skill_router")
class SkillRouterSkill(BaseSkill):
    """
    LLM-powered skill router that composes vision skills dynamically.

    Given a free-text task description, the router asks an Anthropic LLM to
    produce a JSON execution plan, then runs each step via SkillRegistry.

    Requires the ANTHROPIC_API_KEY environment variable to be set.

    Usage:
        skill = SkillRegistry.create("skill_router", {})
        result = skill(image, task="Detect all bolts and count them.")
        # result.data["summary"]      str   — LLM's plan summary
        # result.data["plan"]         list  — planned steps
        # result.data["results"]      list  — SkillResult per step
        # result.data["final_result"] dict  — last step's result.data
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = SkillRouterConfig(**config)
        self._client = None

    def load(self) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.logger.info(
            f"SkillRouter ready — planner: {self.cfg.model}"
        )

    # ------------------------------------------------------------------
    # Override __call__ to handle multi-step orchestration cleanly
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray, task: str = "", **kwargs) -> SkillResult:
        if not self._loaded:
            self.load()
            self._loaded = True

        t0 = time.perf_counter()
        try:
            plan_json = self.preprocess(image, task=task, **kwargs)
            execution_results = self.infer(plan_json, image=image)
            result = self.postprocess(execution_results, plan_json=plan_json)
            result.inference_time_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as e:
            self.logger.exception(f"SkillRouter failed: {e}")
            return SkillResult(
                skill_name="skill_router",
                success=False,
                error=str(e),
                inference_time_ms=(time.perf_counter() - t0) * 1000,
            )

    # ------------------------------------------------------------------
    # BaseSkill interface
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, task: str = "", **kwargs) -> Dict:
        """Call the LLM planner and return the parsed JSON plan."""
        if not task:
            raise ValueError("task= must be provided, e.g. task='Detect bolts'")

        user_msg = f"Task: {task}\n\nProduce the minimal execution plan."
        response = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text.strip()

        # Extract JSON block (LLM may wrap in ```json ... ```)
        if "```" in raw_text:
            start = raw_text.index("{")
            end   = raw_text.rindex("}") + 1
            raw_text = raw_text[start:end]

        plan_json = json.loads(raw_text)
        steps = plan_json.get("plan", [])
        if len(steps) > self.cfg.max_steps:
            plan_json["plan"] = steps[: self.cfg.max_steps]
            self.logger.warning(
                f"Plan truncated to {self.cfg.max_steps} steps "
                f"(LLM returned {len(steps)})."
            )
        self.logger.info(
            f"Plan: {plan_json.get('summary', '')} "
            f"({len(plan_json['plan'])} step(s))"
        )
        return plan_json

    def infer(
        self,
        plan_json: Dict,
        image: np.ndarray = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute each step in the plan sequentially."""
        execution_log: List[Dict[str, Any]] = []
        ctx: Dict[str, Any] = {}  # shared context across steps

        for step in plan_json["plan"]:
            skill_name = step["skill"]
            config = step.get("config", {})
            skill_kwargs = step.get("kwargs", {})

            self.logger.info(
                f"Step {step['step']}: {skill_name} — {step.get('reason', '')}"
            )

            skill = SkillRegistry.create(skill_name, config)
            result = skill(image, **skill_kwargs)

            execution_log.append({
                "step":       step["step"],
                "skill":      skill_name,
                "reason":     step.get("reason", ""),
                "success":    result.success,
                "data":       result.data,
                "time_ms":    result.inference_time_ms,
            })
            ctx.update(result.data)

        return execution_log

    def postprocess(
        self,
        execution_log: List[Dict[str, Any]],
        plan_json: Dict = None,
        **kwargs,
    ) -> SkillResult:
        final_data = execution_log[-1]["data"] if execution_log else {}
        all_ok = all(s["success"] for s in execution_log)

        return SkillResult(
            skill_name="skill_router",
            success=all_ok,
            data={
                "summary":      plan_json.get("summary", "") if plan_json else "",
                "plan":         plan_json.get("plan", []) if plan_json else [],
                "results":      execution_log,
                "final_result": final_data,
            },
            metadata={"planner_model": self.cfg.model},
        )
