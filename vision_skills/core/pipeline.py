from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
from .base_skill import BaseSkill, SkillResult
import logging

logger = logging.getLogger(__name__)


class SkillPipeline:
    """
    Composable pipeline that chains skills sequentially.
    Each step can access results from previous steps via the context dict.

    Example:
        pipeline = SkillPipeline([
            ("depth", depth_skill),
            ("seg",   seg_skill,  lambda img, ctx: {"mode": "auto"}),
            ("pose",  pose_skill, lambda img, ctx: {
                "depth": ctx["depth"].data["depth_map"],
                "mask":  ctx["seg"].data["masks"][0],
            }),
        ])
        results = pipeline.run(image)
    """

    def __init__(self, steps: List[Tuple]):
        """
        steps: list of (name, skill) or (name, skill, kwargs_fn)
        kwargs_fn: Callable[[np.ndarray, Dict[str, SkillResult]], dict]
        """
        self.steps = steps

    def run(
        self,
        image: np.ndarray,
        stop_on_failure: bool = False,
    ) -> Dict[str, SkillResult]:
        context: Dict[str, SkillResult] = {}

        for step in self.steps:
            name, skill = step[0], step[1]
            kwargs_fn: Optional[Callable] = step[2] if len(step) > 2 else None

            kwargs = kwargs_fn(image, context) if kwargs_fn else {}
            result = skill(image, **kwargs)
            context[name] = result

            if not result.success and stop_on_failure:
                logger.warning(f"Pipeline stopped at step '{name}': {result.error}")
                break

        return context

    def __repr__(self):
        step_names = [s[0] for s in self.steps]
        return f"SkillPipeline(steps={step_names})"
