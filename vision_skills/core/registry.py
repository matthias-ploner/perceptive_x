from typing import Dict, Type
from .base_skill import BaseSkill


class SkillRegistry:
    """Central registry for skill discovery and instantiation."""

    _registry: Dict[str, Type[BaseSkill]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a skill class."""
        def decorator(skill_cls: Type[BaseSkill]):
            cls._registry[name] = skill_cls
            return skill_cls
        return decorator

    @classmethod
    def create(cls, name: str, config: dict) -> BaseSkill:
        if name not in cls._registry:
            raise KeyError(
                f"Skill '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](config)

    @classmethod
    def list_skills(cls) -> list:
        return list(cls._registry.keys())


# Convenience alias
register_skill = SkillRegistry.register
