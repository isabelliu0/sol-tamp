"""Extract predefined skill policies from TAMP systems."""

from typing import Any, Dict, Callable
import numpy as np
from numpy.typing import NDArray


class TAMPSkillPolicy:
    """Wraps a TAMP skill as a policy function."""

    def __init__(self, tamp_system, skill_name: str):
        self.tamp_system = tamp_system
        self.skill_name = skill_name
        self.skill = self._find_skill(skill_name)
        self.current_operator = None
        self.is_initialized = False

    def _find_skill(self, skill_name: str):
        for skill in self.tamp_system.skills:
            if skill.__class__.__name__ == skill_name:
                return skill

        available_skills = [s.__class__.__name__ for s in self.tamp_system.skills]
        raise ValueError(
            f"Skill '{skill_name}' not found in TAMP system. "
            f"Available skills: {available_skills}"
        )

    def __call__(self, obs: Any) -> NDArray:
        if not self.is_initialized:
            self._initialize_skill(obs)

        if not self.is_initialized:
            raise AssertionError(f"Skill {self.skill_name} could not be initialized - no valid groundings found")

        return self.skill.get_action(obs)

    def _initialize_skill(self, obs: Any):
        perceiver = self.tamp_system.perceiver
        current_atoms = perceiver.step(obs)
        objects = perceiver._get_objects()

        # Get operator name from skill's _get_operator_name() method
        operator_name = self.skill._get_operator_name()

        for operator in self.tamp_system.operators:
            if operator.name == operator_name:
                groundings = self._get_valid_groundings(operator, objects, current_atoms)
                if not groundings:
                    self.is_initialized = False
                    return
                ground_op = operator.ground(groundings[0])
                self.skill.reset(ground_op)
                self.current_operator = ground_op
                self.is_initialized = True
                return

        print(f"[{self.skill_name}] No matching operator found for name: {operator_name}")
        self.is_initialized = False

    def _get_valid_groundings(self, operator, objects, current_atoms):
        from itertools import product

        param_objects = []
        for param in operator.parameters:
            matching = [obj for obj in objects if obj.type == param.type]
            if not matching:
                return []
            param_objects.append(matching)

        all_groundings = list(product(*param_objects))
        valid_groundings = []

        for grounding in all_groundings:
            ground_op = operator.ground(grounding)
            if ground_op.preconditions.issubset(current_atoms):
                valid_groundings.append(grounding)

        return valid_groundings

    def reset(self):
        self.is_initialized = False
        self.current_operator = None


def get_predefined_skills(
    tamp_system,
    skill_names: list[str]
) -> Dict[str, Callable[[Any], NDArray]]:
    """Create predefined skill policy functions from TAMP system.

    Args:
        tamp_system: TAMP system instance
        skill_names: List of skill names (e.g., ['Grasp', 'Place', 'Reach'])

    Returns:
        Dict mapping 'skill_{name}' to policy functions
    """
    predefined_skills = {}

    print(f"\nAvailable skills in TAMP system:")
    for skill in tamp_system.skills:
        print(f"  - {skill.__class__.__name__}")

    for skill_name in skill_names:
        try:
            policy = TAMPSkillPolicy(tamp_system, skill_name)
            predefined_skills[f"skill_{skill_name}"] = policy
        except ValueError as e:
            print(f"Warning: {e}")
            print(f"Skipping skill '{skill_name}'")

    return predefined_skills
