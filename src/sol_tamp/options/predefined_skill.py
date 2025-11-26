"""Wrap TAMP skills as SOL option policies."""

import numpy as np
from numpy.typing import NDArray
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Skill, GroundOperator


class PredefinedSkillOption:
    """Wraps a TAMP Skill to act as a SOL option policy."""

    def __init__(self, skill: Skill, ground_operator: GroundOperator):
        self.skill = skill
        self.operator = ground_operator
        self.is_initialized = False

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.is_initialized:
            self.skill.reset(self.operator)
            self.is_initialized = True
        return self.skill.get_action(obs)

    def check_preconditions(self, current_atoms: set[GroundAtom]) -> bool:
        return self.operator.preconditions.issubset(current_atoms)

    def check_termination(self, current_atoms: set[GroundAtom]) -> bool:
        add_ok = self.operator.add_effects.issubset(current_atoms)
        delete_ok = self.operator.delete_effects.isdisjoint(current_atoms)
        return add_ok and delete_ok

    def reset(self):
        self.is_initialized = False


class SkillOptionManager:
    """Manages multiple skill options."""

    def __init__(self, tamp_system):
        self.tamp_system = tamp_system
        self.skill_options: dict[str, PredefinedSkillOption] = {}
        self._is_initialized = False

    def _build_skill_options(self, objects):
        """Build skill options from available objects.

        Args:
            objects: Set or list of objects from the perceiver
        """
        objects = list(objects)
        for lifted_operator in self.tamp_system.operators:
            groundings = self._get_valid_groundings(lifted_operator, objects)
            for grounding in groundings:
                ground_op = lifted_operator.ground(grounding)
                for skill in self.tamp_system.skills:
                    if skill.can_execute(ground_op):
                        key = f"{lifted_operator.name}_{ground_op.short_str}"
                        self.skill_options[key] = PredefinedSkillOption(
                            skill, ground_op
                        )

    def _get_valid_groundings(self, operator, objects):
        from itertools import product

        param_objects = []
        for param in operator.parameters:
            matching = [obj for obj in objects if obj.type == param.type]
            if not matching:
                return []
            param_objects.append(matching)
        return list(product(*param_objects))

    def initialize_with_objects(self, objects):
        """Initialize skill options with objects from the environment.

        This should be called after the environment is reset and objects are available.

        Args:
            objects: Set or list of objects from the perceiver
        """
        if not self._is_initialized:
            self._build_skill_options(objects)
            self._is_initialized = True

    def get_skill_names(self) -> list[str]:
        """Get list of skill names. Returns empty list if not initialized."""
        return list(self.skill_options.keys())

    def get_skill_option(self, skill_name: str) -> PredefinedSkillOption:
        """Get a specific skill option by name."""
        return self.skill_options[skill_name]
