"""Extract predefined skill policies from TAMP systems."""

from typing import Any, Dict, Callable
import numpy as np
from numpy.typing import NDArray


class TAMPSkillPolicy:
    """Wraps a TAMP skill as a policy function."""

    def __init__(self, tamp_system, skill_name: str, grounding: tuple = None):
        self.tamp_system = tamp_system
        self.skill_name = skill_name
        self.skill = self._find_skill(skill_name)
        self.grounding = grounding  # Specific object grounding for this skill
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

        # Get operator name from skill's _get_operator_name() method
        operator_name = self.skill._get_operator_name()

        for operator in self.tamp_system.operators:
            if operator.name == operator_name:
                # Check if the specific grounding is valid
                if not self._is_grounding_valid(operator, self.grounding, current_atoms):
                    self.is_initialized = False
                    return

                ground_op = operator.ground(self.grounding)
                self.skill.reset(ground_op)
                self.current_operator = ground_op
                self.is_initialized = True
                return

        print(f"[{self.skill_name}] No matching operator found for name: {operator_name}")
        self.is_initialized = False

    def _is_grounding_valid(self, operator, grounding, current_atoms):
        """Check if a specific grounding satisfies the operator's preconditions."""
        if grounding is None:
            return False

        # Ground the operator with the specific objects
        ground_op = operator.ground(grounding)

        # Check if preconditions are satisfied
        return ground_op.preconditions.issubset(current_atoms)

    def reset(self):
        self.is_initialized = False
        self.current_operator = None


def get_predefined_skills(
    tamp_system,
    skill_names: list[str],
    initial_obs: Any,
    specific_groundings: list[tuple[str, tuple[str, ...]]] = None
) -> Dict[str, Callable[[Any], NDArray]]:
    """Create predefined skill policy functions from TAMP system.

    Args:
        tamp_system: TAMP system instance
        skill_names: List of skill names (e.g., ['Grasp', 'Place', 'Reach'])
        initial_obs: Initial observation to get objects from the environment
        specific_groundings: Optional list of (skill_name, object_names) tuples to create only specific groundings.
                           E.g., [('GraphPickUpSkill', ('robot', 'block1', 'table')), ...]
                           If None, creates all possible groundings.

    Returns:
        Dict mapping 'skill_{name}_{obj1}_{obj2}_...' to policy functions
    """
    from itertools import product

    predefined_skills = {}

    print(f"\nAvailable skills in TAMP system:")
    for skill in tamp_system.skills:
        print(f"  - {skill.__class__.__name__}")

    perceiver = tamp_system.perceiver
    perceiver.step(initial_obs)
    objects = perceiver._get_objects()

    print(f"\nObjects in environment:")
    for obj in objects:
        print(f"  - {obj.name} ({obj.type})")

    obj_by_name = {obj.name: obj for obj in objects}

    if specific_groundings is not None:
        print(f"\nUsing specific groundings (count: {len(specific_groundings)})")
        for skill_name, obj_names in specific_groundings:
            try:
                skill = None
                for s in tamp_system.skills:
                    if s.__class__.__name__ == skill_name:
                        skill = s
                        break

                if skill is None:
                    available_skills = [s.__class__.__name__ for s in tamp_system.skills]
                    raise ValueError(
                        f"Skill '{skill_name}' not found. Available: {available_skills}"
                    )

                grounding = tuple(obj_by_name[name] for name in obj_names)
                grounded_skill_name = f"skill_{skill_name}_{'_'.join(obj_names)}"

                policy = TAMPSkillPolicy(tamp_system, skill_name, grounding)
                predefined_skills[grounded_skill_name] = policy
                print(f"  Created: {grounded_skill_name}")

            except (ValueError, KeyError) as e:
                print(f"Warning: Failed to create {skill_name} with {obj_names}: {e}")
                continue

        print(f"\nTotal grounded skills created: {len(predefined_skills)}")
        return predefined_skills

    for skill_name in skill_names:
        try:
            # Find the skill
            skill = None
            for s in tamp_system.skills:
                if s.__class__.__name__ == skill_name:
                    skill = s
                    break

            if skill is None:
                available_skills = [s.__class__.__name__ for s in tamp_system.skills]
                raise ValueError(
                    f"Skill '{skill_name}' not found in TAMP system. "
                    f"Available skills: {available_skills}"
                )

            # Find the operator for this skill
            if not hasattr(skill, '_get_operator_name'):
                print(f"Warning: Skill '{skill_name}' does not have _get_operator_name() method. Skipping.")
                continue

            operator_name = skill._get_operator_name()
            operator = None
            for op in tamp_system.operators:
                if op.name == operator_name:
                    operator = op
                    break

            if operator is None:
                print(f"Warning: No operator found for skill '{skill_name}' (operator: {operator_name})")
                continue

            # Generate all possible groundings for this operator
            param_objects = []
            for param in operator.parameters:
                matching = [obj for obj in objects if obj.type == param.type]
                if not matching:
                    print(f"Warning: No objects of type '{param.type}' found for skill '{skill_name}'")
                    break
                param_objects.append(matching)

            if len(param_objects) != len(operator.parameters):
                continue

            all_groundings = list(product(*param_objects))

            # Create a policy for each grounding
            for grounding in all_groundings:
                # Create skill name with object names
                obj_names = '_'.join([obj.name for obj in grounding])
                grounded_skill_name = f"skill_{skill_name}_{obj_names}"

                policy = TAMPSkillPolicy(tamp_system, skill_name, grounding)
                predefined_skills[grounded_skill_name] = policy

        except ValueError as e:
            print(f"Warning: {e}")
            print(f"Skipping skill '{skill_name}'")

    print(f"\nTotal grounded skills created: {len(predefined_skills)}")
    return predefined_skills
