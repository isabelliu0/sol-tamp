"""Compute intrinsic rewards from symbolic state changes."""

from typing import Any
from relational_structs import GroundAtom


class IntrinsicRewardComputer:
    """Bridges symbolic TAMP predicates to scalar rewards for RL."""

    def __init__(
        self,
        shortcut_signatures: list[tuple[frozenset[str], frozenset[str]]],
        skill_names: list[str],
    ):
        self.shortcut_signatures = shortcut_signatures
        self.skill_names = skill_names
        self.shortcut_reward_names = [
            f"shortcut_{i}" for i in range(len(shortcut_signatures))
        ]
        self.skill_reward_names = [f"skill_{name}" for name in skill_names]

    def compute_rewards(
        self,
        prev_atoms: set[GroundAtom],
        current_atoms: set[GroundAtom],
        action_info: dict[str, Any],
    ) -> dict[str, float]:
        rewards = {}

        for i, (precond_strs, effect_strs) in enumerate(self.shortcut_signatures):
            reward_name = self.shortcut_reward_names[i]
            precond_satisfied = self._check_predicate_strings(prev_atoms, precond_strs)
            effects_achieved = self._check_predicate_strings(current_atoms, effect_strs)
            rewards[reward_name] = (
                1.0 if precond_satisfied and effects_achieved else 0.0
            )

        for skill_name in self.skill_names:
            reward_name = f"skill_{skill_name}"
            is_this_skill = action_info.get("action_type", "").startswith(
                f"skill_{skill_name}"
            )
            completed = action_info.get("skill_completed", False)
            rewards[reward_name] = 1.0 if (is_this_skill and completed) else 0.0

        return rewards

    def _check_predicate_strings(
        self, atoms: set[GroundAtom], predicate_strs: frozenset[str]
    ) -> bool:
        atom_strs = {str(atom) for atom in atoms}
        for pred_str in predicate_strs:
            if not any(pred_str in atom_str for atom_str in atom_strs):
                return False
        return True

    def get_all_reward_names(self) -> list[str]:
        return self.shortcut_reward_names + self.skill_reward_names
