"""Configurable intrinsic reward computers for SOL."""

from abc import ABC, abstractmethod
from typing import Any


class IntrinsicRewardComputer(ABC):
    """Base class for computing intrinsic rewards."""

    @abstractmethod
    def compute_rewards(
        self,
        prev_obs: Any,
        current_obs: Any,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Compute intrinsic rewards.

        Args:
            prev_obs: Previous observation
            current_obs: Current observation
            action: Action taken
            info: Info dict from environment

        Returns:
            Dict mapping reward names to scalar values
        """
        pass

    @abstractmethod
    def get_reward_names(self) -> list[str]:
        """Get list of all reward names this computer produces."""
        pass


class TAMPPredicateRewardComputer(IntrinsicRewardComputer):
    """Compute rewards based on TAMP predicate changes."""

    def __init__(
        self,
        tamp_system,
        shortcut_specs: list[dict[str, Any]],
        skill_names: list[str],
    ):
        """Initialize TAMP reward computer.

        Args:
            tamp_system: TAMP system instance
            shortcut_specs: List of shortcut specifications, each with:
                - name: reward name (e.g., 'shortcut_push_to_drawer')
                - preconditions: list of predicate strings
                - effects: list of predicate strings
            skill_names: List of skill base names (e.g., ['pick', 'place'])
        """
        self.tamp_system = tamp_system
        self.shortcut_specs = shortcut_specs
        self.skill_names = skill_names

        self.prev_atoms = set()
        self.current_atoms = set()

    def compute_rewards(
        self,
        prev_obs: Any,
        current_obs: Any,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Compute intrinsic rewards from predicate changes and action info."""
        self.prev_atoms = info.get("prev_atoms", set())
        self.current_atoms = info.get("current_atoms", set())

        rewards = {}

        shortcut_reward = 0.0
        for spec in self.shortcut_specs:
            precond_strs = spec.get("preconditions", [])
            effect_strs = spec.get("effects", [])

            precond_satisfied = self._check_predicates(self.prev_atoms, precond_strs)
            effects_achieved = self._check_predicates(self.current_atoms, effect_strs)

            if precond_satisfied and effects_achieved:
                shortcut_reward = 1.0
                break

        rewards["shortcut_"] = shortcut_reward

        for skill_name in self.skill_names:
            reward_name = f"skill_{skill_name}"
            action_type = info.get("action_type", "")
            skill_completed = info.get("skill_completed", False)

            is_this_skill = skill_name.lower() in action_type.lower()
            rewards[reward_name] = 1.0 if (is_this_skill and skill_completed) else 0.0

        return rewards

    def _check_predicates(self, atoms: set, pred_strs: list[str]) -> bool:
        """Check if predicate strings are satisfied in atoms."""
        for pred_str in pred_strs:
            matched = any(pred_str in str(atom) for atom in atoms)
            if not matched:
                return False
        return True

    def get_reward_names(self) -> list[str]:
        """Get all reward names (using metric names with digits removed)."""
        names = ["shortcut_"] if self.shortcut_specs else []
        names.extend([f"skill_{name}" for name in self.skill_names])
        return names


class SimpleRewardComputer(IntrinsicRewardComputer):
    """Simple reward computer using user-provided functions."""

    def __init__(self, reward_functions: dict[str, callable]):
        """Initialize with reward functions.

        Args:
            reward_functions: Dict mapping reward names to functions
                Each function takes (prev_obs, current_obs, action, info) -> float
        """
        self.reward_functions = reward_functions

    def compute_rewards(
        self,
        prev_obs: Any,
        current_obs: Any,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Compute rewards using provided functions."""
        return {
            name: func(prev_obs, current_obs, action, info)
            for name, func in self.reward_functions.items()
        }

    def get_reward_names(self) -> list[str]:
        """Get reward names."""
        return list(self.reward_functions.keys())
