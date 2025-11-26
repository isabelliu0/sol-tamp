"""Main adapter: wraps TAMP environments for SOL compatibility."""

from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
from numpy.typing import NDArray

from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from sol_tamp.adapters.intrinsic_rewards import IntrinsicRewardComputer
from sol_tamp.adapters.observation_encoder import ObservationEncoder
from sol_tamp.options.predefined_skill import SkillOptionManager


class TAMPToSOLEnvironment(gym.Wrapper):
    """Bridges TAMP environments with SOL's hierarchical RL framework.

    Key components for such adaptation:
    - Flatten GraphInstance observations
    - Compute intrinsic rewards from predicate changes
    - Execute skills as options
    - Track symbolic state for reward computation
    """

    def __init__(
        self,
        tamp_system: ImprovisationalTAMPSystem,
        shortcut_signatures: list[tuple[frozenset[str], frozenset[str]]],
        include_symbolic_features: bool = True,
    ):
        self.tamp_system = tamp_system
        self.env = tamp_system.env
        super().__init__(self.env)

        self.skill_manager = SkillOptionManager(tamp_system)
        skill_names = [name.split("_")[0] for name in self.skill_manager.get_skill_names()]
        unique_skill_names = list(dict.fromkeys(skill_names))

        self.reward_computer = IntrinsicRewardComputer(
            shortcut_signatures, unique_skill_names
        )

        self.obs_encoder = ObservationEncoder()
        self.include_symbolic_features = include_symbolic_features

        self.prev_atoms = set()
        self.current_atoms = set()
        self.goal_atoms = set()

        self._setup_observation_space()

    def _setup_observation_space(self):
        obs_dim = self.obs_encoder.get_output_dim(self.include_symbolic_features)
        self.observation_space = DictSpace(
            {
                "observation": Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
            }
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        _, self.current_atoms, self.goal_atoms = self.tamp_system.perceiver.reset(
            obs, info
        )
        self.prev_atoms = self.current_atoms.copy()

        flat_obs = self.obs_encoder.encode(
            obs,
            self.current_atoms if self.include_symbolic_features else None,
            self.goal_atoms if self.include_symbolic_features else None,
        )

        info["intrinsic_rewards"] = {
            name: 0.0 for name in self.reward_computer.get_all_reward_names()
        }
        info["current_atoms"] = self.current_atoms
        info["goal_atoms"] = self.goal_atoms

        return {"observation": flat_obs}, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.prev_atoms = self.current_atoms.copy()
        self.current_atoms = self.tamp_system.perceiver.step(obs)

        intrinsic_rewards = self.reward_computer.compute_rewards(
            self.prev_atoms, self.current_atoms, info
        )
        info["intrinsic_rewards"] = intrinsic_rewards

        flat_obs = self.obs_encoder.encode(
            obs,
            self.current_atoms if self.include_symbolic_features else None,
            self.goal_atoms if self.include_symbolic_features else None,
        )

        info["current_atoms"] = self.current_atoms
        info["goal_atoms"] = self.goal_atoms

        return {"observation": flat_obs}, reward, terminated, truncated, info
