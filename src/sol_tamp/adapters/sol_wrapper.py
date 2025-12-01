"""Generic SOL environment wrapper."""

from typing import Any, Optional, Callable
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace, Graph
import numpy as np
from numpy.typing import NDArray

from sol_tamp.adapters.reward_computers import IntrinsicRewardComputer
from sol_tamp.adapters.observation_encoder import ObservationEncoder


class SOLEnvironmentWrapper(gym.Wrapper):
    """Generic wrapper that adds intrinsic rewards for SOL.

    This wrapper:
    1. Computes intrinsic_rewards using a configurable reward computer
    2. Wraps observations in dict format (required by SOL's HierarchicalWrapper)
    3. Flattens non-Box observations (Graph, etc.) to vectors
    4. Tracks previous observations for reward computation
    """

    def __init__(
        self,
        env: gym.Env,
        reward_computer: IntrinsicRewardComputer,
        observation_encoder: Optional[Callable] = None,
        max_steps: int = 300,
        step_penalty: float = -0.01,
    ):
        """Initialize SOL wrapper.

        Args:
            env: Base environment
            reward_computer: Reward computer instance
            observation_encoder: Optional function to encode observations to flat vectors
        """
        super().__init__(env)
        self.reward_computer = reward_computer
        self.observation_encoder = observation_encoder

        self.prev_obs = None
        self.current_obs = None
        self.max_steps = max_steps
        self.step_count = 0
        self.step_penalty = step_penalty

        self._setup_observation_space()

    def _setup_observation_space(self):
        """Wrap observation space in dict format."""
        if isinstance(self.env.observation_space, DictSpace):
            self.observation_space = self.env.observation_space
        elif isinstance(self.env.observation_space, Box):
            self.observation_space = DictSpace(
                {"observation": self.env.observation_space}
            )
        elif isinstance(self.env.observation_space, Graph):
            if self.observation_encoder is None:
                self.observation_encoder = ObservationEncoder()
            obs_dim = self.observation_encoder.get_output_dim()
            self.observation_space = DictSpace(
                {"observation": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
            )
        else:
            self.observation_space = DictSpace(
                {"observation": self.env.observation_space}
            )

    def _encode_observation(self, obs: Any) -> NDArray[np.float32]:
        """Encode observation to flat vector if needed."""
        if self.observation_encoder is not None:
            return self.observation_encoder.encode(obs)
        return obs

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment."""
        obs, info = self.env.reset(seed=seed, options=options)

        self.prev_obs = obs
        self.current_obs = obs

        info['raw_obs'] = obs

        encoded_obs = self._encode_observation(obs)

        if not isinstance(encoded_obs, dict):
            encoded_obs = {"observation": encoded_obs}

        reward_names = self.reward_computer.get_reward_names()
        info["intrinsic_rewards"] = {name: 0.0 for name in reward_names}
        self.step_count = 0

        return encoded_obs, info

    def step(
        self, action
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step environment."""
        self.step_count += 1
        self.prev_obs = self.current_obs

        # NOTE: SLAP envs by default never truncate episodes, 
        # so we handle max_steps here
        obs, reward, terminated, _, info = self.env.step(action)
        if not terminated:
            reward += self.step_penalty
        self.current_obs = obs

        info['raw_obs'] = obs

        intrinsic_rewards = self.reward_computer.compute_rewards(
            self.prev_obs, self.current_obs, action, info
        )
        info["intrinsic_rewards"] = intrinsic_rewards

        encoded_obs = self._encode_observation(obs)

        if not isinstance(encoded_obs, dict):
            encoded_obs = {"observation": encoded_obs}

        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return encoded_obs, reward, terminated, truncated, info
