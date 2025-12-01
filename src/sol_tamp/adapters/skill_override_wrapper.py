"""Wrapper to override neural network actions with predefined skill policies."""

from typing import Any, Dict, Callable
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class SkillOverrideWrapper(gym.Wrapper):
    """Overrides actions for predefined skills while allowing shortcuts to be learned.

    When a predefined skill's preconditions are not met, returns a no-op action.
    The controller learns from the resulting low task reward that this skill
    should not be selected in this state.
    """

    def __init__(
        self,
        env: gym.Env,
        predefined_skills: Dict[str, Callable[[Any], NDArray]],
        tamp_system=None,
        debug: bool = False,
    ):
        super().__init__(env)
        self.predefined_skills = predefined_skills
        self.tamp_system = tamp_system
        self.last_obs = None
        self.last_raw_obs = None
        self.last_policy = None
        self.debug = debug

        if hasattr(env, 'policies'):
            for skill_name in predefined_skills.keys():
                assert skill_name in env.policies, \
                    f"Skill '{skill_name}' not found in env.policies: {env.policies}"

    @property
    def policies(self):
        """Pass through policies attribute from wrapped environment."""
        if hasattr(self.env, 'policies'):
            return self.env.policies
        return None

    @property
    def current_policy(self):
        """Pass through current_policy attribute from wrapped environment."""
        if hasattr(self.env, 'current_policy'):
            return self.env.current_policy
        return None

    def reset(self, seed=None, options=None):
        if self.debug:
            obs, info = self.env.reset(seed=seed, options={"reset_debug": True})
        else:
            obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        self.last_raw_obs = info.get('raw_obs', None)
        self.last_policy = None

        return obs, info

    def step(self, action):
        if hasattr(self.env, 'current_policy') and self.env.current_policy is not None:
            current_policy_name = self.env.current_policy

            if current_policy_name in self.predefined_skills and current_policy_name != 'controller':
                skill_policy = self.predefined_skills[current_policy_name]

                if current_policy_name != self.last_policy:
                    skill_policy.is_initialized = False
                    self.last_policy = current_policy_name

                skill_obs = self.last_raw_obs if self.last_raw_obs is not None else self.last_obs.get('observation')

                try:
                    predefined_action = skill_policy(skill_obs)

                    # Convert to list for HierarchicalWrapper compatibility
                    if isinstance(action, tuple):
                        action = [predefined_action] + list(action[1:])
                    else:
                        action = list(action) if not isinstance(action, list) else action
                        action[0] = predefined_action

                except (AssertionError, ValueError, RuntimeError, IndexError) as e:
                    # IndexError: PyBullet skills raise this when motion plan is exhausted
                    # AssertionError: TAMP skills raise this when preconditions not met
                    # ValueError/RuntimeError: Other skill execution failures
                    # print(f"[SkillOverrideWrapper] Skill {current_policy_name} failed: {type(e).__name__}: {e}")
                    # base_action_space = self.env.env.env.action_space if hasattr(self.env, 'env') else self.env.action_space
                    base_action_space = self.unwrapped.action_space
                    noop_action = np.zeros(base_action_space.shape, dtype=base_action_space.dtype)

                    # Convert to list for HierarchicalWrapper compatibility
                    if isinstance(action, tuple):
                        action = [noop_action] + list(action[1:])
                    else:
                        action = list(action) if not isinstance(action, list) else action
                        action[0] = noop_action

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs

        if 'raw_obs' in info:
            self.last_raw_obs = info['raw_obs']

        return obs, reward, terminated, truncated, info
