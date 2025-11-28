"""Test environment wrapping."""

import numpy as np

from sol_tamp.tamp_envs import make_tamp_env
from unittest.mock import MagicMock


def test_cluttered_drawer_wrapper():
    cfg = MagicMock()
    cfg.seed = 42
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    assert env.observation_space is not None
    assert env.action_space is not None

    obs, info = env.reset()
    assert "observation" in obs
    assert obs["observation"].shape == env.observation_space["observation"].shape
    assert "intrinsic_rewards" in info
    print(f"intrinsic rewards on reset: {info['intrinsic_rewards']}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert "observation" in obs
    assert obs["observation"].shape == env.observation_space["observation"].shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "intrinsic_rewards" in info

    intrinsic_rewards = info["intrinsic_rewards"]
    assert isinstance(intrinsic_rewards, dict)
    assert "shortcut_0" in intrinsic_rewards
    assert all(isinstance(v, float) for v in intrinsic_rewards.values())
    print(f"intrinsic rewards on step: {info['intrinsic_rewards']}")

    env.close()
