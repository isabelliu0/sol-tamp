"""Test environment wrapping."""

import numpy as np

from tamp_improv.benchmarks.pybullet_cluttered_drawer import (
    ClutteredDrawerTAMPSystem,
)
from sol_tamp.adapters.env_wrapper import TAMPToSOLEnvironment


def test_cluttered_drawer_wrapper():
    tamp_system = ClutteredDrawerTAMPSystem.create_default(seed=42)
    shortcut_signatures = [
        (frozenset({"On(block, table)"}), frozenset({"InDrawer(block)"})),
    ]

    env = TAMPToSOLEnvironment(
        tamp_system=tamp_system,
        shortcut_signatures=shortcut_signatures,
        include_symbolic_features=True,
    )

    assert env.observation_space is not None
    assert env.action_space is not None

    obs, info = env.reset()
    assert "observation" in obs
    assert obs["observation"].shape == env.observation_space["observation"].shape
    assert "intrinsic_rewards" in info
    assert "current_atoms" in info
    assert "goal_atoms" in info
    print(f"intrinsic rewards on reset: {info['intrinsic_rewards']}")
    print(f"current atoms on reset: {info['current_atoms']}")
    print(f"goal atoms on reset: {info['goal_atoms']}")

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
