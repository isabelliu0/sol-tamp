"""Integration tests for SOL-TAMP pipeline."""

import pytest
from unittest.mock import MagicMock

from sol_tamp.tamp_envs import make_tamp_env, TAMP_ENV_SPECS


@pytest.mark.parametrize("env_name", list(TAMP_ENV_SPECS.keys()))
def test_all_tamp_envs_without_sol(env_name):
    """Test all TAMP environments can be created without SOL."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = False

    env = make_tamp_env(f"tamp_{env_name}", cfg, {})

    obs, info = env.reset()
    assert "intrinsic_rewards" in info
    assert "observation" in obs
    assert obs["observation"].shape == env.observation_space["observation"].shape

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert "intrinsic_rewards" in info
    assert isinstance(info["intrinsic_rewards"], dict)
    assert len(info["intrinsic_rewards"]) > 0
    assert "observation" in obs

    env.close()


@pytest.mark.parametrize("env_name", list(TAMP_ENV_SPECS.keys()))
def test_all_tamp_envs_with_sol(env_name):
    """Test all TAMP environments can be wrapped with HierarchicalWrapper."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env(f"tamp_{env_name}", cfg, {})

    obs, info = env.reset()
    assert "observation" in obs
    assert "current_policy" in obs
    assert "rewards" in obs

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert "observation" in obs
    assert isinstance(reward, (int, float))

    env.close()


def test_intrinsic_rewards_structure():
    """Test intrinsic rewards have correct structure."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()
    intrinsic_rewards = info["intrinsic_rewards"]

    assert isinstance(intrinsic_rewards, dict)
    assert "shortcut_0" in intrinsic_rewards
    assert any(k.startswith("skill_") for k in intrinsic_rewards.keys())

    for value in intrinsic_rewards.values():
        assert isinstance(value, float)

    env.close()


def test_hierarchical_wrapper_integration():
    """Test HierarchicalWrapper correctly wraps environment."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 2.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = -1

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()

    assert env.observation_space["observation"] is not None
    assert env.observation_space["current_policy"] is not None
    assert env.observation_space["rewards"] is not None

    assert obs["current_policy"].shape == (1,)
    assert len(obs["rewards"]) == len(env.policies)

    env.close()


def test_episode_rollout():
    """Test complete episode rollout."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = False
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()
    total_steps = 0
    max_steps = 20

    while total_steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        assert "intrinsic_rewards" in info
        assert "observation" in obs

        if terminated or truncated:
            break

    env.close()
