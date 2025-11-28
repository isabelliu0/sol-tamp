"""Integration tests for SOL-TAMP pipeline."""

import pytest
from unittest.mock import MagicMock

from sol_tamp.tamp_envs import make_tamp_env, TAMP_ENV_SPECS


@pytest.mark.parametrize("env_name", list(TAMP_ENV_SPECS.keys()))
def test_all_tamp_envs_without_sol(env_name):
    """Test all TAMP environments can be created without SOL."""
    print(f"\n{'='*60}")
    print(f"Testing {env_name} WITHOUT SOL")
    print(f"{'='*60}")

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = False

    env = make_tamp_env(f"tamp_{env_name}", cfg, {})
    print(f"  Observation space: {env.observation_space['observation'].shape}")
    print(f"  Action space: {env.action_space}")

    obs, info = env.reset()
    assert "intrinsic_rewards" in info
    assert "observation" in obs
    assert obs["observation"].shape == env.observation_space["observation"].shape

    print(f"✓ Environment reset successfully")
    print(f"  Intrinsic rewards: {list(info['intrinsic_rewards'].keys())}")
    print(f"  Num rewards: {len(info['intrinsic_rewards'])}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert "intrinsic_rewards" in info
    assert isinstance(info["intrinsic_rewards"], dict)
    assert len(info["intrinsic_rewards"]) > 0
    assert "observation" in obs

    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.3f}")
    print(f"  Sample intrinsic rewards: {dict(list(info['intrinsic_rewards'].items())[:3])}")

    env.close()
    print(f"✓ Test passed for {env_name}\n")


@pytest.mark.parametrize("env_name", list(TAMP_ENV_SPECS.keys()))
def test_all_tamp_envs_with_sol(env_name):
    """Test all TAMP environments can be wrapped with HierarchicalWrapper."""
    print(f"\n{'='*60}")
    print(f"Testing {env_name} WITH SOL")
    print(f"{'='*60}")

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env(f"tamp_{env_name}", cfg, {})
    print(f"✓ Environment wrapped with SOL HierarchicalWrapper")
    print(f"  Base policies: {env.base_policies}")
    print(f"  All policies: {env.policies}")
    print(f"  Option length: {cfg.sol_num_option_steps}")

    obs, info = env.reset()
    assert "observation" in obs
    assert "current_policy" in obs
    assert "rewards" in obs

    print(f"✓ SOL environment reset successfully")
    print(f"  Current policy index: {obs['current_policy'][0]}")
    print(f"  Current policy name: {env.policies[obs['current_policy'][0]]}")
    print(f"  Rewards shape: {obs['rewards'].shape}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert "observation" in obs
    assert isinstance(reward, (int, float))

    print(f"✓ SOL environment step successful")
    print(f"  Reward: {reward:.3f}")
    print(f"  New policy: {env.policies[obs['current_policy'][0]]}")

    env.close()
    print(f"✓ Test passed for {env_name} with SOL\n")


def test_intrinsic_rewards_structure():
    """Test intrinsic rewards have correct structure."""
    print(f"\n{'='*60}")
    print(f"Testing intrinsic rewards structure")
    print(f"{'='*60}")

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()
    intrinsic_rewards = info["intrinsic_rewards"]

    assert isinstance(intrinsic_rewards, dict)
    assert len(intrinsic_rewards) > 0, "Should have at least some rewards (skills or shortcuts)"
    assert any(k.startswith("skill_") for k in intrinsic_rewards.keys()), "Should have skill rewards"

    print(f"✓ Intrinsic rewards computed")
    print(f"  Total: {len(intrinsic_rewards)}")

    shortcuts = [k for k in intrinsic_rewards if k.startswith('shortcut')]
    skills = [k for k in intrinsic_rewards if k.startswith('skill')]

    print(f"  Shortcuts: {shortcuts if shortcuts else 'None (slap_data/ not found)'}")
    print(f"  Skills: {skills}")

    for value in intrinsic_rewards.values():
        assert isinstance(value, float)

    print(f"✓ All values are floats")
    print(f"  Sample: {dict(list(intrinsic_rewards.items())[:5])}\n")

    env.close()


def test_hierarchical_wrapper_integration():
    """Test HierarchicalWrapper correctly wraps environment."""
    print(f"\n{'='*60}")
    print(f"Testing SOL HierarchicalWrapper integration")
    print(f"{'='*60}")

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

    print(f"✓ Observation space extended with SOL fields")
    print(f"  Has 'current_policy': ✓")
    print(f"  Has 'rewards': ✓")
    print(f"  Total policies: {len(env.policies)}")

    assert obs["current_policy"].shape == (1,)
    assert len(obs["rewards"]) == len(env.policies)

    print(f"✓ Observations have correct shape")
    print(f"  Policy index shape: {obs['current_policy'].shape}")
    print(f"  Rewards shape: {obs['rewards'].shape}\n")

    env.close()


def test_episode_rollout():
    """Test complete episode rollout."""
    print(f"\n{'='*60}")
    print(f"Testing complete episode rollout")
    print(f"{'='*60}")

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = False
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()
    total_steps = 0
    max_steps = 20
    total_reward = 0

    while total_steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_reward += reward

        assert "intrinsic_rewards" in info
        assert "observation" in obs

        if terminated or truncated:
            print(f"  Episode ended at step {total_steps}")
            break

    print(f"✓ Rollout completed successfully")
    print(f"  Steps: {total_steps}/{max_steps}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Avg reward: {total_reward/total_steps:.3f}\n")

    env.close()
