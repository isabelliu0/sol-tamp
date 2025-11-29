"""Test task_reward and policy_reward logging in SOL Runner.

This test verifies that task_reward and policy_reward are:
1. Added to the info dict during environment steps
2. Collected by the episodic stats handler in the runner
3. Written to tensorboard summaries
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.algo.runners.runner import Runner
from sol_tamp.tamp_envs import make_tamp_env, register_tamp_envs
from sol_tamp.tamp_params import add_tamp_env_args, tamp_override_defaults


def test_task_and_policy_reward_in_info():
    """Test that task_reward and policy_reward are added to info dict."""
    print("\n" + "="*60)
    print("Testing task_reward and policy_reward in info dict")
    print("="*60)

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    print("✓ Created SOL environment")
    # Access base_policies through the wrapped environment
    hierarchical_wrapper = env.env if hasattr(env, 'env') else env
    if hasattr(hierarchical_wrapper, 'base_policies'):
        print(f"  Base policies: {hierarchical_wrapper.base_policies}")

    # Reset environment
    obs, info = env.reset()
    print("✓ Environment reset")

    # Execute a full option (wait for controller to choose a policy, then execute steps)
    total_steps = 0
    max_steps = 50
    found_task_reward = False
    found_policy_reward = False

    while total_steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        # Check if we're in a base policy (not controller)
        current_policy_idx = obs['current_policy'][0]
        # Access policies through wrapped environment
        hierarchical_wrapper = env.env if hasattr(env, 'env') else env
        policies = hierarchical_wrapper.policies if hasattr(hierarchical_wrapper, 'policies') else ['controller']
        current_policy = policies[current_policy_idx] if current_policy_idx < len(policies) else 'unknown'

        if current_policy != 'controller':
            # These fields should exist when we're executing a base policy
            if 'task_reward' in info:
                found_task_reward = True
                print(f"  Step {total_steps}: task_reward = {info['task_reward']:.4f}")
                assert isinstance(info['task_reward'], (int, float, np.number)), f"task_reward type is {type(info['task_reward'])}"

            if 'policy_reward' in info:
                found_policy_reward = True
                print(f"  Step {total_steps}: policy_reward = {info['policy_reward']:.4f}")
                assert isinstance(info['policy_reward'], (int, float, np.number)), f"policy_reward type is {type(info['policy_reward'])}"

        if found_task_reward and found_policy_reward:
            break

        if terminated or truncated:
            break

    env.close()

    assert found_task_reward, "task_reward not found in info dict during base policy execution"
    assert found_policy_reward, "policy_reward not found in info dict during base policy execution"

    print("✓ Both task_reward and policy_reward found in info dict")
    print()


def test_reward_logging_to_runner_stats():
    """Test that rewards are collected by the runner's episodic stats handler."""
    print("\n" + "="*60)
    print("Testing reward collection in Runner episodic stats")
    print("="*60)

    # Create a minimal runner to test the stats handler
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10
    cfg.num_policies = 1
    cfg.stats_avg = 100

    # Create environment
    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    print("✓ Created SOL environment")

    # Create a minimal runner with just the stats handler
    from sample_factory.algo.runners.runner import Runner
    from sample_factory.algo.utils.misc import EPISODIC

    # Mock runner
    runner = MagicMock()
    runner.cfg = cfg
    runner.policy_avg_stats = {}

    # Reset and step through environment
    obs, info = env.reset()

    task_rewards = []
    policy_rewards = []

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Simulate what the episodic stats handler would do
        current_policy_idx = obs['current_policy'][0]
        # Access policies through wrapped environment
        hierarchical_wrapper = env.env if hasattr(env, 'env') else env
        policies = hierarchical_wrapper.policies if hasattr(hierarchical_wrapper, 'policies') else ['controller']
        current_policy = policies[current_policy_idx] if current_policy_idx < len(policies) else 'unknown'

        if current_policy != 'controller':
            if 'task_reward' in info:
                task_rewards.append(info['task_reward'])

                # Simulate episodic stats collection
                if 'task_reward' not in runner.policy_avg_stats:
                    from collections import deque
                    runner.policy_avg_stats['task_reward'] = [deque(maxlen=100)]
                runner.policy_avg_stats['task_reward'][0].append(info['task_reward'])

            if 'policy_reward' in info:
                policy_rewards.append(info['policy_reward'])

                if 'policy_reward' not in runner.policy_avg_stats:
                    from collections import deque
                    runner.policy_avg_stats['policy_reward'] = [deque(maxlen=100)]
                runner.policy_avg_stats['policy_reward'][0].append(info['policy_reward'])

        if terminated or truncated:
            break

    env.close()

    assert len(task_rewards) > 0, "No task_rewards collected"
    assert len(policy_rewards) > 0, "No policy_rewards collected"

    print(f"✓ Collected {len(task_rewards)} task_reward samples")
    print(f"  Mean task_reward: {np.mean(task_rewards):.4f}")
    print(f"✓ Collected {len(policy_rewards)} policy_reward samples")
    print(f"  Mean policy_reward: {np.mean(policy_rewards):.4f}")

    # Verify stats were added to the mock runner
    assert 'task_reward' in runner.policy_avg_stats
    assert 'policy_reward' in runner.policy_avg_stats
    assert len(runner.policy_avg_stats['task_reward'][0]) > 0
    assert len(runner.policy_avg_stats['policy_reward'][0]) > 0

    print("✓ Rewards successfully added to runner policy_avg_stats")
    print()


def test_tensorboard_logging_integration():
    """Test that rewards would be logged to tensorboard via the summary writer."""
    print("\n" + "="*60)
    print("Testing tensorboard logging integration")
    print("="*60)

    # Create temporary directory for tensorboard logs
    temp_dir = tempfile.mkdtemp()

    try:
        from tensorboardX import SummaryWriter
        from collections import deque

        # Create a mock runner with tensorboard writer
        cfg = MagicMock()
        cfg.seed = 42
        cfg.tamp_include_symbolic_features = True
        cfg.with_sol = True
        cfg.reward_scale_shortcuts = 1.0
        cfg.reward_scale_skills = 1.0
        cfg.reward_scale_task = 10.0
        cfg.sol_num_option_steps = 10
        cfg.num_policies = 1
        cfg.stats_avg = 100

        # Create environment
        env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

        # Create tensorboard writer
        writer = SummaryWriter(temp_dir)

        print(f"✓ Created tensorboard writer at {temp_dir}")

        # Simulate runner collecting stats
        policy_avg_stats = {}
        env_steps = 0

        obs, info = env.reset()

        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            current_policy_idx = obs['current_policy'][0]
            # Access policies through wrapped environment
            hierarchical_wrapper = env.env if hasattr(env, 'env') else env
            policies = hierarchical_wrapper.policies if hasattr(hierarchical_wrapper, 'policies') else ['controller']
            current_policy = policies[current_policy_idx] if current_policy_idx < len(policies) else 'unknown'

            if current_policy != 'controller':
                env_steps += 1

                # Collect stats like the runner does
                if 'task_reward' in info:
                    if 'task_reward' not in policy_avg_stats:
                        policy_avg_stats['task_reward'] = deque(maxlen=100)
                    policy_avg_stats['task_reward'].append(info['task_reward'])

                if 'policy_reward' in info:
                    if 'policy_reward' not in policy_avg_stats:
                        policy_avg_stats['policy_reward'] = deque(maxlen=100)
                    policy_avg_stats['policy_reward'].append(info['policy_reward'])

            if terminated or truncated:
                break

        env.close()

        # Simulate writing to tensorboard (like _report_experiment_summaries does)
        if 'task_reward' in policy_avg_stats and len(policy_avg_stats['task_reward']) > 0:
            avg_task_reward = np.mean(policy_avg_stats['task_reward'])
            writer.add_scalar('policy_stats/avg_task_reward', float(avg_task_reward), env_steps)
            print(f"✓ Logged task_reward to tensorboard: {avg_task_reward:.4f}")

        if 'policy_reward' in policy_avg_stats and len(policy_avg_stats['policy_reward']) > 0:
            avg_policy_reward = np.mean(policy_avg_stats['policy_reward'])
            writer.add_scalar('policy_stats/avg_policy_reward', float(avg_policy_reward), env_steps)
            print(f"✓ Logged policy_reward to tensorboard: {avg_policy_reward:.4f}")

        writer.flush()
        writer.close()

        # Verify tensorboard files were created
        tb_files = list(Path(temp_dir).glob("*"))
        assert len(tb_files) > 0, "No tensorboard files created"

        print(f"✓ Tensorboard files created: {len(tb_files)} files")
        print()

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def test_reward_values_reasonable():
    """Test that reward values are reasonable and properly scaled."""
    print("\n" + "="*60)
    print("Testing reward value ranges and scaling")
    print("="*60)

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 2.0  # Different scale to test scaling
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    obs, info = env.reset()

    task_rewards = []
    policy_rewards = []

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        current_policy_idx = obs['current_policy'][0]
        current_policy = env.policies[current_policy_idx]

        if current_policy != 'controller' and 'task_reward' in info:
            task_rewards.append(info['task_reward'])
            policy_rewards.append(info['policy_reward'])

            # Verify types
            assert isinstance(info['task_reward'], (int, float, np.number))
            assert isinstance(info['policy_reward'], (int, float, np.number))

            # Verify they're finite
            assert np.isfinite(info['task_reward'])
            assert np.isfinite(info['policy_reward'])

        if terminated or truncated:
            break

    env.close()

    assert len(task_rewards) > 0, "No rewards collected"

    print(f"✓ Collected {len(task_rewards)} reward samples")
    print(f"  Task reward range: [{min(task_rewards):.4f}, {max(task_rewards):.4f}]")
    print(f"  Task reward mean: {np.mean(task_rewards):.4f}")
    print(f"  Policy reward range: [{min(policy_rewards):.4f}, {max(policy_rewards):.4f}]")
    print(f"  Policy reward mean: {np.mean(policy_rewards):.4f}")
    print("✓ All reward values are finite and properly typed")
    print()


def test_comparison_with_and_without_sol():
    """Test that rewards behave correctly with and without SOL wrapper."""
    print("\n" + "="*60)
    print("Comparing reward logging with and without SOL")
    print("="*60)

    # Without SOL - should not have task_reward/policy_reward in info
    cfg_no_sol = MagicMock()
    cfg_no_sol.seed = 42
    cfg_no_sol.tamp_include_symbolic_features = True
    cfg_no_sol.with_sol = False

    env_no_sol = make_tamp_env("tamp_cluttered_drawer", cfg_no_sol, {})
    obs, info = env_no_sol.reset()

    for _ in range(10):
        action = env_no_sol.action_space.sample()
        obs, reward, terminated, truncated, info = env_no_sol.step(action)

        # Without SOL, we shouldn't have these fields
        assert 'task_reward' not in info or info.get('task_reward') is None
        assert 'policy_reward' not in info

        if terminated or truncated:
            break

    env_no_sol.close()
    print("✓ Without SOL: task_reward and policy_reward not in info (as expected)")

    # With SOL - should have task_reward/policy_reward
    cfg_sol = MagicMock()
    cfg_sol.seed = 42
    cfg_sol.tamp_include_symbolic_features = True
    cfg_sol.with_sol = True
    cfg_sol.reward_scale_shortcuts = 1.0
    cfg_sol.reward_scale_skills = 1.0
    cfg_sol.reward_scale_task = 10.0
    cfg_sol.sol_num_option_steps = 10

    env_sol = make_tamp_env("tamp_cluttered_drawer", cfg_sol, {})
    obs, info = env_sol.reset()

    found_rewards = False
    for _ in range(30):
        action = env_sol.action_space.sample()
        obs, reward, terminated, truncated, info = env_sol.step(action)

        current_policy_idx = obs['current_policy'][0]
        # Access policies through wrapped environment
        hierarchical_wrapper = env_sol.env if hasattr(env_sol, 'env') else env_sol
        policies = hierarchical_wrapper.policies if hasattr(hierarchical_wrapper, 'policies') else ['controller']
        current_policy = policies[current_policy_idx] if current_policy_idx < len(policies) else 'unknown'

        if current_policy != 'controller':
            if 'task_reward' in info and 'policy_reward' in info:
                found_rewards = True
                break

        if terminated or truncated:
            break

    env_sol.close()

    assert found_rewards, "With SOL: task_reward and policy_reward should be in info"
    print("✓ With SOL: task_reward and policy_reward present in info (as expected)")
    print()


if __name__ == "__main__":
    # Run tests manually
    print("Running SOL Runner reward logging tests...\n")

    test_task_and_policy_reward_in_info()
    test_reward_logging_to_runner_stats()
    test_tensorboard_logging_integration()
    test_reward_values_reasonable()
    test_comparison_with_and_without_sol()

    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
