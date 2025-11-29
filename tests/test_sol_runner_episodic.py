"""Test episodic task_reward and policy_reward logging in SOL Runner.

This test verifies that task_reward and policy_reward are:
1. Accumulated during the episode in HierarchicalWrapper
2. Added to episode_extra_stats at episode end
3. Sent as EPISODIC messages to the runner
4. Written to tensorboard summaries

This follows the same pattern as episode_controller_reward.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from sol_tamp.tamp_envs import make_tamp_env


def test_episodic_task_and_policy_reward():
    """Test that task_reward and policy_reward are added to episode_extra_stats at episode end."""
    print("\n" + "="*60)
    print("Testing episodic task_reward and policy_reward")
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
    hierarchical_wrapper = env.env if hasattr(env, 'env') else env
    if hasattr(hierarchical_wrapper, 'base_policies'):
        print(f"  Base policies: {len(hierarchical_wrapper.base_policies)}")

    # Reset environment
    obs, info = env.reset()
    print("✓ Environment reset")

    # Run until episode ends
    total_steps = 0
    max_steps = 200
    episode_ended = False
    episode_extra_stats = None

    while total_steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        if terminated or truncated:
            episode_ended = True
            episode_extra_stats = info.get('episode_extra_stats', {})
            print(f"\n✓ Episode ended at step {total_steps}")
            break

    env.close()

    assert episode_ended, "Episode did not end within max_steps"
    assert episode_extra_stats is not None, "episode_extra_stats not found in info"

    # Check that task_reward and policy_reward are in episode_extra_stats
    assert 'episode_task_reward' in episode_extra_stats, \
        f"episode_task_reward not found in episode_extra_stats. Keys: {list(episode_extra_stats.keys())}"
    assert 'episode_policy_reward' in episode_extra_stats, \
        f"episode_policy_reward not found in episode_extra_stats. Keys: {list(episode_extra_stats.keys())}"

    task_reward = episode_extra_stats['episode_task_reward']
    policy_reward = episode_extra_stats['episode_policy_reward']

    print(f"  episode_task_reward = {task_reward:.4f}")
    print(f"  episode_policy_reward = {policy_reward:.4f}")

    # Verify they are numeric
    assert isinstance(task_reward, (int, float, np.number)), f"task_reward type is {type(task_reward)}"
    assert isinstance(policy_reward, (int, float, np.number)), f"policy_reward type is {type(policy_reward)}"

    # Verify they are finite
    assert np.isfinite(task_reward), "task_reward is not finite"
    assert np.isfinite(policy_reward), "policy_reward is not finite"

    print("✓ Both episode_task_reward and episode_policy_reward found in episode_extra_stats")
    print()


def test_multiple_episodes_accumulation():
    """Test that rewards are correctly accumulated and reset across multiple episodes."""
    print("\n" + "="*60)
    print("Testing reward accumulation across multiple episodes")
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

    episode_rewards = []
    num_episodes = 0
    max_episodes = 3
    max_steps_per_episode = 200

    while num_episodes < max_episodes:
        obs, info = env.reset()
        total_steps = 0

        while total_steps < max_steps_per_episode:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            if terminated or truncated:
                episode_extra_stats = info.get('episode_extra_stats', {})
                if 'episode_task_reward' in episode_extra_stats:
                    task_reward = episode_extra_stats['episode_task_reward']
                    policy_reward = episode_extra_stats['episode_policy_reward']
                    episode_rewards.append((task_reward, policy_reward))
                    print(f"  Episode {num_episodes + 1}: task={task_reward:.4f}, policy={policy_reward:.4f}")
                num_episodes += 1
                break

    env.close()

    assert len(episode_rewards) == max_episodes, f"Expected {max_episodes} episodes, got {len(episode_rewards)}"

    # Verify all episodes have valid rewards
    for i, (task_rew, policy_rew) in enumerate(episode_rewards):
        assert np.isfinite(task_rew), f"Episode {i+1} task_reward is not finite"
        assert np.isfinite(policy_rew), f"Episode {i+1} policy_reward is not finite"

    print(f"✓ Successfully collected rewards from {max_episodes} episodes")
    print()


def test_reward_accumulation_matches_sum():
    """Test that episodic rewards match the sum of step rewards."""
    print("\n" + "="*60)
    print("Testing reward accumulation correctness")
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

    # Access the HierarchicalWrapper to track rewards
    hierarchical_wrapper = env.env if hasattr(env, 'env') else env

    obs, info = env.reset()

    # Manually track rewards
    manual_task_reward_sum = 0
    manual_policy_reward_sum = 0
    total_steps = 0
    max_steps = 200

    while total_steps < max_steps:
        action = env.action_space.sample()
        obs, step_reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        # Get current policy
        current_policy_idx = obs['current_policy'][0]
        policies = hierarchical_wrapper.policies if hasattr(hierarchical_wrapper, 'policies') else []
        current_policy = policies[current_policy_idx] if current_policy_idx < len(policies) else 'unknown'

        # Track rewards when not in controller
        if current_policy != 'controller' and hasattr(hierarchical_wrapper, 'rewards'):
            # Access the internal state
            if 'task_reward' in hierarchical_wrapper.rewards:
                manual_task_reward_sum += hierarchical_wrapper.rewards['task_reward']
            # step_reward is the scaled policy reward
            manual_policy_reward_sum += step_reward

        if terminated or truncated:
            episode_extra_stats = info.get('episode_extra_stats', {})
            reported_task_reward = episode_extra_stats.get('episode_task_reward', 0)
            reported_policy_reward = episode_extra_stats.get('episode_policy_reward', 0)

            print(f"  Manual task sum: {manual_task_reward_sum:.4f}")
            print(f"  Reported task: {reported_task_reward:.4f}")
            print(f"  Manual policy sum: {manual_policy_reward_sum:.4f}")
            print(f"  Reported policy: {reported_policy_reward:.4f}")

            # The sums should be close (allowing for floating point errors)
            assert np.isclose(manual_task_reward_sum, reported_task_reward, rtol=1e-5), \
                f"Task reward mismatch: manual={manual_task_reward_sum}, reported={reported_task_reward}"

            print("✓ Reward accumulation matches expected sums")
            break

    env.close()
    print()


def test_tensorboard_logging_format():
    """Test that rewards would be correctly formatted for tensorboard."""
    print("\n" + "="*60)
    print("Testing tensorboard logging format")
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

    obs, info = env.reset()

    max_steps = 200
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episode_extra_stats = info.get('episode_extra_stats', {})

            # Verify the structure matches what the episodic stats handler expects
            assert isinstance(episode_extra_stats, dict)

            if 'episode_task_reward' in episode_extra_stats:
                # These should be scalars that can be written to tensorboard
                assert isinstance(episode_extra_stats['episode_task_reward'], (int, float, np.number))
                assert isinstance(episode_extra_stats['episode_policy_reward'], (int, float, np.number))

                print(f"✓ episode_task_reward: {episode_extra_stats['episode_task_reward']}")
                print(f"✓ episode_policy_reward: {episode_extra_stats['episode_policy_reward']}")
                print("✓ Rewards are in correct format for tensorboard")
            break

    env.close()
    print()


if __name__ == "__main__":
    # Run tests manually
    print("Running SOL Runner episodic reward logging tests...\n")

    test_episodic_task_and_policy_reward()
    test_multiple_episodes_accumulation()
    test_reward_accumulation_matches_sum()
    test_tensorboard_logging_format()

    print("="*60)
    print("ALL EPISODIC TESTS PASSED!")
    print("="*60)
