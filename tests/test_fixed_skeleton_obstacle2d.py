"""Test that a fixed skill sequence can solve obstacle2d task."""

import numpy as np
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from sol_tamp.tamp_envs import make_tamp_env, register_tamp_envs
from unittest.mock import MagicMock


def test_fixed_skill_sequence_obstacle2d():
    """Test obstacle2d with optimal skill sequence."""
    print("\n" + "="*70)
    print("Testing Fixed Skill Sequence on Obstacle2D")
    print("="*70)

    register_tamp_envs()

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 50

    video_dir = Path("videos/fixed_skeleton_test")
    video_dir.mkdir(parents=True, exist_ok=True)
    env = make_tamp_env("tamp_obstacle2d", cfg, {}, render_mode="rgb_array")
    env = RecordVideo(env, str(video_dir), episode_trigger=lambda x: True)

    # Navigate through wrapper stack to find HierarchicalWrapper
    hier_wrapper = env
    while hasattr(hier_wrapper, 'env'):
        if hasattr(hier_wrapper, 'base_policies'):
            break
        hier_wrapper = hier_wrapper.env

    # ['skill_GraphPickUpSkill', 'skill_GraphPickUpFromTargetSkill', 'skill_GraphPutDownSkill', 'skill_GraphPutDownOnTargetSkill']
    base_policies = hier_wrapper.base_policies if hasattr(hier_wrapper, 'base_policies') else []
    # ['skill_GraphPickUpSkill', 'skill_GraphPickUpFromTargetSkill', 'skill_GraphPutDownSkill', 'skill_GraphPutDownOnTargetSkill', 'controller']
    all_policies = hier_wrapper.policies if hasattr(hier_wrapper, 'policies') else []
    print(f"  Base policies: {base_policies}")
    print(f"  All policies: {all_policies}")

    # Check wrapper stack
    print(f"\n  Wrapper stack:")
    temp_env = env
    depth = 0
    while hasattr(temp_env, 'env'):
        print(f"    {depth}: {type(temp_env).__name__}")
        depth += 1
        temp_env = temp_env.env
    print(f"    {depth}: {type(temp_env).__name__} (base)")

    skill_sequence = [
        "skill_GraphPickUpFromTargetSkill_robot_block2_target_area",
        "skill_GraphPutDownSkill_robot_block2_table",
        "skill_GraphPickUpSkill_robot_block1_table",
        "skill_GraphPutDownOnTargetSkill_robot_block1_target_area",
    ]

    print(f"\n  Target skill sequence: {skill_sequence}")

    obs, info = env.reset()
    print(f"\n✓ Environment reset")

    total_reward = 0
    total_steps = 0
    max_steps = 300

    skill_idx = 0
    steps_in_current_skill = 0
    current_policy = None

    print(f"\n{'Step':<6} {'Policy':<40} {'Reward':<8} {'Done':<6}")
    print("-" * 70)

    for step in range(max_steps):
        if 'current_policy' in obs:
            policy_idx = int(obs['current_policy'][0])
            current_policy = all_policies[policy_idx] if policy_idx < len(all_policies) else 'unknown'

        if current_policy == 'controller':
            if skill_idx >= len(skill_sequence):
                print(f"\n  Ran out of skills in sequence at step {step}")
                break

            next_skill = skill_sequence[skill_idx]
            if next_skill not in base_policies:
                raise ValueError(f"Skill {next_skill} not in base_policies: {base_policies}")

            policy_choice = base_policies.index(next_skill)
            base_action = np.zeros(3, dtype=np.float32)
            action = (base_action, np.array([policy_choice]))

            if step < 5:
                print(f"\nDEBUG step {step}: Controller choosing {next_skill}")
                print(f"  action = {action}, type = {type(action)}")

            print(f"{step:<6} CONTROLLER -> {next_skill:<27} ", end="")
            skill_idx += 1
            steps_in_current_skill = 0
        else:
            base_action = np.zeros(3, dtype=np.float32)
            policy_choice = np.array([0])
            action = (base_action, policy_choice)
            steps_in_current_skill += 1

            if step < 10 and steps_in_current_skill == 1:
                print(f"\nDEBUG step {step}: Skill {current_policy} executing")
                print(f"  action before step = {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

        if current_policy != 'controller':
            if steps_in_current_skill == 1:
                print(f"{step:<6} {current_policy:<40} {reward:<8.2f} ", end="")
            elif steps_in_current_skill % 10 == 0:
                print(f".", end="", flush=True)

        if terminated or truncated:
            print(f"\n\n{'='*70}")
            if terminated:
                print(f"✓ Episode TERMINATED at step {step}")
            else:
                print(f"  Episode TRUNCATED at step {step}")

            print(f"\nFinal Statistics:")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Total steps: {total_steps}")
            print(f"  Skills executed: {skill_idx}/{len(skill_sequence)}")

            if 'episode_extra_stats' in info:
                print(f"\n  Episode stats:")
                for key, value in info['episode_extra_stats'].items():
                    print(f"    {key}: {value:.2f}")

            break
    else:
        print(f"\n\n  Reached max steps ({max_steps}) without episode ending")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Skills executed: {skill_idx}/{len(skill_sequence)}")

    print(f"\n{'='*70}")
    print("Validation:")

    if terminated:
        print("✓ Episode terminated (task completed)")
    else:
        print("✗ Episode did not terminate")

    env.close()
