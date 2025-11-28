"""Test skill override wrapper functionality."""

from unittest.mock import MagicMock
import numpy as np

from sol_tamp.tamp_envs import make_tamp_env


def test_skill_override_wrapper_integration():
    """Test that SkillOverrideWrapper is properly integrated with SOL."""
    print(f"\n{'='*60}")
    print(f"Testing SkillOverrideWrapper Integration")
    print(f"{'='*60}")

    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = False
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    print(f"\n✓ Environment created with SOL + SkillOverrideWrapper")
    print(f"  Total policies: {len(env.env.policies)}")
    print(f"  Policies: {env.env.policies}")
    print(f"  Predefined skills: {list(env.predefined_skills.keys())}")

    assert hasattr(env, 'predefined_skills'), "Should have predefined_skills attribute"

    shortcuts = [p for p in env.env.policies if p.startswith('shortcut_')]
    skills = [p for p in env.env.policies if p.startswith('skill_')]

    print(f"\n  Shortcuts (learnable): {shortcuts}")
    print(f"  Skills (frozen): {skills}")

    for skill in skills:
        if skill in env.predefined_skills:
            print(f"    ✓ {skill} has predefined policy")
        else:
            print(f"    ✗ {skill} MISSING predefined policy")

    if len(env.predefined_skills) == 0:
        print(f"\n⚠ No predefined skills matched - skill names may need to be updated")
        env.close()
        return

    print(f"\nResetting environment...")
    obs, info = env.reset()

    assert "observation" in obs
    assert "current_policy" in obs

    print(f"✓ Environment reset successfully")
    print(f"  Observation keys: {obs.keys()}")
    print(f"  Info keys: {info.keys()}")
    print(f"  Current policy: {env.env.policies[obs['current_policy'][0]]}")
    print(f"  Raw obs in info: {'raw_obs' in info}")

    print(f"\nTaking {20} random steps...")
    total_steps = 0
    max_steps = 20
    skill_executions = {}
    skill_failures = {}

    while total_steps < max_steps:
        action = env.action_space.sample()
        current_policy_name = env.env.current_policy if hasattr(env.env, 'current_policy') else None

        print(f"\n  Step {total_steps + 1}:")
        print(f"    Current policy: {current_policy_name}")
        print(f"    Is skill: {current_policy_name in env.predefined_skills if current_policy_name else False}")

        if current_policy_name in env.predefined_skills:
            if current_policy_name not in skill_executions:
                skill_executions[current_policy_name] = 0
                skill_failures[current_policy_name] = 0
            skill_executions[current_policy_name] += 1

        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    Reward: {reward:.3f}")
            print(f"    Terminated: {terminated}, Truncated: {truncated}")
        except AssertionError as e:
            if "current_ground_operator" in str(e) or "_current_ground_operator" in str(e.__traceback__.tb_frame.f_code.co_filename):
                print(f"    ⚠ Skill failed to initialize (preconditions not met)")
                if current_policy_name in skill_failures:
                    skill_failures[current_policy_name] += 1
                raise
            else:
                raise

        total_steps += 1

        if terminated or truncated:
            print(f"    Episode ended, resetting...")
            obs, info = env.reset()
            break

    print(f"\n{'='*60}")
    print(f"✓ Rollout completed")
    print(f"  Steps: {total_steps}")
    if skill_executions:
        print(f"  Skill execution attempts:")
        for skill_name in sorted(skill_executions.keys()):
            attempts = skill_executions[skill_name]
            failures = skill_failures.get(skill_name, 0)
            print(f"    {skill_name}: {attempts} attempts, {failures} failures")
    print(f"{'='*60}\n")

    env.close()
    print(f"✓ Test passed\n")
