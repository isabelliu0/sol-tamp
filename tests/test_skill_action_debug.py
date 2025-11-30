"""Debug script to see what actions a skill returns."""

from sol_tamp.tamp_systems import get_tamp_system
from sol_tamp.tamp_envs import make_tamp_env, register_tamp_envs
from sol_tamp.adapters.skill_policies import TAMPSkillPolicy
from unittest.mock import MagicMock

register_tamp_envs()

# Create environment
cfg = MagicMock()
cfg.seed = 42
cfg.tamp_include_symbolic_features = True
cfg.with_sol = False  # Just basic env
cfg.reward_scale_shortcuts = 1.0
cfg.reward_scale_skills = 1.0
cfg.reward_scale_task = 10.0

env = make_tamp_env("tamp_obstacle2d", cfg, {}, render_mode="rgb_array")
obs, info = env.reset()

print(f"Initial observation type: {type(obs)}")
print(f"Info keys: {info.keys() if isinstance(info, dict) else 'not a dict'}")

# Get TAMP system
tamp_system = get_tamp_system("obstacle2d")

# Create skill policy
skill = TAMPSkillPolicy(tamp_system, "GraphPickUpFromTargetSkill")

# Try to get action
print("\n" + "="*70)
print("Testing GraphPickUpFromTargetSkill")
print("="*70)

for step in range(5):
    print(f"\nStep {step}:")
    action = skill(obs)
    print(f"  Returned action: {action}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action dtype: {action.dtype}")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.4f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        break

env.close()
