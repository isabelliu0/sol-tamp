"""TAMP environment registration and factory for SOL."""

from typing import Optional

from sample_factory.envs.env_utils import register_env
from sol.hierarchical import HierarchicalWrapper

from tamp_improv.benchmarks.pybullet_cluttered_drawer import (
    ClutteredDrawerTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_cleanup_table import (
    CleanupTableTAMPSystem,
)
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from sol_tamp.adapters.env_wrapper import TAMPToSOLEnvironment


TAMP_ENV_SPECS = {
    "cluttered_drawer": {
        "system_class": ClutteredDrawerTAMPSystem,
        "shortcut_signatures": [
            (frozenset(["On(block, table)"]), frozenset(["InDrawer(block)"])),
        ],
    },
    "obstacle_tower": {
        "system_class": GraphObstacleTowerTAMPSystem,
        "shortcut_signatures": [
            (frozenset(["On(block, table)"]), frozenset(["On(block, target)"])),
        ],
    },
    "cleanup_table": {
        "system_class": CleanupTableTAMPSystem,
        "shortcut_signatures": [
            (frozenset(["On(toy, table)"]), frozenset(["InBin(toy)"])),
        ],
    },
    "obstacle2d": {
        "system_class": GraphObstacle2DTAMPSystem,
        "shortcut_signatures": [
            (frozenset(["Holding(robot, ball)"]), frozenset(["At(ball, target)"])),
        ],
    },
}


def make_tamp_env(
    full_env_name: str,
    cfg,
    env_config,
    render_mode: Optional[str] = None,
    **kwargs
):
    """Factory function to create TAMP environments for SOL.

    Args:
        full_env_name: Environment name (e.g., 'tamp_cluttered_drawer')
        cfg: Sample Factory config
        env_config: Additional environment config
        render_mode: Rendering mode for PyBullet envs
    """
    env_name = full_env_name.replace("tamp_", "")

    if env_name not in TAMP_ENV_SPECS:
        raise ValueError(
            f"Unknown TAMP environment: {env_name}. "
            f"Available: {list(TAMP_ENV_SPECS.keys())}"
        )

    spec = TAMP_ENV_SPECS[env_name]

    tamp_system = spec["system_class"].create_default(
        seed=cfg.seed if hasattr(cfg, "seed") else None,
        render_mode=render_mode,
    )

    env = TAMPToSOLEnvironment(
        tamp_system=tamp_system,
        shortcut_signatures=spec["shortcut_signatures"],
        include_symbolic_features=cfg.tamp_include_symbolic_features,
    )

    if cfg.with_sol:
        reward_scale = {
            "shortcut_": cfg.reward_scale_shortcuts,
            "task_reward": cfg.reward_scale_task,
            "controller": 1.0,
        }
        base_policies = []

        for i in range(len(spec["shortcut_signatures"])):
            base_policies.append(f"shortcut_{i}")

        all_skill_names = env.skill_manager.get_skill_names()
        unique_skill_types = set()
        for name in all_skill_names:
            parts = name.split("_")
            if len(parts) > 0:
                unique_skill_types.add(parts[0])

        for skill_type in unique_skill_types:
            reward_scale[f"skill_{skill_type}"] = cfg.reward_scale_skills
            base_policies.append(f"skill_{skill_type}")

        controller_reward_key = "task_reward"

        env = HierarchicalWrapper(
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            cfg.sol_num_option_steps,
        )

    return env


def register_tamp_envs():
    """Register all TAMP environments with Sample Factory."""
    for env_name in TAMP_ENV_SPECS.keys():
        full_name = f"tamp_{env_name}"
        register_env(full_name, make_tamp_env)
