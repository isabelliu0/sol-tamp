"""TAMP environment registration and factory for SOL."""

import pickle
from pathlib import Path
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


def _load_trained_signatures(system_name: str) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Load trained shortcut signatures from pickle file.

    Args:
        system_name: Name of the TAMP system (e.g., 'ClutteredDrawerTAMPSystem')

    Returns:
        List of shortcut signatures as (preconditions, effects) tuples.
        Each signature is converted from ShortcutSignature objects to frozenset pairs
        containing predicate names with placeholder objects.
    """
    signatures_path = Path("slap_data") / system_name / "trained_signatures.pkl"

    if not signatures_path.exists():
        print(
            f"Warning: Trained signatures not found at {signatures_path}. "
            "Using empty signature list. Run TAMP training data collection to generate signatures."
        )
        return []

    with open(signatures_path, "rb") as f:
        trained_sigs = pickle.load(f)

    print(f"Loaded {len(trained_sigs)} trained shortcut signatures for {system_name}")

    # Convert ShortcutSignature objects to (preconditions, effects) format
    # For now, we use predicate names with generic object placeholders
    # This allows the intrinsic reward computer to match based on predicates
    converted_signatures = []
    for sig in trained_sigs:
        # Create predicate strings with placeholder objects
        # e.g., "Holding" -> "Holding(obj)"
        source_preds = frozenset(
            f"{pred}(obj)" if pred not in ["GripperEmpty", "NotGripperEmpty"] else pred
            for pred in sig.source_predicates
        )
        target_preds = frozenset(
            f"{pred}(obj)" if pred not in ["GripperEmpty", "NotGripperEmpty"] else pred
            for pred in sig.target_predicates
        )
        converted_signatures.append((source_preds, target_preds))

    return converted_signatures


TAMP_ENV_SPECS = {
    "cluttered_drawer": {
        "system_class": ClutteredDrawerTAMPSystem,
        "system_name": "ClutteredDrawerTAMPSystem",
    },
    "obstacle_tower": {
        "system_class": GraphObstacleTowerTAMPSystem,
        "system_name": "GraphObstacleTowerTAMPSystem",
    },
    "cleanup_table": {
        "system_class": CleanupTableTAMPSystem,
        "system_name": "CleanupTableTAMPSystem",
    },
    "obstacle2d": {
        "system_class": GraphObstacle2DTAMPSystem,
        "system_name": "GraphObstacle2DTAMPSystem",
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

    # Load trained shortcut signatures from pickle files
    shortcut_signatures = _load_trained_signatures(spec["system_name"])

    env = TAMPToSOLEnvironment(
        tamp_system=tamp_system,
        shortcut_signatures=shortcut_signatures,
        include_symbolic_features=cfg.tamp_include_symbolic_features,
    )

    # Initialize skill manager by doing a temporary reset to get objects
    # This is needed because some perceivers (e.g., GraphObstacle2D) only
    # provide objects after reset
    temp_obs, temp_info = env.env.reset()
    objects, _, _ = tamp_system.perceiver.reset(temp_obs, temp_info)
    env.skill_manager.initialize_with_objects(objects)

    if cfg.with_sol:
        reward_scale = {
            "shortcut_": cfg.reward_scale_shortcuts,
            "task_reward": cfg.reward_scale_task,
            "controller": 1.0,
        }
        base_policies = []

        for i in range(len(shortcut_signatures)):
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
