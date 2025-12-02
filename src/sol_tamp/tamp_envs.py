"""TAMP environment registration and factory for SOL."""

import pickle
from pathlib import Path
from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Graph

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
from sol_tamp.adapters.sol_wrapper import SOLEnvironmentWrapper
from sol_tamp.adapters.reward_computers import TAMPPredicateRewardComputer
from sol_tamp.adapters.observation_encoder import ObservationEncoder
from sol_tamp.adapters.skill_override_wrapper import SkillOverrideWrapper
from sol_tamp.adapters.skill_policies import get_predefined_skills


def _load_trained_signatures(system_name: str, tamp_system) -> list[dict[str, any]]:
    """Load trained shortcut signatures from pickle file."""
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

    predicate_types = {}
    for pred in tamp_system.predicates:
        type_names = [str(t.name) for t in pred.types]
        predicate_types[pred.name] = type_names

    def _format_predicate(pred_name: str) -> str:
        """Format predicate with its actual type signature."""
        if pred_name in predicate_types:
            types = predicate_types[pred_name]
            if types:
                return f"{pred_name}({', '.join(types)})"
            else:
                return pred_name
        else:
            return pred_name

    shortcut_specs = []
    for i, sig in enumerate(trained_sigs):
        preconditions = [_format_predicate(pred) for pred in sig.source_predicates]
        effects = [_format_predicate(pred) for pred in sig.target_predicates]

        shortcut_specs.append({
            "name": f"shortcut_{i}",
            "preconditions": preconditions,
            "effects": effects,
        })

    return shortcut_specs


TAMP_ENV_SPECS = {
    "cluttered_drawer": {
        "system_class": ClutteredDrawerTAMPSystem,
        "system_name": "ClutteredDrawerTAMPSystem",
        "skill_names": [
            "ReachSkill",
            "GraspFrontBackSkill",
            "GraspFullClearSkill",
            "PlaceRightObjectSkill",
            "PlaceTargetSkill",
            "PlaceFrontObjectSkill",
            "GraspNonTargetSkill",
            "GraspLeftRightSkill",
            "PlaceBackObjectSkill",
            "PlaceLeftObjectSkill",
        ],
    },
    "obstacle_tower": {
        "system_class": GraphObstacleTowerTAMPSystem,
        "system_name": "GraphObstacleTowerTAMPSystem",
        "skill_names": ["Grasp", "Place", "Reach"],
    },
    "cleanup_table": {
        "system_class": CleanupTableTAMPSystem,
        "system_name": "CleanupTableTAMPSystem",
        "skill_names": ["Grasp", "Place", "Reach"],
    },
    "obstacle2d": {
        "system_class": GraphObstacle2DTAMPSystem,
        "system_name": "GraphObstacle2DTAMPSystem",
        "skill_names": [
            "GraphPickUpSkill",
            "GraphPickUpFromTargetSkill",
            "GraphPutDownSkill",
            "GraphPutDownOnTargetSkill",
        ],
    },
}


def _detect_graph_dimensions(env: gym.Env) -> tuple[int, int]:
    """Detect max_nodes and feature_dim from environment's observation space."""
    obs_space = env.observation_space
    if isinstance(obs_space, Graph):
        obs, _ = env.reset()
        nodes = obs.nodes if hasattr(obs, "nodes") else obs["nodes"]
        num_nodes = nodes.shape[0]
        feature_dim = nodes.shape[1] if len(nodes.shape) > 1 else nodes.shape[0]
        return num_nodes, feature_dim
    return None, None


def _create_observation_encoder(
    env: gym.Env, spec: dict
) -> Optional[ObservationEncoder]:
    """Create observation encoder with correct dimensions."""
    if "max_nodes" in spec and "feature_dim" in spec:
        return ObservationEncoder(
            max_nodes=spec["max_nodes"],
            feature_dim=spec["feature_dim"],
        )

    max_nodes, feature_dim = _detect_graph_dimensions(env)
    assert max_nodes is not None and feature_dim is not None
    return ObservationEncoder(
        max_nodes=max_nodes,
        feature_dim=feature_dim,
    )


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

    if cfg.use_shortcuts:
        shortcut_specs = _load_trained_signatures(spec["system_name"], tamp_system)
    else:
        shortcut_specs = []

    reward_computer = TAMPPredicateRewardComputer(
        tamp_system=tamp_system,
        shortcut_specs=shortcut_specs,
        skill_names=spec["skill_names"],
    )

    observation_encoder = _create_observation_encoder(tamp_system.env, spec)

    env = SOLEnvironmentWrapper(
        env=tamp_system.env,
        reward_computer=reward_computer,
        observation_encoder=observation_encoder,
    )

    if cfg.with_sol:
        # Get initial observation to enumerate objects for grounded skills
        initial_obs, _ = tamp_system.env.reset()
        predefined_skills = get_predefined_skills(tamp_system, spec["skill_names"], initial_obs)

        reward_scale = {
            "task_reward": cfg.reward_scale_task,
            "controller": 1.0,
            "shortcut_": cfg.reward_scale_shortcuts,
        }
        base_policies = []

        for spec_item in shortcut_specs:
            shortcut_name = spec_item["name"]
            base_policies.append(shortcut_name)

        # Add all grounded skills to base_policies and reward_scale
        for grounded_skill_name in predefined_skills.keys():
            # Add to base_policies with full grounded name
            base_policies.append(grounded_skill_name)
            # Add to reward_scale with full grounded name (each grounding gets its own scale)
            reward_scale[grounded_skill_name] = cfg.reward_scale_skills

        controller_reward_key = "task_reward"

        env = HierarchicalWrapper(
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            cfg.sol_num_option_steps,
        )

        env = SkillOverrideWrapper(env, predefined_skills, debug=cfg.debug)

    return env


def register_tamp_envs():
    """Register all TAMP environments with Sample Factory."""
    for env_name in TAMP_ENV_SPECS.keys():
        full_name = f"tamp_{env_name}"
        register_env(full_name, make_tamp_env)
