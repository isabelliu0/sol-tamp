"""TAMP environment registration and factory for SOL."""

from typing import Optional, Callable, Any
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


TAMP_ENV_SPECS = {
    "cluttered_drawer": {
        "system_class": ClutteredDrawerTAMPSystem,
        "shortcut_specs": [
            {
                "name": "shortcut_0",
                "preconditions": ["On(block, table)"],
                "effects": ["InDrawer(block)"],
            }
        ],
        "skill_names": ["Grasp", "Place", "Reach"],
    },
    "obstacle_tower": {
        "system_class": GraphObstacleTowerTAMPSystem,
        "shortcut_specs": [
            {
                "name": "shortcut_0",
                "preconditions": ["On(block, table)"],
                "effects": ["On(block, target)"],
            }
        ],
        "skill_names": ["Grasp", "Place", "Reach"],
    },
    "cleanup_table": {
        "system_class": CleanupTableTAMPSystem,
        "shortcut_specs": [
            {
                "name": "shortcut_0",
                "preconditions": ["On(toy, table)"],
                "effects": ["InBin(toy)"],
            }
        ],
        "skill_names": ["Grasp", "Place", "Reach"],
    },
    "obstacle2d": {
        "system_class": GraphObstacle2DTAMPSystem,
        "shortcut_specs": [
            {
                "name": "shortcut_0",
                "preconditions": ["Holding(robot, ball)"],
                "effects": ["At(ball, target)"],
            }
        ],
        "skill_names": ["PickUp", "PutDown"],
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

    reward_computer = TAMPPredicateRewardComputer(
        tamp_system=tamp_system,
        shortcut_specs=spec["shortcut_specs"],
        skill_names=spec["skill_names"],
    )

    observation_encoder = _create_observation_encoder(tamp_system.env, spec)

    env = SOLEnvironmentWrapper(
        env=tamp_system.env,
        reward_computer=reward_computer,
        observation_encoder=observation_encoder,
    )

    if cfg.with_sol:
        reward_scale = {
            "shortcut_": cfg.reward_scale_shortcuts,
            "task_reward": cfg.reward_scale_task,
            "controller": 1.0,
        }
        base_policies = []

        for spec_item in spec["shortcut_specs"]:
            base_policies.append(spec_item["name"])

        for skill_name in spec["skill_names"]:
            reward_scale[f"skill_{skill_name}"] = cfg.reward_scale_skills
            base_policies.append(f"skill_{skill_name}")

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
