"""Test TAMP environment registration and factory."""

from unittest.mock import MagicMock

from sol_tamp.tamp_envs import make_tamp_env, register_tamp_envs, TAMP_ENV_SPECS


def test_env_specs_defined():
    """Test that all expected environments are defined."""
    expected_envs = ["cluttered_drawer", "obstacle_tower", "cleanup_table", "obstacle2d"]
    for env_name in expected_envs:
        assert env_name in TAMP_ENV_SPECS
        assert "system_class" in TAMP_ENV_SPECS[env_name]
        assert "shortcut_specs" in TAMP_ENV_SPECS[env_name]
        assert "skill_names" in TAMP_ENV_SPECS[env_name]


def test_shortcut_specs_format():
    """Test that shortcut specs are properly formatted."""
    for env_name, spec in TAMP_ENV_SPECS.items():
        shortcuts = spec["shortcut_specs"]
        assert isinstance(shortcuts, list)
        for shortcut in shortcuts:
            assert "name" in shortcut
            assert "preconditions" in shortcut
            assert "effects" in shortcut
            assert isinstance(shortcut["preconditions"], list)
            assert isinstance(shortcut["effects"], list)


def test_make_tamp_env_cluttered_drawer():
    """Test creating cluttered drawer environment."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.with_sol = False

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    assert env is not None
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")
    assert hasattr(env, "reset")
    assert hasattr(env, "step")

    env.close()


def test_make_tamp_env_with_sol():
    """Test creating environment with SOL wrapper."""
    cfg = MagicMock()
    cfg.seed = 42
    cfg.tamp_include_symbolic_features = True
    cfg.with_sol = True
    cfg.reward_scale_shortcuts = 1.0
    cfg.reward_scale_skills = 1.0
    cfg.reward_scale_task = 10.0
    cfg.sol_num_option_steps = 10

    env = make_tamp_env("tamp_cluttered_drawer", cfg, {})

    assert env is not None
    obs, info = env.reset()

    assert "observation" in obs
    assert "current_policy" in obs
    assert "rewards" in obs

    env.close()


def test_register_tamp_envs():
    """Test that environment registration doesn't crash."""
    register_tamp_envs()
