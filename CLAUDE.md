# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOL-TAMP is a hierarchical RL baseline that integrates:
- **SOL** (Scalable Option Learning) from Meta's sol repository
- **TAMP Improvisation** from the SLAP repository

The system uses pre-defined TAMP (Task-and-Motion Planning) skills as fixed option policies, while SOL learns new option policies for shortcut transitions. A high-level controller coordinates all options.

## Build and Development Commands

### Installation
```bash
# Install this package
pip install -e .

# Install development dependencies
pip install -e ".[develop]"

# Install SOL dependencies (required external dependency)
cd /path/to/sol
pip install -r requirements.txt
cd sol
python setup.py build_ext --inplace

# Add SOL to PYTHONPATH
export PYTHONPATH=/path/to/sol:$PYTHONPATH
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_env_wrapper.py

# Run specific test
pytest tests/test_integration.py::test_all_tamp_envs_with_sol

# Run tests with verbose output
pytest tests/ -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/ experiments/

# Sort imports
isort src/ tests/ experiments/

# Type checking
mypy src/

# Linting
pylint src/
```

### Training

**Via SOL's launch.py:**
```bash
cd /path/to/sol
python launch.py --expfile /path/to/sol-tamp/configs/sol_tamp.yaml --mode local --debug
```

**Direct Sample Factory:**
```bash
python experiments/train_tamp.py \
    --env=tamp_cluttered_drawer \
    --experiment=sol_cluttered_drawer \
    --with_sol=True \
    --sol_num_option_steps=-1 \
    --reward_scale_shortcuts=1.0 \
    --reward_scale_skills=1.0 \
    --reward_scale_task=10.0
```

## Architecture

### Core Components

**Environment Bridge (`src/sol_tamp/adapters/env_wrapper.py`)**
- `TAMPToSOLEnvironment`: Main wrapper that bridges TAMP environments with SOL's hierarchical RL framework
- Coordinates three specialized adapters: observation encoding, intrinsic reward computation, and skill option management
- Tracks symbolic state (current atoms, goal atoms) for reward computation
- Flattens GraphInstance observations into vectors suitable for neural networks

**Observation Processing (`src/sol_tamp/adapters/observation_encoder.py`)**
- `ObservationEncoder`: Converts GraphInstance observations to flat vectors
- Handles graph observations (nodes padded to `max_nodes=100`, features truncated to `feature_dim=64`)
- Optionally encodes symbolic atoms as one-hot vectors (50 current + 50 goal atoms)
- Dynamically assigns indices to atoms as they are encountered

**Intrinsic Rewards (`src/sol_tamp/adapters/intrinsic_rewards.py`)**
- `IntrinsicRewardComputer`: Bridges symbolic TAMP predicates to scalar RL rewards
- Tracks two reward types:
  - **Shortcut rewards**: Given when preconditions + effects are achieved (e.g., block moves from table to drawer)
  - **Skill rewards**: Given when a predefined skill successfully completes
- Matches predicate strings against current symbolic state to determine reward

**Skill Options (`src/sol_tamp/options/predefined_skill.py`)**
- `PredefinedSkillOption`: Wraps TAMP skills as SOL option policies
- Checks preconditions and termination conditions using symbolic predicates
- `SkillOptionManager`: Automatically grounds all lifted operators with valid object combinations
- Skill naming: `{operator_name}_{grounding}` (e.g., `Pick_block0`)

**Environment Registration (`src/sol_tamp/tamp_envs.py`)**
- `TAMP_ENV_SPECS`: Maps environment names to their TAMP system classes and shortcut signatures
- `make_tamp_env`: Factory that creates wrapped environments, optionally with SOL's `HierarchicalWrapper`
- When `with_sol=True`, creates reward scales and base policies for SOL's hierarchical learning
- `register_tamp_envs`: Registers all environments with Sample Factory

### Supported Environments

All environments follow the naming pattern `tamp_{env_name}`:
- `tamp_cluttered_drawer`: ClutteredDrawer from SLAP
- `tamp_obstacle_tower`: ObstacleTower from SLAP
- `tamp_cleanup_table`: CleanupTable from SLAP
- `tamp_obstacle2d`: Obstacle2D from SLAP

Each environment defines shortcut signatures that specify valid symbolic transitions for learning.

### Configuration

**Sample Factory Parameters (`src/sol_tamp/tamp_params.py`)**
- `tamp_override_defaults`: Provides TAMP-specific defaults for Sample Factory (learning rate, batch size, PPO parameters, etc.)
- `add_tamp_env_args`: Adds TAMP-specific command-line arguments:
  - `--tamp_include_symbolic_features`: Include symbolic atoms in observations
  - `--reward_scale_shortcuts`: Scale for shortcut intrinsic rewards
  - `--reward_scale_skills`: Scale for skill intrinsic rewards
  - `--reward_scale_task`: Scale for task completion reward

**Config File (`configs/sol_tamp.yaml`)**
- SOL launch configuration with hyperparameter sweeps for all environments
- Default reward scales: shortcuts=1.0, skills=1.0, task=10.0
- Training: 10M env steps with async RL, 8 workers, 4 envs per worker

### Data Flow

1. TAMP system provides raw observations (GraphInstance) and symbolic state (GroundAtoms)
2. `ObservationEncoder` flattens graph + symbolic features into vectors
3. Agent selects actions (either low-level or option selection depending on SOL mode)
4. `IntrinsicRewardComputer` analyzes predicate changes and action info to compute reward dict
5. `TAMPToSOLEnvironment` packages everything for Sample Factory/SOL
6. When wrapped with `HierarchicalWrapper`, SOL manages option policies and controller

### External Dependencies

**Critical**: This codebase requires two external repositories:
- **SOL** (https://github.com/facebookresearch/sol): Must be installed and in PYTHONPATH. Provides `HierarchicalWrapper` and option learning.
- **TAMP Improvisation** (https://github.com/isabelliu0/SLAP): Installed as git dependency. Provides TAMP environments and skills.
