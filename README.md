# SOL-TAMP Baseline

Integration of Scalable Option Learning (SOL) with Task-and-Motion Planning (TAMP) environments.

This repository provides a baseline implementation that combines:

- **SOL** (from Meta's [sol](https://github.com/facebookresearch/sol)) - Scalable hierarchical RL
- **TAMP Improvisation** (from [SLAP](https://github.com/isabelliu0/SLAP)) - Symbolic planning with learned skills

For this hierarchical RL baseline,

1. Pre-defined TAMP skills act as fixed option policies
2. SOL learns new option policies for shortcut transitions
3. A high-level controller learns to coordinate all options

## Installation

1. Install this package:

```bash
# Assume we are working with uv
git submodule update --init
uv venv --python=3.11
uv pip install -e .
```

2. Install SOL dependencies (required):

```bash
uv pip install -r third-party/sol/requirements.txt
cd third-party/sol/sol
python setup.py build_ext --inplace
```

3. Add SOL to PYTHONPATH:

```bash
export PYTHONPATH=/home/airlabbw/NeSy/skill_refactor_branches/sol-tamp/third-party/sol:$PYTHONPATH
```

4. Download Shortcut Signatures from [SLAP-Shortcuts](http://slap-data.s3-website.us-east-2.amazonaws.com/#training_data/multi_rl/) and [SLAP-Policies](http://slap-data.s3-website.us-east-2.amazonaws.com/#trained_policies/multi_rl/)

Desired structure
```
slap_data/
    CleanupTableTAMPSystem/
        config.json
        current_atoms.pkl
        pattern_n7-to-n132_b5e0169c_0e5ec8e8.pkl
        policy_n7-to-n132_b5e0169c_0e5ec8e8.zip
        ...
    ClutteredDrawerTAMPSystem/
```


## Usage

Train SOL on TAMP environments using SOL's launch.py:

```bash
cd /path/to/sol
python launch.py --expfile /home/airlabbw/NeSy/skill_refactor_branches/sol-tamp/configs/sol_tamp.yaml --mode local --debug
```

Or directly with Sample Factory:

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

Run tests:

```bash
pytest tests/
```

## Supported Environments

- `tamp_cluttered_drawer` - ClutteredDrawer from SLAP
- `tamp_obstacle_tower` - ObstacleTower from SLAP
- `tamp_cleanup_table` - CleanupTable from SLAP
- `tamp_obstacle2d` - Obstacle2D from SLAP
