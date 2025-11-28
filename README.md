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

### 1. Clone Repository with Submodules

```bash
git clone --recursive <your-repo-url>
cd sol-tamp

# If you already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Create Python Environment

```bash
# Using conda (recommended)
conda create -n sol-tamp python=3.11
conda activate sol-tamp

# Or using venv
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install sol-tamp Package

```bash
pip install -e .
```

### 4. Setup SOL

```bash
# Get absolute path to SOL submodule
SOL_PATH="$(pwd)/third-party/sol"

# Add SOL to Python path via .pth file
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$SOL_PATH" > "$SITE_PACKAGES/sol-tamp.pth"

# Install SOL dependencies
cd third-party/sol
pip install -r requirements.txt

# Build Cython extensions
cd sol
python setup.py build_ext --inplace
cd ../../..
```

### 5. Download Shortcut Training Data (Optional)

Download from [SLAP-Shortcuts](http://slap-data.s3-website.us-east-2.amazonaws.com/#training_data/multi_rl/) and organize as:

```
slap_data/
├── ClutteredDrawerTAMPSystem/
│   ├── trained_signatures.pkl
│   ├── current_atoms.pkl
│   └── goal_atoms.pkl
├── GraphObstacleTowerTAMPSystem/
│   └── trained_signatures.pkl
├── CleanupTableTAMPSystem/
│   └── trained_signatures.pkl
└── GraphObstacle2DTAMPSystem/
    └── trained_signatures.pkl
```

**Note:** If `slap_data/` doesn't exist, the system will still work but with empty shortcut lists.

## Usage

### Training with SOL

```bash
python -m sample_factory.algorithms.appo.train_appo \
    --env=tamp_cluttered_drawer \
    --experiment=sol_cluttered_drawer \
    --with_sol=True \
    --sol_num_option_steps=10 \
    --reward_scale_shortcuts=1.0 \
    --reward_scale_skills=1.0 \
    --reward_scale_task=10.0 \
    --num_workers=8 \
    --num_envs_per_worker=4 \
    --train_for_env_steps=10000000
```

**Key parameters:**

- `--with_sol=True`: Enable hierarchical SOL learning
- `--sol_num_option_steps`: Max steps per option (-1 for unlimited)
- `--reward_scale_*`: Scaling for different reward types

### Training without SOL (Flat RL Baseline)

```bash
python -m sample_factory.algorithms.appo.train_appo \
    --env=tamp_cluttered_drawer \
    --experiment=flat_rl_cluttered_drawer \
    --with_sol=False
```

### Run Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_integration.py -v -s
```

## Supported Environments

- `tamp_cluttered_drawer` - ClutteredDrawer from SLAP
- `tamp_obstacle_tower` - ObstacleTower from SLAP
- `tamp_cleanup_table` - CleanupTable from SLAP
- `tamp_obstacle2d` - Obstacle2D from SLAP
