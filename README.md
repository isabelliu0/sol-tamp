# SOL-TAMP Baseline

Integration of Scalable Option Learning (SOL) with Task-and-Motion Planning (TAMP) environments.

This repository provides a baseline implementation that combines:
- **SOL** (from Meta's [sol](https://github.com/facebookresearch/sol)) - Scalable hierarchical RL
- **TAMP Improvisation** (from [SLAP](https://github.com/isabelliu0/SLAP)) - Symbolic planning with learned skills

For this hierarchical RL baseline,
1. Pre-defined TAMP skills act as fixed option policies
2. SOL learns new option policies for shortcut transitions
3. A high-level controller learns to coordinate all options