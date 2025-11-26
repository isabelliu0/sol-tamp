"""Training script for SOL on TAMP environments."""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from sol_tamp.tamp_envs import register_tamp_envs
from sol_tamp.tamp_params import add_tamp_env_args, tamp_override_defaults


def parse_tamp_cfg(argv=None, evaluation=False):
    """Parse configuration for TAMP environments."""
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_tamp_env_args(partial_cfg.env, parser)
    tamp_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_tamp_envs()
    cfg = parse_tamp_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
