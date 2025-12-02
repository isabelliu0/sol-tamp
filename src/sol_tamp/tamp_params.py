"""Sample Factory configuration parameters for TAMP environments."""

from sample_factory.utils.utils import str2bool


def tamp_override_defaults(env, parser):
    """Override default Sample Factory parameters for TAMP environments."""
    parser.set_defaults(
        batched_sampling=False,
        num_workers=8,
        num_envs_per_worker=4,
        worker_num_splits=2,
        train_for_env_steps=10000000,
        encoder_mlp_layers=[128, 128],
        env_frameskip=1,
        nonlinearity="relu",
        batch_size=2048,
        use_rnn=True,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1.0,
        rollout=128,
        max_grad_norm=0.1,
        num_epochs=4,
        num_batches_per_epoch=2,
        ppo_clip_ratio=0.2,
        value_loss_coeff=0.5,
        exploration_loss_coeff=0.0001, # from point-maze
        learning_rate=0.0003,
        lr_schedule="kl_adaptive_epoch",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=True,
        recurrence=128,
        normalize_input=False,
        normalize_returns=False,
        value_bootstrap=True,
        experiment_summaries_interval=10,
        save_every_sec=120,
        serial_mode=False,
        async_rl=True,
        with_sol=True,
        sol_num_option_steps=50,
    )


def add_tamp_env_args(env, parser):
    """Add TAMP-specific command line arguments."""
    p = parser
    p.add_argument(
        "--reward_scale_shortcuts",
        type=float,
        default=0.00001,
        help="Scaling for shortcut option intrinsic rewards",
    )
    p.add_argument(
        "--reward_scale_skills",
        type=float,
        default=0.00001,
        help="Scaling for skill option intrinsic rewards",
    )
    p.add_argument(
        "--reward_scale_task",
        type=float,
        default=1.0,
        help="Scaling for task reward",
    )
    p.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="if debug, the environment will always be reset to a default state"
    )
    p.add_argument(
        "--use_shortcuts",
        type=str2bool,
        default=True,
        help="Whether to load and use shortcut policies (set to False for skills-only training)"
    )
    p.add_argument(
        "--oracle_skills_only",
        type=str2bool,
        default=False,
        help="Only create 4 oracle grounded skills for obstacle2d (testing/debugging)"
    )
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=300,
        help="Maximum steps per episode before truncation"
    )
