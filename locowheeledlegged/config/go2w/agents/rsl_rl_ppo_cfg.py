from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 500
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,   # 【修复】1e-3 -> 3e-4，降低续训时分布突变导致的梯度爆炸风险
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    experiment_name = "locowheeledlegged_go2w"
    # 默认改回 tensorboard：无需 wandb 账号即可离线记录。
    # 若要使用 wandb，请改回 logger = "wandb" 并先执行 `wandb login`。
    logger = "tensorboard"
    # logger = "wandb"
    wandb_project = "LocoWheeledLegged"















