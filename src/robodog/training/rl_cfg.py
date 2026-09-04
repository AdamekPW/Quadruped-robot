"""
Konfiguracja RL (PPO / rsl-rl)

"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def silver_badger_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Tworzy konfigurację runnera RL dla zadania velocity Silver Badgera."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            # Normalizacja obserwacji (running mean/std) — ogranicza skalę wejść sieci, aby uniknąć NaN w gradiencie
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.02,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="silver_badger_velocity",
        logger="wandb",
        wandb_project="robodog",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


# Ścieżki importu własnych sieci (rozwiązywane przez rsl-rl `resolve_callable`).
_CNN_ACTOR = "robodog.algorithms.networks.cnn.cnn_actor:CNN_actor"
_CNN_CRITIC = "robodog.algorithms.networks.cnn.cnn_critic:CNN_critic"


def silver_badger_cnn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Wariant runnera z WŁASNYMI sieciami CNN (enkoder nad mapą terenu) """
    cfg = silver_badger_ppo_runner_cfg()
    cfg.actor.class_name = _CNN_ACTOR
    cfg.critic.class_name = _CNN_CRITIC
    return cfg
