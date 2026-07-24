"""Konfiguracja RL (PPO / rsl-rl) dla zadania velocity Silver Badgera.

Na razie kopia ustawień Go1 (sprawdzone hiperparametry baseline). Zmieniona
tylko nazwa eksperymentu — logi lądują w logs/rsl_rl/silver_badger_velocity.
Docelowo tu podmienimy runner na własną implementację PPO, jeśli promotor
wskaże ją jako wymagany wkład (patrz notatki projektu).
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
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
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
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
