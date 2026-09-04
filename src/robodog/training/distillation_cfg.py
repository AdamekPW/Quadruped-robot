"""
Konfiguracja DISTYLACJI (teacher/student)

"""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg

# Ścieżki importu klas (rozwiązywane przez rsl-rl `resolve_callable`).
_CNN_ACTOR = "robodog.algorithms.networks.cnn.cnn_actor:CNN_actor"
_DEPTH_STUDENT = "robodog.algorithms.networks.cnn.depth_student:DepthStudent"
_DISTILLATION = "rsl_rl.algorithms.distillation:Distillation"

_GAUSSIAN = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
}


@dataclass
class RslRlDistillationAlgorithmCfg:
    """Hiperparametry algorytmu distylacji (kwargs `Distillation.__init__`)."""

    num_learning_epochs: int = 1
    gradient_length: int = 15
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"
    class_name: str = _DISTILLATION


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Konfiguracja runnera distylacji (odpowiednik on-policy, ale student/teacher)."""

    student: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(_GAUSSIAN),
            rnn_type="lstm",
            rnn_hidden_dim=128,
            rnn_num_layers=1,
            class_name=_DEPTH_STUDENT,
        )
    )

    teacher: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(_GAUSSIAN),
            class_name=_CNN_ACTOR,
        )
    )

    algorithm: RslRlDistillationAlgorithmCfg = field(
        default_factory=RslRlDistillationAlgorithmCfg

    )
    
    teacher_checkpoint: str | None = None

def distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
    """Tworzy konfigurację distylacji Silver Badgera (teacher CNN → student depth)."""
    return RslRlDistillationRunnerCfg(
        obs_groups={"student": ("student_proprio",), "teacher": ("actor",)},
        experiment_name="silver_badger_distillation",
        logger="wandb",
        wandb_project="robodog",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=5_000,
    )
