""" Rejestracja zadań Silver Badgera w rejestrze mjlab """

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robodog.algorithms.runners.distillation_runner import MjlabDistillationRunner
from robodog.environment.variants import as_play, baseline_cfg, cnn_cfg, distill_cfg
from robodog.training.distillation_cfg import distillation_runner_cfg
from robodog.training.rl_cfg import cnn_ppo_runner_cfg, ppo_runner_cfg

# Baseline MLP: teren wklejony w płaski wektor obserwacji.
register_mjlab_task(
    task_id="Mjlab-Silver-Badger",
    env_cfg=baseline_cfg(),
    play_env_cfg=as_play(baseline_cfg()),
    rl_cfg=ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Wariant CNN: teren jako obraz 2D + własne sieci konwolucyjne. Ten sam robot
# i teren co w baseline, inny sposób przetwarzania otoczenia — do porównania.
register_mjlab_task(
    task_id="Mjlab-Silver-Badger-CNN",
    env_cfg=cnn_cfg(),
    play_env_cfg=as_play(cnn_cfg()),
    rl_cfg=cnn_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Distylacja teacher -> student. Teacher wczytuje się z checkpointu CNN przez
# `--agent.resume True --agent.load-run <run>`.
register_mjlab_task(
    task_id="Mjlab-Silver-Badger-Distill",
    env_cfg=distill_cfg(),
    play_env_cfg=as_play(distill_cfg()),
    rl_cfg=distillation_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)
