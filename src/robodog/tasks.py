"""Rejestracja zadań velocity dla Silver Badgera w rejestrze mjlab.

Import tego modułu (pośrednio przez `import robodog`) rejestruje ID zadania,
dzięki czemu widzą je `mjlab.scripts.train` / `play`.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robodog.algorithms.runners.distillation_runner import MjlabDistillationRunner
from robodog.environment.env_cfg import env_cfg
from robodog.training.distillation_cfg import distillation_runner_cfg
from robodog.training.rl_cfg import cnn_ppo_runner_cfg, ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Silver-Badger",
    env_cfg=env_cfg(lock_spine=True),
    play_env_cfg=env_cfg(play=True, lock_spine=True),
    rl_cfg=ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger-CNN",
    env_cfg= env_cfg(lock_spine=True, terrain_as_image=True),
    play_env_cfg=env_cfg(
        play=True, lock_spine=True, terrain_as_image=True
    ),
    rl_cfg=cnn_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

_distill_env_cfg = env_cfg(
    lock_spine=True, terrain_as_image=True, with_depth=True
)
# Render głębi + kolizje korpusu + height_scan = ciężkie; startowo mało środowisk
# (do dostrojenia pod VRAM). Distylacja i tak potrzebuje mniej danych niż RL.
_distill_env_cfg.scene.num_envs = 256
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger-Distill",
    env_cfg=_distill_env_cfg,
    play_env_cfg=env_cfg(
        play=True, lock_spine=True, terrain_as_image=True, with_depth=True
    ),
    rl_cfg=distillation_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)
