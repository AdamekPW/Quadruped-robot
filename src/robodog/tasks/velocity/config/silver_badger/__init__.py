"""Rejestracja zadań velocity dla Silver Badgera w rejestrze mjlab.

Import tego modułu (pośrednio przez `import robodog`) rejestruje ID zadania,
dzięki czemu widzą je `mjlab.scripts.train` / `play`.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robodog.algorithms.runners.distillation_runner import MjlabDistillationRunner

from .environment.flat_env_cfg import silver_badger_flat_env_cfg
from .environment.rough_env_cfg import silver_badger_rough_env_cfg
from .reinforcement_learning.distillation_cfg import (
    silver_badger_distillation_runner_cfg,
)
from .reinforcement_learning.rl_cfg import (
    silver_badger_cnn_ppo_runner_cfg,
    silver_badger_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Silver-Badger",
    env_cfg=silver_badger_flat_env_cfg(),
    play_env_cfg=silver_badger_flat_env_cfg(play=True),
    rl_cfg=silver_badger_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Wariant rough z USZTYWNIONYM kręgosłupem — bieżący cel eksperymentów (pierwsza
# „noga" studium porównawczego). Kręgosłup ruchomy wróci osobnym zadaniem później.
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger",
    env_cfg=silver_badger_rough_env_cfg(lock_spine=True),
    play_env_cfg=silver_badger_rough_env_cfg(play=True, lock_spine=True),
    rl_cfg=silver_badger_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Wariant CNN: teren jako obraz 2D (terrain_as_image=True) + własne sieci
# konwolucyjne (class_name w rl_cfg). Ten sam usztywniony robot i rough terrain,
# inny sposób przetwarzania otoczenia — do porównania z baseline'em MLP.
# num_envs dziedziczy z bazy (2048) — po włączeniu kolizji korpusu to VRAM-owy
# sufit dla obu wariantów; ta sama liczba env = uczciwe porównanie MLP vs CNN.
_cnn_env_cfg = silver_badger_rough_env_cfg(lock_spine=True, terrain_as_image=True)
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger-CNN",
    env_cfg=_cnn_env_cfg,
    play_env_cfg=silver_badger_rough_env_cfg(
        play=True, lock_spine=True, terrain_as_image=True
    ),
    rl_cfg=silver_badger_cnn_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Zadanie DISTYLACJI (teacher CNN → student depth+proprio). Środowisko wystawia
# JEDNOCZEŚNIE height_scan (teacher) i głębię (student). Render kamery jest drogi,
# więc dużo mniej środowisk niż w RL. Teacher wczytuje się z checkpointu CNN przez
# --agent.resume True --agent.load-run <run>. Student docelowo dostanie LSTM (Adam).
_distill_env_cfg = silver_badger_rough_env_cfg(
    lock_spine=True, terrain_as_image=True, with_depth=True
)
# Render głębi + kolizje korpusu + height_scan = ciężkie; startowo mało środowisk
# (do dostrojenia pod VRAM). Distylacja i tak potrzebuje mniej danych niż RL.
_distill_env_cfg.scene.num_envs = 256
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger-Distill",
    env_cfg=_distill_env_cfg,
    play_env_cfg=silver_badger_rough_env_cfg(
        play=True, lock_spine=True, terrain_as_image=True, with_depth=True
    ),
    rl_cfg=silver_badger_distillation_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)
