"""Rejestracja zadań velocity dla Silver Badgera w rejestrze mjlab.

Import tego modułu (pośrednio przez `import robodog`) rejestruje ID zadania,
dzięki czemu widzą je `mjlab.scripts.train` / `play`.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .environment.flat_env_cfg import silver_badger_flat_env_cfg
from .environment.rough_env_cfg import silver_badger_rough_env_cfg
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
_cnn_env_cfg = silver_badger_rough_env_cfg(lock_spine=True, terrain_as_image=True)
# CNN (aktywacje conv + GroupNorm dla wszystkich środowisk) zjada więcej VRAM niż
# MLP. Na RTX 5060 Ti (15 GB): 4096 = OOM (~15 GB, w grafie Warp), 2048 ≈ 8 GB,
# 3200 ≈ 11-12 GB z marginesem. Więcej env = mniejsza wariancja gradientu PPO.
_cnn_env_cfg.scene.num_envs = 3200
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Silver-Badger-CNN",
    env_cfg=_cnn_env_cfg,
    play_env_cfg=silver_badger_rough_env_cfg(
        play=True, lock_spine=True, terrain_as_image=True
    ),
    rl_cfg=silver_badger_cnn_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
