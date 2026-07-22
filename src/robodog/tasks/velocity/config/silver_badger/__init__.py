"""Rejestracja zadań velocity dla Silver Badgera w rejestrze mjlab.

Import tego modułu (pośrednio przez `import robodog`) rejestruje ID zadania,
dzięki czemu widzą je `mjlab.scripts.train` / `play`.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import silver_badger_flat_env_cfg
from .rl_cfg import silver_badger_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Silver-Badger",
    env_cfg=silver_badger_flat_env_cfg(),
    play_env_cfg=silver_badger_flat_env_cfg(play=True),
    rl_cfg=silver_badger_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
