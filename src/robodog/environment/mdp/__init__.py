"""Rozszerzenia MDP dla zadania velocity — to, czego nie ma w `mjlab`.

Układ katalogu naśladuje `mjlab.tasks.velocity.mdp`, żeby import wyglądał znajomo:

    from robodog.environment import mdp as robodog_mdp
"""

from .actions import DelayedJointPositionAction, DelayedJointPositionActionCfg
from .curriculums import dr_curriculum
from .domain_randomization import gravity
from .observations import depth_image, height_scan_image
from .terrain_placement import assign_terrain_grid, required_num_envs

__all__ = [
    "DelayedJointPositionAction",
    "DelayedJointPositionActionCfg",
    "assign_terrain_grid",
    "depth_image",
    "dr_curriculum",
    "gravity",
    "height_scan_image",
    "required_num_envs",
]
