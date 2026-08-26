"""Rozszerzenia MDP dla zadania velocity — to, czego nie ma w `mjlab`.

Układ katalogu naśladuje `mjlab.tasks.velocity.mdp`, żeby import wyglądał znajomo:

    from robodog.tasks.velocity import mdp as robodog_mdp
"""

from .actions import DelayedJointPositionAction, DelayedJointPositionActionCfg
from .curriculums import dr_curriculum
from .domain_randomization import gravity

__all__ = [
    "DelayedJointPositionAction",
    "DelayedJointPositionActionCfg",
    "dr_curriculum",
    "gravity",
]
