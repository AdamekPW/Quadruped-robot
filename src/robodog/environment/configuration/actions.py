""" Akcje - co dokładnie polityka steruje i z jakim opóźnieniem """

from dataclasses import fields

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ActionTermCfg

from .. import mdp as robodog_mdp
from robodog.assets.robots.silver_badger.constants import (
    SILVER_BADGER_ACTION_SCALE,
    SPINE_JOINT_NAME,
)

# Opóźnienie akcji: do jednego kroku sterowania (20 ms przy 50 Hz), z 5% szansą
# na dodatkowy jitter - tyle mniej więcej wynosi opóźnienie realnego łańcucha.
_MAX_DELAY_STEPS = 1
_JITTER_PROBABILITY = 0.05


def prepare_actions(actions: dict[str, ActionTermCfg], *, lock_spine: bool) -> None:
    """Podmienia człon akcji na wariant z opóźnieniem i ustawia skalę per-siłownik.

    Args:
        lock_spine: gdy True model nie ma stawu kręgosłupa.
    """
    base_action = actions["joint_pos"]
    assert isinstance(base_action, JointPositionActionCfg)

    # Przepisujemy wszystkie pola bazowego członu i dokładamy opóźnienie.
    d = {}
    for f in fields(base_action):
        d[f.name] = getattr(base_action, f.name)

    joint_pos_action = robodog_mdp.DelayedJointPositionActionCfg(
        **d,
        max_delay_steps=_MAX_DELAY_STEPS,
        jitter_probability=_JITTER_PROBABILITY,
    )
    
    actions["joint_pos"] = joint_pos_action

    action_scale = dict(SILVER_BADGER_ACTION_SCALE)
    if lock_spine:
        del action_scale[SPINE_JOINT_NAME]
    joint_pos_action.scale = action_scale
