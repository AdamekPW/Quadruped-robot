""" Akcje - co dokładnie polityka steruje i z jakim opóźnieniem """

from dataclasses import fields

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ActionTermCfg

from .. import mdp as robodog_mdp
from ..constants import LEG_ACTUATOR_PATTERNS
from robodog.assets.robots.silver_badger.constants import SILVER_BADGER_ACTION_SCALE

# Opóźnienie akcji: do jednego kroku sterowania (20 ms przy 50 Hz), z 5% szansą
# na dodatkowy jitter — tyle mniej więcej wynosi opóźnienie realnego łańcucha.
_MAX_DELAY_STEPS = 1
_JITTER_PROBABILITY = 0.05


def prepare_actions(actions: dict[str, ActionTermCfg], *, lock_spine: bool) -> None:
    """Podmienia człon akcji na wariant z opóźnieniem i ustawia skalę per-siłownik.

    Args:
        lock_spine: gdy True polityka NIE dostaje kanału na kręgosłup — steruje
            tylko 12 stawami nóg, a `spine_joint` trzyma neutralny kąt swoim PD.
    """
    base_action = actions["joint_pos"]
    assert isinstance(base_action, JointPositionActionCfg)

    # Przepisujemy wszystkie pola bazowego członu i dokładamy opóźnienie.
    joint_pos_action = robodog_mdp.DelayedJointPositionActionCfg(
        **{f.name: getattr(base_action, f.name) for f in fields(base_action)},
        max_delay_steps=_MAX_DELAY_STEPS,
        jitter_probability=_JITTER_PROBABILITY,
    )
    actions["joint_pos"] = joint_pos_action

    action_scale = dict(SILVER_BADGER_ACTION_SCALE)
    if lock_spine:
        joint_pos_action.actuator_names = LEG_ACTUATOR_PATTERNS
        action_scale.pop("spine_joint", None)
    joint_pos_action.scale = action_scale
