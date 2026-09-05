""" Zdarzenia (domain randomization) - losowanie warunków przy każdym resecie """

from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg

from .. import mdp as robodog_mdp
from ..constants import FOOT_GEOMS


def prepare_events(events: dict[str, EventTermCfg]) -> None:
    """Dostraja i uzupełnia słownik zdarzeń w miejscu."""
    # `base_com` usunięty: `pseudo_inertia` nadpisuje `body_ipos` (liczy je od wartości
    # domyślnych) i odpala się po nim, więc kasował jego efekt. COM przesuwa teraz sam
    # `pseudo_inertia` przez `t1..t3_range`.
    events.pop("base_com")
    events["encoder_bias"].mode = "reset"
    events["foot_friction"].mode = "interval"
    events["foot_friction"].interval_range_s = (4.0, 10.0)
    events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    events["pd_gains"] = EventTermCfg(
        mode="reset",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "operation": "scale",
        },
    )
    events["pseudo_inertia"] = EventTermCfg(
        mode="reset",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
            "alpha_range": (-0.08, 0.08),
            # Przesunięcie COM tułowia po `base_com`. `t1..t3` to parametry perturbacji
            # pseudo-inercji, nie metry wprost, ale skala wychodzi 1:1 (0.025 -> 0.025 m).
            "t1_range": (-0.025, 0.025),
            "t2_range": (-0.025, 0.025),
            "t3_range": (-0.03, 0.03),
        }
    )
    events["joint_friction"] = EventTermCfg(
        mode="interval",
        interval_range_s=(4.0, 10.0),
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "ranges": (0.7, 1.3)
        }
    )
    events["joint_armature"] = EventTermCfg(
        mode="reset",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "ranges": (0.7, 1.3)
        }
    )
    # Przechył grawitacji jako odpowiednik pochyłości terenu
    events["gravity"] = EventTermCfg(
        mode="reset",
        func=robodog_mdp.gravity,
        params={
            "xy_range": (-0.5, 0.5),
            "z_scale_range": (0.9, 1.1),
        },
    )
