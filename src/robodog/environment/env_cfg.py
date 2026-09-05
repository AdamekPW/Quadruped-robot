from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from robodog.assets.robots.silver_badger.constants import (
    SILVER_BADGER_ACTION_SCALE,
    TERRAIN_SCAN_SITE_HEIGHT,
    TERRAIN_SCAN_SITE_NAME,
    get_silver_badger_robot_cfg,
)
from robodog.environment import mdp as robodog_mdp

from .constants import (
    FOOT_SITES,
    FOOT_GEOMS,
    LEG_ACTUATOR_PATTERNS,
)
from .metrics import add_velocity_metrics


def env_cfg(*, lock_spine: bool = True) -> ManagerBasedRlEnvCfg:
    """Bazowe zadanie velocity Silver Badgera na terenie nierównym (+ curriculum).
    Args:
        lock_spine: gdy True kręgosłup jest usztywniony
    """

    cfg = make_velocity_env_cfg()

    cfg.scene.num_envs = 2048

    cfg.scene.entities = {"robot": get_silver_badger_robot_cfg(full_collision=True)}

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500
    
    cfg.sim.nconmax = 80
    cfg.sim.njmax = 260

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            # NIE tułów: promienie lecą pionowo w dół z fizycznej pozycji ramki,
            # więc startując z tułowia wchodzą pod powierzchnię zbocza stromszego
            # niż ~22° i raportują ścianę jako płaską podłogę.
            sensor.frame.type = "site"
            sensor.frame.name = TERRAIN_SCAN_SITE_NAME
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(
            mode="geom", pattern=("trunk_collision", "rear_collision"), entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        trunk_ground_cfg,
    )

    # --- Curriculum terenu włączony ---
    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    if cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = 0

    # --- Komendy: więcej marszu NA WPROST + rzadsza zmiana kierunku ---
    twist_cmd = cfg.commands["twist"]
    twist_cmd.rel_forward_envs = 0.4
    twist_cmd.resampling_time_range = (5.0, 10.0)
    twist_cmd.rel_standing_envs = 0.05

    add_velocity_metrics(cfg.metrics)

    cfg.observations["actor"].nan_policy = "sanitize"
    cfg.observations["critic"].nan_policy = "sanitize"

    # Skan mierzy teraz wysokość od site'u, a nie od tułowia, więc każda wartość
    # jest o TERRAIN_SCAN_SITE_HEIGHT większa. Odejmujemy to, żeby obserwacja
    # została w dokładnie tej samej skali co wcześniej: na płaskim nadal ~0.32 m.
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["height_scan"].params["offset"] = (
            TERRAIN_SCAN_SITE_HEIGHT
        )


    # --- Akcje: skala per-siłownik; kręgosłup ewentualnie poza polityką ---
    base_action = cfg.actions["joint_pos"]
    assert isinstance(base_action, JointPositionActionCfg)
    joint_pos_action = robodog_mdp.DelayedJointPositionActionCfg(
        **{f.name: getattr(base_action, f.name) for f in fields(base_action)},
        max_delay_steps=1,
        jitter_probability=0.05,
    )
    cfg.actions["joint_pos"] = joint_pos_action

    action_scale = dict(SILVER_BADGER_ACTION_SCALE)
    if lock_spine:
        # Polityka nie dostaje kanału na kręgosłup: sterujemy tylko nogami.
        joint_pos_action.actuator_names = LEG_ACTUATOR_PATTERNS
        action_scale.pop("spine_joint", None)
    joint_pos_action.scale = action_scale

    # --- Nagrody: podpięcie pod nazwy Silver Badgera ---
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    # Na terenie nierównym „pion" liczymy względem lokalnego nachylenia terenu.
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = 0.0

    cfg.rewards["track_linear_velocity"].weight = 4.0  # było 2.0
    cfg.rewards["pose"].weight = 0.5  # było 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0  # było 2.0
    cfg.rewards["air_time"].weight = 0.5  # było 0.0 (nagroda za stawianie kroków)
    del cfg.rewards["angular_momentum"]

    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

    cfg.rewards["pose"].params["std_standing"] = {
        r"spine_joint": 0.05,
        r".*_(hip|thigh)_joint": 0.05,
        r".*_calf_joint": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r"spine_joint": 0.3,
        r".*_(hip|thigh)_joint": 0.3,
        r".*_calf_joint": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r"spine_joint": 0.3,
        r".*_(hip|thigh)_joint": 0.3,
        r".*_calf_joint": 0.6,
    }

    # `base_com` usunięty: `pseudo_inertia` nadpisuje `body_ipos` (liczy je od wartości
    # domyślnych) i odpala się po nim, więc kasował jego efekt. COM przesuwa teraz sam
    # `pseudo_inertia` przez `t1..t3_range`.
    cfg.events.pop("base_com")
    cfg.events["encoder_bias"].mode = "reset"
    cfg.events["foot_friction"].mode = "interval"
    cfg.events["foot_friction"].interval_range_s = (4.0, 10.0)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["pd_gains"] = EventTermCfg(
        mode="reset",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "operation": "scale",
        },
    )
    cfg.events["pseudo_inertia"] = EventTermCfg(
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
    cfg.events["joint_friction"] = EventTermCfg(
        mode="interval",
        interval_range_s=(4.0, 10.0),
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "ranges": (0.7, 1.3)
        }
    )
    cfg.events["joint_armature"] = EventTermCfg(
        mode="reset",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "ranges": (0.7, 1.3)
        }
    )
    # Przechył grawitacji: darmowy odpowiednik pochyłości terenu / bocznego wiatru
    cfg.events["gravity"] = EventTermCfg(
        mode="reset",
        func=robodog_mdp.gravity,
        params={
            "xy_range": (-0.5, 0.5),
            "z_scale_range": (0.9, 1.1),
        },
    )

    # --- Curriculum: siła domain randomization rośnie 0 -> 100% ---
    curriculum_events = [
        "encoder_bias",
        "foot_friction",
        "pd_gains",
        "joint_friction",
        "gravity",
        "push_robot",
        "pseudo_inertia",
        "joint_armature",
    ]

    cfg.curriculum["domain_randomization"] = CurriculumTermCfg(
        func=robodog_mdp.dr_curriculum,
        params={
            "event_names": tuple(curriculum_events),
            "start_step": 500 * 24,
            "end_step": 4_000 * 24,
            "action_name": "joint_pos",
        },
    )

    # --- Podgląd ---
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # --- Terminacje ---
    cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    cfg.terminations["trunk_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": trunk_ground_cfg.name},
    )

    return cfg
