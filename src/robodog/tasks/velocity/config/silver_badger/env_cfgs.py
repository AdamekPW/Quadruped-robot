"""Konfiguracja zadania velocity (śledzenie zadanej prędkości) dla Silver Badgera.

Wariant FLAT (płaski teren), wzorowany na `config/go1/env_cfgs.py` z mjlab, ale
z nazwami geometrii/site/ciał Silver Badgera i bez elementów specyficznych dla
Go1 (granularne kolizje ud/goleni, których nasz model nie ma).

Uwaga o kręgosłupie: `spine_joint` jest tu traktowany jak zwykły siłownik —
polityka steruje nim razem z nogami (wariant RUCHOMY). Wariant USZTYWNIONY
(blokada kręgosłupa) dodamy osobno — to niezależna zmienna studium porównawczego.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from robodog.assets.robots.silver_badger.silver_badger_constants import (
    SILVER_BADGER_ACTION_SCALE,
    get_silver_badger_robot_cfg,
)

# Nazwy stóp w modelu Silver Badgera (geomy kolizyjne i site pokrywają się nazwą).
FOOT_SITES = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")
FOOT_GEOMS = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")


def silver_badger_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Tworzy konfigurację zadania velocity Silver Badger na płaskim terenie."""
    cfg = make_velocity_env_cfg()

    # --- Robot ---
    cfg.scene.entities = {"robot": get_silver_badger_robot_cfg()}

    # --- Płaski teren ---
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # --- Dostrojenie symulacji pod płaski teren (jak go1 flat) ---
    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    # --- Sensory ---
    # Usuwamy skan terenu (siatka rzutów) — niepotrzebny na płaskim.
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    
    # Skan wysokości stóp podpinamy pod site stóp Silver Badgera.
    for sensor in cfg.scene.sensors:
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    # Sensor kontaktu stopa–ziemia (potrzebny obserwacjom i nagrodom chodu).
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

    # --- Obserwacje: brak skanu terenu na płaskim ---
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Zabezpieczenie przed NaN: gdyby pojedynczy env wpadł w NaN/Inf (rzadki,
    # późny brzeg numeryczny przy zbieżności), sanityzuj obserwacje do 0 zamiast
    # zatruwać sieć. Menedżer nagród sanityzuje NaN sam z siebie. Bez tego std
    # rozkładu akcji może stać się NaN i cały trening wywala się w torch.normal.
    cfg.observations["actor"].nan_policy = "sanitize"
    cfg.observations["critic"].nan_policy = "sanitize"

    # --- Akcje: skala per-siłownik (0.25 * limit_siły / sztywność) ---
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SILVER_BADGER_ACTION_SCALE

    # --- Nagrody: podpięcie pod nazwy Silver Badgera ---
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)  # flat
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0
    # Brak sensora root_angmom w naszym modelu — usuwamy człon (w go1 ma wagę 0).
    del cfg.rewards["angular_momentum"]

    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

    # Docelowe odchylenia pozy (stój/chód/bieg) — kręgosłup + nogi.
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

    # --- Zdarzenia (domain randomization) ---
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    # Losowanie tarcia stóp z condim=6 wymaga wariantu per-oś (jak go1) — na
    # razie pomijamy DR tarcia dla prostoty pierwszej wersji.
    cfg.events.pop("foot_friction", None)

    # --- Podgląd ---
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # --- Terminacje (płaski teren) ---
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # "fell_over" (bad_orientation 70°) zostaje z bazy.
    # Reset env, którego stan fizyki wpadł w NaN/Inf (druga warstwa ochrony).
    cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)

    # --- Curriculum: bez poziomów terenu na płaskim ---
    cfg.curriculum.pop("terrain_levels", None)

    # --- Tryb play (podgląd nauczonej polityki) ---
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
