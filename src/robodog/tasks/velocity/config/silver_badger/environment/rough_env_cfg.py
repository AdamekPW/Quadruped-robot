from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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
    get_silver_badger_robot_cfg,
)

from .constants import (
    FOOT_SITES,
    FOOT_GEOMS,
    LEG_ACTUATOR_PATTERNS,
    TERRAIN_SCAN_GRID_HW,
)
from .observations import height_scan_image


def silver_badger_rough_env_cfg(
    play: bool = False,
    lock_spine: bool = True,
    terrain_as_image: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Zadanie velocity Silver Badger na terenie NIEPŁASKIM (rough + curriculum).

    W przeciwieństwie do wariantu flat NIE usuwamy skanu terenu — sensor
    `terrain_scan` (siatka rzutów wokół tułowia) i obserwacja `height_scan` są
    surowym wejściem o otoczeniu (docelowo pod sieć konwolucyjną). Teren
    generowany proceduralnie z curriculum poziomów trudności (mechanizm mjlab
    `terrain_levels_vel` — robot awansuje/spada między wierszami zależnie od
    przebytej drogi).

    Args:
        play: tryb podglądu (długi epizod, bez zakłóceń/pchania, teren losowany).
        lock_spine: gdy True (domyślnie) kręgosłup jest USZTYWNIONY — polityka
            steruje tylko 12 stawami nóg, a `spine_joint` trzyma się neutralnego
            kąta 0 przez swój siłownik PD (kp=20). To pierwsza „noga" studium
            porównawczego. Ustaw False, by wrócić do kręgosłupa RUCHOMEGO.
        terrain_as_image: gdy False (domyślnie) `height_scan` jest płaskim
            wektorem sklejonym z resztą obserwacji (baseline pod MLP). Gdy True,
            skan terenu jest WYDZIELONY do osobnej grupy obserwacji `height_scan`
            o kształcie obrazu `(B, 1, H, W)` — pod własną sieć konwolucyjną.
            Wymaga wtedy ustawienia `obs_groups` w rl_cfg, żeby runner podał tę grupę modelowi.
    """
    cfg = make_velocity_env_cfg()

    cfg.scene.num_envs = 4096

    # --- Robot ---
    cfg.scene.entities = {"robot": get_silver_badger_robot_cfg()}

    # --- Dostrojenie symulacji pod kontakty na terenie nierównym (jak go1 rough) ---
    # Więcej iteracji CCD i większy budżet dopasowań kontaktu, bo heightfield/bloki
    # generują znacznie więcej par kontaktowych niż płaska podłoga.
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    # --- Sensory: skan terenu (zostaje!) podpięty pod tułów ---
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "trunk"
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
    # Uwaga: nasz model ma kolizje TYLKO w stopach (reszta contype=0), więc nie
    # dodajemy sensorów kolizji ud/goleni/tułowia ani kar za nie (jak w go1).

    # --- Curriculum terenu włączony ---
    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    # Start od NAJŁATWIEJSZEGO terenu (poziom 0), nie rozrzut 0-5. Inaczej część
    # robotów trafia na starcie na trudny teren, gdzie losowa polityka się wywraca
    # i uczy się „nie ruszaj się = bezpiecznie" — a raz zbiegłszy do bezruchu, nie
    # wychodzi z niego nawet po degradacji na łatwy teren. Startując od 0 robot
    # najpierw uczy się chodzić (jak na flat), a curriculum podnosi trudność potem.
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = 0

    # --- Komendy: więcej marszu NA WPROST + rzadsza zmiana kierunku ---
    # Awans w curriculum terenu wymaga >4 m przemieszczenia NETTO od punktu startu
    # w epizodzie (`terrain_levels_vel`, próg = size/2). Domyślnie tylko 20% env-ów
    # dostaje komendę „prosto", a kierunek losuje się co 3-8 s → robot dużo chodzi,
    # ale netto <4 m i teren tkwi na poziomie 0. Zwiększamy udział marszu prosto i
    # wydłużamy okno komendy, żeby robot faktycznie pokonywał dystans i awansował.
    twist_cmd = cfg.commands["twist"]
    twist_cmd.rel_forward_envs = 0.4
    twist_cmd.resampling_time_range = (5.0, 10.0)
    twist_cmd.rel_standing_envs = 0.05

    # Zabezpieczenie przed NaN (patrz wariant flat).
    cfg.observations["actor"].nan_policy = "sanitize"
    cfg.observations["critic"].nan_policy = "sanitize"

    if terrain_as_image:
        # Zdejmij płaski height_scan z actora i critica (zostaje sama propriocepcja), zachowując oryginalną skalę (1/max_distance sensora).
        scan_scale = cfg.observations["actor"].terms["height_scan"].scale
        for group_name in ("actor", "critic"):
            cfg.observations[group_name].terms.pop("height_scan", None)
        # Dodaj skan jako osobną grupę o kształcie obrazu (B, 1, H, W).
        cfg.observations["height_scan"] = ObservationGroupCfg(
            terms={
                "scan": ObservationTermCfg(
                    func=height_scan_image,
                    params={
                        "sensor_name": "terrain_scan",
                        "grid_hw": TERRAIN_SCAN_GRID_HW,
                    },
                    scale=scan_scale,
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
        )
        # Uwaga: samo wydzielenie grupy nie wystarcza — w rl_cfg trzeba wskazać
        # `obs_groups`, np. {"actor": ["actor", "height_scan"],
        # "critic": ["critic", "height_scan"]}, żeby runner podał ją modelowi.

    # --- Akcje: skala per-siłownik; kręgosłup ewentualnie poza polityką ---
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
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

    # Przeciw „zamrożonemu staniu": na rough robot zbiegał do bezruchu, bo pion +
    # poza + śledzenie ~zerowych obrotów dawały ~3/krok BEZ jazdy, a raczkujący
    # chód (ryzyko upadku) mniej. Aby zapobiec:
    #  - mocniej nagradzamy jazdę liniową,
    #  - odbierz „darmowy" zysk z trzymania domyślnej pozy,
    #  - ścij nagrodę za obrót (robot wyłudzał ją stojąc, matchując ~0 yaw),
    #  - WŁĄCZ nagrodę za czas w powietrzu stóp = bezpośrednia zachęta do KROKU.
    # (Wagi do dostrojenia — patrz notatka o zamrożonym staniu.)
    cfg.rewards["track_linear_velocity"].weight = 4.0  # było 2.0
    cfg.rewards["pose"].weight = 0.5  # było 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0  # było 2.0
    cfg.rewards["air_time"].weight = 0.5  # było 0.0 (nagroda za stawianie kroków)
    # Brak sensora root_angmom w naszym modelu — usuwamy człon (w go1 ma wagę 0).
    del cfg.rewards["angular_momentum"]

    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

    # Docelowe odchylenia pozy (stój/chód/bieg) — kręgosłup + nogi. Gdy kręgosłup
    # jest usztywniony, jego człon i tak jest stały (trzyma się 0), więc nie szkodzi.
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
    # DR tarcia stóp (condim=6) wymaga wariantu per-oś jak w go1 — na razie pomijamy.
    cfg.events.pop("foot_friction", None)

    # --- Podgląd ---
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # --- Terminacje ---
    # Nasz model nie ma sensorów kontaktu na korpusie/nogach, więc jedynym
    # sygnałem upadku jest orientacja tułowia — zostawiamy „fell_over" z bazy
    # (70°). `out_of_terrain_bounds` zostaje (napędza awans w curriculum terenu).
    cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)

    # --- Tryb play (podgląd nauczonej polityki) ---
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.scene.num_envs = 50  # podgląd: kilka robotów wystarczy
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        # W play chcemy przekrój różnych terenów, nie curriculum — losujemy teren
        # przy resecie i wyłączamy narastanie trudności.
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if (
            cfg.scene.terrain is not None
            and cfg.scene.terrain.terrain_generator is not None
        ):
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg
