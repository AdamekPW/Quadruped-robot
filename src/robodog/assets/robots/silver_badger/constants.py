"""Stałe i konfiguracja encji robota Silver Badger dla mjlab.

Odpowiednik `go1_constants.py` z mjlab, ale dla Silver Badgera (MAB Robotics):
czworonóg z DODATKOWYM aktywnym stawem kręgosłupa (`spine_joint`, 1-DOF w
płaszczyźnie strzałkowej) — łącznie 13 siłowników (kręgosłup + 12 nóg).

Wartości siłowników (sztywność kp=20, tłumienie kv=0.5, limity siły: kręgosłup
±48 N·m, nogi ±16 N·m) i poza nominalna pochodzą z oryginalnego modelu MJCF
(repo one_policy_to_run_them_all — patrz README obok). Kolizje: tylko stopy,
bo w oryginale reszta geometrii miała contype=0 (kontakt szedł jawnymi parami).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


SILVER_BADGER_XML: Path = Path(__file__).parent / "xmls" / "silver_badger.xml"
assert SILVER_BADGER_XML.exists(), f"Brak pliku modelu: {SILVER_BADGER_XML}"


def get_spec() -> mujoco.MjSpec:
    """Wczytuje MjSpec z czystego (mjlab-owego) XML-a Silver Badgera."""
    return mujoco.MjSpec.from_file(str(SILVER_BADGER_XML))



# Siłowniki (sterowanie pozycyjne PD).
# Sztywność i tłumienie z oryginalnego <position kp="20" kv="0.5">.
STIFFNESS = 20.0 # jak mocno silnik "ciągnie" do zadanego kąta (jak sztywność sprężyny)
DAMPING = 0.5 # jak mocno hamuje ruch (aby staw nie oscylował)
ARMATURE = 0.013122 # bezwładność wirnika "odbita" przez przekładnie

SPINE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("spine_joint",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=48.0,  # forcerange kręgosłupa w oryginale
    armature=ARMATURE,
)
LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=16.0,  # forcerange nóg w oryginale
    armature=ARMATURE,
)

# Poza startowa na początku każdego epizodu i punkt odniesienia dla akcji.
# qpos bazy: z=0.316; stawy per-noga (znaki różne z powodu lustrzanej geometrii).
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.316), # pozycja tułowia w metrach (x, y, h), chcemy aby stał na nogach
    joint_pos={ # kąt każdego stawu w radianiach (geometria lustrzana dlatego znaki)
        "spine_joint": 0.0,
        "RL_hip_joint": -0.1, "RL_thigh_joint": -0.8, "RL_calf_joint": 1.5,
        "RR_hip_joint": 0.1, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.5,
        "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "FL_hip_joint": 0.1, "FL_thigh_joint": -0.8, "FL_calf_joint": 1.5,
    },
    joint_vel={".*": 0.0},
)


# Kolizje - tylko stopy (geomy RL_foot, RR_foot, FR_foot, FL_foot)
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=("^[FR][LR]_foot$",),
    contype=0, # maski bitowe kolizji
    conaffinity=1, # maski bitowe kolizji
    condim=6, # ile "kierunków" tarcia model liczy
    priority=1,
    friction=(1.1, 5e-3, 1e-4), # współczynniki tarcia (ślizgowe, obrotowe, wałkowe)
    solimp=(0.015, 1.0, 0.02),
    disable_other_geoms=True, # wyłączamy kolizję całej reszty robota
)

# Złożenie konfiguracji siłowników
SILVER_BADGER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SPINE_ACTUATOR_CFG, LEG_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9, # miękki limit zakresu stawów, polityka jest zniechęcana do wchodzenia w ostatnie 10% zakresu
)

# Złożenie całej konfiguracji robota
def get_silver_badger_robot_cfg() -> EntityCfg:
    """Zwraca świeżą konfigurację encji Silver Badgera.

    Nowa instancja przy każdym wywołaniu — żeby współdzielenie configu w wielu
    miejscach nie prowadziło do mutacji (tak samo jak w go1_constants).
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=SILVER_BADGER_ARTICULATION,
    )

# Mapowanie wyniku sieci (z zakresu [-1, 1]) na sensowny kąt
# Skala akcji per siłownik: 0.25 * limit_siły / sztywność
SILVER_BADGER_ACTION_SCALE: dict[str, float] = {}
for _a in SILVER_BADGER_ARTICULATION.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    assert _a.effort_limit is not None
    for _name in _a.target_names_expr:
        SILVER_BADGER_ACTION_SCALE[_name] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
    # Headless-safe: kompiluje encję i wypisuje diagnostykę (bez okna GLFW).
    from mjlab.entity.entity import Entity

    robot = Entity(get_silver_badger_robot_cfg())
    model = robot.spec.compile()
    print(f"Encja Silver Badger OK: nq={model.nq} nu={model.nu} njnt={model.njnt}")
    print("Skala akcji:", SILVER_BADGER_ACTION_SCALE)
