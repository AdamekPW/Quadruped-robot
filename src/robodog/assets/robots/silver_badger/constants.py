"""
Stałe i konfiguracja encji robota Silver Badger dla mjlab.

"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


SILVER_BADGER_XML: Path = Path(__file__).parent / "xmls" / "silver_badger.xml"
assert SILVER_BADGER_XML.exists(), f"Brak pliku modelu: {SILVER_BADGER_XML}"


# --- Punkt startowy skanu terenu ---
# `terrain_scan` rzuca promienie PIONOWO w dół, a mjlab startuje je zawsze z
# fizycznej pozycji ramki — nie ma w configu żadnego offsetu pionowego. Startując
# z tułowia (0.316 m nad gruntem), promienie sięgające 0.8 m do przodu wchodzą POD
# powierzchnię zbocza stromszego niż ~22°: trafiają w podstawę heightfieldu i
# raportują ścianę jako płaską podłogę pół metra niżej. Podniesiony site przesuwa
# ten próg poza 45° (na zboczu 45° z tułowiem trzymanym poziomo zapas to +0.34 m).
TERRAIN_SCAN_SITE_NAME = "terrain_scan_origin"
TERRAIN_SCAN_SITE_HEIGHT = 0.8


def _add_terrain_scan_site(spec: mujoco.MjSpec) -> None:
    """Dopina do tułowia site, z którego startuje skan terenu."""
    spec.body("trunk").add_site(
        name=TERRAIN_SCAN_SITE_NAME,
        pos=(0.0, 0.0, TERRAIN_SCAN_SITE_HEIGHT),
        size=(0.01, 0.01, 0.01),
        group=5,  # grupa wizualizacyjna: nie koliduje i nie trafiają w nią promienie
    )


def get_spec() -> mujoco.MjSpec:
    """Wczytuje MjSpec z czystego (mjlab-owego) XML-a Silver Badgera."""
    spec = mujoco.MjSpec.from_file(str(SILVER_BADGER_XML))
    _add_terrain_scan_site(spec)
    return spec



STIFFNESS = 20.0 # jak mocno silnik "ciągnie" do zadanego kąta (jak sztywność sprężyny)
DAMPING = 0.5 # jak mocno hamuje ruch (aby staw nie oscylował)
ARMATURE = 0.013122 # bezwładność wirnika "odbita" przez przekładnie

SPINE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("spine_joint",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=48.0,
    armature=ARMATURE,
)

LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=16.0,
    armature=ARMATURE,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.316), # pozycja tułowia w metrach (x, y, h), chcemy aby stał na nogach
    joint_pos={
        "spine_joint": 0.0,
        "RL_hip_joint": -0.1, "RL_thigh_joint": -0.8, "RL_calf_joint": 1.5,
        "RR_hip_joint": 0.1, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.5,
        "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "FL_hip_joint": 0.1, "FL_thigh_joint": -0.8, "FL_calf_joint": 1.5,
    },
    joint_vel={".*": 0.0},
)


_FEET_REGEX = "^[FR][LR]_foot$"

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_FEET_REGEX,),
    contype=0,
    conaffinity=1,
    condim=6, # ile "kierunków" tarcia model liczy
    priority=1,
    friction=(1.1, 5e-3, 1e-4), # współczynniki tarcia (ślizgowe, obrotowe, wałkowe)
    solimp=(0.015, 1.0, 0.02),
    disable_other_geoms=True, # wyłączamy kolizję całej reszty robota
)

_BODY_REGEX = (
    "^trunk_collision$",
    "^rear_collision$",
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(_FEET_REGEX, *_BODY_REGEX),
    contype=0,
    conaffinity=1,
    condim={_FEET_REGEX: 6, ".*": 1},
    priority={_FEET_REGEX: 1},
    friction={_FEET_REGEX: (1.1, 5e-3, 1e-4)},
    solimp={_FEET_REGEX: (0.015, 1.0, 0.02)},
    solref=(0.01, 1),
    disable_other_geoms=True,
)

SILVER_BADGER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SPINE_ACTUATOR_CFG, LEG_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9, # miękki limit zakresu stawów, polityka jest zniechęcana do wchodzenia w ostatnie 10% zakresu
)

def get_silver_badger_robot_cfg(full_collision: bool = False) -> EntityCfg:
    """Zwraca świeżą konfigurację encji Silver Badgera.

    Nowa instancja przy każdym wywołaniu - żeby współdzielenie configu w wielu miejscach nie prowadziło do konfliktów

    Args:
        full_collision: gdy False (domyślnie) kolizje mają tylko stopy - lekkie, pod teren płaski. Gdy True włącza
        kolizje całego korpusu (`FULL_COLLISION`) - tułów/kręgosłup/uda/golenie zderzają się z terenem zamiast przez niego przenikać. 
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION if full_collision else FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=SILVER_BADGER_ARTICULATION,
    )

# Mapowanie wyniku sieci (z zakresu [-1, 1]) na sensowny kąt
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
