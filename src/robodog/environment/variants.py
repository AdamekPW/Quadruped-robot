"""Warianty środowiska Silver Badgera zbudowane na wspólnej bazie z `env_cfg`.

Zamiast jednej funkcji z pękiem flag logicznych mamy tu jedną bazę i kilka
funkcji, które ją przekształcają. Dzięki temu odpowiedź na pytanie „czym różni
się distylacja od CNN" mieści się w kilku linijkach zamiast w rozsianych `if`-ach.

Warianty różnią się tym, CO polityka widzi:

  * `baseline_cfg`  — skan terenu wklejony w płaski wektor obserwacji (MLP),
  * `cnn_cfg`       — ten sam skan wystawiony jako obraz 2D pod własną sieć CNN,
  * `distill_cfg`   — jak CNN, plus kamera głębi i grupy obserwacji dla studenta.

`as_play` jest prostopadłe do powyższych: przełącza DOWOLNY wariant w tryb
podglądu nauczonej polityki i z założenia stosuje się jako ostatnie.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.spec_config import CameraCfg

from robodog.assets.robots.silver_badger.constants import TERRAIN_SCAN_SITE_HEIGHT
from robodog.environment import mdp as robodog_mdp

from .constants import (
    DEPTH_CAMERA_FOVY,
    DEPTH_CAMERA_HW,
    DEPTH_CAMERA_NAME,
    DEPTH_CAMERA_POS,
    DEPTH_CAMERA_QUAT,
    DEPTH_CUTOFF,
    PLAY_ROBOTS_PER_TERRAIN_CELL,
    TERRAIN_SCAN_GRID_HW,
)
from .env_cfg import env_cfg
from .observations import depth_image, height_scan_image

# Render głębi + kolizje korpusu + height_scan = ciężkie; startowo mało środowisk
# (do dostrojenia pod VRAM). Distylacja i tak potrzebuje mniej danych niż RL.
_DISTILL_NUM_ENVS = 256


def baseline_cfg(*, lock_spine: bool = True) -> ManagerBasedRlEnvCfg:
    """Baseline MLP: teren jako płaski wektor w obserwacji actora i critica."""
    return env_cfg(lock_spine=lock_spine)


def cnn_cfg(*, lock_spine: bool = True) -> ManagerBasedRlEnvCfg:
    """Wariant CNN: ten sam robot i teren, skan wystawiony jako obraz 2D."""
    cfg = baseline_cfg(lock_spine=lock_spine)
    _use_terrain_image(cfg)
    return cfg


def distill_cfg(*, lock_spine: bool = True) -> ManagerBasedRlEnvCfg:
    """Distylacja: środowisko wystawia JEDNOCZEŚNIE skan (teacher) i głębię (student)."""
    # KOLEJNOŚĆ JEST ISTOTNA: `_use_terrain_image` musi pójść pierwsze, bo zdejmuje
    # `height_scan` z grupy actora, a `_add_depth_camera` kopiuje tę grupę na
    # `student_proprio`. Odwrotna kolejność po cichu dałaby studentowi skan terenu,
    # czyli dokładnie tę informację, której ma się nauczyć odtwarzać z głębi.
    cfg = cnn_cfg(lock_spine=lock_spine)
    _add_depth_camera(cfg)
    cfg.scene.num_envs = _DISTILL_NUM_ENVS
    return cfg


def as_play(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Przełącza wariant w tryb podglądu: długi epizod, bez zakłóceń, siatka terenu.

    Stosuj JAKO OSTATNIE — funkcja usuwa zdarzenia i curriculum, więc puszczona
    przed budową wariantu nie miałaby czego usuwać. Modyfikuje `cfg` w miejscu
    i zwraca je dla wygody zapisu `as_play(cnn_cfg())`.
    """
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}

    assert cfg.scene.terrain is not None
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    generator.curriculum = True
    generator.num_rows = 10
    generator.border_width = 20.0
    # `num_cols` celowo nie ustawiamy: przy `curriculum=True` generator i tak
    # je ignoruje i robi jedną kolumnę na typ terenu. Wiersz = poziom trudności,
    # kolumna = rodzaj przeszkody.
    num_cols = len(generator.sub_terrains)

    # Równa obsada każdego pola siatki zamiast losowego rozrzutu: cały
    # curriculum widać naraz. Liczba środowisk MUSI wyjść z siatki, inaczej
    # `assign_terrain_grid` zgłosi błąd.
    cfg.scene.num_envs = robodog_mdp.required_num_envs(
        generator.num_rows, num_cols, PLAY_ROBOTS_PER_TERRAIN_CELL
    )
    # Tryb `startup`, nie `reset`: odpala się raz, PRZED pierwszym resetem, więc
    # `reset_base` od razu widzi właściwe `env_origins`. Zdarzenie `reset`
    # dopisane na końcu słownika zadziałałoby dopiero na następny epizod.
    cfg.events["assign_terrain_grid"] = EventTermCfg(
        mode="startup",
        func=robodog_mdp.assign_terrain_grid,
        params={"robots_per_cell": PLAY_ROBOTS_PER_TERRAIN_CELL},
    )
    return cfg


def _use_terrain_image(cfg: ManagerBasedRlEnvCfg) -> None:
    """Przenosi skan terenu z płaskiego wektora do osobnej grupy o kształcie obrazu."""
    # Zdejmij płaski height_scan z actora i critica (zostaje sama propriocepcja),
    # zachowując oryginalną skalę (1/max_distance sensora).
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
                    "offset": TERRAIN_SCAN_SITE_HEIGHT,
                },
                scale=scan_scale,
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
    )


def _add_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
    """Dokłada kamerę głębi: sensor, grupy obserwacji studenta i DR ustawienia kamery."""
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.cameras = robot_cfg.cameras + (
        CameraCfg(
            name=DEPTH_CAMERA_NAME,
            body="trunk",
            pos=DEPTH_CAMERA_POS,
            quat=DEPTH_CAMERA_QUAT,
            fovy=DEPTH_CAMERA_FOVY,
        ),
    )
    depth_cam = CameraSensorCfg(
        name=DEPTH_CAMERA_NAME,
        camera_name=f"robot/{DEPTH_CAMERA_NAME}",
        width=DEPTH_CAMERA_HW[1],
        height=DEPTH_CAMERA_HW[0],
        data_types=("depth",),
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_cam,)
    cfg.observations["depth"] = ObservationGroupCfg(
        terms={
            "depth": ObservationTermCfg(
                func=depth_image,
                params={
                    "sensor_name": DEPTH_CAMERA_NAME,
                    "cutoff_distance": DEPTH_CUTOFF,
                },
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
    )
    # Student to ta sama propriocepcja co actor, ale BEZ `base_lin_vel` — prędkości
    # liniowej korpusu nie da się zmierzyć na prawdziwym robocie.
    student_terms = {
        name: deepcopy(term)
        for name, term in cfg.observations["actor"].terms.items()
        if name != "base_lin_vel"
    }
    cfg.observations["student_proprio"] = ObservationGroupCfg(
        terms=student_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        nan_policy="sanitize",
    )

    # DR ustawienia kamery: na prawdziwym robocie nigdy nie siedzi idealnie tam,
    # gdzie w modelu. Bez tego student przeuczy się na jedno ustawienie.
    cfg.events["cam_pos"] = EventTermCfg(
        mode="startup",
        func=dr.cam_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", camera_names=(DEPTH_CAMERA_NAME,)),
            "operation": "add",  # przesunięcie w metrach: +-5 mm
            "ranges": (-0.005, 0.005),
        },
    )
    cfg.events["cam_quat"] = EventTermCfg(
        mode="startup",
        func=dr.cam_quat,
        params={
            "asset_cfg": SceneEntityCfg("robot", camera_names=(DEPTH_CAMERA_NAME,)),
            "roll_range": (-0.02, 0.02),
            "pitch_range": (-0.02, 0.02),
            "yaw_range": (-0.02, 0.02),
        },
    )
