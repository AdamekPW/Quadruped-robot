""" Scena - co fizycznie stoi w świecie i jak liczy się fizyka """

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainGeneratorCfg

from robodog.assets.robots.silver_badger.constants import get_silver_badger_robot_cfg

# Strojenie solvera pod kontakty na terenie nierównym (wzorowane na go1 rough).
_CCD_ITERATIONS = 500  # wykrywanie kolizji ciągłych: dużo, bo stopy są małe i szybkie
_IMPRATIO = 10  # sztywność tarcia względem sztywności normalnej — mniej ślizgania
_CONE = "elliptic"  # stożek tarcia eliptyczny: dokładniejszy niż piramidalny
_CONTACT_SENSOR_MAXMATCH = 500  # ile par kontaktów sensor zdąży obsłużyć
_NCONMAX = 80  # limit kontaktów na środowisko
_NJMAX = 260  # limit ograniczeń solvera na środowisko

# Od którego poziomu trudności startują roboty. 0 = wszyscy od najłatwiejszego
_MAX_INIT_TERRAIN_LEVEL = 0

# Maksymalne nachylenie zboczy (tangens, nie stopnie) na najwyższym poziomie
# curriculum. Bazowe 1.0 to 45 stopni — nie do wejścia przy jakimkolwiek realnym
# tarciu stóp, więc górne wiersze tych dwóch kolumn stały puste, a `..._inv`
# (miska, z której trzeba wyjść pod górę) dławiło się w okolicy poziomu 1.5.
# 0.4 to 22 stopnie wzdłuż osi i 31 po przekątnej — powierzchnia jest iloczynem
# dwóch "namiotów", więc po skosie jest do sqrt(2) razy stromiej niż mówi ta liczba.
_MAX_TERRAIN_SLOPE = 0.4

# Kolumny generatora, których dotyczy `_MAX_TERRAIN_SLOPE`.
_SLOPED_SUB_TERRAINS = ("hf_pyramid_slope", "hf_pyramid_slope_inv")


def prepare_scene(cfg: ManagerBasedRlEnvCfg) -> None:
    """ Wstawia robota, stroi solver i włącza curriculum terenu """
    cfg.scene.entities = {"robot": get_silver_badger_robot_cfg(full_collision=True)}

    cfg.sim.mujoco.ccd_iterations = _CCD_ITERATIONS
    cfg.sim.mujoco.impratio = _IMPRATIO
    cfg.sim.mujoco.cone = _CONE
    cfg.sim.contact_sensor_maxmatch = _CONTACT_SENSOR_MAXMATCH
    cfg.sim.nconmax = _NCONMAX
    cfg.sim.njmax = _NJMAX

    if cfg.scene.terrain is not None:
        # Wiersz siatki = poziom trudności, kolumna = rodzaj przeszkody.
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = True
            _limit_slopes(cfg.scene.terrain.terrain_generator)
        cfg.scene.terrain.max_init_terrain_level = _MAX_INIT_TERRAIN_LEVEL


def _limit_slopes(generator: TerrainGeneratorCfg) -> None:
    """Ścina maksymalne nachylenie zboczy do `_MAX_TERRAIN_SLOPE`.

    Podmienia CAŁY słownik na nowy zamiast poprawiać wpisy w miejscu: mjlab kopiuje
    `ROUGH_TERRAINS_CFG` płytko (`dataclasses.replace`), więc sam słownik
    `sub_terrains` i konfiguracje w nim są współdzielone przez wszystkie zadania
    w procesie. Zmiana w miejscu wyciekłaby naraz do baseline'u, CNN i distylacji.
    """
    sub_terrains = dict(generator.sub_terrains)
    for name in _SLOPED_SUB_TERRAINS:
        sub_terrains[name] = replace(
            sub_terrains[name], slope_range=(0.0, _MAX_TERRAIN_SLOPE)
        )
    generator.sub_terrains = sub_terrains
