""" Scena - co fizycznie stoi w świecie i jak liczy się fizyka """

from mjlab.envs import ManagerBasedRlEnvCfg

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
        cfg.scene.terrain.max_init_terrain_level = _MAX_INIT_TERRAIN_LEVEL
