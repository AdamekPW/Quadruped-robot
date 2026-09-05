from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from robodog.assets.robots.silver_badger.constants import TERRAIN_SCAN_SITE_HEIGHT

from .configuration.actions import prepare_actions
from .configuration.curriculum import prepare_curriculum
from .configuration.events import prepare_events
from .configuration.metrics import add_velocity_metrics
from .configuration.rewards import prepare_rewards
from .configuration.scene import prepare_scene
from .configuration.sensors import prepare_sensors
from .constants import TRUNK_CONTACT_SENSOR_NAME


def env_cfg(*, lock_spine: bool = True) -> ManagerBasedRlEnvCfg:
    """ Bazowe zadanie velocity Silver Badgera na terenie nierównym (+ curriculum) """

    cfg = make_velocity_env_cfg()

    cfg.scene.num_envs = 2048
    prepare_scene(cfg)
    prepare_sensors(cfg.scene)

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

    prepare_actions(cfg.actions, lock_spine=lock_spine)

    prepare_rewards(cfg.rewards)

    prepare_events(cfg.events)

    # po prepare_events: curriculum wymienia nazwy zdarzeń, muszą już istnieć.
    prepare_curriculum(cfg.curriculum)

    # --- Podgląd ---
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # --- Terminacje ---
    cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    cfg.terminations["trunk_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": TRUNK_CONTACT_SENSOR_NAME},
    )

    return cfg
