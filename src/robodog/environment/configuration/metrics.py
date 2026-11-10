""" Metryki diagnostyczne: co logujemy, żeby widzieć postęp treningu """

from mjlab.managers import MetricsTermCfg, SceneEntityCfg

from ..mdp.metrics import (
    TWIST_COMMAND_NAME,
    ang_vel_error_yaw,
    commanded_lin_vel_x,
    lin_vel_error_xy,
    realized_lin_vel_x,
    spine_angle,
)


def add_velocity_metrics(
    metrics: dict[str, MetricsTermCfg],
    command_name: str = TWIST_COMMAND_NAME,
) -> None:
    """ Dopisuje metryki śledzenia prędkości do słownika metryk środowiska """
    params = {"command_name": command_name}
    metrics["vel_error_xy"] = MetricsTermCfg(func=lin_vel_error_xy, params=params)
    metrics["vel_error_yaw"] = MetricsTermCfg(func=ang_vel_error_yaw, params=params)
    metrics["vel_cmd_x"] = MetricsTermCfg(func=commanded_lin_vel_x, params=params)
    metrics["vel_realized_x"] = MetricsTermCfg(func=realized_lin_vel_x)

    spine_cfg = SceneEntityCfg("robot", joint_names=("spine_joint",))
    metrics["spine_abs_angle"] = MetricsTermCfg(func=spine_angle, params={"asset_cfg": spine_cfg})
