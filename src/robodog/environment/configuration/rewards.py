""" Nagrody - dostrojenie bazowych członów mjlab pod Silver Badgera """

from mjlab.managers import RewardTermCfg

from ..constants import FOOT_SITES, TERRAIN_SCAN_SENSOR_NAME

# Docelowe odchylenia standardowe pozy - osobno dla stania, marszu i biegu.
# Im większa wartość, tym luźniej polityka może odejść od pozy referencyjnej.
_POSE_STD_STANDING = {
    r"spine_joint": 0.05,
    r".*_(hip|thigh)_joint": 0.05,
    r".*_calf_joint": 0.1,
}
_POSE_STD_MOVING = {
    r"spine_joint": 0.3,
    r".*_(hip|thigh)_joint": 0.3,
    r".*_calf_joint": 0.6,
}


def prepare_rewards(rewards: dict[str, RewardTermCfg]) -> None:
    """Dostraja słownik nagród w miejscu."""
    rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    # Na terenie nierównym „pion" liczymy względem lokalnego nachylenia terenu.
    rewards["upright"].params["terrain_sensor_names"] = (TERRAIN_SCAN_SENSOR_NAME,)
    rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    rewards["body_ang_vel"].weight = 0.0

    rewards["track_linear_velocity"].weight = 4.0  # było 2.0
    rewards["pose"].weight = 0.5  # było 1.0
    rewards["track_angular_velocity"].weight = 1.0  # było 2.0
    rewards["air_time"].weight = 0.5  # było 0.0 (nagroda za stawianie kroków)

    # Brak sensora momentu pędu w naszym modelu (w go1 człon ma i tak wagę 0).
    del rewards["angular_momentum"]

    for reward_name in ("foot_clearance", "foot_slip"):
        rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

    rewards["pose"].params["std_standing"] = dict(_POSE_STD_STANDING)
    rewards["pose"].params["std_walking"] = dict(_POSE_STD_MOVING)
    rewards["pose"].params["std_running"] = dict(_POSE_STD_MOVING)
