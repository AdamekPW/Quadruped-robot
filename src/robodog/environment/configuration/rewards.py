""" Nagrody - dostrojenie bazowych członów mjlab pod Silver Badgera """

from mjlab.managers import RewardTermCfg

from robodog.assets.robots.silver_badger.constants import SPINE_JOINT_NAME

from ..constants import FOOT_SITES, TERRAIN_SCAN_SENSOR_NAME

# Docelowe odchylenia standardowe pozy - osobno dla stania, marszu i biegu.
# Im większa wartość, tym luźniej polityka może odejść od pozy referencyjnej.
_POSE_STD_STANDING = {
    SPINE_JOINT_NAME: 0.05,
    r".*_(hip|thigh)_joint": 0.05,
    r".*_calf_joint": 0.1,
}
_POSE_STD_MOVING = {
    SPINE_JOINT_NAME: 0.3,
    r".*_(hip|thigh)_joint": 0.3,
    r".*_calf_joint": 0.6,
}


def _pose_std(std: dict[str, float], *, lock_spine: bool) -> dict[str, float]:
    """Kopia słownika `std` pozy; bez wpisu kręgosłupa, gdy model nie ma tego stawu.

    Nagroda `pose` zgłasza błąd na wzorzec, który nie pasuje do żadnego stawu.
    """
    std = dict(std)
    if lock_spine:
        del std[SPINE_JOINT_NAME]
    return std


def prepare_rewards(rewards: dict[str, RewardTermCfg], *, lock_spine: bool) -> None:
    """Dostraja słownik nagród w miejscu.

    Args:
        lock_spine: gdy True model nie ma stawu kręgosłupa (pomijamy go w `pose`).
    """
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

    pose_params = rewards["pose"].params
    pose_params["std_standing"] = _pose_std(_POSE_STD_STANDING, lock_spine=lock_spine)
    pose_params["std_walking"] = _pose_std(_POSE_STD_MOVING, lock_spine=lock_spine)
    pose_params["std_running"] = _pose_std(_POSE_STD_MOVING, lock_spine=lock_spine)
