"""Testy wymiarów obserwacji środowiska Silver Badgera.

Służą do sprawdzania, jak wyglądają wymiary wejścia sieci: płaski baseline
(wszystko sklejone w jeden wektor) vs wariant z terenem jako obrazem 2D pod
własną sieć konwolucyjną.

INTEGRACYJNE — budują środowisko (kompilacja MuJoCo Warp), więc są wolniejsze
niż smoke testy z `environment_version_test.py` i wymagają GPU CUDA. Odpalaj
świadomie, a flagą `-s` zobaczysz wypisane rozbicie wymiarów na człony:

    pytest tests/environment_test.py -s
"""

import pytest

# Oczekiwane wymiary (Silver Badger). Zmiana modelu/sensorów je ruszy —
# i wtedy test ma o tym głośno powiedzieć.
#
# Propriocepcja = korpus(9: prędkość liniowa, kątowa, grawitacja) + komenda(3)
# + 3 x liczba stawów (kąty, prędkości, poprzednia akcja):
#   kręgosłup usztywniony: 12 stawów -> 12 + 36 = 48
#   kręgosłup ruchomy:     13 stawów -> 12 + 39 = 51
_EXPECTED_PROPRIO = {True: 48, False: 51}  # klucz: lock_spine
_EXPECTED_TERRAIN_POINTS = 187  # 11 x 17 rzutów siatki terenu
_EXPECTED_CRITIC_PRIVILEGED = 24  # człony uprzywilejowane krytyka (stopy)
_NUM_ENVS = 2

_LOCK_SPINE_VARIANTS = pytest.mark.parametrize(
    "lock_spine", [True, False], ids=["spine_locked", "spine_free"]
)


def _require_gpu():
    """Budowa środowiska mjlab (MuJoCo Warp) wymaga GPU CUDA."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("brak GPU CUDA — środowisko mjlab nie zbuduje się na CPU")


def _build_env(cfg):
    """Buduje środowisko z podanej konfiguracji, z małą liczbą env do inspekcji."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg.scene.num_envs = _NUM_ENVS
    return ManagerBasedRlEnv(cfg=cfg, device="cuda")


def _print_observation_dims(env) -> None:
    """Wypisuje rozbicie wymiarów obserwacji na grupy i człony (widoczne z -s)."""
    om = env.observation_manager
    print("\n--- wymiary obserwacji (grupa -> człony) ---")
    for group in om.active_terms:
        names = om.active_terms[group]
        dims = om.group_obs_term_dim[group]
        print(f"  grupa {group!r}: łącznie {om.group_obs_dim[group]}")
        for name, dim in zip(names, dims):
            print(f"      {name:22s} {dim}")


@_LOCK_SPINE_VARIANTS
def test_baseline_observation_dims_flat(lock_spine: bool):
    """Baseline MLP: obserwacje to płaskie wektory propriocepcja + teren (+ uprzywilejowane)."""
    _require_gpu()
    from robodog.environment.variants import baseline_cfg

    env = _build_env(baseline_cfg(lock_spine=lock_spine))
    try:
        obs, _ = env.reset()
        _print_observation_dims(env)
        actor_dim = _EXPECTED_PROPRIO[lock_spine] + _EXPECTED_TERRAIN_POINTS
        assert tuple(obs["actor"].shape) == (_NUM_ENVS, actor_dim)
        assert tuple(obs["critic"].shape) == (
            _NUM_ENVS,
            actor_dim + _EXPECTED_CRITIC_PRIVILEGED,
        )
        # W baseline teren jest wtopiony w wektor, brak osobnej grupy obrazu.
        assert "height_scan" not in obs
    finally:
        del env


@_LOCK_SPINE_VARIANTS
def test_cnn_observation_dims_terrain_as_image(lock_spine: bool):
    """Wariant CNN: height_scan wydzielony jako obraz 2D (B, 1, H, W)."""
    _require_gpu()
    from robodog.environment.constants import TERRAIN_SCAN_GRID_HW
    from robodog.environment.variants import cnn_cfg

    env = _build_env(cnn_cfg(lock_spine=lock_spine))
    try:
        obs, _ = env.reset()
        _print_observation_dims(env)
        height, width = TERRAIN_SCAN_GRID_HW
        # Teren jako obraz jednokanałowy.
        assert tuple(obs["height_scan"].shape) == (_NUM_ENVS, 1, height, width)
        # Bez terenu w wektorze zostaje sama propriocepcja (+ uprzywilejowane krytyka).
        proprio = _EXPECTED_PROPRIO[lock_spine]
        assert tuple(obs["actor"].shape) == (_NUM_ENVS, proprio)
        assert tuple(obs["critic"].shape) == (
            _NUM_ENVS,
            proprio + _EXPECTED_CRITIC_PRIVILEGED,
        )
    finally:
        del env


@_LOCK_SPINE_VARIANTS
def test_spine_variant_shapes_robot_and_action(lock_spine: bool):
    """Usztywniony kręgosłup = brak stawu w modelu; ruchomy = staw + kanał akcji + metryka."""
    _require_gpu()
    from robodog.assets.robots.silver_badger.constants import SPINE_JOINT_NAME
    from robodog.environment.variants import baseline_cfg

    env = _build_env(baseline_cfg(lock_spine=lock_spine))
    try:
        robot = env.scene["robot"]
        expected_joints = 12 if lock_spine else 13
        assert len(robot.joint_names) == expected_joints
        assert env.action_manager.total_action_dim == expected_joints
        assert (SPINE_JOINT_NAME in robot.joint_names) is not lock_spine
        assert ("spine_abs_angle" in env.metrics_manager.active_terms) is not lock_spine
    finally:
        del env


def test_terrain_grid_shape_matches_ray_count():
    """Stała TERRAIN_SCAN_GRID_HW musi zgadzać się z liczbą rzutów sensora."""
    from robodog.environment.constants import (
        TERRAIN_SCAN_GRID_HW,
    )

    height, width = TERRAIN_SCAN_GRID_HW
    assert height * width == _EXPECTED_TERRAIN_POINTS
