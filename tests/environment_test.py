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

# Oczekiwane wymiary (rough, Silver Badger). Zmiana modelu/sensorów je ruszy —
# i wtedy test ma o tym głośno powiedzieć.
_EXPECTED_ACTOR_FLAT = 237  #  propriocepcja(50) + height_scan(187)
_EXPECTED_CRITIC_FLAT = 261  # propriocepcja(50) + height_scan(187) + człony uprzywilejowane krytyka(24)
_EXPECTED_TERRAIN_POINTS = 187  # 11 x 17 rzutów siatki terenu
_NUM_ENVS = 2


def _require_gpu():
    """Budowa środowiska mjlab (MuJoCo Warp) wymaga GPU CUDA."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("brak GPU CUDA — środowisko mjlab nie zbuduje się na CPU")


def _build_rough_env(**cfg_overrides):
    """Buduje środowisko rough Silver Badgera z małą liczbą env do inspekcji."""
    import robodog  # noqa: F401 — import rejestruje zadania i moduły configu
    from mjlab.envs import ManagerBasedRlEnv
    from robodog.environment.env_cfg import env_cfg

    cfg = env_cfg(**cfg_overrides)
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


def test_rough_observation_dims_flat():
    """Baseline rough: obserwacje to płaskie wektory (actor 237, critic 261)."""
    _require_gpu()
    env = _build_rough_env(terrain_as_image=False)
    try:
        obs, _ = env.reset()
        _print_observation_dims(env)
        assert tuple(obs["actor"].shape) == (_NUM_ENVS, _EXPECTED_ACTOR_FLAT)
        assert tuple(obs["critic"].shape) == (_NUM_ENVS, _EXPECTED_CRITIC_FLAT)
        # W baseline teren jest wtopiony w wektor, brak osobnej grupy obrazu.
        assert "height_scan" not in obs
    finally:
        del env


def test_rough_observation_dims_terrain_as_image():
    """Wariant CNN: height_scan wydzielony jako obraz 2D (B, 1, H, W)."""
    _require_gpu()
    from robodog.environment.constants import (
        TERRAIN_SCAN_GRID_HW,
    )

    env = _build_rough_env(terrain_as_image=True)
    try:
        obs, _ = env.reset()
        _print_observation_dims(env)
        height, width = TERRAIN_SCAN_GRID_HW
        # Teren jako obraz jednokanałowy.
        assert tuple(obs["height_scan"].shape) == (_NUM_ENVS, 1, height, width)
        # Propriocepcja bez terenu: 237-187=50 (actor), 261-187=74 (critic).
        assert tuple(obs["actor"].shape) == (
            _NUM_ENVS,
            _EXPECTED_ACTOR_FLAT - _EXPECTED_TERRAIN_POINTS,
        )
        assert tuple(obs["critic"].shape) == (
            _NUM_ENVS,
            _EXPECTED_CRITIC_FLAT - _EXPECTED_TERRAIN_POINTS,
        )
    finally:
        del env


def test_terrain_grid_shape_matches_ray_count():
    """Stała TERRAIN_SCAN_GRID_HW musi zgadzać się z liczbą rzutów sensora."""
    from robodog.environment.constants import (
        TERRAIN_SCAN_GRID_HW,
    )

    height, width = TERRAIN_SCAN_GRID_HW
    assert height * width == _EXPECTED_TERRAIN_POINTS
