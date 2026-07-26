"""Obserwacje specyficzne dla RoboDoga.

Na razie jedna funkcja: skan terenu jako obraz 2D pod sieć konwolucyjną.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp


def height_scan_image(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    grid_hw: tuple[int, int],
    offset: float = 0.0,
    miss_value: float | None = None,
) -> torch.Tensor:
    """Skan wysokości terenu jako obraz 2D `(B, 1, H, W)`.

    Owija wbudowany `height_scan` (zwraca płaski wektor `(B, H*W)`) i zwija go do
    siatki 2D, żeby sieć konwolucyjna widziała strukturę przestrzenną terenu
    zamiast worka luźnych liczb.

    Kolejność punktów jest zgodna z `GridPatternCfg`: rzuty są generowane
    `meshgrid(x, y, indexing="xy")` i spłaszczane wierszowo, więc reshape do
    `(H, W)` z `H` = liczba wartości po osi Y (bocznej) i `W` = po osi X (do
    przodu) odtwarza spójną przestrzennie mapę. Wartości `grid_hw` bierz ze
    stałej `TERRAIN_SCAN_GRID_HW` (dopasowanej do sensora).

    Args:
        env: środowisko.
        sensor_name: nazwa sensora RayCast (siatka rzutów), np. "terrain_scan".
        grid_hw: kształt siatki `(H, W)`; iloczyn musi równać się liczbie rzutów.
        offset: stały offset odejmowany od wysokości (jak w bazowym height_scan).
        miss_value: wartość dla chybionych rzutów (domyślnie max_distance sensora).

    Returns:
        Tensor `(B, 1, H, W)` — pojedynczy kanał, gotowy pod Conv2d.
    """
    flat = envs_mdp.height_scan(
        env, sensor_name=sensor_name, offset=offset, miss_value=miss_value
    )
    height, width = grid_hw
    batch = flat.shape[0]
    assert flat.shape[-1] == height * width, (
        f"Skan terenu ma {flat.shape[-1]} punktów, a siatka {height}x{width}="
        f"{height * width}. Zaktualizuj TERRAIN_SCAN_GRID_HW pod GridPatternCfg "
        f"(size/resolution) sensora {sensor_name!r}."
    )
    return flat.reshape(batch, 1, height, width)
