""" Deterministyczne rozstawianie robotów po siatce terenu """

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def required_num_envs(num_rows: int, num_cols: int, robots_per_cell: int) -> int:
    """Ile środowisk trzeba, by obsadzić siatkę `num_rows` x `num_cols` po równo."""
    return num_rows * num_cols * robots_per_cell


def assign_terrain_grid(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robots_per_cell: int = 3,
) -> None:
    """Sadza dokładnie `robots_per_cell` robotów na każdym polu siatki terenu.
    Args:
        env: Środowisko — musi mieć teren z generatora.
        env_ids: Środowiska do rozstawienia; `None` oznacza wszystkie.
        robots_per_cell: Ilu robotów na jedno pole siatki.
    """
    if robots_per_cell < 1:
        raise ValueError(
            f"assign_terrain_grid: robots_per_cell musi być >= 1, dostano {robots_per_cell}."
        )

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise ValueError(
            "assign_terrain_grid wymaga terenu z generatora "
            "(terrain_type='generator'); płaski `plane` nie ma siatki do obsadzenia."
        )

    # Kształt tablicy, NIE config: przy `curriculum=True` generator ignoruje
    # `num_cols` i robi jedną kolumnę na typ terenu, więc config potrafi kłamać.
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    needed = required_num_envs(num_rows, num_cols, robots_per_cell)
    if env.num_envs != needed:
        raise ValueError(
            f"assign_terrain_grid: siatka {num_rows}x{num_cols} po {robots_per_cell} "
            f"robotów na pole wymaga {needed} środowisk, a scena ma {env.num_envs}."
        )

    ids = resolve_env_ids(env, env_ids).long()
    cell = ids // robots_per_cell
    levels = cell // num_cols
    types = cell % num_cols

    terrain.terrain_levels[ids] = levels
    terrain.terrain_types[ids] = types
    # `scene.env_origins` zwraca ten sam tensor, więc `reset_base` zobaczy zmianę.
    terrain.env_origins[ids] = terrain.terrain_origins[levels, types]
