from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg


def _expand_gravity_per_env(env: ManagerBasedRlEnv) -> None:
    """Zamienia współdzieloną grawitację `(1,)` na tablicę per-środowisko `(num_envs,)` """
    sim = env.sim
    option = sim.wp_model.opt
    gravity = option.gravity
    if gravity.shape[0] >= sim.num_envs:
        return  # Już per-świat (albo trenujemy na jednym środowisku).

    expanded = wp.zeros(
        shape=(sim.num_envs,), dtype=gravity.dtype, device=gravity.device
    )
    wp.to_torch(expanded)[:] = wp.to_torch(gravity)[0]
    option.gravity = expanded

    sim.model.clear_cache()
    sim.create_graph()


class gravity:
    """ Losuje wektor grawitacji osobno dla każdego środowiska """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg  # Zakresy przychodzą przy każdym wywołaniu (curriculum je zmienia).
        _expand_gravity_per_env(env)
        self._nominal_z = float(env.sim.model.opt.gravity[0, 2].item())

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        xy_range: tuple[float, float] = (0.0, 0.0),
        z_scale_range: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        gravity_field = env.sim.model.opt.gravity
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        count = int(env_ids.numel())

        sampled = torch.empty(count, 3, device=env.device)
        sampled[:, :2].uniform_(*xy_range) # przechył w poziomie
        sampled[:, 2].uniform_(*z_scale_range).mul_(self._nominal_z) # siła ciążenia
        gravity_field[env_ids] = sampled
