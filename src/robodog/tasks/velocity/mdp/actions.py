"""
Akcje z opóźnieniem - symulacja lagu pętli sterowania.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Konfiguracja sterowania pozycyjnego z losowym opóźnieniem akcji."""

    max_delay_steps: int = 1
    jitter_probability: float = 0.05

    def build(self, env: ManagerBasedRlEnv) -> "DelayedJointPositionAction":
        return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
    """Sterowanie pozycyjne stawów z buforem historii akcji."""

    cfg: DelayedJointPositionActionCfg # type: ignore

    def __init__(
        self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg=cfg, env=env)

        if cfg.max_delay_steps < 0:
            raise ValueError(
                f"max_delay_steps musi być >= 0, dostano {cfg.max_delay_steps}."
            )
        if not 0.0 <= cfg.jitter_probability <= 1.0:
            raise ValueError(
                f"jitter_probability musi być w [0, 1], dostano {cfg.jitter_probability}."
            )

        # Bufor cykliczny: wiersz -1 to akcja bieżąca, -1-k to sprzed k kroków.
        self._history = torch.zeros(
            cfg.max_delay_steps + 1,
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        # Ile kroków opóźnienia ma konkretne środowisko
        self._delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Czy w danym środowisku losujemy opóźnienie w każdym epizodzie
        self._jitter = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._env_index = torch.arange(self.num_envs, device=self.device)

        self.curriculum_coeff: float = 1.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        ids = self._resolve_env_ids(env_ids)
        self._history[:, ids] = 0.0
        self._resample_delay(ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)

        if self.cfg.max_delay_steps == 0:
            return

        self._history = torch.roll(self._history, shifts=-1, dims=0)
        self._history[-1] = self._processed_actions

        delay = self._delay_steps
        if self.cfg.jitter_probability > 0.0:
            delay = torch.where(self._jitter, self._sample_delay(self.num_envs), delay)

        # Curriculum ściąga opóźnienie do zera na starcie treningu
        delay = torch.round(delay * self.curriculum_coeff).long()
        delay = delay.clamp_(0, self.cfg.max_delay_steps)

        rows = self._history.shape[0] - 1 - delay
        self._processed_actions = self._history[rows, self._env_index]

    def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids
        if env_ids is None:
            return self._env_index
        return self._env_index[env_ids]

    def _sample_delay(self, count: int) -> torch.Tensor:
        return torch.randint(
            0,
            self.cfg.max_delay_steps + 1,
            (count,),
            device=self.device,
            dtype=torch.long,
        )

    def _resample_delay(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        if count == 0:
            return
        self._delay_steps[env_ids] = self._sample_delay(count)
        self._jitter[env_ids] = (
            torch.rand(count, device=self.device) < self.cfg.jitter_probability
        )
