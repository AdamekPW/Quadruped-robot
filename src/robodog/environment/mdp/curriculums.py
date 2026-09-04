"""
Curriculum płynnie rozkręcający siłę domain randomization.

"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

Range = tuple[float, float]


def _is_range(value: Any) -> bool:
    """Czy wartość wygląda jak zakres losowania `(min, max)`?"""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _is_range_param(key: str, value: Any) -> bool:
    """Czy dany parametr członu zdarzenia jest zakresem (albo słownikiem zakresów)?
    """
    if not (key == "ranges" or key.endswith("_range") or key.endswith("_ranges")):
        return False
    if _is_range(value):
        return True
    return isinstance(value, dict) and all(_is_range(item) for item in value.values())


def _shrink(value: Any, coeff: float) -> Any:
    """Ściąga zakres (lub słownik zakresów) w stronę jego środka."""
    if isinstance(value, dict):
        return {key: _shrink(item, coeff) for key, item in value.items()}
    low, high = value
    middle = 0.5 * (low + high)
    return (middle + (low - middle) * coeff, middle + (high - middle) * coeff)


class dr_curriculum:
    """
    Rozkręca domain randomization od zera do pełnej siły w oknie kroków treningu.
    
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        event_names: tuple[str, ...] = tuple(cfg.params["event_names"])
        action_name: str | None = cfg.params.get("action_name")

        if cfg.params["start_step"] >= cfg.params["end_step"]:
            raise ValueError(
                "dr_curriculum: start_step musi być mniejsze niż end_step "
                f"(dostano {cfg.params['start_step']} i {cfg.params['end_step']})."
            )

        self._full_ranges: dict[str, dict[str, Any]] = {}
        for name in event_names:
            params = env.event_manager.get_term_cfg(name).params
            ranges = {
                key: deepcopy(value)
                for key, value in params.items()
                if _is_range_param(key, value)
            }
            if not ranges:
                raise ValueError(
                    f"dr_curriculum: człon zdarzenia {name!r} nie ma żadnego "
                    "parametru wyglądającego jak zakres losowania. Sprawdź nazwę."
                )
            self._full_ranges[name] = ranges

        self._action_term = (
            env.action_manager.get_term(action_name) if action_name else None
        )
        self._last_coeff: float | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        event_names: tuple[str, ...],
        start_step: int,
        end_step: int,
        action_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        del env_ids, event_names, action_name  # Rozwiązane raz w `__init__`.

        progress = (env.common_step_counter - start_step) / (end_step - start_step)
        coeff = min(max(progress, 0.0), 1.0)

        if coeff != self._last_coeff:
            self._last_coeff = coeff
            for name, ranges in self._full_ranges.items():
                params = env.event_manager.get_term_cfg(name).params
                for key, value in ranges.items():
                    params[key] = _shrink(value, coeff)
            if self._action_term is not None:
                self._action_term.curriculum_coeff = coeff

        return {"strength": torch.tensor(coeff)}
