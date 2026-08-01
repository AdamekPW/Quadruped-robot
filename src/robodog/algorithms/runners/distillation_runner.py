"""
Warstwa spinająca distylację rsl-rl z mjlab.

"""

from __future__ import annotations

import torch
from rsl_rl.runners import DistillationRunner

from mjlab.rl import RslRlVecEnvWrapper


class MjlabDistillationRunner(DistillationRunner):
    """`DistillationRunner` z czyszczeniem konfigów modeli i jawnym ładowaniem teachera."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)

        # Zdejmij None-owe opcje modeli, żeby MLPModel ich nie dostał
        for key in ("student", "teacher"):
            if key not in train_cfg:
                continue

            for opt in ("cnn_cfg", "distribution_cfg"):
                if train_cfg[key].get(opt) is None:
                    train_cfg[key].pop(opt, None)
            if train_cfg[key].get("rnn_type") is None:
                for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                    train_cfg[key].pop(opt, None)

        super().__init__(env, train_cfg, log_dir, device)

        if teacher_checkpoint:
            loaded = torch.load(teacher_checkpoint, weights_only=False, map_location=device)
            self.alg.load(loaded, load_cfg={"teacher": True, "iteration": False}, strict=True)
            print(f"[INFO] Załadowano teachera z: {teacher_checkpoint}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Tłumaczy `load_cfg` z mjlab (`{"actor": True}`) na słownik distylacji"""
        if (load_cfg is not None) and ("actor" in load_cfg) and ("student" not in load_cfg):
            load_cfg = {
                "student": bool(load_cfg["actor"]),
                "teacher": True,
                "iteration": load_cfg.get("iteration", False),
            }
        return super().load(path, load_cfg, strict, map_location)
