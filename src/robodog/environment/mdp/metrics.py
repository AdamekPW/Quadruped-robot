"""
Metryki diagnostyczne śledzenia komendy prędkości.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Nazwa członu komendy prędkości w konfiguracji velocity mjlab.
TWIST_COMMAND_NAME = "twist"


def lin_vel_error_xy(
    env: ManagerBasedRlEnv,
    command_name: str = TWIST_COMMAND_NAME,
    asset_name: str = "robot",
) -> torch.Tensor:
    """ Błąd śledzenia zadanej prędkości liniowej w płaszczyźnie XY [m/s] """
    command = env.command_manager.get_command(command_name)  # (B, 3)
    lin_vel = env.scene[asset_name].data.root_link_lin_vel_b  # (B, 3)
    return torch.norm(command[:, :2] - lin_vel[:, :2], dim=-1)  # type: ignore # (B,)


def ang_vel_error_yaw(
    env: ManagerBasedRlEnv,
    command_name: str = TWIST_COMMAND_NAME,
    asset_name: str = "robot",
) -> torch.Tensor:
    """ Błąd śledzenia zadanej prędkości obrotowej wokół osi pionowej [rad/s] """
    command = env.command_manager.get_command(command_name)  # (B, 3)
    ang_vel = env.scene[asset_name].data.root_link_ang_vel_b  # (B, 3)
    return torch.abs(command[:, 2] - ang_vel[:, 2])  # type: ignore # (B,)


def commanded_lin_vel_x(
    env: ManagerBasedRlEnv,
    command_name: str = TWIST_COMMAND_NAME,
) -> torch.Tensor:
    """ Zadana prędkość do przodu [m/s] ze znakiem """
    return env.command_manager.get_command(command_name)[:, 0]  # type: ignore # (B,)


def realized_lin_vel_x(
    env: ManagerBasedRlEnv,
    asset_name: str = "robot",
) -> torch.Tensor:
    """ Faktycznie osiągana prędkość do przodu [m/s] ze znakiem """
    return env.scene[asset_name].data.root_link_lin_vel_b[:, 0]  # (B,)
