# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the WF-TRON2A balance and locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_planar_velocity_exp(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Reward tracking the commanded forward and lateral body velocities."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - robot.data.root_lin_vel_b.torch[:, :2]), dim=1)
    return torch.exp(-error / 0.25)


def track_yaw_velocity_exp(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Reward tracking the commanded yaw rate."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - robot.data.root_ang_vel_b.torch[:, 2])
    return torch.exp(-error / 0.25)


def upright_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize roll and pitch away from upright."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.projected_gravity_b.torch[:, :2]), dim=1)


def fallen(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.35) -> torch.Tensor:
    """End episodes when the torso tips too far or drops close to the floor."""
    robot = env.scene[asset_cfg.name]
    tipped = torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:, :2], dim=1) > 0.75
    too_low = robot.data.root_pos_w.torch[:, 2] < min_height
    return tipped | too_low
