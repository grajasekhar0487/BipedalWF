# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WF-TRON2A balancing and velocity-tracking environment."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as core_mdp
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from BipedalWF.assets import BIPEDALWF_ASSETS_DIR

from ... import mdp


JOINT_NAMES = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "wheel_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
    "wheel_R_Joint",
]


@configclass
class Tron2APhysicsCfg(PresetCfg):
    """Supported simulator presets for WF-TRON2A."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True))
    default = newton_mjwarp


@configclass
class Tron2ASceneCfg(InteractiveSceneCfg):
    """Ground and robot scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BIPEDALWF_ASSETS_DIR / "robots" / "WF_TRON2A" / "usd" / "robot.usd"),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # With the URDF's zero joint pose, the wheel radius places contact at z ~= 0.75 m.
            pos=(0.0, 0.0, 0.75),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "leg_joints": ImplicitActuatorCfg(
                joint_names_expr=["proximal_pitch_[LR]_Joint", "proximal_roll_[LR]_Joint", "knee_[LR]_Joint"],
                stiffness=0.0,
                damping=0.0,
                joint_effort_limit=150.0,
                joint_velocity_limit=18.3,
            ),
            "yaw_joints": ImplicitActuatorCfg(
                joint_names_expr=["proximal_yaw_[LR]_Joint"],
                stiffness=0.0,
                damping=0.0,
                joint_effort_limit=60.0,
                joint_velocity_limit=22.52,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_[LR]_Joint"],
                stiffness=0.0,
                damping=0.0,
                joint_effort_limit=22.0,
                joint_velocity_limit=41.89,
            ),
        },
        soft_joint_pos_limit_factor=0.9,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class CommandsCfg:
    """Velocity commands for balancing while driving and turning."""

    base_velocity: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 7.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.6, 0.6)
        ),
    )


@configclass
class ActionsCfg:
    """Joint torque commands with conservative scales below the URDF actuator limits."""

    joint_effort = core_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        scale={
            "proximal_pitch_[LR]_Joint": 40.0,
            "proximal_roll_[LR]_Joint": 25.0,
            "proximal_yaw_[LR]_Joint": 15.0,
            "knee_[LR]_Joint": 40.0,
            "wheel_[LR]_Joint": 10.0,
        },
    )


@configclass
class ObservationsCfg:
    """Base IMU state, joint state, velocity targets, and previous action."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=core_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=core_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=core_mdp.projected_gravity)
        velocity_command = ObsTerm(func=core_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=core_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=core_mdp.joint_vel_rel)
        actions = ObsTerm(func=core_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset the robot close to its upright standing configuration."""

    reset_base = EventTerm(
        func=core_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.15, 0.15)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTerm(
        func=core_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.04, 0.04),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Rewards encourage upright balance, command tracking, and efficient control."""

    alive = RewTerm(func=core_mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=core_mdp.is_terminated, weight=-5.0)
    planar_velocity = RewTerm(func=mdp.track_planar_velocity_exp, weight=2.0)
    yaw_velocity = RewTerm(func=mdp.track_yaw_velocity_exp, weight=0.5)
    upright = RewTerm(
        func=mdp.upright_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    angular_velocity = RewTerm(
        func=core_mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques = RewTerm(
        func=core_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=core_mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """End episodes after falling or when the time limit is reached."""

    time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
    fallen = DoneTerm(func=mdp.fallen, params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.35})


@configclass
class Tron2AEnvCfg(ManagerBasedRLEnvCfg):
    """WF-TRON2A balancing and velocity-command locomotion environment."""

    scene: Tron2ASceneCfg = Tron2ASceneCfg(num_envs=512, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics = Tron2APhysicsCfg()
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(3.5, 3.5, 2.5), lookat=(0.0, 0.0, 0.8))
        self.sim.physics_material.static_friction = 1.2
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
