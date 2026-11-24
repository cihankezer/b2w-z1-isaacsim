"""Shared policy observation of the benchmark suite (95 dims; T1 2.1.0, T2, T3; T3 protocol D40).

Order: joint positions (22), joint velocities (22), base angular velocity (3), projected gravity (3), previous
action (22), base-velocity goal (3; zero in T1/T2), TCP goal in B (7: position + quaternion wxyz), measured TCP
pose in B (7), TCP pose error (6). The goal command is named per task (T1 `ee_pose`, T2/T3 `tcp_target`); it
must provide the TCP goal in B as its command and `tcp_pose_w()` (measured TCP pose in W).
"""

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, subtract_frame_transforms

from .runtime import JOINTS, zero_velocity_goal

_JOINTS = SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)


def tcp_pose_body(env, command_name="tcp_target"):
    """Measured TCP pose in B: position + quaternion wxyz (7)."""
    cmd, d = env.command_manager.get_term(command_name), env.scene["robot"].data
    p_w, q_w = cmd.tcp_pose_w()
    p, q = subtract_frame_transforms(d.root_pos_w, d.root_quat_w, p_w, q_w)
    return torch.cat([p, q], dim=1)


def tcp_pose_error_body(env, command_name="tcp_target"):
    """Target minus measured TCP in B: position (3) and rotation as axis-angle of q_goal * q_tcp^-1 (3)."""
    goal, tcp = env.command_manager.get_command(command_name), tcp_pose_body(env, command_name)
    rot = axis_angle_from_quat(quat_mul(goal[:, 3:], quat_conjugate(tcp[:, 3:])))
    return torch.cat([goal[:, :3] - tcp[:, :3], rot], dim=1)


@configclass
class SuitePolicyCfg(ObservationGroupCfg):
    """The 95-dim policy group; subclasses only change the command names (term order is fixed here)."""

    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS})
    joint_vel = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": _JOINTS})
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
    projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
    previous_action = ObservationTermCfg(func=mdp.last_action)
    base_velocity_goal = ObservationTermCfg(func=zero_velocity_goal)
    tcp_goal_body = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "tcp_target"})
    tcp_pose_body = ObservationTermCfg(func=tcp_pose_body, params={"command_name": "tcp_target"})
    tcp_pose_error_body = ObservationTermCfg(func=tcp_pose_error_body, params={"command_name": "tcp_target"})

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
