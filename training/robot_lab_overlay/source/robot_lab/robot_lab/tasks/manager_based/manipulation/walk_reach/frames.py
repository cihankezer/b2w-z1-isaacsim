"""T3 frames (protocol v0.3 §5) in torch. No Isaac Lab import. Quaternions wxyz.

F_T: origin (x_b, y_b, z_terrain + h_T), axes = base yaw, z up. Follows base x/y and yaw only.
F_H: heading frame for velocities (base yaw, z up).
"""

import torch

from ..reach_height.torch_targets import matrix_to_quat, yaw_from_quat, yaw_matrix

__all__ = ["matrix_to_quat", "yaw_from_quat", "yaw_matrix", "quat_mul", "quat_conjugate", "quat_apply",
           "yaw_quat", "ft_origin", "ft_to_world", "world_to_ft", "heading_velocity", "spherical", "tilt"]


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def yaw_quat(yaw):
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)


def ft_origin(base_pos_w, h_T, terrain_z):
    """F_T origin (N, 3) for base positions (N, 3) and terrain height under the base (N,)."""
    return torch.stack([base_pos_w[:, 0], base_pos_w[:, 1], terrain_z + h_T], dim=1)


def ft_to_world(base_pos_w, base_quat_w, p_T, q_T, h_T, terrain_z):
    """Target pose in F_T -> world: p_w = o_T + R_z(psi) p_T, q_w = q_z(psi) * q_T."""
    qz = yaw_quat(yaw_from_quat(base_quat_w))
    return ft_origin(base_pos_w, h_T, terrain_z) + quat_apply(qz, p_T), quat_mul(qz, q_T)


def world_to_ft(base_pos_w, base_quat_w, p_w, q_w, h_T, terrain_z):
    """Inverse of ft_to_world."""
    qz_inv = quat_conjugate(yaw_quat(yaw_from_quat(base_quat_w)))
    return quat_apply(qz_inv, p_w - ft_origin(base_pos_w, h_T, terrain_z)), quat_mul(qz_inv, q_w)


def heading_velocity(base_quat_w, lin_vel_w, ang_vel_w):
    """(v_H,x, v_H,y, omega_z): world velocity rotated by -yaw, world yaw rate (§5.2)."""
    v = quat_apply(quat_conjugate(yaw_quat(yaw_from_quat(base_quat_w))), lin_vel_w)
    return v[:, 0], v[:, 1], ang_vel_w[:, 2]


def spherical(p):
    """(l, pitch, yaw) of points (..., 3) in F_T: pitch positive up, yaw positive left (§5.4)."""
    l = p.norm(dim=-1)
    return l, torch.atan2(p[..., 2], p[..., :2].norm(dim=-1)), torch.atan2(p[..., 1], p[..., 0])


def tilt(base_quat_w):
    """Angle between the base z-axis and world z [rad]."""
    z = quat_apply(base_quat_w, torch.tensor([0.0, 0.0, 1.0], dtype=base_quat_w.dtype,
                                             device=base_quat_w.device).expand(base_quat_w.shape[:-1] + (3,)))
    return torch.acos(z[..., 2].clamp(-1.0, 1.0))
