"""Z1 + base-offset forward kinematics in torch (protocol v0.3 §5.3, §6.2). No Isaac Lab import.

A goal g = (q_arm (6), b (3)) with b = (dh, pitch, roll). The base pose in F_T is
  position (0, 0, h_nom + dh - h_T), rotation Ry(pitch) Rx(roll)   (base x/y = F_T origin, yaw 0 in F_T),
the T2 base model (wholebody.base_pose). The chain base_link -> link06 and the TCP offset come from
data/kinematics.json (exported from the USD joint tree by evaluation/measure_t3.py).
"""

import torch

from . import protocol as P
from .frames import matrix_to_quat


def _axis_angle_matrix(axis, angle):
    """Rotation matrices (N, 3, 3) about a fixed unit axis (3,) by angles (N,)."""
    k = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                     dtype=angle.dtype, device=angle.device)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + s * k + (1 - c) * (k @ k)


class ArmFK:
    """Batched TCP pose in F_T for goals (q_arm, b)."""

    def __init__(self, device="cpu", dtype=torch.float32, data=P.KINEMATICS):
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
        self.steps = [(t(s["local0"]), None if s["axis"] is None else [float(a) for a in s["axis"]], t(s["local1_inv"]))
                      for s in data["steps"]]
        self.tcp = torch.eye(4, dtype=dtype, device=device)
        self.tcp[:3, 3] = t(data["tcp_offset_link06"])
        self.base_z = float(data["h_nom"]) - float(data["h_T"])
        self.soft_limits = t(data["soft_limits_rad"])
        self.nominal_q = t(data["nominal_arm_q"])
        self.b_nom = t(data["b_nom"])
        self.T_base_link00 = t(data["T_base_link00"])
        self.dtype, self.device = dtype, device

    def base(self, b):
        """Base transforms in F_T (N, 4, 4) for offsets b (N, 3)."""
        n = len(b)
        T = torch.zeros(n, 4, 4, dtype=b.dtype, device=b.device)
        Ry = _axis_angle_matrix([0.0, 1.0, 0.0], b[:, 1])
        Rx = _axis_angle_matrix([1.0, 0.0, 0.0], b[:, 2])
        T[:, :3, :3] = Ry @ Rx
        T[:, 2, 3] = self.base_z + b[:, 0]
        T[:, 3, 3] = 1.0
        return T

    def tcp_matrix(self, q, b):
        """TCP transforms in F_T (N, 4, 4) for arm joints q (N, 6) and base offsets b (N, 3)."""
        T, j = self.base(b), 0
        for local0, axis, local1_inv in self.steps:
            T = T @ local0
            if axis is not None:
                R = torch.zeros(len(q), 4, 4, dtype=q.dtype, device=q.device)
                R[:, :3, :3] = _axis_angle_matrix(axis, q[:, j])
                R[:, 3, 3] = 1.0
                T = T @ R
                j += 1
            T = T @ local1_inv
        return T @ self.tcp

    def tcp_pose(self, q, b):
        """TCP position (N, 3) and quaternion wxyz (N, 4) in F_T."""
        T = self.tcp_matrix(q, b)
        return T[:, :3, 3], matrix_to_quat(T[:, :3, :3])


def interpolate(g_start, g_goal, s):
    """Joint-space goal interpolation g(s) = (1 - s) g_start + s g_goal, s in [0, 1] (N,) or (N, K)."""
    s = s.clamp(0.0, 1.0)
    while s.dim() < g_start.dim():
        s = s.unsqueeze(-1)
    return g_start + s * (g_goal - g_start)
