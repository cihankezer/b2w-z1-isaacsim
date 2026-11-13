"""Vectorised (torch) T2 target sampling and frames (protocol v0.4 §4.1-4.3). No Isaac Lab import.

Same distribution as the numpy reference in sampler.py: band ~ weights -> uniform reachable voxel of the
band -> uniform point in the voxel, re-drawn until |azimuth| <= 60 deg -> Mode A canonical orientation.
Quaternions are wxyz. G = ground frame: origin on the ground below the base at reset, yaw of the base.
"""

import torch

from . import protocol as P


def approach_angle(z):
    """Torch version of protocol.approach_angle (piecewise linear in z, clamped at the profile ends)."""
    zs = [p[0] for p in P.APPROACH_PROFILE]
    angles = [p[1] for p in P.APPROACH_PROFILE]
    out = torch.full_like(z, angles[0])
    for (z0, a0), (z1, a1) in zip(zip(zs[:-1], angles[:-1]), zip(zs[1:], angles[1:])):
        seg = (z >= z0) & (z < z1)
        out = torch.where(seg, a0 + (z - z0) * (a1 - a0) / (z1 - z0), out)
    return torch.where(z >= zs[-1], torch.full_like(z, angles[-1]), out)


def canonical_matrix(theta, z):
    """Mode A orientation in G (N,3,3), columns = link06 axes; matches wholebody.canonical_rotation."""
    a = approach_angle(z)
    c, s, zero = torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)
    radial = torch.stack([c, s, zero], dim=1)
    tangential = torch.stack([-s, c, zero], dim=1)
    up = torch.stack([zero, zero, torch.ones_like(theta)], dim=1)
    x = torch.cos(a)[:, None] * radial - torch.sin(a)[:, None] * up
    zax = torch.sin(a)[:, None] * radial + torch.cos(a)[:, None] * up
    return torch.stack([x, tangential, zax], dim=2)


def matrix_to_quat(m):
    """Rotation matrices (N,3,3) -> unit quaternions wxyz (N,4) with w >= 0."""
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], 1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], 1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], 1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1),
    ], dim=1)                                                     # (N, 4 candidates, 4)
    diag = torch.stack([trace, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], dim=1)
    best = cands[torch.arange(len(m), device=m.device), diag.argmax(dim=1)]
    q = best / best.norm(dim=1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


def yaw_matrix(yaw):
    c, s, zero, one = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, zero], 1), torch.stack([s, c, zero], 1), torch.stack([zero, zero, one], 1)], 1)


def yaw_from_quat(q):
    """Yaw [rad] of wxyz quaternions (Z-Y-X extrinsic yaw, as isaaclab.utils.math.yaw_quat)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def g_to_world(origin_w, yaw, pos_G, rot_G):
    """Target pose G -> W: positions (N,3) and rotation matrices (N,3,3) -> (pos_w, quat_w wxyz)."""
    Rz = yaw_matrix(yaw)
    return origin_w + (Rz @ pos_G[..., None])[..., 0], matrix_to_quat(Rz @ rot_G)


class TorchReachableSet:
    """Reachable voxel centres per band on `device`, built from a sampler.ReachableSet."""

    def __init__(self, rs, device):
        self.resolution = rs.resolution
        self.device = device
        self.centres = [torch.as_tensor(rs.centres[v], dtype=torch.float32, device=device) for v in rs.by_band]

    def sample(self, n, band_weights=(1.0, 1.0, 1.0), generator=None, bands=None):
        """n targets: band (long), position_G (n,3), rot_G (n,3,3), quat_G (n,4 wxyz)."""
        dev = self.device
        if bands is None:
            w = torch.as_tensor(band_weights, dtype=torch.float32, device=dev)
            bands = torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)
        centres = torch.empty(n, 3, device=dev)
        for b, c in enumerate(self.centres):
            sel = (bands == b).nonzero(as_tuple=True)[0]
            if len(sel):
                centres[sel] = c[torch.randint(len(c), (len(sel),), device=dev, generator=generator)]
        pos = centres.clone()
        bad = torch.ones(n, dtype=torch.bool, device=dev)
        while bool(bad.any()):
            idx = bad.nonzero(as_tuple=True)[0]
            pos[idx] = centres[idx] + (torch.rand(len(idx), 3, device=dev, generator=generator) - 0.5) * self.resolution
            bad[idx] = torch.atan2(pos[idx, 1], pos[idx, 0]).abs() > P.AZIMUTH_MAX_RAD
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        rot = canonical_matrix(theta, pos[:, 2])
        return dict(band=bands, position_G=pos, rot_G=rot, quat_G=matrix_to_quat(rot))
