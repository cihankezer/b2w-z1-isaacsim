"""CPU pose mathematics. Quaternions at every public boundary use wxyz."""

import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, quaternion):
    q = np.asarray(quaternion, dtype=float)
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError("Invalid pose")
    if abs(np.linalg.norm(q) - 1.0) > 1e-3:
        raise ValueError("Quaternion must be unit length")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = p
    return result


def pose(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat(canonical=True)
    return np.r_[matrix[:3, 3], q[[3, 0, 1, 2]]]


def pose_error(actual, desired):
    """Return Euclidean metres, geodesic radians, matrix-log Frobenius norm."""
    dp = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    angle = float(Rotation.from_matrix(desired[:3, :3].T @ actual[:3, :3]).magnitude())
    return dp, angle, np.sqrt(2.0) * angle


def quintic(start, finish, velocity, acceleration, duration, time):
    """C² translation from rest to specified endpoint derivatives (SI units)."""
    p0, p1, v1, a1 = map(np.asarray, (start, finish, velocity, acceleration))
    rhs = np.stack([p1 - p0, duration * v1, duration**2 * a1])
    a, b, c = np.linalg.solve([[1, 1, 1], [3, 4, 5], [6, 12, 20]], rhs)
    s = np.clip(time / duration, 0.0, 1.0)
    return (p0 + a*s**3 + b*s**4 + c*s**5,
            (3*a*s**2 + 4*b*s**3 + 5*c*s**4) / duration,
            (6*a*s + 12*b*s**2 + 20*c*s**3) / duration**2)


def circle(target, time):
    omega = 2 * np.pi * target["frequency_hz"]
    u, v = np.asarray(target["u"]), np.asarray(target["v"])
    r, theta = target["radius_m"], omega * time
    result = transform(target["centre_world_xyz"], target["quaternion_world_wxyz"])
    result[:3, 3] += r * (u*np.cos(theta) + v*np.sin(theta))
    velocity = r * omega * (-u*np.sin(theta) + v*np.cos(theta))
    acceleration = -r * omega**2 * (u*np.cos(theta) + v*np.sin(theta))
    return result, velocity, acceleration


def reference(target, episode_time, settled_pose, settle_s=2.0):
    """Return desired transform, phase, phase time, and the tracking-window mask.

    Samples represent interval endpoints: (start, end]. draft.7 windows at 50 Hz:
    fixed = the whole hold incl. the transient (250 samples for 5 s); circle = the
    scored window (20, 40] s (1000 samples). See steady_state() for the fixed steady window.
    """
    t = episode_time - settle_s
    if t <= 1e-9:
        return settled_pose.copy(), "settle", max(0.0, episode_time), False
    if target["family"] == "fixed":
        desired = transform(target["position_world_xyz"], target["quaternion_world_wxyz"])
        return desired, "fixed", t, 1e-9 < t <= target["hold_s"] + 1e-9
    duration = target["transition_s"]
    initial, velocity, acceleration = circle(target, 0.0)
    if t <= duration + 1e-9:
        desired = np.eye(4)
        desired[:3, 3] = quintic(settled_pose[:3, 3], initial[:3, 3], velocity,
                                acceleration, duration, t)[0]
        s = np.clip(t / duration, 0.0, 1.0)
        blend = 10*s**3 - 15*s**4 + 6*s**5
        delta = Rotation.from_matrix(settled_pose[:3, :3].T @ initial[:3, :3]).as_rotvec()
        desired[:3, :3] = settled_pose[:3, :3] @ Rotation.from_rotvec(blend * delta).as_matrix()
        return desired, "transition", t, False
    t -= duration
    scored = target["steady_state_start_s"] + 1e-9 < t <= target["duration_s"] + 1e-9
    return circle(target, t)[0], "circle", t, scored


def steady_state(target, episode_time, settle_s=2.0):
    """Fixed targets: the last steady_state_last_s of the hold (100 samples for 2 s). Circles: none."""
    if target is None or target["family"] != "fixed":
        return False
    t = episode_time - settle_s
    return target["hold_s"] - target["steady_state_last_s"] + 1e-9 < t <= target["hold_s"] + 1e-9
