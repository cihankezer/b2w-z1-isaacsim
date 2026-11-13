"""Whole-body kinematics for the T2 reachability analysis (protocol v0.4 §4.3). CPU only.

Uses the T1 USD joint tree (standing_ee.kinematics.UsdArmKinematics). Base posture = (dh, pitch)
with roll 0, yaw of the ground frame G and base x/y fixed; legs solved so that each wheel centre
(foot-joint anchor) stays at its nominal world position; arm solved by batched damped least squares
for the 6D link06 pose of a TCP target.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..standing_ee.clearance import classification, moved_shape, surface_clearance
from ..standing_ee.kinematics import MOVING, boxes_overlap, moved_bounds
from ..standing_ee.targets import nominal_state, soft_limits
from . import protocol as P

LEGS = ("FR", "FL", "RR", "RL")
ARM = tuple(f"joint{i}" for i in range(1, 7))


def _rotations(axis, angles):
    T = np.zeros((len(angles), 4, 4))
    T[:, :3, :3] = Rotation.from_rotvec(np.asarray(angles)[:, None] * axis[None]).as_matrix()
    T[:, 3, 3] = 1.0
    return T


class Chain:
    """Serial chain from base_link to body `tip`; optionally ending at the anchor (parent side) of `end_joint`."""

    def __init__(self, model, tip, end_joint=None):
        by_child = {model.bodies[e[1]]: e for e in model.edges}
        links, name = [], tip
        while name != "base_link":
            edge = by_child[name]
            links.append(edge)
            name = model.bodies[edge[0]]
        links.reverse()
        self.steps = [(e[3], e[5], np.linalg.inv(e[4])) for e in links]
        self.joint_names = [e[2] for e in links if e[5] is not None]
        self.limits = np.array([e[6] for e in links if e[5] is not None])
        self.end = None if end_joint is None else next(e[3] for e in model.edges if e[2] == end_joint)

    def fk(self, q):
        """Tip transforms in the base frame (N, 4, 4), joint origins and axes (lists of (N, 3))."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        T = np.tile(np.eye(4), (len(q), 1, 1))
        origins, axes, j = [], [], 0
        for local0, axis, local1_inv in self.steps:
            T = T @ local0
            if axis is not None:
                origins.append(T[:, :3, 3].copy())
                axes.append(T[:, :3, :3] @ axis)
                T = T @ _rotations(axis, q[:, j])
                j += 1
            T = T @ local1_inv
        if self.end is not None:
            T = T @ self.end
        return T, origins, axes


def point_jacobian(point, origins, axes):
    """Geometric Jacobian (N, 6, n) of a point on the chain: rows = linear (3), angular (3)."""
    lin = [np.cross(a, point - o) for o, a in zip(origins, axes)]
    return np.concatenate([np.stack(lin, axis=2), np.stack(axes, axis=2)], axis=1)


def arm_ik(chain, targets, q_init, limits, iterations=200, damping=0.05, max_step=0.3,
           tol_p=P.REACHABILITY["ik_tolerance"]["position_m"], tol_r=P.REACHABILITY["ik_tolerance"]["orientation_rad"]):
    """Batched DLS IK for link06 transforms `targets` (N, 4, 4) in the base frame, joint limits clamped.
    Returns q (N, 6), converged (N,), position error (N,), orientation error (N,)."""
    q = np.clip(np.array(q_init, dtype=float), limits[:, 0], limits[:, 1])
    active = np.ones(len(q), dtype=bool)
    ep = np.full(len(q), np.inf)
    er = np.full(len(q), np.inf)
    for _ in range(iterations + 1):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        T, origins, axes = chain.fk(q[idx])
        e_p = targets[idx, :3, 3] - T[:, :3, 3]
        e_r = Rotation.from_matrix(targets[idx, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
        ep[idx], er[idx] = np.linalg.norm(e_p, axis=1), np.linalg.norm(e_r, axis=1)
        done = (ep[idx] <= tol_p) & (er[idx] <= tol_r)
        active[idx[done]] = False
        idx, e_p, e_r = idx[~done], e_p[~done], e_r[~done]
        if not len(idx):
            break
        J = point_jacobian(T[~done, :3, 3], [o[~done] for o in origins], [a[~done] for a in axes])
        err = np.concatenate([e_p, e_r], axis=1)
        A = J @ J.transpose(0, 2, 1) + damping**2 * np.eye(6)
        dq = (J.transpose(0, 2, 1) @ np.linalg.solve(A, err[..., None]))[..., 0]
        scale = np.minimum(1.0, max_step / np.maximum(np.abs(dq).max(axis=1), 1e-12))
        q[idx] = np.clip(q[idx] + dq * scale[:, None], limits[:, 0], limits[:, 1])
    return q, (ep <= tol_p) & (er <= tol_r), ep, er


def leg_ik(chain, target, q0, limits, tol=P.REACHABILITY["ik_tolerance"]["position_m"]):
    """Leg joints placing the wheel centre at `target` (base frame). Returns (q, ok)."""
    def residual(q):
        return chain.fk(q)[0][0, :3, 3] - target
    lo, hi = limits[:, 0], limits[:, 1]
    solved = least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                           max_nfev=200)
    return solved.x, bool(np.linalg.norm(residual(solved.x)) <= tol)


def yaw_of(matrix):
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def ground_frame(base):
    """G: origin = base origin projected onto z = 0, axes = base yaw."""
    G = np.eye(4)
    G[:3, :3] = Rotation.from_euler("z", yaw_of(base)).as_matrix()
    G[:2, 3] = base[:2, 3]
    return G


def base_pose(G, height, pitch, roll=P.REACHABILITY["base_roll_rad"]):
    """World base transform for a posture: x/y = G origin, z = height, orientation = yaw_G * pitch * roll."""
    T = G.copy()
    T[:3, :3] = G[:3, :3] @ Rotation.from_euler("xy", [roll, pitch]).as_matrix()   # extrinsic: Ry(pitch) Rx(roll)
    T[2, 3] = height
    return T


class WholeBody:
    """Nominal state, chains and limits of the T2 robot model (soft limits as Isaac Lab clamps them)."""

    def __init__(self, model):
        self.model = model
        self.base_nominal, self.joints_nominal = nominal_state(model)
        frames = model.forward(self.base_nominal, self.joints_nominal)
        self.G = ground_frame(self.base_nominal)
        self.arm = Chain(model, "link06")
        if tuple(self.arm.joint_names) != ARM:
            raise ValueError(f"Unexpected arm chain {self.arm.joint_names}")
        self.arm_limits = soft_limits(self.arm.limits, P.REACHABILITY["joint_limit_factor"])
        self.legs, self.leg_limits, self.wheel_world = {}, {}, {}
        for leg in LEGS:
            chain = Chain(model, f"{leg}_calf", end_joint=f"{leg}_foot_joint")
            if chain.joint_names != [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf")]:
                raise ValueError(f"Unexpected leg chain {chain.joint_names}")
            self.legs[leg] = chain
            self.leg_limits[leg] = soft_limits(chain.limits, P.REACHABILITY["joint_limit_factor"])
            q = [self.joints_nominal[n] for n in chain.joint_names]
            self.wheel_world[leg] = (self.base_nominal @ chain.fk(q)[0][0])[:3, 3]
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = P.TCP_OFFSET_M
        self.nominal_tcp_world = frames[P.TCP_PARENT] @ self.tcp
        # Shoulder (joint2 origin) in the base frame at the nominal arm pose, for distance pruning.
        _, origins, _ = self.arm.fk([self.joints_nominal[n] for n in ARM])
        self.shoulder_base = origins[1][0]

    def solve_legs(self, base):
        """Leg joints for a world base transform with the wheel centres fixed. Returns (joints dict, reason)."""
        joints, inv = {}, np.linalg.inv(base)
        for leg, chain in self.legs.items():
            target = (inv @ np.r_[self.wheel_world[leg], 1.0])[:3]
            q0 = [self.joints_nominal[n] for n in chain.joint_names]
            q, ok = leg_ik(chain, target, q0, self.leg_limits[leg])
            if not ok:
                return None, f"leg_ik:{leg}"
            joints.update(zip(chain.joint_names, q.tolist()))
        return joints, None

    def posture(self, dh, pitch):
        """Feasible posture: (base transform, joints with nominal arm, None) or (base, None, reason)."""
        base = base_pose(self.G, P.H_NOM_M + dh, pitch)
        legs, reason = self.solve_legs(base)
        if reason:
            return base, None, reason
        joints = dict(self.joints_nominal, **legs)
        reason = clearance_violation(self.model, self.model.forward(base, joints))
        return base, (None if reason else joints), reason

    def tcp_to_link06(self, tcp_world):
        """link06 world transforms (N, 4, 4) for TCP world transforms (N, 4, 4)."""
        return tcp_world @ np.linalg.inv(self.tcp)

    def reach_radius(self, samples=20000, seed=0):
        """Upper bound of |TCP - shoulder| over the arm's soft joint range (Monte Carlo + 2 cm margin)."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(self.arm_limits[:, 0], self.arm_limits[:, 1], size=(samples, 6))
        T, origins, _ = self.arm.fk(q)
        tcp = (T @ self.tcp)[:, :3, 3]
        return float(np.linalg.norm(tcp - origins[1], axis=1).max() + 0.02)


def canonical_rotation(theta, z):
    """Mode A orientation in G for azimuths `theta` (N,) and target heights `z` (N,): columns = link06 axes."""
    theta = np.asarray(theta, dtype=float)
    c, s, zero = np.cos(theta), np.sin(theta), np.zeros_like(theta)
    radial = np.stack([c, s, zero], axis=1)
    tangential = np.stack([-s, c, zero], axis=1)
    up = np.tile([0.0, 0.0, 1.0], (len(theta), 1))
    a = np.broadcast_to(P.approach_angle(z), theta.shape)[:, None]
    # Tool axis pitched `a` below horizontal about the tangential axis (a = 0: radial, a = pi/2: straight down).
    x = np.cos(a) * radial - np.sin(a) * up
    zax = np.sin(a) * radial + np.cos(a) * up
    return np.stack([x, tangential, zax], axis=2)


def tcp_world(G, positions, rotations):
    """TCP world transforms (N, 4, 4) from G-frame positions and rotations."""
    T = np.tile(np.eye(4), (len(positions), 1, 1))
    T[:, :3, :3] = G[:3, :3] @ rotations
    T[:, :3, 3] = positions @ G[:3, :3].T + G[:3, 3]
    return T


ARM_BODIES = frozenset(MOVING) | {"link00"}   # Z1 arm incl. its mount; margin classes arm_arm / arm_body


def _rigid_with_base(model):
    """Bodies welded to base_link by fixed joints (constant relative geometry)."""
    group, grew = {"base_link"}, True
    while grew:
        grew = False
        for parent, child, _, _, _, axis, _ in model.edges:
            a, b = model.bodies[parent], model.bodies[child]
            if axis is None and (a in group) != (b in group):
                group |= {a, b}
                grew = True
    return group


def _body_pairs(model):
    """Non-arm body pairs PhysX can collide whose distance can change: not joint-adjacent (incl. T1 known
    mounts), not USD-filtered, not both welded to base_link."""
    cached = getattr(model, "_t2_body_pairs", None)
    if cached is not None:
        return cached
    filtered = set()
    for entry in model.physics_filtering["filtered_pairs"]:
        a = model.bodies.get(entry["path"])
        for t in entry["targets"]:
            b = model.bodies.get(t)
            if a and b:
                filtered.add(frozenset((a, b)))
    rigid = _rigid_with_base(model)
    names = sorted(n for n in model.shapes if n not in ARM_BODIES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if frozenset((a, b)) not in model.adjacent and frozenset((a, b)) not in filtered
             and not (a in rigid and b in rigid)]
    model._t2_body_pairs = pairs
    return pairs


def clearance_violation(model, frames, margins=None):
    """First clearance violation of a whole-body configuration, or None (protocol v0.4 §4.3 step 5).

    Ground and arm pairs come from the T1 report (measured with the largest margin, then judged per class);
    body-body pairs (legs, wheels, base, head, lidar, arm mount) are measured here. Unresolved geometry fails.
    """
    m = dict(P.CLEARANCE_MARGINS_M if margins is None else margins)
    for row in model.clearance_report(frames, margin=max(m.values())):
        a, b = row["pair"]
        cls = "ground" if b == "ground" else ("arm_arm" if a in ARM_BODIES and b in ARM_BODIES else "arm_body")
        d = row.get("clearance_m")
        if row["classification"] == "unresolved" or d is None:
            return f"unresolved:{cls}:{a}:{b}"
        if d < m[cls] - 1e-8:
            return f"{classification(d, m[cls])}:{cls}:{a}:{b}"
    margin = m["body_body"]
    for a, b in _body_pairs(model):
        ba, bb = moved_bounds(model.bounds[a], frames[a]), moved_bounds(model.bounds[b], frames[b])
        if not np.all(np.minimum(ba[1], bb[1]) + margin > np.maximum(ba[0], bb[0])):
            continue
        for ia, sa in enumerate(model.shapes[a]):
            for ib, sb in enumerate(model.shapes[b]):
                if not boxes_overlap(model.collider_bounds[a][ia], frames[a], model.collider_bounds[b][ib], frames[b], margin):
                    continue
                d = surface_clearance(moved_shape(sa, frames[a]), moved_shape(sb, frames[b]))["clearance_m"]
                if d < margin - 1e-8:
                    return f"{classification(d, margin)}:body_body:{a}:{b}"
    return None


def full_joints(legs_joints, arm_q):
    """Joint dict for the clearance check: posture joints with the given arm configuration."""
    return dict(legs_joints, **dict(zip(ARM, np.asarray(arm_q, dtype=float).tolist())))

