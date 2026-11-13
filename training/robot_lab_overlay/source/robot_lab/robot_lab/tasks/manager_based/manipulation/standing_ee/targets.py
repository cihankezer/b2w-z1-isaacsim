"""Seeded joint-space target sampler (protocol draft.7) and circle path audit.

Fixed targets: arm joints uniform within the 90 % soft limits (legs/base/gripper nominal) -> FK -> 6D
link06 pose; filters only (self/ground clearance >= 5 mm on the monitor geometry, EE not below the
base). No IK and no workspace box. Circles: centre and constant orientation from such a sample; the
whole transition + circle is IK/clearance-checked at 10 Hz. A discrete check, not a continuous-
collision proof. Uniform in joint space is not uniform in task space (accepted choice).
"""

from collections import Counter

import numpy as np

from .geometry import pose, reference, transform
from .protocol import (CLEARANCE_POLICY, NOMINAL_ARM_POSE, NOMINAL_BASE_POSITION_XYZ, NOMINAL_BASE_QUAT_WXYZ,
                       NOMINAL_LEG_JOINT_POS, SAMPLER, SOFT_LIMIT_FACTOR, VERSION, target_digest, validate_targets)

PLANES = {"XY": ([1, 0, 0], [0, 1, 0]), "XZ": ([1, 0, 0], [0, 0, 1]), "YZ": ([0, 1, 0], [0, 0, 1])}


def soft_limits(limits, factor=SOFT_LIMIT_FACTOR):
    """Isaac Lab soft limits: the hard range shrunk about its centre by `factor`."""
    limits = np.asarray(limits, dtype=float)
    centre, half = limits.mean(axis=1), (limits[:, 1] - limits[:, 0]) / 2 * factor
    return np.c_[centre - half, centre + half]


def nominal_state(model):
    """Nominal base transform (env frame) and joint positions; the closed gripper (0 rad) is
    clamped to its soft limit as Isaac Lab does at reset."""
    joints = {**NOMINAL_LEG_JOINT_POS, **NOMINAL_ARM_POSE}
    for _, _, name, _, _, axis, limits in model.edges:
        if axis is not None and name not in joints:
            joints[name] = float(np.clip(0.0, *soft_limits([limits])[0]))
    return transform(NOMINAL_BASE_POSITION_XYZ, NOMINAL_BASE_QUAT_WXYZ), joints


def check(model, base, joints, q):
    """Return (frames, None) or (frames, rejection reason)."""
    frames = model.forward(base, dict(joints, **dict(zip(model.arm_names, q))))
    if frames["link06"][2, 3] < frames["base_link"][2, 3]:
        return frames, "ee_below_base"
    return frames, model.collisions(frames)


def audit_circle(model, target, base, joints, settled):
    """10 Hz over the transition and circle (2 + 40 s); IK, clearance filters, branch/velocity cap."""
    q = np.array([joints[n] for n in model.arm_names])
    path = []
    for t in 2.0 + np.arange(0, 42.0 + 0.05, 0.1):
        desired = reference(target, t, settled)[0]
        if desired[2, 3] < base[2, 3]:
            return None, f"t={t:.2f}:ee_below_base"
        solution, reason = model.solve(desired, base, joints, initial=q)
        if reason:
            return None, f"t={t:.2f}:{reason}"
        if path and np.max(np.abs(solution - q)) > 2.5 * 0.1 + 1e-6:
            return None, f"t={t:.2f}:ik_branch_or_velocity"
        q = solution
        path.append({"episode_time_s": float(t), "arm_joint_pos_rad": q.tolist()})
    return path, None


def category(reason):
    reason = reason.split(":", 1)[1] if reason.startswith("t=") else reason
    if reason.startswith("ee_below_base") or reason.startswith("ik_"):
        return reason.split(":")[0]
    kind, scope = reason.split(":")[:2]          # e.g. penetration:self:link02:link06
    return f"{scope}_{'unresolved' if kind == 'unresolved' else 'clearance'}"


def sample(model, seed, hashes, fixed_count=SAMPLER["fixed_count"], circle_count=SAMPLER["circle_count"]):
    """Draw one target set from `seed`. Raises if a target cannot be found within the attempt cap."""
    rng = np.random.default_rng(seed)
    base, joints = nominal_state(model)
    settled = model.forward(base, joints)["link06"]
    lo, hi = soft_limits(model.limits).T
    rejections = {"fixed": Counter(), "circle": Counter()}
    pairs = Counter()
    attempts = {"fixed": 0, "circle": 0}
    targets = []

    def draw(family):
        attempts[family] += 1
        if attempts[family] > SAMPLER["max_attempts_per_target"] * (fixed_count if family == "fixed" else circle_count):
            raise ValueError(f"{family}: attempt cap reached; revise the sampler explicitly")
        q = rng.uniform(lo, hi)
        frames, reason = check(model, base, joints, q)
        return q, frames, reason

    def reject(family, reason):
        rejections[family][category(reason)] += 1
        pairs[reason.split(":", 1)[1] if reason.startswith("t=") else reason] += 1

    def finish(target, n, **evidence):
        target["sampler_audit"] = dict(attempt_index=n, target_sha256=target_digest(target), **evidence)
        targets.append(target)

    for i in range(fixed_count):
        while True:
            q, frames, reason = draw("fixed")
            if reason:
                reject("fixed", reason)
                continue
            goal = pose(frames["link06"])
            finish(dict(target_id=f"fixed-{i:02d}", family="fixed", position_world_xyz=goal[:3].tolist(),
                        quaternion_world_wxyz=goal[3:].tolist(), joint_space_sample_rad=q.tolist(),
                        hold_s=SAMPLER["fixed"]["hold_s"], steady_state_last_s=SAMPLER["fixed"]["steady_state_last_s"]),
                   attempts["fixed"])
            break
    c = SAMPLER["circle"]
    for i in range(circle_count):
        plane = c["planes"][i % 3]
        u, v = PLANES[plane]
        while True:
            q, frames, reason = draw("circle")
            if reason:
                reject("circle", reason)
                continue
            goal = pose(frames["link06"])
            target = dict(target_id=f"circle-{i:02d}-{plane}", family="circle", plane=plane, u=u, v=v,
                          centre_world_xyz=goal[:3].tolist(), quaternion_world_wxyz=goal[3:].tolist(),
                          joint_space_sample_rad=q.tolist(), radius_m=c["radius_m"], frequency_hz=c["frequency_hz"],
                          transition_s=c["transition_s"], duration_s=c["duration_s"],
                          steady_state_start_s=c["steady_state_start_s"])
            path, reason = audit_circle(model, target, base, joints, settled)
            if reason:
                reject("circle", reason)
                continue
            finish(target, attempts["circle"], ik_path=path)
            break
    approx = sorted({s["path"]: s.get("cooking_issue") for shapes in model.shapes.values() for s in shapes
                     if s.get("source") == "approx_hull"}.items())
    return validate_targets(dict(
        benchmark="T1-STANDING-6D-EE", protocol_version=VERSION, robot_variant="B2W", end_effector_frame="link06",
        seed=int(seed), settle_s=2.0, sampler=SAMPLER, counts={"fixed": fixed_count, "circle": circle_count},
        protocol_counts=(fixed_count, circle_count) == (SAMPLER["fixed_count"], SAMPLER["circle_count"]),
        asset_hashes=hashes, nominal_base_pose_env=pose(base).tolist(), nominal_joint_pos=joints,
        nominal_ee_pose_env=pose(settled).tolist(), soft_limits_rad=dict(zip(model.arm_names, np.c_[lo, hi].tolist())),
        clearance_policy=CLEARANCE_POLICY, approx_hull_colliders=[dict(path=p, cooking_issue=i) for p, i in approx],
        attempts=attempts, rejections={f: dict(sorted(r.items())) for f, r in rejections.items()},
        rejection_details=dict(pairs.most_common()), targets=targets))
