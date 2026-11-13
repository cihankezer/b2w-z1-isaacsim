"""T1 protocol constants (draft.7), sampled-target-set contract and per-target statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import circle, transform

VERSION = "1.0.0"   # protocol 1.0.0 (2026-11-12): draft.7 + oracle controller, fast shutdown, metrics_export completeness
TASK_ID = "RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0"
# Clearance policy (draft.5; draft.7 adds the approx_hull geometry for uncookable mesh colliders).
CLEARANCE_POLICY = {
    "threshold_m": 0.005,
    "self": {"mode": "monitor", "geometry": "physx_collider_surfaces",
             "method": "capsule analytic; capsule/convex certified QP (hpp-fcl GJK/EPA fallback); convex/convex hpp-fcl "
                       "GJK/EPA (signed); cylinders as circumscribed 96-gon prisms (conservative, <= r*(1/cos(pi/96)-1))",
             "pairs": "non-adjacent pairs with a moving Z1 link or gripper body; adjacency = joints + asset filters",
             "termination": "physical self-contact via collision_arm_contact (PhysX self-collision enabled)"},
    "ground": {"mode": "terminate", "geometry": "physx_collider_surfaces"},
    "mesh_colliders": {"geometry": "approx_hull",
                       "definition": "exact convex hull of the collider's USD mesh points (each mesh in its own frame)",
                       "reason": "PhysX cooked hull unreadable: request_convex_collision_representation returns "
                                 "RESULT_ERROR_COOKING_FAILED on the Xform CollisionAPI prims of this asset",
                       "rows": "clearance rows touching such a collider carry geometry='approx_hull'"},
}

# Nominal initial state (draft.5): deterministic; the seed governs target sampling only.
# Base height/orientation and leg/wheel joints: final sample (t = 30.0 s) of
# diag-standing30-20261110T000140Z-c935ca00; base x/y at the env origin.
NOMINAL_BASE_POSITION_XYZ = (0.0, 0.0, 0.6038742661476135)
NOMINAL_BASE_QUAT_WXYZ = (0.9998885989189148, -0.007302512880414724, 0.00014560901036020368, -0.013016185723245144)
NOMINAL_LEG_JOINT_POS = {
    "FL_hip_joint": 0.013796, "FR_hip_joint": 0.013154, "RL_hip_joint": -0.014676, "RR_hip_joint": -0.0151,
    "FL_thigh_joint": 0.770373, "FR_thigh_joint": 0.826215, "RL_thigh_joint": 0.778484, "RR_thigh_joint": 0.814459,
    "FL_calf_joint": -1.569885, "FR_calf_joint": -1.560587, "RL_calf_joint": -1.587931, "RR_calf_joint": -1.540878,
    "FL_foot_joint": 0.313908, "FR_foot_joint": -0.150282, "RL_foot_joint": 0.324509, "RR_foot_joint": -0.155925,
}
# Nominal arm ready pose (draft.5, approved), verified by evaluation/diagnostics/nominal_pose_check.py.
NOMINAL_ARM_POSE = {"joint1": 0.0, "joint2": 1.0, "joint3": -1.3, "joint4": 0.3, "joint5": 0.0, "joint6": 0.0}
SOFT_LIMIT_FACTOR = 0.9   # Isaac Lab soft_joint_pos_limit_factor of the robot config

# Target sampler (draft.7). Frozen here; every evaluation run draws its own target set from its seed.
SAMPLER = {
    "version": "t1-joint-space-sampler-1",
    "fixed_count": 50, "circle_count": 10,
    "arm_joint_distribution": "uniform within the 90 % soft joint limits; legs/base/gripper at the nominal state",
    "task_space_note": "uniform in joint space is not uniform in task space (accepted choice)",
    "filters": {"self_clearance_min_m": 0.005, "ground_clearance_min_m": 0.005,
                "self_geometry": "monitor geometry (collider surfaces, approx_hull for mesh colliders)",
                "ee_min_height": "link06 origin z >= base_link origin z (world)"},
    "fixed": {"hold_s": 5.0, "steady_state_last_s": 2.0, "ik": False},
    "circle": {"radius_m": 0.1, "frequency_hz": 0.05, "transition_s": 2.0, "duration_s": 40.0,
               "steady_state_start_s": 20.0, "planes": ["XY", "XZ", "YZ"], "plane_assignment": "index mod 3",
               "centre": "link06 position of a joint-space sample",
               "orientation": "constant; link06 orientation of that joint-space sample (circle start)",
               "audit": "6D IK + clearance filters at 10 Hz over transition and circle; branch jump <= 0.25 rad/0.1 s"},
    "max_attempts_per_target": 400,
}


def target_digest(target):
    definition = {k: v for k, v in target.items() if k != "sampler_audit"}
    return hashlib.sha256(json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()


def asset_hashes(path):
    """Hash the robot asset tree, including externally composed USD layers/meshes."""
    root = Path(path).resolve().parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def validate_targets(data):
    """A sampled target set: sampler frozen by the protocol, targets drawn from data['seed']."""
    if data.get("protocol_version") != VERSION or data.get("robot_variant") != "B2W":
        raise ValueError("Target set must match this B2W protocol version")
    if data.get("end_effector_frame") != "link06" or data.get("settle_s") != 2.0:
        raise ValueError("Unexpected TCP or settling period")
    if data.get("sampler") != SAMPLER or not isinstance(data.get("seed"), int):
        raise ValueError("Sampler definition or seed differs from the protocol")
    targets = data["targets"]
    if len({t["target_id"] for t in targets}) != len(targets):
        raise ValueError("Duplicate target IDs")
    counts = data.get("counts", {})
    if [sum(t["family"] == f for t in targets) for f in ("fixed", "circle")] != [counts.get("fixed"), counts.get("circle")]:
        raise ValueError("Target counts differ from the recorded counts")
    for t in targets:
        if t.get("sampler_audit", {}).get("target_sha256") != target_digest(t):
            raise ValueError(f"Target changed after sampling: {t['target_id']}")
        if t["family"] == "fixed":
            transform(t["position_world_xyz"], t["quaternion_world_wxyz"])
            if (t["hold_s"], t["steady_state_last_s"]) != (SAMPLER["fixed"]["hold_s"], SAMPLER["fixed"]["steady_state_last_s"]):
                raise ValueError("Fixed scoring window changed")
        elif t["family"] == "circle":
            axes = np.array([t["u"], t["v"]], dtype=float)
            if not np.allclose(axes @ axes.T, np.eye(2), atol=1e-8):
                raise ValueError("Circle axes must be orthonormal")
            c = SAMPLER["circle"]
            if (t["radius_m"], t["frequency_hz"], t["transition_s"], t["duration_s"], t["steady_state_start_s"]) != \
                    (c["radius_m"], c["frequency_hz"], c["transition_s"], c["duration_s"], c["steady_state_start_s"]):
                raise ValueError("Circle protocol changed")
            circle(t, 0.0)
        else:
            raise ValueError("Unknown target family")
    return data


def load_targets(path):
    return validate_targets(json.loads(Path(path).read_text()))


METRICS = ("position_m", "orientation_geodesic_rad", "orientation_paper_frobenius")


def target_metrics(trial):
    """Per target (draft.7). Fixed: tracking = mean/RMS over the whole 5 s hold incl. the transient;
    steady = mean over its last 2 s. Circle: tracking = mean/RMS over the scored window (20, 40] s.
    tracking_*_max: per-trial maximum over the tracking window (trial_summary diagnostic)."""
    result = {}
    for window in ("tracking", "steady"):
        values = np.asarray(trial.get(window) or [], dtype=float).reshape(-1, 3)
        if len(values) and not np.isfinite(values).all():
            raise ValueError("Non-finite scored error")
        for axis, name in enumerate(METRICS):
            column = values[:, axis]
            result[f"{window}_{name}_mean"] = float(column.mean()) if len(column) else None
            if window == "tracking":
                result[f"{window}_{name}_rms"] = float(np.sqrt(np.mean(column**2))) if len(column) else None
                result[f"{window}_{name}_max"] = float(column.max()) if len(column) else None
    return result


def mean_std(values):
    """Sample SD (ddof 1) over the given units (targets or seeds); None for fewer than 2 units."""
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values)}


def summarize(trials):
    """Success over all started trials; error statistics are mean ± std over successful targets."""
    result = {"protocol_version": VERSION, "statistics_convention": {
        "failure_handling": "exclude_whole_failed_trial_from_errors",
        "success": "completed_without_failure / all_started_trials",
        "unit": "target (per-target mean/RMS first, then mean ± std over targets)",
        "std": "sample_ddof_1", "primary_orientation": "geodesic", "paper_exact_reproduction": False},
        "families": {}}
    for family in ("fixed", "circle"):
        rows = [r for r in trials if r["family"] == family]
        valid = [r for r in rows if r["completed"] and not r["failure"]]
        per_target = [target_metrics(r) for r in valid]
        keys = sorted({k for m in per_target for k, v in m.items() if v is not None})
        result["families"][family] = {
            "n_trials": len(rows), "n_failures": sum(bool(r["failure"]) for r in rows),
            "n_incomplete": sum(not r["completed"] and not r["failure"] for r in rows),
            "n_successes": len(valid),
            "success_rate": len(valid) / len(rows) if rows else None,
            "metrics": {k: mean_std([m[k] for m in per_target]) for k in keys}}
    return result


def summarize_seeds(aggregates):
    """Mean ± std over seeds of each per-seed family statistic (success rate and target means)."""
    result = {}
    for family in ("fixed", "circle"):
        per_seed = [a["families"][family] for a in aggregates]
        keys = sorted({k for f in per_seed for k in f["metrics"]})
        result[family] = {"success_rate": mean_std([f["success_rate"] for f in per_seed]),
                          "metrics": {k: mean_std([f["metrics"].get(k, {}).get("mean") for f in per_seed]) for k in keys}}
    return {"protocol_version": VERSION, "n_seeds": len(aggregates), "families": result}
