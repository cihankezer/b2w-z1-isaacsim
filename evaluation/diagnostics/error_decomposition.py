"""Steady-state error decomposition for successful fixed T1 trials (CPU, USD FK, no physics).

    error_decomposition.py <seed-dir> <out.json>
Rows: steady samples (last 2 s of the 5 s hold) of successful fixed trials in <seed-dir>/steps.csv.
Per row, link06 error to the world-fixed goal from USD FK with
  measured      : actual base pose,  actual arm joints (should equal the logged error)
  nominal_base  : nominal base pose, actual arm joints        -> removes base motion
  perfect_pd    : actual base pose,  applied (clipped) targets -> removes joint tracking error
  unclipped     : actual base pose,  IK command before clipping -> removes soft-limit clipping
  ik_only       : nominal base pose, IK command before clipping
Joint vectors in steps.csv follow runtime.JOINTS (arm = columns 16..21); the IK command is
default_joint_pos + action_command (StandingAction.process_actions).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
import numpy as np

csv.field_size_limit(1 << 30)
p, g, k, t = (usd_loader.t1_module(n) for n in ("protocol", "geometry", "kinematics", "targets"))


class r:  # runtime.JOINTS order (runtime.py:12-16); runtime.py itself imports Isaac Lab
    LEGS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
    ARM = [f"joint{i}" for i in range(1, 7)]
    JOINTS = LEGS + [f"{leg}_foot_joint" for leg in ("FR", "FL", "RR", "RL")] + ARM
seed_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
from pxr import Usd  # noqa: E402

model = k.UsdArmKinematics(Usd.Stage.Open(str(usd_loader.ASSET)), "/Unitree_B2W")
_, nominal_joints = t.nominal_state(model)
targets = {x["target_id"]: x for x in p.load_targets(seed_dir/"targets.json")["targets"]}
with (seed_dir/"trial_summary.csv").open(newline="") as f:
    success = {row["trial_id"]: row for row in csv.DictReader(f)
               if row["family"] == "fixed" and row["success"] == "True"}
lo, hi = t.soft_limits(model.limits).T
default_arm = np.array([p.NOMINAL_ARM_POSE[n] for n in r.ARM])
VARIANTS = ("measured", "nominal_base", "perfect_pd", "unclipped", "ik_only")
rows = defaultdict(lambda: defaultdict(list))   # trial -> field -> list
with (seed_dir/"steps.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        if row["trial_id"] not in success or row["steady"] != "True":
            continue
        tr = rows[row["trial_id"]]
        q = np.array(json.loads(row["joint_pos"]))
        q_act, q_app = q[16:], np.array(json.loads(row["applied_joint_targets"]))[16:]
        q_cmd = default_arm + np.array(json.loads(row["action_command"]))[16:]
        target = targets[row["target_id"]]
        goal_w = g.transform(json.loads(row["ee_goal_pos_world_xyz"]), json.loads(row["ee_goal_quat_world_wxyz"]))
        origin = np.array(json.loads(row["ee_goal_pos_world_xyz"])) - np.array(target["position_world_xyz"])
        base_act = g.transform(json.loads(row["base_pos_world_xyz"]), json.loads(row["base_quat_world_wxyz"]))
        base_nom = g.transform(origin + np.array(p.NOMINAL_BASE_POSITION_XYZ), p.NOMINAL_BASE_QUAT_WXYZ)
        legs = dict(nominal_joints, **dict(zip(r.JOINTS, q.tolist())))
        for name, base, arm in (("measured", base_act, q_act), ("nominal_base", base_nom, q_act),
                                ("perfect_pd", base_act, q_app), ("unclipped", base_act, q_cmd),
                                ("ik_only", base_nom, q_cmd)):
            ee = model.forward(base, dict(legs, **dict(zip(r.ARM, arm.tolist()))))["link06"]
            tr[name].append(g.pose_error(ee, goal_w)[:2])
        tr["logged"].append([float(row["position_error_m"]), float(row["orientation_error_geodesic_rad"])])
        tr["pd_abs"].append(np.abs(q_app - q_act))
        tr["cmd_abs"].append(np.abs(q_cmd - q_act))
        tr["clip_abs"].append(np.abs(q_cmd - q_app))
        tr["sample_abs"].append(np.abs(np.array(target["joint_space_sample_rad"]) - q_act))
        tr["base_dev"].append([float(np.linalg.norm(base_act[:3, 3] - base_nom[:3, 3])),
                               float(g.pose_error(base_act, base_nom)[1])])
        tr["sample_margin"] = float(np.min(np.minimum(np.array(target["joint_space_sample_rad"]) - lo,
                                                      hi - np.array(target["joint_space_sample_rad"]))))


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else None, max=float(v.max()), n=len(v))


per_trial = {}
for trial, tr in rows.items():
    clip = np.array(tr["clip_abs"])
    per_trial[trial] = dict(
        target_id=success[trial]["target_id"], samples=len(tr["logged"]),
        **{f"{n}_position_m": float(np.mean(np.array(tr[n])[:, 0])) for n in VARIANTS + ("logged",)},
        **{f"{n}_orientation_rad": float(np.mean(np.array(tr[n])[:, 1])) for n in VARIANTS + ("logged",)},
        pd_tracking_abs_rad=np.mean(tr["pd_abs"], axis=0).tolist(), command_minus_actual_abs_rad=np.mean(tr["cmd_abs"], axis=0).tolist(),
        sample_minus_actual_abs_rad=np.mean(tr["sample_abs"], axis=0).tolist(),
        base_position_deviation_m=float(np.mean(np.array(tr["base_dev"])[:, 0])),
        base_orientation_deviation_rad=float(np.mean(np.array(tr["base_dev"])[:, 1])),
        clipped=bool((clip > 1e-6).any()), clipped_joints=[r.ARM[j] for j in np.where((clip > 1e-6).any(axis=0))[0]],
        clipped_fraction=float((clip > 1e-6).any(axis=1).mean()), max_clip_rad=float(clip.max()),
        joint_space_sample_min_soft_limit_margin_rad=tr["sample_margin"])
trials = list(per_trial.values())
summary = dict(
    seed_dir=str(seed_dir), protocol_version=p.VERSION, successful_fixed_trials=len(trials),
    statistic="per-trial mean over the steady window, then mean ± std (ddof 1) / max over trials",
    error={f"{n}_{a}": stats([x[f"{n}_{a}"] for x in trials]) for n in VARIANTS + ("logged",)
           for a in ("position_m", "orientation_rad")},
    per_arm_joint_abs_rad={what: dict(zip(r.ARM, (stats([x[what][j] for x in trials]) for j in range(6))))
                           for what in ("pd_tracking_abs_rad", "command_minus_actual_abs_rad", "sample_minus_actual_abs_rad")},
    base_deviation=dict(position_m=stats([x["base_position_deviation_m"] for x in trials]),
                        orientation_rad=stats([x["base_orientation_deviation_rad"] for x in trials])),
    clipping=dict(targets_with_any_clip=sum(x["clipped"] for x in trials),
                  joints=dict(sorted(((j, sum(j in x["clipped_joints"] for x in trials)) for j in r.ARM))),
                  max_clip_rad=max(x["max_clip_rad"] for x in trials)),
    per_trial=per_trial)
out.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "per_trial"}, indent=1))
