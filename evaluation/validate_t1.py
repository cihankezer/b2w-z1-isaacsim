"""Run the eight T1 completion checks inside an approved Isaac Lab container.

No Docker lifecycle, setup, asset repair or training. --cpu-only runs only the
synthetic metric check and deliberately reports the runtime checks NOT_RUN.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee"
CHECKS = ("robot_physics", "coordinate_frames", "error_metrics", "fixed_pose_generator",
          "circle_generator", "standing_termination", "reproducibility", "metrics_export")
TOLERANCES = dict(position_m=1e-4, orientation_rad=1e-4, reference=1e-9, export_error=1e-8,
                  fk_position_m=1e-3, fk_orientation_rad=1e-2)
RUNTIME_IDENTITY = ("robot_variant", "end_effector_frame", "seed", "physics_dt_s", "control_dt_s",
                    "actuator_types", "loaded_joint_parameters", "asset_hashes", "image_id", "gpu_uuid", "git", "versions")
KEYS = ("stage", "repetition", "env_id", "target_id", "family", "episode_time_s", "phase", "phase_time_s", "scored", "steady", "validation_case")
POSE_FIELDS = ("ee_pos_world_xyz", "ee_quat_world_wxyz", "ee_goal_pos_world_xyz", "ee_goal_quat_world_wxyz")
METRIC_FIELDS = ("position_error_m", "orientation_error_geodesic_rad", "orientation_error_paper_frobenius",
                 "base_pos_world_xyz", "base_quat_world_wxyz", "base_lin_vel_world_xyz", "base_ang_vel_body_xyz",
                 "base_height_m", "tilt_rad", "base_roll_pitch_rad", "base_xy_displacement_m", "foot_contact_flags",
                 "foot_contact_forces", "foot_positions_world_xyz", "foot_drift_m", "wheel_rotation_rad",
                 "wheel_travel_rad", "standing_violation", "terminated", "truncated", "failure", "failure_reason")


def pure_modules():
    package = types.ModuleType("_t1_validation")
    package.__path__ = [str(TASK)]
    sys.modules[package.__name__] = package
    return tuple(importlib.import_module(package.__name__ + "." + n) for n in ("geometry", "protocol"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_steps(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            elif value == "":
                row[key] = None
            else:
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    return rows


def independent_errors(actual_position, actual_quaternion, target_position, target_quaternion):
    """Quaternion-product angle, independent of the task's SciPy rotation metric."""
    a, b = np.asarray(actual_quaternion, dtype=float), np.asarray(target_quaternion, dtype=float)
    for q in (a, b):
        require(q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1) < 1e-3,
                "Invalid exported quaternion")
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    vector = b[0]*a[1:] - a[0]*b[1:] - np.cross(b[1:], a[1:])
    angle = 2*math.atan2(float(np.linalg.norm(vector)), abs(float(np.dot(a, b))))
    delta = np.asarray(actual_position, dtype=float) - np.asarray(target_position, dtype=float)
    require(delta.shape == (3,) and np.isfinite(delta).all(), "Invalid exported position")
    return np.array([np.linalg.norm(delta), angle, math.sqrt(2)*angle])


def check_error_metrics(geometry):
    actual = geometry.transform([.03, .04, 0], [math.sqrt(.5), 0, 0, math.sqrt(.5)])
    errors = geometry.pose_error(actual, np.eye(4))
    expected = [.05, math.pi/2, math.pi/math.sqrt(2)]
    np.testing.assert_allclose(errors, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(independent_errors([.03,.04,0], [math.sqrt(.5),0,0,math.sqrt(.5)],
                                                 [0,0,0], [1,0,0,0]), expected, atol=1e-12, rtol=0)
    return dict(position_error_m=errors[0], geodesic_rad=errors[1], paper_frobenius=errors[2])


def check_hold(directory):
    meta = read_json(directory/"interface.json")
    expected = ([f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                 for joint in ("hip", "thigh", "calf")] +
                [f"{leg}_foot_joint" for leg in ("FR", "FL", "RR", "RL")] + [f"joint{i}" for i in range(1,7)])
    require(meta["robot_variant"] == "B2W" and meta["end_effector_frame"] == "tcp", "Robot/TCP mismatch")
    require(meta["num_envs"] == 1 and meta["seed"] == 1101, "Hold must use one environment and seed 1101")
    # draft.4 gripper asset: the 22 action joints plus at most jointGripper (as StandingAction, runtime.py:26-28)
    extra = sorted(set(meta["joint_order"]) - set(expected))
    require(meta["action_order"] == expected and set(expected) <= set(meta["joint_order"])
            and extra in ([], ["jointGripper"]), "22-joint mapping mismatch")
    require(meta["actuator_types"], "Missing actuator configuration")
    params = meta["loaded_joint_parameters"]
    indices = [meta["joint_order"].index(n) for n in expected]
    for name in ("joint_stiffness", "joint_damping", "joint_vel_limits", "joint_effort_limits"):
        values = np.asarray(params[name], dtype=float)[indices]
        require(np.isfinite(values).all(), f"Invalid loaded {name}")
        if name == "joint_stiffness":
            require(np.all(values[np.r_[0:12,16:22]] > 0) and np.all(values[12:16] == 0), "Position/wheel gains mismatch")
        else:
            require(np.all(values > 0), f"Nonpositive loaded {name}")
    require(meta["physics_dt_s"] == .005 and meta["control_dt_s"] == .02 and meta["hold_duration_s"] == 30,
            "Hold must use 200 Hz physics / 50 Hz control for 30 s")
    rows = read_steps(directory/"steps.csv")
    require(len(rows) == 1500, "30 s hold requires 1500 controller samples")
    for tick, row in enumerate(rows, 1):
        require(abs(row["episode_time_s"] - tick*.02) < 1e-9, "Hold clock/reset discontinuity")
        require(not row["failure"] and not row["terminated"] and not row["truncated"], "Nominal hold failed/reset")
        for field in ("joint_pos", "joint_vel", "action_command", "applied_joint_targets", *POSE_FIELDS, *METRIC_FIELDS[:-1]):
            require(np.isfinite(np.asarray(row[field], dtype=float)).all(), f"NaN/Inf in {field}")
        require(np.all(np.asarray(row["applied_joint_targets"])[12:16] == 0), "Nonzero wheel target")
        if row["episode_time_s"] > 2:
            require(not row["standing_violation"], "Standing compliance failed after settling")
    return dict(duration_s=30, samples=len(rows), joint_order=expected, tcp="tcp", actuator_types=meta["actuator_types"])


def tcp_ref(geometry, sampled, target, t, settled):
    """T1 2.0.0 reference: rigid map of the link06 reference by the TCP offset stated in the target file."""
    o = np.asarray(sampled["tcp_offset_m"], dtype=float)
    return geometry.tcp_reference(target, t, settled, geometry.transform(o, [1, 0, 0, 0]),
                                  geometry.transform(-o, [1, 0, 0, 0]))


def check_references(sampled, family, geometry, protocol):
    """Reference geometry of one sampled target set (draft.7 sampler); kinematic audit evidence present."""
    protocol.validate_targets(sampled)
    settled = geometry.transform(sampled["nominal_ee_pose_env"][:3], sampled["nominal_ee_pose_env"][3:])
    targets = [t for t in sampled["targets"] if t["family"] == family]
    require(len(targets) == sampled["counts"][family], "Sampled count mismatch")
    for target in targets:
        q = np.asarray(target["joint_space_sample_rad"])
        require(q.shape == (6,) and np.isfinite(q).all(), "Missing/invalid joint-space sample")
        if family == "fixed":
            expected = geometry.transform(target["position_world_xyz"], target["quaternion_world_wxyz"])
            for t in np.arange(1,round(target["hold_s"]/.02)+1)*.02 + 2:
                np.testing.assert_allclose(tcp_ref(geometry,sampled,target,t,settled)[0], expected, atol=1e-12, rtol=0)
        else:
            path = target["sampler_audit"]["ik_path"]
            require(len(path) == 421 and all(np.asarray(p["arm_joint_pos_rad"]).shape == (6,) and
                                             np.isfinite(p["arm_joint_pos_rad"]).all() for p in path), "Missing/invalid audited IK path")
            u, v = np.array(target["u"]), np.array(target["v"])
            normal = np.cross(u,v)
            plane_normal = {"XY":[0,0,1], "XZ":[0,1,0], "YZ":[1,0,0]}[target["plane"]]
            require(abs(abs(normal@plane_normal)-1) < 1e-10,"Circle axes disagree with named world plane")
            orientation = geometry.transform([0,0,0],target["quaternion_world_wxyz"])[:3,:3]
            for t in np.linspace(0,40,401):
                pose = geometry.circle(target,t)[0]
                delta = pose[:3,3] - target["centre_world_xyz"]
                require(abs(np.linalg.norm(delta)-.1) < 1e-10 and abs(delta@normal) < 1e-10, "Circle radius/plane mismatch")
                np.testing.assert_allclose(pose[:3,:3],orientation,atol=1e-12,rtol=0)
            np.testing.assert_allclose(geometry.circle(target,0)[0], geometry.circle(target,20)[0], atol=1e-12, rtol=0)
            # Independent finite differences check C2 translation and smooth orientation at entry.
            h = 1e-3
            poses = [tcp_ref(geometry,sampled,target,4+j*h,settled)[0] for j in range(-3,4)]
            velocity_left = (3*poses[3][:3,3]-4*poses[2][:3,3]+poses[1][:3,3])/(2*h)
            velocity_right = (-3*poses[3][:3,3]+4*poses[4][:3,3]-poses[5][:3,3])/(2*h)
            np.testing.assert_allclose(velocity_left, velocity_right, atol=1e-5, rtol=0)
            acceleration_left = (2*poses[3][:3,3]-5*poses[2][:3,3]+4*poses[1][:3,3]-poses[0][:3,3])/h**2
            acceleration_right = (2*poses[3][:3,3]-5*poses[4][:3,3]+4*poses[5][:3,3]-poses[6][:3,3])/h**2
            np.testing.assert_allclose(acceleration_left, acceleration_right, atol=1e-4, rtol=0)
            np.testing.assert_allclose(poses[2][:3,:3], poses[4][:3,:3], atol=1e-7, rtol=0)
            start = [tcp_ref(geometry,sampled,target,2+j*h,settled)[0] for j in range(4)]
            np.testing.assert_allclose(start[0],settled,atol=1e-12,rtol=0)
            np.testing.assert_allclose((-3*start[0][:3,3]+4*start[1][:3,3]-start[2][:3,3])/(2*h),0,atol=1e-5,rtol=0)
            np.testing.assert_allclose((2*start[0][:3,3]-5*start[1][:3,3]+4*start[2][:3,3]-start[3][:3,3])/h**2,
                                       0,atol=1e-4,rtol=0)
            np.testing.assert_allclose(start[1][:3,:3],settled[:3,:3],atol=1e-7,rtol=0)
    return dict(target_count=len(targets), seed=sampled["seed"], reference_check="PASS",
                rejections=sampled["rejections"][family], attempts=sampled["attempts"][family])


def compare_repetitions(one, two):
    require(one and len(one) == len(two), "Empty or unequal repetition lengths")
    maxima = dict(position_m=0., orientation_rad=0.)
    for a,b in zip(one,two):
        for key in ("env_id", "target_id", "family", "episode_time_s", "phase", "scored", "terminated", "truncated", "failure", "failure_reason"):
            require(a[key] == b[key], f"Repetition structure differs: {key}")
        for field in POSE_FIELDS[2:]:
            np.testing.assert_allclose(a[field],b[field],atol=TOLERANCES["reference"],rtol=0,err_msg="Reference changed")
        for field, units in (("position_error_m","position_m"), ("base_height_m","position_m"),
                             ("foot_drift_m","position_m"), ("base_xy_displacement_m","position_m"),
                             ("orientation_error_geodesic_rad","orientation_rad"), ("tilt_rad","orientation_rad"),
                             ("wheel_rotation_rad","orientation_rad")):
            difference = float(np.max(np.abs(np.asarray(a[field])-np.asarray(b[field]))))
            require(math.isfinite(difference) and difference <= TOLERANCES[units], f"Repeatability failed: {field} ({difference})")
            maxima[units] = max(maxima[units], difference)
        require(a["foot_contact_flags"] == b["foot_contact_flags"] and a["standing_violation"] == b["standing_violation"],
                "Standing/contact flags differ between repetitions")
    return dict(samples_per_repetition=len(one), max_differences=maxima, tolerances=TOLERANCES)


def match_configuration(one, two):
    for key in RUNTIME_IDENTITY:
        require(one[key] == two[key],f"Runtime configuration changed between stages: {key}")


def check_metrics(rows, summaries, sampled, geometry, origins):
    require(rows and summaries and len(summaries) == len(sampled["targets"]), "Missing trial samples/summaries")
    require(len({s["target_id"] for s in summaries}) == len(summaries), "Duplicate trial summaries")
    require({s["target_id"] for s in summaries} == {t["target_id"] for t in sampled["targets"]}, "Trial IDs mismatch")
    for row in rows:
        require(all(k in row and row[k] is not None for k in (*KEYS[2:-1], *POSE_FIELDS, *METRIC_FIELDS)), "Missing required per-step field")
        for field in METRIC_FIELDS[:-1]:
            require(np.isfinite(np.asarray(row[field],dtype=float)).all(),f"Nonfinite stability/error field: {field}")
        actual = independent_errors(row[POSE_FIELDS[0]], row[POSE_FIELDS[1]], row[POSE_FIELDS[2]], row[POSE_FIELDS[3]])
        logged = [row[n] for n in METRIC_FIELDS[:3]]
        np.testing.assert_allclose(logged,actual,atol=TOLERANCES["export_error"],rtol=0,err_msg="Exported pose error mismatch")
    outcomes = dict(completed=0, terminated=0)
    for target in sampled["targets"]:
        # protocol 1.0.0/2.0.0: every trial (completed or terminated) must be complete and self-consistent.
        trial = [r for r in rows if r["target_id"] == target["target_id"]]
        summary = next(s for s in summaries if s["target_id"] == target["target_id"])
        require(trial, "Trial has no samples")
        duration = 2+target["hold_s"] if target["family"] == "fixed" else 44
        full, last = round(duration/.02), trial[-1]
        require(not any(r["terminated"] or r["truncated"] or r["failure"] for r in trial[:-1]), "Trial continued after its end")
        if summary["success"]:
            require(len(trial) == full, "Incomplete completed rollout")
            require(last["truncated"] and not last["terminated"] and not last["failure"], "Completed trial flags incorrect")
            require(summary["completed"] and not summary["failure"] and not summary["failure_reason"], "Incorrect trial summary flags")
            outcomes["completed"] += 1
        else:
            require(len(trial) <= full, "Terminated trial longer than its episode")
            require(last["terminated"] and not last["truncated"] and last["failure"] and last["failure_reason"],
                    "Terminated trial lacks terminal flags/reason")
            require(not summary["completed"] and summary["failure"] and summary["failure_reason"] == last["failure_reason"],
                    "Terminated trial summary disagrees with its terminal sample")
            outcomes["terminated"] += 1
        scored, steady = [], []
        origin = np.asarray(origins[trial[0]["env_id"]])
        shifted = dict(target)
        position_key = "position_world_xyz" if target["family"] == "fixed" else "centre_world_xyz"
        shifted[position_key] = (np.asarray(target[position_key])+origin).tolist()
        settled_row = next((r for r in trial if abs(r["episode_time_s"]-2) < 1e-9), None)
        require(settled_row is not None or trial[-1]["episode_time_s"] < 2, "Missing settled sample")
        settled = geometry.transform(settled_row[POSE_FIELDS[0]],settled_row[POSE_FIELDS[1]]) if settled_row else None
        for tick,row in enumerate(trial,1):
            require(abs(row["episode_time_s"]-tick*.02) < 1e-9, "Trial clock/reset discontinuity")
            require(row["scored"] == geometry.reference(target,tick*.02,np.eye(4))[3], "Incorrect tracking mask")
            require(row["steady"] == geometry.steady_state(target,tick*.02), "Incorrect steady-state mask")
            if row["episode_time_s"] > 2:
                expected = geometry.pose(tcp_ref(geometry,sampled,shifted,tick*.02,settled)[0])
                reference_error = independent_errors(row[POSE_FIELDS[2]],row[POSE_FIELDS[3]],expected[:3],expected[3:])
                require(np.max(reference_error) < TOLERANCES["export_error"],"Logged reference differs from sampled target")
            if row["scored"]:
                scored.append([row[n] for n in METRIC_FIELDS[:3]])
            if row["steady"]:
                steady.append([row[n] for n in METRIC_FIELDS[:3]])
        require((summary["tracking_sample_count"],summary["steady_sample_count"]) == (len(scored),len(steady)),
                "Summary sample count disagrees with the per-step mask")
        names = ("position_m", "orientation_geodesic_rad", "orientation_paper_frobenius")
        if not summary["success"]:
            require(all(summary.get(f"{w}_{n}_{k}") is None for w in ("tracking","steady") for n in names
                        for k in ("mean","rms","max")), "Terminated trial reports aggregate errors")
            continue
        expected_counts = (250, 100) if target["family"] == "fixed" else (1000, 0)
        require((len(scored),len(steady)) == expected_counts, "Incorrect tracking/steady sample count")
        for axis,name in enumerate(names):
            values = np.asarray(scored)[:,axis]
            expected = {f"tracking_{name}_mean":values.mean(),f"tracking_{name}_rms":np.sqrt(np.mean(values**2)),
                        f"tracking_{name}_max":values.max()}
            if steady:
                expected[f"steady_{name}_mean"] = np.asarray(steady)[:,axis].mean()
            for key,value in expected.items():
                require(abs(summary[key]-value) < TOLERANCES["export_error"], f"Incorrect summary: {key}")
    return dict(trials=len(summaries), samples=len(rows), outcomes=outcomes, error_recomputation="independent quaternion product",
                windows="fixed: whole hold + last 2 s; circle: (20, 40] s")


def export_rows(out, sources):
    """Preserve partial traces; exports do not by themselves imply validation."""
    with (out/"trajectories.csv").open("w",newline="") as trajectory, (out/"metrics.csv").open("w",newline="") as metrics, (out/"trial_summary.csv").open("w",newline="") as summary:
        tw = csv.DictWriter(trajectory,fieldnames=(*KEYS,*POSE_FIELDS));tw.writeheader()
        mw = csv.DictWriter(metrics,fieldnames=(*KEYS,*METRIC_FIELDS,"collision_geometry","self_clearance","self_clearance_episode"));mw.writeheader()
        sw = None
        for stage,repetition,directory in sources:
            if (directory/"steps.csv").exists():
                for row in read_steps(directory/"steps.csv"):
                    row.update(stage=stage,repetition=repetition)
                    for writer,fields in ((tw,(*KEYS,*POSE_FIELDS)),(mw,(*KEYS,*METRIC_FIELDS,"collision_geometry","self_clearance","self_clearance_episode"))):
                        writer.writerow({k:json.dumps(row.get(k),allow_nan=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in fields})
            if (directory/"trial_summary.csv").exists():
                for row in read_steps(directory/"trial_summary.csv"):
                    row.update(stage=stage,repetition=repetition)
                    if sw is None:
                        sw=csv.DictWriter(summary,fieldnames=list(row));sw.writeheader()
                    sw.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        if sw is None:
            csv.writer(summary).writerow(("stage","repetition","target_id","family","success","failure","valid_sample_count"))


def run_child(out, name, script, arguments):
    directory = out/"runs"/name
    directory.mkdir(parents=True)
    command = [sys.executable,str(script),"--out",str(directory/"artifacts"),*arguments]
    write_json(directory/"command.json",command)
    start = time.monotonic()
    print(f"T1 validation: {name}",flush=True)
    try:
        with (directory/"stdout.log").open("w") as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        process = dict(exit_code=result.returncode,timed_out=False)
    except (subprocess.TimeoutExpired,OSError) as exc:
        process = dict(exit_code=None,timed_out=isinstance(exc,subprocess.TimeoutExpired),error=str(exc))
    process["wall_time_s"] = time.monotonic()-start
    write_json(directory/"process.json",process)
    return directory/"artifacts",process


def completed(directory, process):
    if (directory/"error.json").exists():
        raise ValueError(f"Child recorded an error: {read_json(directory/'error.json').get('message')}")
    # draft.7 decision C: judged by completion.json (written before teardown); the known native
    # teardown abort (non-zero exit after completion) is recorded in process.json, not overridden.
    require(not process["timed_out"], "Child process timed out; inspect stdout.log/process.json")
    require((directory/"completion.json").exists() and read_json(directory/"completion.json")["status"] == "finished",
            f"Child did not finish (exit code {process['exit_code']}); inspect stdout.log/process.json")


def new_report(cpu_only):
    return dict(schema_version=1,mode="cpu_only" if cpu_only else "full",all_tests_passed=False,
                status="INCOMPLETE",progress=dict(passed=0,total=8),
                checks={n:dict(status="NOT_RUN",reason="Not executed") for n in CHECKS},errors=[])


def save_report(out, report):
    report["progress"]["passed"] = sum(v["status"] == "PASS" for v in report["checks"].values())
    report["all_tests_passed"] = report["mode"] == "full" and report["progress"]["passed"] == 8 and not report["errors"]
    report["status"] = "PASS" if report["all_tests_passed"] else "FAIL" if report["errors"] or any(v["status"] == "FAIL" for v in report["checks"].values()) else "INCOMPLETE"
    report["artifacts"] = {name:dict(exists=(out/name).is_file(),sha256=sha256(out/name) if (out/name).is_file() else None)
                           for name in ("targets.json","trajectories.csv","metrics.csv","config.yaml","trial_summary.csv")}
    report["runs"] = {p.parent.name:dict(**read_json(p),stdout_log=str((p.parent/"stdout.log").relative_to(out)),
                                       artifacts=str((p.parent/"artifacts").relative_to(out)))
                      for p in sorted((out/"runs").glob("*/process.json"))}
    # Required files cannot be bypassed by manually setting eight statuses.
    if report["all_tests_passed"] and not all(a["exists"] for a in report["artifacts"].values()):
        report.update(all_tests_passed=False,status="FAIL")
        report["errors"].append("Required artifacts missing")
    write_json(out/"validation_report.json",report)


def record_check(report, name, function):
    try:
        report["checks"][name] = dict(status="PASS",evidence=function())
    except Exception as exc:
        report["checks"][name] = dict(status="FAIL",reason=f"{type(exc).__name__}: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True,help="New persistent validation directory")
    parser.add_argument("--cpu-only",action="store_true",help="Synthetic metrics only; never an overall PASS")
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    report = new_report(args.cpu_only)
    config = dict(protocol_version="2.1.0",robot="B2W+Z1",tcp="tcp (link06 + 0.171 m x)",seed=1101,device=args.device,
                  nominal_hold_s=30,physics_dt_s=.005,control_dt_s=.02,controller="oracle",tolerances=TOLERANCES,
                  tolerances_status="PROPOSED; fixed before the run",mode=report["mode"],
                  image_id=os.environ.get("T1_IMAGE_ID"),gpu_uuid=os.environ.get("ISAAC_GPU_UUID"),
                  argv=sys.argv if argv is None else argv,source_hashes={})
    sources = []
    try:
        source = args.out/"source";source.mkdir()
        for path in [Path(__file__),ROOT/"evaluation/eval_t1.py",ROOT/"evaluation/t1_runtime_checks.py",ROOT/"docs/protocols/TASK_T1.md",*sorted(TASK.glob("*.py"))]:
            shutil.copy2(path,source/path.name)
            config["source_hashes"][str(path.relative_to(ROOT))] = sha256(path)
        (args.out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
        export_rows(args.out,sources);save_report(args.out,report)
        geometry,protocol = pure_modules()
        record_check(report,"error_metrics",lambda:check_error_metrics(geometry));save_report(args.out,report)
        if args.cpu_only:
            return 1
        require(report["checks"]["error_metrics"]["status"] == "PASS", "Synthetic metric prerequisite failed")
        require(config["image_id"] and config["gpu_uuid"] and config["gpu_uuid"] != "all",
                "Set inspected T1_IMAGE_ID and allocated ISAAC_GPU_UUID; run inside the approved Isaac container")
        evaluator = ROOT/"evaluation/eval_t1.py"
        hold,hp = run_child(args.out,"hold",evaluator,["--mode","smoke","--controller","hold","--hold-seconds","30","--device",args.device])
        sources.append(("hold",0,hold))
        def robot_check():
            completed(hold,hp)
            return check_hold(hold)
        record_check(report,"robot_physics",robot_check)
        if (hold/"interface.json").exists():
            config["runtime"] = read_json(hold/"interface.json")
            config["resolved_env_config"] = str((hold/"run_config.yaml").relative_to(args.out))
        (args.out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
        save_report(args.out,report)
        # draft.7 order: fault checks first, then FK/frames, each in its own process.
        for name in ("standing_termination","coordinate_frames"):
            worker,wp = run_child(args.out,f"runtime_{name}",ROOT/"evaluation/t1_runtime_checks.py",
                                  ["--checks",name,"--device",args.device])
            sources.append((f"runtime_{name}",0,worker))
            def worker_check(name=name,worker=worker,wp=wp):
                completed(worker,wp)
                evidence = read_json(worker/"checks.json")[name]
                require(evidence["status"] == "PASS",evidence.get("reason","Runtime check failed"))
                return evidence
            record_check(report,name,worker_check)
        save_report(args.out,report)
        if all(report["checks"][n]["status"] == "PASS" for n in ("robot_physics","coordinate_frames","standing_termination")):
            seed = 1101
            repetitions = []
            for repetition in (1,2):
                directory,process = run_child(args.out,f"evaluate_{repetition}",evaluator,["--mode","evaluate","--controller","oracle",
                                                                                         "--seeds",str(seed),"--device",args.device])
                sources.append(("evaluate",repetition,directory/f"seed-{seed}"));repetitions.append((directory,process))
            first = repetitions[0][0]
            sampled = None
            try:
                completed(*repetitions[0])
                sampled = protocol.load_targets(first/f"seed-{seed}"/"targets.json")
                shutil.copy2(first/f"seed-{seed}"/"targets.json",args.out/"targets.json")
            except Exception as exc:
                for name in ("fixed_pose_generator","circle_generator"):
                    report["checks"][name] = dict(status="FAIL",reason=f"Target sampling failed: {exc}")
            if sampled:
                for family,name in (("fixed","fixed_pose_generator"),("circle","circle_generator")):
                    record_check(report,name,lambda family=family:check_references(sampled,family,geometry,protocol))
                def reproduction_check():
                    for directory,process in repetitions:
                        completed(directory,process)
                        require(sha256(directory/f"seed-{seed}"/"targets.json") == sha256(args.out/"targets.json"),
                                "Seeded target set changed between repetitions")
                    configs = [read_json(d/"interface.json") for d,_ in repetitions]
                    for meta in configs:
                        match_configuration(config["runtime"],meta)
                    for key in ("num_envs","controller","environment_origins_world_xyz"):
                        require(configs[0][key] == configs[1][key],f"Repetition config changed: {key}")
                    return dict(target_set="byte-identical",
                                traces=compare_repetitions(*(read_steps(d/f"seed-{seed}"/"steps.csv") for d,_ in repetitions)))
                record_check(report,"reproducibility",reproduction_check)
                def metrics_check():
                    evidence = []
                    for directory,process in repetitions:
                        completed(directory,process)
                        origins = read_json(directory/"interface.json")["environment_origins_world_xyz"]
                        evidence.append(check_metrics(read_steps(directory/f"seed-{seed}"/"steps.csv"),
                                                      read_steps(directory/f"seed-{seed}"/"trial_summary.csv"),sampled,geometry,origins))
                    return evidence
                record_check(report,"metrics_export",metrics_check)
        for name,value in report["checks"].items():
            if value["status"] == "NOT_RUN":
                report["checks"][name] = dict(status="BLOCKED",reason="A prerequisite did not pass; inspect preceding checks")
        for path,digest in config["source_hashes"].items():
            require(sha256(ROOT/path) == digest,"Source changed during validation")
    except (Exception,KeyboardInterrupt) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            export_rows(args.out,sources)
        except Exception as exc:
            report["errors"].append(f"Export failed: {exc}")
        if not (args.out/"config.yaml").exists():
            (args.out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
        save_report(args.out,report)
        print(f"T1 validation: {report['status']} ({report['progress']['passed']}/8); {args.out/'validation_report.json'}",flush=True)
    return 0 if report["all_tests_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
