"""Run a scripted T1 smoke hold, or evaluate seeded target sets (protocol draft.7).

Evaluate: for each seed, draw a target set with the frozen sampler (targets.sample), run it in
batches of --num-envs environments (fixed targets first, then circles), and export per-seed steps,
trial summaries and aggregates, plus mean ± std over seeds. The initial state is deterministic;
the seed governs target sampling only.

Execute with Isaac Lab's Python, after approval for the simulator/GPU workload.
No setup, Docker lifecycle, checkpoint loading or policy training occurs here.
"""

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.standing_ee"


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def git_info(path):
    def read(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={path.resolve()}", "-C", str(path), *args],
                                text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": read("rev-parse", "HEAD"), "status": read("status", "--porcelain"),
            "version": read("describe", "--tags", "--always", "--dirty")}


class CsvLog:
    def __init__(self, path):
        self.stream = path.open("x", newline="")
        self.writer = None

    def append(self, row):
        row = json_safe(row)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()})

    def close(self):
        self.stream.close()


def export_trials(out, trials, protocol):
    write_json(out/"aggregate_summary.json", protocol.summarize(trials))
    summary = CsvLog(out/"trial_summary.csv")
    try:
        for trial in trials:
            row = {k: v for k, v in trial.items() if k not in ("tracking", "steady", "contact_counts")}
            row["success"] = trial["completed"] and not trial["failure"]
            row["tracking_sample_count"] = len(trial["tracking"])
            row["steady_sample_count"] = len(trial["steady"])
            row["foot_contact_fractions"] = ([n/trial["diagnostic_samples"] for n in trial["contact_counts"]]
                                             if trial["diagnostic_samples"] else None)
            metrics = protocol.target_metrics(trial)
            row.update({k: (v if row["success"] else None) for k, v in metrics.items()})
            summary.append(row)
    finally:
        summary.close()


EXPECTED_SAMPLES = {"fixed": (250, 100), "circle": (1000, 0)}   # (tracking, steady) at 50 Hz


def new_trial(target, trial_id, batch, env_id):
    return dict(target_id=target["target_id"], family=target["family"], trial_id=trial_id, batch=batch, env_id=env_id,
                failure=False, completed=False, failure_reason=[], tracking=[], steady=[], samples=0,
                standing_violation=False, evaluated_duration_s=0.0, diagnostic_samples=0,
                contact_counts=[0]*4, max_base_xy_displacement_m=0.0, max_tilt_rad=0.0,
                min_base_height_m=None, max_foot_drift_m=0.0, max_wheel_rotation_rad=0.0,
                max_abs_roll_rad=0.0, max_abs_pitch_rad=0.0, wheel_travel_rad=[0.0]*4)


def update_trial(trial, sample):
    """Fold one controller sample into its trial; return True when the trial ended."""
    trial["samples"] += 1
    trial["evaluated_duration_s"] = sample["episode_time_s"]
    if sample["phase"] != "settle":
        trial["standing_violation"] |= sample["standing_violation"]
        if sample["base_height_m"] is not None:
            trial["diagnostic_samples"] += 1
            trial["contact_counts"] = [n+int(c) for n, c in zip(trial["contact_counts"], sample["foot_contact_flags"])]
            trial["max_base_xy_displacement_m"] = max(trial["max_base_xy_displacement_m"], sample["base_xy_displacement_m"])
            trial["max_tilt_rad"] = max(trial["max_tilt_rad"], sample["tilt_rad"])
            trial["min_base_height_m"] = (min(trial["min_base_height_m"], sample["base_height_m"])
                                         if trial["min_base_height_m"] is not None else sample["base_height_m"])
            trial["max_foot_drift_m"] = max(trial["max_foot_drift_m"], max(sample["foot_drift_m"]))
            trial["max_wheel_rotation_rad"] = max(trial["max_wheel_rotation_rad"], max(map(abs, sample["wheel_rotation_rad"])))
            trial["wheel_travel_rad"] = sample["wheel_travel_rad"]
            trial["max_abs_roll_rad"] = max(trial["max_abs_roll_rad"], abs(sample["base_roll_pitch_rad"][0]))
            trial["max_abs_pitch_rad"] = max(trial["max_abs_pitch_rad"], abs(sample["base_roll_pitch_rad"][1]))
    errors = [sample["position_error_m"], sample["orientation_error_geodesic_rad"], sample["orientation_error_paper_frobenius"]]
    if sample["scored"]:
        trial["tracking"].append(errors)
    if sample["steady"]:
        trial["steady"].append(errors)
    if not (sample["terminated"] or sample["truncated"]):
        return False
    trial.update(failure=sample["failure"], failure_reason=sample["failure_reason"],
                 completed=sample["truncated"] and not sample["terminated"])
    if trial["completed"] and (len(trial["tracking"]), len(trial["steady"])) != EXPECTED_SAMPLES[trial["family"]]:
        trial.update(failure=True, failure_reason=["invalid_scoring_sample_count"])
    return True


def oracle_arm_q(target, episode_time, nominal_q, quintic, settle_s=2.0):
    """Oracle controller (validation/demos): arm joint target at `episode_time`.
    Fixed: quintic joint-space transition (2 s) from the nominal arm pose to the target's sampled
    configuration, then hold. Circle: linear interpolation of the sampler audit's IK joint path
    (10 Hz over transition + circle, stored in targets.json). Nominal pose during settling."""
    import numpy as np
    if episode_time < settle_s - 1e-9:
        return np.asarray(nominal_q, dtype=float)
    if target["family"] == "fixed":
        blend = quintic(0.0, 1.0, 0.0, 0.0, 2.0, episode_time - settle_s)[0]
        return (1 - blend) * np.asarray(nominal_q) + blend * np.asarray(target["joint_space_sample_rad"])
    path = target["sampler_audit"]["ik_path"]
    times = [p["episode_time_s"] for p in path]
    qpath = np.array([p["arm_joint_pos_rad"] for p in path])
    return np.array([np.interp(episode_time, times, qpath[:, j]) for j in range(qpath.shape[1])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New persistent run directory; never overwritten")
    parser.add_argument("--controller", choices=("oracle", "ik", "hold"), default="oracle",
                        help="evaluate: oracle (validation/demo default), ik (scripted DLS-IK) or hold; smoke always holds")
    parser.add_argument("--seeds", type=str, default="1101", help="evaluate: comma-separated target-sampling seeds")
    parser.add_argument("--num-envs", type=int, default=4, help="evaluate: parallel environments per batch")
    parser.add_argument("--fixed-count", type=int, default=None, help="evaluate: override the protocol count (flagged)")
    parser.add_argument("--circle-count", type=int, default=None, help="evaluate: override the protocol count (flagged)")
    parser.add_argument("--hold-seconds", type=float, default=6.0,
                        help="Duration of smoke-mode nominal hold (validation uses 30 s)")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("--hold-seconds must be finite and positive")
    if args.mode != "smoke" and args.hold_seconds != 6.0:
        parser.error("--hold-seconds applies only to smoke mode")
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or args.num_envs < 1:
        parser.error("--seeds must be distinct; --num-envs >= 1")
    args.headless = True
    # The launcher caller must supply actual host image/GPU provenance.
    import os
    if not os.environ.get("T1_IMAGE_ID") or not os.environ.get("ISAAC_GPU_UUID"):
        parser.error("Set T1_IMAGE_ID to the inspected Docker image ID and ISAAC_GPU_UUID to the allocated UUID")
    if os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("T1 requires one allocated GPU UUID")
    args.out.mkdir(parents=True, exist_ok=False)
    # Retain the source even if simulator initialization or the asset audit fails.
    source = args.out / "source"
    source.mkdir()
    for path in sorted((OVERLAY/"standing_ee").glob("*.py")):
        shutil.copy2(path, source/path.name)
    shutil.copy2(__file__, source/Path(__file__).name)
    shutil.copy2(ROOT/"docs/protocols/TASK_T1.md", source/"TASK_T1.md")
    write_json(args.out/"startup.json", dict(argv=sys.argv, image_id=os.environ["T1_IMAGE_ID"],
               gpu_uuid=os.environ["ISAAC_GPU_UUID"], git=git_info(ROOT)))
    app = env = log = None
    trials, seed_dir = [], None
    try:
        app = AppLauncher(args).app   # Isaac Lab default fast_shutdown=True (teardown closed, protocol 1.0.0)
        import gymnasium as gym
        import numpy as np
        import torch
        import omni.usd
        import robot_lab.tasks  # noqa: F401
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils import math as mu

        # Load only the new package from the overlay without running setup.sh.
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        if str(OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(OVERLAY))
        importlib.import_module(PACKAGE)
        geometry = importlib.import_module(PACKAGE + ".geometry")
        protocol = importlib.import_module(PACKAGE + ".protocol")
        from_cfg = importlib.import_module(PACKAGE + ".env_cfg").StandingEnvCfg
        kinematics = importlib.import_module(PACKAGE + ".kinematics")
        sampler = importlib.import_module(PACKAGE + ".targets")
        evaluate = args.mode == "evaluate"
        counts = dict(fixed_count=protocol.SAMPLER["fixed_count"] if args.fixed_count is None else args.fixed_count,
                      circle_count=protocol.SAMPLER["circle_count"] if args.circle_count is None else args.circle_count)
        cfg = from_cfg()
        cfg.sim.device = args.device
        cfg.seed = seeds[0]
        cfg.scene.num_envs = args.num_envs if evaluate else 1
        env = gym.make(protocol.TASK_ID, cfg=cfg).unwrapped
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joints")
        command = env.command_manager.get_term("ee_pose")
        stage = omni.usd.get_context().get_stage()
        model = kinematics.UsdArmKinematics(stage, "/World/envs/env_0/Robot")
        hashes = protocol.asset_hashes(robot.cfg.spawn.usd_path)
        versions = {}
        for name in ("isaacsim", "isaaclab", "robot-lab", "torch", "numpy", "scipy", "rsl-rl-lib"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        metadata = dict(protocol_version=protocol.VERSION, run_id=args.out.name, task_id=protocol.TASK_ID,
            robot_variant="B2W", end_effector_frame="tcp", mode=args.mode, controller=args.controller,
            seed=cfg.seed, seeds=seeds if evaluate else [cfg.seed], seed_role="target sampling only; initial state deterministic",
            sampler=protocol.SAMPLER, target_counts=counts,
            protocol_counts=(counts["fixed_count"], counts["circle_count"]) == (protocol.SAMPLER["fixed_count"], protocol.SAMPLER["circle_count"]),
            num_envs=env.num_envs, physics_dt_s=env.physics_dt, control_dt_s=env.step_dt,
            payload_mass_kg=0.0, image_id=os.environ["T1_IMAGE_ID"], gpu_uuid=os.environ["ISAAC_GPU_UUID"],
            image_tag=os.environ.get("T1_IMAGE_TAG"),
            gpu_name=torch.cuda.get_device_name(env.device) if str(env.device).startswith("cuda") else None,
            machine=platform.platform(), versions=versions,
            git={"project": git_info(ROOT), "IsaacLab": git_info(ROOT/"repos/IsaacLab"),
                 "robot_lab": git_info(ROOT/"repos/robot_lab")},
            asset_hashes=hashes, joint_order=robot.joint_names,
            environment_origins_world_xyz=env.scene.env_origins.tolist(),
            action_order=[robot.joint_names[j] for j in action.joint_ids],
            action_units="nominal offsets rad; wheel inputs ignored; wheel targets rad/s=0",
            actuator_config={name: ac.to_dict() for name, ac in robot.cfg.actuators.items()},
            loaded_joint_parameters={name: getattr(robot.data, name)[0].tolist() for name in (
                "joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff",
                "joint_pos_limits", "joint_vel_limits", "joint_effort_limits")},
            body_names=robot.body_names, loaded_body_masses_kg=robot.data.default_mass[0].tolist(),
            loaded_body_inertias=robot.data.default_inertia[0].tolist(),
            collision_bound_sources=model.bounds_source,
            collision_filtering=model.physics_filtering,
            geometric_exclusions=sorted(sorted(pair) for pair in model.adjacent),
            clearance_policy=protocol.CLEARANCE_POLICY,
            approx_hull_colliders=[dict(path=s["path"], cooking_issue=s.get("cooking_issue")) for shapes in model.shapes.values()
                                   for s in shapes if s.get("source") == "approx_hull"],
            unresolved_colliders=[dict(path=s["path"], issue=s["issue"]) for shapes in model.shapes.values()
                                  for s in shapes if not s.get("resolved", True)],
            resolved_env_config=cfg.to_dict(), source_snapshot="source/", runtime_validation="in_progress")
        dump_yaml(str(args.out/"run_config.yaml"), metadata)
        # Plain JSON interface for the validation harness; full config stays in YAML.
        interface = {k: v for k, v in metadata.items() if k not in ("actuator_config", "resolved_env_config")}
        interface["actuator_types"] = {name: ac.class_type.__module__ + "." + ac.class_type.__qualname__
                                       for name, ac in robot.cfg.actuators.items()}
        interface["hold_duration_s"] = args.hold_seconds if args.mode == "smoke" else None
        write_json(args.out/"interface.json", interface)
        start = time.monotonic()

        def step_env(offsets):
            with torch.inference_mode():
                env.step(offsets)
            return env.t1_samples

        if not evaluate:
            env.reset(seed=cfg.seed)
            log = CsvLog(args.out/"steps.csv")
            max_steps = round(args.hold_seconds / env.step_dt)
            executed_steps = 0
            for _ in range(max_steps):
                if not app.is_running():
                    break
                samples = step_env(torch.zeros(env.num_envs, 22, device=env.device))
                executed_steps += 1
                for sample in samples:
                    sample.update(protocol_version=protocol.VERSION, robot_variant="B2W", trial_id=0, seed=cfg.seed)
                    log.append(sample)
                    if sample["failure"]:
                        raise ValueError(f"Nominal hold failed: {sample['failure_reason']}")
            log.close(); log = None
            complete = executed_steps == max_steps
            write_json(args.out/"completion.json", dict(status="finished" if complete else "incomplete", mode=args.mode,
                       physics_samples=executed_steps, physics_integration_steps=executed_steps*cfg.decimation,
                       imagined_samples=0, wall_time_s=time.monotonic()-start))
            if not complete:
                raise RuntimeError("Run interrupted/incomplete; partial artifacts preserved")
            print(f"T1 artifacts: {args.out}", flush=True)
            return

        nominal_q = np.array([protocol.NOMINAL_ARM_POSE[n] for n in model.arm_names])
        per_seed, physics_samples = [], 0
        for seed in seeds:
            seed_dir = args.out/f"seed-{seed}"
            seed_dir.mkdir()
            t0 = time.monotonic()
            sampled = sampler.sample(model, seed, hashes, **counts)
            write_json(seed_dir/"targets.json", sampled)
            sampling_s = time.monotonic() - t0
            ordered = sorted(sampled["targets"], key=lambda t: t["family"] != "fixed")   # stable: fixed first
            batches = [ordered[j:j+env.num_envs] for j in range(0, len(ordered), env.num_envs)]
            trials = []
            log = CsvLog(seed_dir/"steps.csv")
            with torch.inference_mode():     # resets after inference-mode steps must stay in inference mode
                env.reset(seed=seed)
            for b, batch in enumerate(batches):
                for i in range(env.num_envs):
                    command.assign(i, batch[i] if i < len(batch) else None)
                with torch.inference_mode():
                    env.reset()
                active = [new_trial(t, len(trials)+i, b, i) for i, t in enumerate(batch)]
                trials.extend(active)
                done = [i >= len(batch) for i in range(env.num_envs)]
                memory = robot.data.joint_pos[:, action.joint_ids[16:]].clone()
                max_steps = round(max(command.duration(i) for i in range(len(batch))) / env.step_dt) + 1
                for _ in range(max_steps):
                    if not app.is_running() or all(done):
                        break
                    offsets = torch.zeros(env.num_envs, 22, device=env.device)
                    for i in range(len(batch)):
                        t = float(env.episode_length_buf[i])*env.step_dt
                        if args.controller == "hold" or done[i] or t < 2.0 - 1e-9:
                            continue
                        target = command.targets[i]
                        if args.controller == "oracle":
                            desired_q = torch.as_tensor(oracle_arm_q(target, t, nominal_q, geometry.quintic),
                                                        device=env.device, dtype=torch.float32)
                        else:
                            # DLS-IK acts on the link06 body: map the TCP reference back to the flange (2.0.0).
                            desired = torch.as_tensor(geometry.pose(command.evaluate(i)[0] @ protocol.TCP_INV),
                                                      device=env.device, dtype=torch.float32)
                            d = robot.data
                            dp, dr = mu.compute_pose_error(d.body_pos_w[i:i+1, command.hand], d.body_quat_w[i:i+1, command.hand],
                                                           desired[None, :3], desired[None, 3:], rot_error_type="axis_angle")
                            jac = robot.root_physx_view.get_jacobians()[i, command.hand, :, :][:, [6+j for j in action.joint_ids[16:]]]
                            error = torch.cat([dp[0], dr[0]])
                            delta = jac.T @ torch.linalg.solve(jac @ jac.T + 0.03**2*torch.eye(6, device=env.device), error)
                            measured = d.joint_pos[i, action.joint_ids[16:]]
                            memory[i] = (memory[i] + delta.clamp(-0.04, 0.04)).clamp(measured-0.2, measured+0.2)
                            desired_q = memory[i]
                        offsets[i, 16:] = desired_q - robot.data.default_joint_pos[i, action.joint_ids[16:]]
                    samples = step_env(offsets)
                    physics_samples += env.num_envs
                    for i, sample in enumerate(samples[:len(batch)]):
                        if done[i]:
                            continue
                        sample.update(protocol_version=protocol.VERSION, robot_variant="B2W", seed=seed, batch=b,
                                      trial_id=active[i]["trial_id"])
                        log.append(sample)
                        done[i] = update_trial(active[i], sample)
            log.close(); log = None
            export_trials(seed_dir, trials, protocol)
            per_seed.append(protocol.summarize(trials))
            complete = all(t["completed"] or t["failure"] for t in trials)
            write_json(seed_dir/"completion.json", dict(status="finished" if complete else "incomplete", seed=seed,
                       targets=len(trials), batches=len(batches), sampling_wall_s=sampling_s,
                       rollout_wall_s=time.monotonic()-t0-sampling_s, wall_time_s=time.monotonic()-t0))
            if not complete:
                raise RuntimeError(f"Seed {seed} interrupted/incomplete; partial trial artifacts preserved")
            trials, seed_dir = [], None
        write_json(args.out/"seeds_summary.json", protocol.summarize_seeds(per_seed))
        write_json(args.out/"completion.json", dict(status="finished", mode=args.mode, seeds=seeds,
                   physics_samples=physics_samples, physics_integration_steps=physics_samples*cfg.decimation,
                   imagined_samples=0, wall_time_s=time.monotonic()-start))
        print(f"T1 artifacts: {args.out}", flush=True)
    except BaseException as exc:
        if log:
            log.close(); log = None
        if trials and seed_dir is not None and not (seed_dir/"trial_summary.csv").exists():
            export_trials(seed_dir, trials, protocol)
        write_json(args.out/"error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if log:
            log.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
