"""T2 GPU smoke test (protocol v0.4 §10 step 3, acceptance tests A6 and A8). Zero actions only, no policy.

  --phase eval   T2ReachEvalEnvCfg, 16 envs: observation size, target-map consistency, A6 terminations
  --phase train  T2ReachEnvCfg, 4096 envs: A8 throughput, cost of the ground-clearance term and its broad phase

Run via evaluation/diagnostics/launch.sh (LAUNCH_TASK=t2, mode gpu, one allocated GPU UUID).
Writes /t1-run/artifacts/smoke_<phase>.json.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.reach_height"
OUT = Path("/t1-run/artifacts")


def write(phase, result):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"smoke_{phase}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")


def eval_phase(env, mods, result, torch, np):
    P, S, W = mods["protocol"], mods["sampler"], mods["wholebody"]
    from scipy.spatial.transform import Rotation
    cmd = env.command_manager.get_term("tcp_target")
    robot = env.scene["robot"]
    n = env.num_envs
    zero = torch.zeros(n, 22, device=env.device)

    obs, _ = env.reset(seed=1)
    om = env.observation_manager
    result["observation"] = dict(
        policy_shape=list(obs["policy"].shape), expected_dim=95,
        terms=dict(zip(om.active_terms["policy"], [list(d) for d in om.group_obs_term_dim["policy"]])),
        pass_=bool(obs["policy"].shape[-1] == 95))
    obs, *_ = env.step(zero)                      # §4.4: one control period after reset

    # Target-map consistency: 125 resamples of all envs -> 2000 targets, checked against the numpy reference.
    rs = S.ReachableSet.load(cmd.reachability_path)
    ids = torch.arange(n, device=env.device)
    d = robot.data
    rows = []
    for _ in range(125):
        cmd._resample_command(ids)
        cmd._update_command()
        rows.append({k: v.detach().cpu().numpy().copy() for k, v in dict(
            band=cmd.band_id, pos_G=cmd.position_G, quat_G=cmd.quat_G, pos_w=cmd.position_w, quat_w=cmd.quat_w,
            origin=cmd.origin_w, yaw=cmd.yaw, command=cmd.command, root_pos=d.root_pos_w, root_quat=d.root_quat_w,
            env_origin=env.scene.env_origins).items()})
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    pos = cat["pos_G"].astype(float)
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    canon = W.canonical_rotation(theta, pos[:, 2])
    R_G = Rotation.from_quat(cat["quat_G"][:, [1, 2, 3, 0]])
    rot_err_G = (Rotation.from_matrix(canon).inv() * R_G).magnitude()
    Rz = Rotation.from_euler("z", cat["yaw"])
    pos_w_ref = cat["origin"] + Rz.apply(pos)
    rot_err_w = (Rotation.from_quat(cat["quat_w"][:, [1, 2, 3, 0]]).inv() * (Rz * R_G)).magnitude()
    root = Rotation.from_quat(cat["root_quat"][:, [1, 2, 3, 0]])
    cmd_pos_ref = root.inv().apply(cat["pos_w"] - cat["root_pos"])
    cmd_rot_err = (Rotation.from_quat(cat["command"][:, [4, 5, 6, 3]]).inv()
                   * (root.inv() * Rotation.from_quat(cat["quat_w"][:, [1, 2, 3, 0]]))).magnitude()
    root_yaw = np.array([math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) for w, x, y, z in cat["root_quat"]])
    freq = np.bincount(cat["band"], minlength=3) / len(cat["band"])
    in_largest = rs.contains(pos)
    c = dict(
        n_targets=int(len(pos)), band_frequency=freq.tolist(),
        all_in_sampling_set=bool(in_largest.all()),
        all_in_band=bool((P.band_index(pos[:, 2]) == cat["band"]).all()),
        max_abs_azimuth_rad=float(np.abs(theta).max()),
        max_orientation_error_G_rad=float(rot_err_G.max()),
        max_world_position_error_m=float(np.abs(cat["pos_w"] - pos_w_ref).max()),
        max_world_orientation_error_rad=float(rot_err_w.max()),
        max_command_position_error_m=float(np.abs(cat["command"][:, :3] - cmd_pos_ref).max()),
        max_command_orientation_error_rad=float(cmd_rot_err.max()),
        G_origin_xy_vs_root_max_m=float(np.abs(cat["origin"][:, :2] - cat["root_pos"][:, :2]).max()),
        G_origin_z_vs_env_origin_max_m=float(np.abs(cat["origin"][:, 2] - cat["env_origin"][:, 2]).max()),
        G_yaw_vs_root_max_rad=float(np.abs(np.angle(np.exp(1j * (cat["yaw"] - root_yaw)))).max()),
        sampling_set_dropped_voxels=rs.dropped, reachability_sha256=cmd.reachability_sha256)
    c["pass_"] = bool(c["all_in_sampling_set"] and c["all_in_band"] and c["max_abs_azimuth_rad"] <= P.AZIMUTH_MAX_RAD + 1e-5
                      and c["max_orientation_error_G_rad"] < 1e-3 and c["max_world_position_error_m"] < 1e-4
                      and c["max_world_orientation_error_rad"] < 1e-3 and c["max_command_position_error_m"] < 1e-4
                      and c["max_command_orientation_error_rad"] < 1e-3 and np.abs(freq - 1 / 3).max() < 0.05)
    # TCP definition sanity: measured TCP in G after one zero-action step vs. the map's nominal TCP.
    tcp_w, _ = cmd.tcp_pose_w()
    tcp_G = Rotation.from_euler("z", cat["yaw"][:n]).inv().apply(tcp_w.cpu().numpy() - cat["origin"][:n])
    c["measured_tcp_G_mean"] = tcp_G.mean(axis=0).tolist()
    result["target_map"] = c

    # A6: teleport env 0 by 0.4 m sideways, pitch env 1 by 1.2 rad (nose down); one zero-action step.
    env.reset()
    env.step(zero)
    pose = torch.cat([d.root_pos_w, d.root_quat_w], dim=1).clone()
    yaw0 = float(cmd.yaw[0])
    pose[0, 0] += -0.4 * math.sin(yaw0)
    pose[0, 1] += 0.4 * math.cos(yaw0)
    q1 = Rotation.from_quat(pose[1, [4, 5, 6, 3]].cpu().numpy()) * Rotation.from_euler("y", 1.2)
    pose[1, 3:] = torch.tensor(q1.as_quat()[[3, 0, 1, 2]], device=env.device, dtype=pose.dtype)
    robot.write_root_pose_to_sim(pose[:2], env_ids=torch.tensor([0, 1], device=env.device))
    robot.write_root_velocity_to_sim(torch.zeros(2, 6, device=env.device), env_ids=torch.tensor([0, 1], device=env.device))
    env.step(zero)
    tm = env.termination_manager
    terms = {name: tm.get_term(name).cpu().numpy().astype(bool) for name in tm.active_terms}
    a6 = dict(env0_sideways_terms=[k for k, v in terms.items() if v[0]],
              env1_pitch_terms=[k for k, v in terms.items() if v[1]],
              other_envs_terminated={k: int(v[2:].sum()) for k, v in terms.items()},
              ground_clearance_reasons_env1=tm.get_term_cfg("ground_clearance").func.reasons[1])
    a6["pass_"] = bool(terms["walk_away"][0] and terms["fall_tilt"][1]
                       and not terms["walk_away"][2:].any() and not terms["fall_tilt"][2:].any())
    result["A6"] = a6


def train_phase(env, mods, result, torch, np):
    n = env.num_envs
    zero = torch.zeros(n, 22, device=env.device)
    tm = env.termination_manager
    clearance = tm.get_term_cfg("ground_clearance")
    term = clearance.func
    counts = {k: 0 for k in tm.active_terms}

    def run(steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(steps):
            env.step(zero)
            for k in counts:
                counts[k] += int(tm.get_term(k).sum())
        torch.cuda.synchronize()
        return time.perf_counter() - t

    env.reset(seed=1)
    run(20)                                              # warm-up
    steps = 200
    wall = run(steps)
    result["A8"] = dict(num_envs=n, steps=steps, wall_s=wall, env_steps_per_s=n * steps / wall,
                        control_steps_per_s=steps / wall, terminations=dict(counts),
                        gpu_mem_max_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, pass_=True)

    # Broad phase in isolation at 4096 envs.
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        flagged = term.broad_phase(env, mods["protocol"].CLEARANCE_M)
    torch.cuda.synchronize()
    broad_ms = (time.perf_counter() - t) / 100 * 1e3
    # Profiled term inside steps (broad + narrow), then throughput with the term replaced by a no-op.
    term.profile = True
    profiled_wall = run(100)
    term.profile = False
    s = term.stats
    original = clearance.func
    class NoClearance:   # class terms need reset(); returns "no termination"
        def __call__(self, env, **kw):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        def reset(self, env_ids=None):
            pass

    clearance.func = NoClearance()
    wall_off = run(steps)
    clearance.func = original
    result["clearance_cost"] = dict(
        num_envs=n, collider_corner_points=int(term.points.shape[0]),
        broad_phase_isolated_ms_per_call=broad_ms, flagged_envs_last_isolated_call=int(flagged.sum()),
        profiled_steps=s["calls"], broad_phase_in_step_ms_per_call=s["broad_s"] / max(s["calls"], 1) * 1e3,
        narrow_phase_ms_per_call=s["narrow_s"] / max(s["calls"], 1) * 1e3,
        flagged_envs_per_step=s["flagged_envs"] / max(s["calls"], 1),
        step_ms_with_term=wall / steps * 1e3, step_ms_without_term=wall_off / steps * 1e3,
        step_ms_profiled=profiled_wall / 100 * 1e3,
        term_share_of_step=1 - wall_off / wall)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("eval", "train"), required=True)
    parser.add_argument("--reachability", required=True, help="container path of the frozen v0.4 npz")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("needs one allocated ISAAC_GPU_UUID")
    result = dict(phase=args.phase, gpu_uuid=os.environ["ISAAC_GPU_UUID"], image_id=os.environ.get("T1_IMAGE_ID"),
                  reachability=args.reachability, status="started")
    write(args.phase, result)
    app = env = None
    try:
        app = AppLauncher(args).app
        import gymnasium as gym
        import numpy as np
        import torch
        import robot_lab.tasks  # noqa: F401
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        if str(OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(OVERLAY))
        importlib.import_module(PACKAGE)
        mods = {m: importlib.import_module(f"{PACKAGE}.{m}") for m in ("protocol", "sampler", "wholebody", "env_cfg")}
        cfg = (mods["env_cfg"].T2ReachEvalEnvCfg if args.phase == "eval" else mods["env_cfg"].T2ReachEnvCfg)()
        cfg.sim.device = args.device
        cfg.commands.tcp_target.reachability_path = args.reachability
        task = ("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0" if args.phase == "eval"
                else "RobotLab-Isaac-T2-Reach-Unitree-B2W-Z1-v0")
        result.update(task=task, num_envs=cfg.scene.num_envs, protocol_version=mods["protocol"].VERSION,
                      pinned_sha256=cfg.commands.tcp_target.reachability_sha256,
                      torch=torch.__version__, device=torch.cuda.get_device_name(0))
        t = time.perf_counter()
        env = gym.make(task, cfg=cfg).unwrapped
        result["make_s"] = time.perf_counter() - t
        (eval_phase if args.phase == "eval" else train_phase)(env, mods, result, torch, np)
        result["status"] = "completed"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        write(args.phase, result)
        print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=1, default=float), flush=True)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
