"""T3 evaluation runner (protocol v0.6 §7-9) with scripted controllers; acceptance tests A11, A12, A18. GPU.

Each env runs one evaluation-set episode (deterministic nominal reset, 2 s settle with V0 at p_home, 10 s test).
The runner records every alive step, stops after 600 control steps and computes the §9.1 episode rows with
walk_reach.metrics (target = the command after the step: same state time as the measured TCP, and the value the
policy observes, A18).

Controllers (no policy): zero  = zero action (nominal PD targets, wheels 0 rad/s);
                         drive = zero joint offsets, open-loop differential wheel speeds from the velocity command
                                 (wheel i: (vx* - wz* y_i) / r, y_i from the nominal wheel centres; a_w = speed / s_w);
                         carry = drive, plus the base carried kinematically before every step: root pose advanced by
                                 the commanded twist over one control step (x/y along the heading, yaw += wz* dt; height,
                                 roll, pitch kept) and the twist written as root velocity. A scripted test input for A18:
                                 open-loop skid steering does not turn the base, and a velocity write alone is stopped
                                 by wheel friction within the step.
Options: --cells V0-S1,V1-S1 | core | all;  --repeats 2 (A12: same episodes again after env.reset, max |diff| per
column);  --a18 (per step: target in the observation vs. the metrics target vs. the debug marker, in B, tol 1e-6;
plus the F_T schedule recomputed independently in numpy/float64, tol 2e-5).
Writes episodes.csv (first repeat), summary.json to --out.
"""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PKG = "robot_lab.tasks.manager_based.manipulation"
LOGS = ROOT / "repos/robot_lab/logs/t3"
TASK = "RobotLab-Isaac-T3-WalkReach-Eval-Unitree-B2W-Z1-v0"
WHEEL_RADIUS_M = 0.108


def select_cells(spec, P):
    cells = [f"{v}-{s}" for v in list(P.PROFILES) + list(P.STRESS_PROFILES) for s in P.SCENARIOS]
    if spec == "all":
        return cells
    if spec == "core":
        return [c for c in cells if c.split("-")[0] in P.PROFILES]
    out = spec.split(",")
    if not set(out) <= set(cells):
        raise ValueError(f"unknown cells {set(out) - set(cells)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    ap.add_argument("--controller", choices=("zero", "drive", "carry"), default="zero")
    ap.add_argument("--cells", default="core")
    ap.add_argument("--episodes", type=int, default=64, help="per cell (first n of the 64)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--a18", action="store_true")
    ap.add_argument("--eval-set", default=str(LOGS / "diag-eval-set-v5-20261124T073730Z-cf80791b/artifacts/t3_eval_set_v5.npz"))
    ap.add_argument("--eval-pool", default="")
    ap.add_argument("--train-pool", default="")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        ap.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(status="started", controller=args.controller, cells=args.cells, episodes=args.episodes,
                   repeats=args.repeats, a18=args.a18, gpu_uuid=os.environ["ISAAC_GPU_UUID"],
                   image_id=os.environ.get("T1_IMAGE_ID"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import robot_lab.tasks  # noqa: F401
        parent = importlib.import_module(PKG)
        if str(OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(OVERLAY))
        importlib.import_module(f"{PKG}.walk_reach")
        P = importlib.import_module(f"{PKG}.walk_reach.protocol")
        POOL = importlib.import_module(f"{PKG}.walk_reach.pool")
        MET = importlib.import_module(f"{PKG}.walk_reach.metrics")
        FR = importlib.import_module(f"{PKG}.walk_reach.frames")
        KIN = importlib.import_module(f"{PKG}.walk_reach.kinematics")
        RT = importlib.import_module(f"{PKG}.walk_reach.runtime")
        EC = importlib.import_module(f"{PKG}.walk_reach.env_cfg")
        JOINTS = importlib.import_module(f"{PKG}.standing_ee.runtime").JOINTS
        from isaaclab.utils.math import subtract_frame_transforms

        ev = np.load(args.eval_set)
        arrays = {k: ev[k] for k in ev.files if k != "metadata"}
        if POOL.digest(arrays) != P.EVAL_SET_SHA256:
            raise ValueError("evaluation set digest differs from protocol.EVAL_SET_SHA256")
        meta = json.loads(str(ev["metadata"]))
        pool_arrays, _ = POOL.load(args.eval_pool or str(LOGS / P.POOL_FILES["eval"]), P.EVAL_POOL_SHA256)
        cells = select_cells(args.cells, P)
        rows_idx = [int(i) for c in cells for i in np.flatnonzero(
            (arrays["cell_V"] == c.split("-")[0]) & (arrays["cell_S"] == c.split("-")[1])
            & (arrays["episode_index"] < args.episodes))]
        n = len(rows_idx)
        summary.update(eval_set_version=meta["version"], eval_set_sha256=P.EVAL_SET_SHA256, num_envs=n,
                       protocol_version=P.VERSION)

        cfg = EC.T3WalkReachEvalEnvCfg()
        cfg.scene.num_envs = n
        cfg.sim.device = args.device
        if args.train_pool:
            cfg.commands.tcp_target.pool_path = args.train_pool
        cfg.commands.tcp_target.debug_vis = args.a18
        t0 = time.perf_counter()
        env = gym.make(TASK, cfg=cfg).unwrapped
        summary["make_s"] = time.perf_counter() - t0
        dev, robot = env.device, env.scene["robot"]
        cmd, vel = env.command_manager.get_term(RT.COMMAND), env.command_manager.get_term(RT.VELOCITY)
        om = env.observation_manager
        names, dims = om.active_terms["policy"], [d[0] for d in om.group_obs_term_dim["policy"]]
        start = int(np.sum(dims[:names.index("tcp_goal_body")]))
        goal_slice = slice(start, start + 7)
        summary["observation"] = dict(terms=dict(zip(names, dims)), dim=int(sum(dims)))
        jid = robot.find_joints(JOINTS, preserve_order=True)[0]
        non_wheel = jid[:12] + jid[16:]
        feet = robot.find_bodies([f"{l}_foot" for l in ("FR", "FL", "RR", "RL")], preserve_order=True)[0]
        wheel_y = torch.tensor([c[1] for c in P.WHEEL_CENTRE_BODY], device=dev)
        cmd_arr = torch.as_tensor(arrays["cmd"][rows_idx], dtype=torch.float32, device=dev)

        def assign():
            vel.assign(torch.arange(n, device=dev), cmd_arr)
            for scen in ("S1", "S3", "S4"):
                ids = [j for j, r in enumerate(rows_idx) if arrays["cell_S"][r] == scen]
                if not ids:
                    continue
                rr = [rows_idx[j] for j in ids]
                if scen != "S4":
                    cmd.assign(ids, scen, target_T=arrays["target_T"][rr])
                else:
                    nodes = arrays["s4_nodes"][rr]
                    g = np.c_[pool_arrays["q_arm"], pool_arrays["b"]]
                    chain = np.concatenate([np.repeat(g[None, 0:1], len(rr), 0),
                                            g[np.where(nodes >= 0, nodes, 0)]], axis=1)
                    cmd.assign(ids, "S4", chain_g=chain, chain_T=arrays["s4_T_traj"][rr], chain_H=arrays["s4_hold"][rr])

        def controller(obs, alive):
            a = torch.zeros(n, 22, device=dev)
            if args.controller in ("drive", "carry"):
                c = env.command_manager.get_command(RT.VELOCITY)
                a[:, 12:16] = (c[:, :1] - c[:, 2:3] * wheel_y[None]) / WHEEL_RADIUS_M / P.WHEEL_ACTION_SCALE_RAD_S
            if args.controller == "carry":
                c, d, dt = env.command_manager.get_command(RT.VELOCITY), robot.data, env.step_dt
                yaw = FR.yaw_from_quat(d.root_quat_w)
                pose = d.root_pose_w.clone()
                pose[:, 0] += c[:, 0] * torch.cos(yaw) * dt
                pose[:, 1] += c[:, 0] * torch.sin(yaw) * dt
                pose[:, 3:] = FR.quat_mul(FR.yaw_quat(c[:, 2] * dt), d.root_quat_w)
                v = d.root_vel_w.clone()
                v[:, 0], v[:, 1] = c[:, 0] * torch.cos(yaw), c[:, 0] * torch.sin(yaw)
                v[:, 3:5], v[:, 5] = 0.0, c[:, 2]
                ids = alive.nonzero(as_tuple=True)[0]        # terminated envs are left alone (no repeated failures)
                robot.write_root_pose_to_sim(pose[ids], env_ids=ids)
                robot.write_root_velocity_to_sim(v[ids], env_ids=ids)
            return a

        def run():
            obs, _ = env.reset(seed=P.EVAL_SET_SEED)
            assign()
            K = int(round(P.TEST_S / P.CONTROL_DT_S))
            F = dict(tcp_pos_w=3, tcp_quat_w=4, target_pos_w=3, target_quat_w=4, tcp_pos_T=3, target_pos_T=3,
                     base_quat_w=4, base_lin_vel_w=3, base_ang_vel_w=3, base_lin_vel_b=3, base_height=1,
                     wheel_pos_body_xy=8, joint_pos=18, joint_vel=22, torque=22, target_height=1, target_quat_T=4,
                     base_pos=3)
            buf = {k: torch.full((n, K, d), float("nan"), device=dev) for k, d in F.items()}
            alive = torch.ones(n, dtype=torch.bool, device=dev)
            k_alive = torch.zeros(n, dtype=torch.long, device=dev)
            term_id, t_term = [None] * n, [float("nan")] * n
            term_all, bodies = [""] * n, [""] * n
            a18 = dict(steps=0, obs_vs_metric_pos=0.0, obs_vs_metric_quat=0.0, obs_vs_marker_pos=0.0,
                       obs_vs_marker_quat=0.0, sched_pos=0.0, sched_quat=0.0, marker_calls=0) if args.a18 else None
            obs = env.observation_manager.compute()
            wall = time.perf_counter()
            for j in range(SETTLE + K):
                obs, _, terminated, truncated, _ = env.step(controller(obs, alive))
                if bool(truncated.any()):
                    raise RuntimeError("time-out before step 600")
                dead = (terminated & alive).nonzero(as_tuple=True)[0].tolist()
                for i in dead:
                    ids = RT.termination_ids(env, i)
                    term_id[i], t_term[i] = (ids[0] if ids else "unknown"), (j + 1 - SETTLE) * P.CONTROL_DT_S
                    term_all[i] = "|".join(ids)
                    bodies[i] = "|".join(sorted({b for name in ("body_contact", "arm_contact")
                                                 for b in env.termination_manager.get_term_cfg(name).func.reasons[i]}))
                alive &= ~terminated
                d = robot.data
                o = env.scene.env_origins
                if args.a18:
                    cmd._debug_vis_callback(None)
                    a18_step(obs["policy"][:, goal_slice], d, o, alive, a18, j)
                if j < SETTLE:
                    continue
                k = j - SETTLE
                tcp_p, tcp_q = cmd.tcp_pose_w()
                tcp_T, _ = FR.world_to_ft(d.root_pos_w, d.root_quat_w, tcp_p, tcp_q, P.H_T_M, o[:, 2])
                fp = subtract_frame_transforms(d.root_pos_w[:, None].expand(-1, 4, -1).reshape(-1, 3),
                                               d.root_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4),
                                               d.body_pos_w[:, feet].reshape(-1, 3))[0].reshape(n, 4, 3)
                vals = dict(tcp_pos_w=tcp_p - o, tcp_quat_w=tcp_q, target_pos_w=cmd.position_w - o,
                            target_quat_w=cmd.quat_w, tcp_pos_T=tcp_T, target_pos_T=cmd.position_T,
                            target_quat_T=cmd.quat_T, base_quat_w=d.root_quat_w, base_lin_vel_w=d.root_lin_vel_w,
                            base_ang_vel_w=d.root_ang_vel_w, base_lin_vel_b=d.root_lin_vel_b,
                            base_height=(d.root_pos_w[:, 2] - o[:, 2])[:, None],
                            wheel_pos_body_xy=fp[..., :2].reshape(n, 8), joint_pos=d.joint_pos[:, non_wheel],
                            joint_vel=d.joint_vel[:, jid], torque=d.applied_torque[:, jid],
                            target_height=cmd.target_height()[:, None], base_pos=d.root_pos_w - o)
                for key, v in vals.items():
                    buf[key][alive, k] = v[alive].float()
                k_alive += alive.long()
            torch.cuda.synchronize()
            wall = time.perf_counter() - wall
            host = {key: v.cpu().numpy().astype(float) for key, v in buf.items()}
            lim = robot.data.joint_pos_limits[0, non_wheel].cpu().numpy().astype(float)
            eff = robot.data.joint_effort_limits[0, jid].cpu().numpy().astype(float)
            rows = []
            for i, r in enumerate(rows_idx):
                K_i = int(k_alive[i])
                ep = {key: host[key][i, :K_i] for key in host}
                ep["wheel_pos_body_xy"] = ep["wheel_pos_body_xy"].reshape(K_i, 4, 2)
                ep["base_height"], ep["target_height"] = ep["base_height"][:, 0], ep["target_height"][:, 0]
                ep.update(joint_limits=lim, effort_limit=eff, termination_id=term_id[i], t_termination=t_term[i])
                m = MET.episode_metrics(ep, str(arrays["cell_S"][r]), tuple(arrays["cmd"][r]))
                yaw = np.unwrap(MET.yaw(ep["base_quat_w"])) if K_i else np.zeros(1)
                diag = dict(termination_all=term_all[i], contact_bodies=bodies[i],
                            base_yaw_change_rad=float(yaw[-1] - yaw[0]),
                            base_xy_travel_m=float(np.linalg.norm(np.diff(ep["base_pos"][:, :2], axis=0), axis=1).sum()))
                rows.append(dict(method=f"scripted-{args.controller}", policy_seed=0, eval_set_version=meta["version"],
                                 eval_set_hash=P.EVAL_SET_SHA256, cell_V=str(arrays["cell_V"][r]),
                                 cell_S=str(arrays["cell_S"][r]), episode_index=int(arrays["episode_index"][r]), **m, **diag))
            if args.a18:
                a18.update(schedule_check(host, k_alive.cpu().numpy(), rows_idx))
            return rows, wall, a18, host

        def a18_step(goal_obs, d, o, alive, acc, j):
            """Target in B three ways (float64 from the stored float32 world poses): obs vs metrics vs marker."""
            idx = alive.nonzero(as_tuple=True)[0]
            if not len(idx):
                return
            root_p, root_q = (d.root_pos_w[idx] - o[idx]).double(), d.root_quat_w[idx].double()

            def to_b(pw, qw):
                return subtract_frame_transforms(root_p, root_q, (pw[idx] - o[idx]).double(), qw[idx].double())

            def qdist(a, b):
                return torch.minimum((a - b).abs().amax(dim=1), (a + b).abs().amax(dim=1)).max().item()

            pm, qm = to_b(cmd.position_w, cmd.quat_w)
            mk = cmd.marker_pose_w
            pk, qk = to_b(mk[0], mk[1])
            g = goal_obs[idx].double()
            acc["steps"] += 1
            acc["marker_calls"] += 1
            acc["obs_vs_metric_pos"] = max(acc["obs_vs_metric_pos"], (g[:, :3] - pm).abs().max().item())
            acc["obs_vs_metric_quat"] = max(acc["obs_vs_metric_quat"], qdist(g[:, 3:], qm))
            acc["obs_vs_marker_pos"] = max(acc["obs_vs_marker_pos"], (g[:, :3] - pk).abs().max().item())
            acc["obs_vs_marker_quat"] = max(acc["obs_vs_marker_quat"], qdist(g[:, 3:], qk))

        def schedule_check(host, k_alive, rows_idx):
            """Expected F_T targets from the evaluation set (numpy / float64 FK) vs. the metrics target in F_T."""
            fk = KIN.ArmFK("cpu", dtype=torch.float64)
            g_all = np.c_[pool_arrays["q_arm"], pool_arrays["b"]]
            dt, worst_p, worst_q = P.CONTROL_DT_S, 0.0, 0.0
            for i, r in enumerate(rows_idx):
                K_i = int(k_alive[i])
                if not K_i:
                    continue
                k = np.arange(K_i)
                scen = str(arrays["cell_S"][r])
                if scen in ("S1", "S3"):
                    sch = P.S1_SCHEDULE if scen == "S1" else P.S3_SCHEDULE
                    j = np.minimum(k // int(round(sch["hold_s"] / dt)), sch["points"] - 1)
                    exp = arrays["target_T"][r][j]
                else:
                    T, H = arrays["s4_T_traj"][r], arrays["s4_hold"][r]
                    used = ~np.isnan(T)
                    T, H = T[used], H[used]
                    nodes = arrays["s4_nodes"][r][used]
                    starts = np.r_[0.0, np.cumsum(T + H)[:-1]]
                    tau = (k + 1) * dt
                    seg = np.clip(np.searchsorted(starts, tau, side="left") - 1, 0, len(T) - 1)
                    s = np.where(T[seg] > 0, np.clip((tau - starts[seg]) / np.where(T[seg] > 0, T[seg], 1), 0, 1), 1.0)
                    prev = np.r_[0, nodes][seg]
                    gg = g_all[prev] + s[:, None] * (g_all[nodes[seg]] - g_all[prev])
                    p, q = fk.tcp_pose(torch.as_tensor(gg[:, :6]), torch.as_tensor(gg[:, 6:]))
                    exp = np.c_[p.numpy(), q.numpy()]
                got_p, got_q = host["target_pos_T"][i, :K_i], host["target_quat_T"][i, :K_i]
                worst_p = max(worst_p, float(np.abs(got_p - exp[:, :3]).max()))
                dq = np.minimum(np.abs(got_q - exp[:, 3:]).max(axis=1), np.abs(got_q + exp[:, 3:]).max(axis=1))
                worst_q = max(worst_q, float(dq.max()))
            return dict(sched_pos=worst_p, sched_quat=worst_q)

        SETTLE = RT.SETTLE_STEPS
        results = []
        for rep in range(args.repeats):
            rows, wall, a18, host = run()
            results.append(rows)
            summary[f"repeat_{rep}"] = dict(wall_s=wall, env_steps_per_s=n * (SETTLE + 500) / wall,
                                            aggregate=MET.aggregate(rows),
                                            terminations={k: sum(r["termination_id"] == k for r in rows)
                                                          for k in sorted({r["termination_id"] for r in rows})})
            if a18 is not None:
                tol = 1e-6
                a18["pass_"] = bool(max(a18[k] for k in ("obs_vs_metric_pos", "obs_vs_metric_quat",
                                                          "obs_vs_marker_pos", "obs_vs_marker_quat")) <= tol
                                    and a18["steps"] > 0)
                a18["schedule_pass_"] = bool(max(a18["sched_pos"], a18["sched_quat"]) <= 2e-5)
                summary[f"repeat_{rep}"]["A18"] = a18
        rows = results[0]
        keys = list(rows[0])
        with open(args.out / "episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        per_cell = {}
        for c in cells:
            rc = [r for r in rows if f"{r['cell_V']}-{r['cell_S']}" == c]
            per_cell[c] = dict(n=len(rc), terminations={k: sum(r["termination_id"] == k for r in rc)
                                                         for k in sorted({r["termination_id"] for r in rc})},
                               **{k: float(np.nanmean([r[k] for r in rc])) if np.isfinite([r[k] for r in rc]).any()
                                  else float("nan")
                                  for k in ("e_vx_mean", "e_vy_mean", "e_w_mean", "e_pos_mean_all", "e_ori_mean_all",
                                            "base_h_min", "tilt_max", "primary_success_rate", "base_yaw_change_rad",
                                            "base_xy_travel_m")})
            per_cell[c]["contact_bodies"] = sorted({r["contact_bodies"] for r in rc} - {""})
        summary["per_cell"] = per_cell
        if args.repeats > 1:
            diffs = {}
            for k in keys:
                a = [r[k] for r in results[0]]
                b = [r[k] for r in results[1]]
                if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in a + b):
                    x, y = np.asarray(a, float), np.asarray(b, float)
                    same_nan = np.isnan(x) == np.isnan(y)
                    diffs[k] = float(np.nanmax(np.abs(x - y))) if same_nan.all() and np.isfinite(x).any() else (
                        0.0 if same_nan.all() else float("inf"))
                else:
                    diffs[k] = 0.0 if a == b else float("inf")
            summary["A12"] = dict(max_abs_diff=diffs, max_over_columns=max(diffs.values()),
                                  identical=all(v == 0.0 for v in diffs.values()))
        summary["status"] = "completed"
    except Exception:
        summary.update(status="error", error=traceback.format_exc())
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("error",)}, indent=1, default=float)[:20000],
              flush=True)
        if env is not None:
            env.close()
        app.close()
    raise SystemExit(0 if summary["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
