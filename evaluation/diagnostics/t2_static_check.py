"""T2 diagnostic (GPU): validate eval_t2.static_torques against the settled nominal pose (pass: |static - applied| <= 5 % of
each effort limit for all 18 position joints in all envs; exit 1 otherwise).

T2ReachEvalEnvCfg, 16 envs, zero action (hold the T1 nominal pose) for 2 s, then at the settled state:
static_torques (PhysX gravity forces + Jacobians, min-norm wheel contact forces) vs. the applied PD torques
and the measured wheel contact forces (mean over the last 0.2 s). Also dumps the raw PhysX arrays of env 0
(g, Jacobians of the feet, body/COM positions, masses) to static_check.npz for offline analysis.
Writes /t1-run/artifacts/static_check.{json,npz}. Run via launch.sh (LAUNCH_TASK=t2, mode gpu).
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.reach_height"
OUT = Path("/t1-run/artifacts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability", required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("needs one allocated ISAAC_GPU_UUID")
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(status="started")
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
        E = importlib.import_module("eval_t2")
        cfg = importlib.import_module(f"{PACKAGE}.env_cfg").T2ReachEvalEnvCfg()
        cfg.sim.device = args.device
        cfg.commands.tcp_target.reachability_path = args.reachability
        env = gym.make("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
        robot = env.scene["robot"]
        n = env.num_envs
        jid = env.action_manager.get_term("joints").joint_ids
        names = [robot.joint_names[j] for j in jid]
        feet = E.foot_body_ids(robot)
        contacts = env.scene["contacts"]
        sensor_feet = [contacts.body_names.index(f"{leg}_foot") for leg in E.LEGS]
        zero = torch.zeros(n, 22, device=env.device)
        env.reset()
        tau_hist, f_hist = [], []
        for k in range(100):
            env.step(zero)
            if k >= 90:
                tau_hist.append(robot.data.applied_torque[:, jid].cpu().numpy())
                f_hist.append(contacts.data.net_forces_w[:, sensor_feet].cpu().numpy())
        applied = np.mean(tau_hist, axis=0)                     # (n, 22)
        f_meas = np.mean(f_hist, axis=0)                        # (n, 4, 3)
        tau, statics = E.static_torques(env, robot, torch, np, n, jid, feet)
        view = robot.root_physx_view
        g = view.get_gravity_compensation_forces()[:n].cpu().numpy()
        J = view.get_jacobians()[:n].cpu().numpy()
        mass = robot.data.default_mass[0].cpu().numpy()
        np.savez(OUT / "static_check.npz", g=g[0], J=J[0], joint_names=np.array(robot.joint_names),
                 body_names=np.array(robot.body_names), body_pos_w=robot.data.body_pos_w[0].cpu().numpy(),
                 body_quat_w=robot.data.body_quat_w[0].cpu().numpy(),
                 body_com_pos_w=robot.data.body_com_pos_w[0].cpu().numpy(), mass=mass,
                 root_pos_w=robot.data.root_pos_w[0].cpu().numpy(), root_quat_w=robot.data.root_quat_w[0].cpu().numpy(),
                 ground_z=env.command_manager.get_term("tcp_target").origin_w[0, 2].item(),
                 applied=applied[0], f_meas=f_meas[0], tau=tau[0], jid=np.array(jid), feet=np.array(feet),
                 joint_vel_abs_max=np.abs(robot.data.joint_vel[0].cpu().numpy()).max())
        weight = float(mass.sum()) * 9.81
        err = np.abs(tau - applied)
        pos = list(range(12)) + list(range(16, 22))
        result.update(
            g_shape=list(g.shape), J_shape=list(J.shape), num_joints=robot.num_joints, num_bodies=robot.num_bodies,
            weight_N=weight, g_base_env0=g[0, :6].tolist(),
            measured_fz_total_over_weight=float(f_meas[..., 2].sum(axis=1).mean() / weight),
            measured_wheel_force_env0=f_meas[0].tolist(),
            solved_vs_static={k: float(np.median(v)) for k, v in statics.items()},
            per_joint_env0={names[j]: dict(static=float(tau[0, j]), applied=float(applied[0, j])) for j in pos},
            abs_err_max_per_joint={names[j]: float(err[:, j].max()) for j in pos},
            rel_err_max=float((err[:, pos] / np.maximum(np.abs(applied[:, pos]), 1.0)).max()))
        limits = robot.data.joint_effort_limits[:, jid].cpu().numpy()
        frac = err[:, pos] / limits[:, pos]
        # Pass: static torques agree with the settled applied torques within 5 % of each effort limit
        # (internal contact forces are indeterminate, so the solve need not reproduce the settled split exactly).
        result["abs_err_over_limit_max_per_joint"] = {names[j]: float(frac[:, c].max()) for c, j in enumerate(pos)}
        result["pass_"] = bool(frac.max() <= 0.05)
        result["status"] = "completed" if result["pass_"] else "failed_agreement"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        (OUT / "static_check.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=1, default=float), flush=True)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
