"""T1 2.1.0 check (T3 D43): the T1 policy observation equals the T3 policy observation for the same state. GPU.

Two processes (one simulator each), both 1 env, deterministic nominal reset, zero action:
  --phase t1  T1 environment holding T1 fixed target `fixed-00` (T1 2.0.0 reference target file) for 2 s settle +
              5 s; per step: robot state, the goal command (TCP goal in B) and the 95-dim policy observation.
  --phase t3  T3 evaluation environment, velocity command V0; per step: the robot state is compared with the T1
              record (bitwise), the T3 goal command is set to the recorded T1 command (T1's target schedule is not
              a T3 scenario) and the policy observation is recomputed and compared with the T1 observation.
Pass: same term names, order and dims (95), identical states, max |obs_T1 - obs_T3| over all steps <= 1e-6 (and
reported; 0 expected). Writes obs_t1.npz / obs_check.json to --out.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PKG = "robot_lab.tasks.manager_based.manipulation"
TARGETS = ROOT / "repos/robot_lab/logs/t3/diag-t1v2-validate-20261123T204134Z-26af8a39/artifacts/validation/targets.json"
STEPS = 340


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("t1", "t3"), required=True)
    ap.add_argument("--t1-record", default="", help="obs_t1.npz of the t1 phase (t3 phase)")
    ap.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        ap.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(phase=args.phase, status="started")
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
        if args.phase == "t1":
            importlib.import_module(f"{PKG}.standing_ee")
            cfg = importlib.import_module(f"{PKG}.standing_ee.env_cfg").StandingEnvCfg()
            task, cmd_name = "RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0", "ee_pose"
        else:
            importlib.import_module(f"{PKG}.walk_reach")
            cfg = importlib.import_module(f"{PKG}.walk_reach.env_cfg").T3WalkReachEvalEnvCfg()
            task, cmd_name = "RobotLab-Isaac-T3-WalkReach-Eval-Unitree-B2W-Z1-v0", "tcp_target"
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        env = gym.make(task, cfg=cfg).unwrapped
        robot, om = env.scene["robot"], env.observation_manager
        terms = list(zip(om.active_terms["policy"], [d[0] for d in om.group_obs_term_dim["policy"]]))
        cmd = env.command_manager.get_term(cmd_name)
        env.reset(seed=1)
        if args.phase == "t1":
            target = next(t for t in json.loads(TARGETS.read_text())["targets"] if t["target_id"] == "fixed-00")
            cmd.assign(0, target)
        else:
            env.command_manager.get_term("base_velocity").assign([0], [[0.0, 0.0]])
            rec = np.load(args.t1_record)
        zero = torch.zeros(1, 22, device=env.device)
        rows = dict(obs=[], command=[], root=[], joint_pos=[], joint_vel=[], body=[])
        worst = dict(state=0.0, obs=0.0, obs_term={})
        for k in range(STEPS):
            obs, _, terminated, truncated, _ = env.step(zero)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError(f"episode ended at step {k + 1}")
            d = robot.data
            state = dict(root=d.root_state_w[0], joint_pos=d.joint_pos[0], joint_vel=d.joint_vel[0],
                         body=d.body_state_w[0].reshape(-1))
            if args.phase == "t1":
                for key, v in state.items():
                    rows[key].append(v.cpu().numpy())
                rows["obs"].append(obs["policy"][0].cpu().numpy())
                rows["command"].append(cmd.command[0].cpu().numpy())
                continue
            for key, v in state.items():
                worst["state"] = max(worst["state"], float(np.abs(v.cpu().numpy() - rec[key][k]).max()))
            cmd._command[0] = torch.as_tensor(rec["command"][k], device=env.device)
            o3 = om.compute()["policy"][0].cpu().numpy()
            diff = np.abs(o3 - rec["obs"][k])
            worst["obs"] = max(worst["obs"], float(diff.max()))
            s = 0
            for name, dim in terms:
                worst["obs_term"][name] = max(worst["obs_term"].get(name, 0.0), float(diff[s:s + dim].max()))
                s += dim
        result["terms"] = terms
        if args.phase == "t1":
            np.savez(args.out / "obs_t1.npz", terms=json.dumps(terms), **{k: np.stack(v) for k, v in rows.items()})
            result.update(steps=STEPS, obs_dim=int(sum(d for _, d in terms)),
                          goal_command_range=[float(np.min(rows["command"])), float(np.max(rows["command"]))])
        else:
            t1_terms = [tuple(t) for t in json.loads(str(rec["terms"]))]
            result.update(steps=STEPS, t1_terms=t1_terms, same_layout=t1_terms == [tuple(t) for t in terms],
                          max_state_diff=worst["state"], max_obs_diff=worst["obs"], max_obs_diff_per_term=worst["obs_term"])
            result["pass_"] = bool(result["same_layout"] and sum(d for _, d in terms) == 95 and worst["obs"] <= 1e-6)
        result["status"] = "completed"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        name = "obs_t1.json" if args.phase == "t1" else "obs_check.json"
        (args.out / name).write_text(json.dumps(result, indent=1, default=float))
        print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=1, default=float), flush=True)
        if env is not None:
            env.close()
        app.close()
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
