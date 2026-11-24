"""T3 environment tests in Isaac Sim (protocol v0.6 §11). GPU, scripted inputs only.

--phase checks  T3WalkReachEvalEnvCfg, 5 envs:
    layout   policy observation = T2's terms, order and dims (95); the velocity-goal slot equals the command
    A8       env0: base pitched 1.2 rad, raised 0.6 m -> no termination on steps 1-4, F2 (only) on step 5;
             env1: base at 0.35 m rolled 1.3 rad (tips onto its side) -> first termination is a fall ID;
             env2: control, nominal stand under random velocity commands -> no termination;
             F3: cached height injection (0.2 m) into env2 for 5 evaluations of the fall_height term -> fires on the
             5th only, never in another env
    S2       env3: Z1 commanded to an arm pose that penetrates the trunk (found with the T1 clearance model; arm
             base excluded, arm >= 0.25 m above ground) -> first termination IDs == ["S2"]
--phase speed   --task t1|t2|t3 --num-envs N --steps S: zero-action throughput (env steps/s). t3 also: time in
                the target generator (profiled block), stranding counters (D32) and termination counts over all steps.
Writes <phase>[_<task>].json to --out.
"""

import argparse
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
T2_MAP = ROOT / "repos/robot_lab/logs/t2/diag-reach-A1-v04r3-20261113T084127Z-1d029cc2/artifacts/reachability_T2_protocol.npz"


def find_s2_pose(model, targets, np, seed=7, tries=3000):
    """Arm pose (dict joint -> rad) whose moving arm link penetrates a non-arm body by >= 2 cm, arm base and
    PhysX-filtered pairs excluded, all arm links >= 0.25 m above ground; nearest to nominal in joint space."""
    from scipy.spatial.transform import Rotation  # noqa: F401
    base, joints = targets.nominal_state(model)
    lo, hi = targets.soft_limits(model.limits).T
    q0 = np.array([joints[n] for n in model.arm_names])
    rng = np.random.default_rng(seed)
    arm = {f"link0{i}" for i in range(2, 7)} | {"gripperStator", "gripperMover"}
    best = None
    for _ in range(tries):
        q = rng.uniform(lo, hi)
        frames = model.forward(base, dict(joints, **dict(zip(model.arm_names, q))))
        if min(frames[b][2, 3] for b in arm) < 0.25:
            continue
        rows = [r for r in model.clearance_report(frames, 0.005) if r["pair"][1] != "ground"
                and r.get("clearance_m") is not None and r["clearance_m"] < -0.02
                and (r["pair"][0] in arm) != (r["pair"][1] in arm)
                and not ({r["pair"][0], r["pair"][1]} & {"link00", "link01", "head_link", "lidar_link"})]
        if rows:
            d = float(np.linalg.norm(q - q0))
            if best is None or d < best[0]:
                best = (d, q, [r["pair"] for r in rows])
    if best is None:
        raise RuntimeError("no S2 pose found")
    return dict(zip(model.arm_names, best[1].tolist())), best[2], best[0]


def checks(env, mods, result, torch, np):
    RT, P = mods["runtime"], mods["protocol"]
    T2E = importlib.import_module(f"{PKG}.reach_height.env_cfg")
    robot, tm, dev, n = env.scene["robot"], env.termination_manager, env.device, env.num_envs
    om = env.observation_manager
    names, dims = om.active_terms["policy"], [d[0] for d in om.group_obs_term_dim["policy"]]
    t2_names = [k for k, v in T2E.ObservationsCfg.PolicyCfg().__dict__.items()
                if v is not None and type(v).__name__ == "ObservationTermCfg"]
    obs, _ = env.reset(seed=1)
    obs, *_ = env.step(torch.zeros(n, 22, device=dev))
    s = int(np.sum(dims[:names.index("base_velocity_goal")]))
    slot = obs["policy"][:, s:s + 3]
    result["layout"] = dict(terms=dict(zip(names, dims)), t2_terms=t2_names, dim=int(sum(dims)),
                            velocity_slot_equals_command=bool(torch.equal(
                                slot, env.command_manager.get_command(RT.VELOCITY))),
                            velocity_slot_nonzero_envs=int((slot != 0).any(dim=1).sum()))
    result["layout"]["pass_"] = bool(names == t2_names and sum(dims) == 95 and dims[names.index("base_velocity_goal")] == 3
                                     and result["layout"]["velocity_slot_equals_command"])

    # S2 target pose (CPU clearance model of env 0)
    import omni.usd
    kin = importlib.import_module(f"{PKG}.standing_ee.kinematics")
    targets = importlib.import_module(f"{PKG}.standing_ee.targets")
    model = kin.UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
    t = time.perf_counter()
    s2_q, s2_pairs, s2_dist = find_s2_pose(model, targets, np)
    result["S2_search"] = dict(q=s2_q, penetrating_pairs=s2_pairs, joint_distance_rad=s2_dist,
                               search_s=time.perf_counter() - t)

    # A8 + S2 scripted episode
    env.reset(seed=2)
    root = robot.data.root_pose_w.clone()
    root[0, 2] += 0.6
    root[0, 3:] = torch.tensor([np.cos(0.6), 0.0, np.sin(0.6), 0.0], device=dev)          # pitch 1.2 rad
    root[1, 2] = env.scene.env_origins[1, 2] + 0.35
    root[1, 3:] = torch.tensor([np.cos(0.65), np.sin(0.65), 0.0, 0.0], device=dev)         # roll 1.3 rad
    robot.write_root_pose_to_sim(root[:2], env_ids=torch.arange(2, device=dev))
    robot.write_root_velocity_to_sim(torch.zeros(2, 6, device=dev), env_ids=torch.arange(2, device=dev))
    env.sim.forward()
    env.scene.update(env.physics_dt)
    jid = robot.find_joints(mods["JOINTS"], preserve_order=True)[0]
    action = torch.zeros(n, 22, device=dev)
    default = robot.data.default_joint_pos[3, jid].cpu().numpy()
    for k, name in enumerate(mods["JOINTS"]):
        if name in s2_q:
            action[3, k] = float(s2_q[name] - default[k])
    first = {}
    per_step = []
    for step in range(1, 151):
        _, _, terminated, truncated, _ = env.step(action)
        rec = {}
        for i in range(n):
            if i in first:
                continue
            ids = RT.termination_ids(env, i)
            if bool(terminated[i]) or bool(truncated[i]):
                contact = {name: list(tm.get_term_cfg(name).func.reasons[i]) for name in ("body_contact", "arm_contact")}
                first[i] = dict(step=step, ids=ids, truncated=bool(truncated[i]), contact_bodies=contact)
                if i == 3:
                    action[3] = 0.0
            rec[i] = ids
        if step <= 5:
            per_step.append(rec)
        if all(i in first for i in (0, 1, 3)) and step >= 100:
            break
    a8 = dict(env0_tilt=first.get(0), env0_ids_steps_1_to_5=[r.get(0) for r in per_step],
              env1_tip_over=first.get(1), env2_control=first.get(2), steps_run=step)
    e0, e1 = first.get(0), first.get(1)
    a8["tilt_pass"] = bool(e0 and e0["step"] == 5 and e0["ids"] == ["F2"]
                           and all(not r.get(0) for r in per_step[:4]))
    a8["tip_over_pass"] = bool(e1 and e1["ids"] and e1["ids"][0] in ("F1", "F2", "F3"))
    a8["control_pass"] = 2 not in first

    # F3 cached-height injection (no physics), as T1 2.0.0 runtime check
    env.reset(seed=3)
    term = tm.get_term_cfg("fall_height")
    fired = []
    with torch.inference_mode():
        pos = robot.data.root_pos_w
        before = pos[2].clone()
        try:
            for _ in range(5):
                pos[2, 2] = env.scene.env_origins[2, 2] + 0.2
                out = term.func(env, **term.params)
                fired.append(out.nonzero(as_tuple=True)[0].tolist())
        finally:
            pos[2] = before
    a8["F3_injection_fired_envs_per_eval"] = fired
    a8["F3_pass"] = fired == [[], [], [], [], [2]]
    a8["pass_"] = bool(a8["tilt_pass"] and a8["tip_over_pass"] and a8["control_pass"] and a8["F3_pass"])
    result["A8"] = a8
    e3 = first.get(3)
    result["S2"] = dict(env3=e3, pass_=bool(e3 and e3["ids"] == ["S2"]))


def speed(env, mods, result, torch, np, task, steps):
    n, dev = env.num_envs, env.device
    zero = torch.zeros(n, 22, device=dev)
    tm = env.termination_manager
    counts = {k: 0 for k in tm.active_terms}

    def run(k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(k):
            env.step(zero)
            for name in counts:
                counts[name] += int(tm.get_term(name).sum())
        torch.cuda.synchronize()
        return time.perf_counter() - t

    env.reset(seed=1)
    run(20)
    wall = run(steps)
    result["throughput"] = dict(task=task, num_envs=n, steps=steps, wall_s=wall, env_steps_per_s=n * steps / wall,
                                ms_per_step=wall / steps * 1e3)
    if task == "t3":
        cmd = env.command_manager.get_term(mods["runtime"].COMMAND)
        cmd.profile = True
        k = min(200, steps)
        wall_p = run(k)
        cmd.profile = False
        s = cmd.stats
        result["generator"] = dict(profiled_steps=k, generator_ms_per_step=s["generator_s"] / s["calls"] * 1e3,
                                   share_of_profiled_step=s["generator_s"] / wall_p,
                                   profiled_ms_per_step=wall_p / k * 1e3)
        result["stranding"] = cmd.sampler.rates()
        result["stranding"]["sim_steps"] = steps + 20 + k
        result["stranding"]["sim_time_s_per_env"] = (steps + 20 + k) * env.step_dt
    result["terminations"] = counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("checks", "speed"), required=True)
    ap.add_argument("--task", choices=("t1", "t2", "t3"), default="t3")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        ap.error("needs one allocated ISAAC_GPU_UUID")
    name = args.phase if args.phase == "checks" else f"speed_{args.task}"
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(phase=args.phase, task=args.task, gpu_uuid=os.environ["ISAAC_GPU_UUID"],
                  image_id=os.environ.get("T1_IMAGE_ID"), status="started")
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
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
        mods = {}
        if args.phase == "checks" or args.task == "t3":
            importlib.import_module(f"{PKG}.walk_reach")
            mods = {m: importlib.import_module(f"{PKG}.walk_reach.{m}") for m in ("protocol", "runtime", "env_cfg")}
            mods["JOINTS"] = importlib.import_module(f"{PKG}.standing_ee.runtime").JOINTS
            E = mods["env_cfg"]
            cfg = E.T3WalkReachEvalEnvCfg() if args.phase == "checks" else E.T3WalkReachEnvCfg()
            task = ("RobotLab-Isaac-T3-WalkReach-Eval-Unitree-B2W-Z1-v0" if args.phase == "checks"
                    else "RobotLab-Isaac-T3-WalkReach-Unitree-B2W-Z1-v0")
            cfg.scene.num_envs = 5 if args.phase == "checks" else args.num_envs
            result["protocol_version"] = mods["protocol"].VERSION
        elif args.task == "t2":
            importlib.import_module(f"{PKG}.reach_height")
            E = importlib.import_module(f"{PKG}.reach_height.env_cfg")
            cfg, task = E.T2ReachEnvCfg(), "RobotLab-Isaac-T2-Reach-Unitree-B2W-Z1-v0"
            cfg.commands.tcp_target.reachability_path = os.environ.get("T2_REACHABILITY_NPZ", str(T2_MAP))
            cfg.scene.num_envs = args.num_envs
        else:
            importlib.import_module(f"{PKG}.standing_ee")
            E = importlib.import_module(f"{PKG}.standing_ee.env_cfg")
            cfg, task = E.StandingEnvCfg(), "RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0"
            cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        result.update(gym_task=task, num_envs=cfg.scene.num_envs, torch=torch.__version__,
                      device=torch.cuda.get_device_name(0))
        t = time.perf_counter()
        env = gym.make(task, cfg=cfg).unwrapped
        result["make_s"] = time.perf_counter() - t
        if args.phase == "checks":
            checks(env, mods, result, torch, np)
        else:
            speed(env, mods, result, torch, np, args.task, args.steps)
        result["status"] = "completed"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
        print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=1, default=float), flush=True)
        if env is not None:
            env.close()
        app.close()
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
