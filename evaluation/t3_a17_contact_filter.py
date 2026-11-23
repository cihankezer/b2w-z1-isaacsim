"""T3 acceptance test A17 (protocol v0.4 §4.3, D21): can a ground-filtered contact sensor separate terrain
contact (F1) from self-contact (S2) on the multi-body B2W+Z1 sensor? GPU (Isaac Sim), no training.

Scene: the T2 evaluation environment (T1 robot, flat ground at /World/ground) with every termination except the
time-out removed, plus one extra robot contact sensor per tested ground filter pattern. Isaac Lab documents
filtered reporting as one-to-many only (one sensor body per env), so the multi-body case is what is tested.

Scripted cases (one env each, state written at reset, zero action = hold nominal PD targets):
  stand     nominal state: only the four wheels touch the ground
  arm_zero  nominal legs, Z1 joints written to 0 (CAD zero pose; T1 found link02-link06 self-contact there)
  side      base written at 0.35 m with roll 1.3 rad: the robot falls on its side, trunk/leg bodies hit the ground
For each recorded step and body: unfiltered net force (> 1 N = contact), filtered ground force per pattern, and the
geometric classification (lowest collider-AABB corner of the body within 1 cm of the ground = terrain).
Pass for a pattern: on every step, the set of bodies with filtered ground force > 1 N equals the set of
bodies with net contact whose geometric class is terrain, and no arm body in arm_zero shows ground force.
Writes a17.json to --out.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PATTERNS = {"ground_root": ["/World/ground"], "ground_children": ["/World/ground/.*"],
            "ground_deep": ["/World/ground/.*/.*"]}
CASES = ("stand", "arm_zero", "side")
STEPS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        ap.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(test="A17", status="started", patterns=PATTERNS, cases=CASES, steps=STEPS)
    (args.out / "a17.json").write_text(json.dumps(result, indent=1))
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.math import quat_from_euler_xyz
        import robot_lab.tasks  # noqa: F401
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        if str(OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(OVERLAY))
        env_cfg = importlib.import_module("robot_lab.tasks.manager_based.manipulation.reach_height.env_cfg")
        kin = importlib.import_module("robot_lab.tasks.manager_based.manipulation.standing_ee.kinematics")
        cfg = env_cfg.T2ReachEvalEnvCfg()
        cfg.sim.device = args.device
        cfg.scene.num_envs = len(CASES)
        cfg.commands.tcp_target.reachability_path = str(ROOT / ("repos/robot_lab/logs/t2/diag-reach-A1-v04r3-"
                                                                "20261113T084127Z-1d029cc2/artifacts/reachability_T2_protocol.npz"))
        for name in ("fall_tilt", "body_contact", "arm_contact", "invalid_state", "walk_away", "ground_clearance"):
            setattr(cfg.terminations, name, None)
        cfg.episode_length_s = 60.0
        for name, pat in PATTERNS.items():
            setattr(cfg.scene, f"filt_{name}", ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1,
                                                                update_period=0.0, filter_prim_paths_expr=pat))
        sensors_ok, errors = {}, {}
        try:
            env = gym.make("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
        except Exception as exc:          # a pattern the contact view rejects: drop it and report
            errors["make"] = repr(exc)
            raise
        robot = env.scene["robot"]
        import omni.usd
        model = kin.UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
        corners = {}
        for n, bounds in model.collider_bounds.items():
            if n in robot.body_names:
                corners[robot.body_names.index(n)] = torch.as_tensor(
                    np.concatenate([kin.corners(np.asarray(b)) for b in bounds]), dtype=torch.float32, device=env.device)
        ground_prims = [str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse()
                        if str(p.GetPath()).startswith("/World/ground")][:20]
        env.reset()
        d = robot.data
        ids = torch.arange(len(CASES), device=env.device)
        root = d.default_root_state[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        jp, jv = d.default_joint_pos[ids].clone(), torch.zeros_like(d.default_joint_pos[ids])
        arm = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
        jp[1, arm] = 0.0
        root[2, 2] = env.scene.env_origins[2, 2] + 0.35
        root[2, 3:7] = quat_from_euler_xyz(torch.tensor([1.3], device=env.device), torch.zeros(1, device=env.device),
                                           torch.zeros(1, device=env.device))[0]
        robot.write_root_state_to_sim(root, ids)
        robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
        robot.set_joint_position_target(jp, env_ids=ids)
        env.scene.write_data_to_sim()
        zero = torch.zeros(len(CASES), env.action_manager.total_action_dim, device=env.device)
        net_sensor = env.scene["contacts"]
        names = net_sensor.body_names
        idx_in_robot = [robot.body_names.index(n) for n in names]
        arm_bodies = {n for n in names if n.startswith("link0") or n.startswith("gripper")}
        steps, t0 = [], time.perf_counter()
        for k in range(STEPS):
            env.step(zero)
            if k < 3:    # arm_zero: re-write the zero arm pose (the action term targets the nominal pose)
                robot.write_joint_state_to_sim(jp[1:2, arm], torch.zeros(1, 6, device=env.device), joint_ids=arm,
                                               env_ids=ids[1:2])
            net = net_sensor.data.net_forces_w.norm(dim=-1)                       # (E, B)
            low = torch.stack([((d.body_pos_w[:, j, None] + torch.einsum("eij,kj->eki", _rot(d.body_quat_w[:, j]),
                                                                    corners[j]))[..., 2].min(dim=1).values
                                - env.scene.env_origins[:, 2]) if j in corners else torch.full((len(CASES),), 9.0, device=env.device)
                               for j in idx_in_robot], dim=1)                       # (E, B) lowest corner height
            row = dict(step=k, cases={})
            for e, case in enumerate(CASES):
                contact = {names[b] for b in range(len(names)) if net[e, b] > 1.0}
                geo_terrain = {names[b] for b in range(len(names)) if net[e, b] > 1.0 and low[e, b] < 0.01}
                entry = dict(net_contact=sorted(contact), geometric_terrain=sorted(geo_terrain), filtered={})
                for pname in PATTERNS:
                    s = env.scene[f"filt_{pname}"]
                    fm = s.data.force_matrix_w
                    if fm is None:
                        entry["filtered"][pname] = None
                        continue
                    g = fm[e].norm(dim=-1).sum(dim=-1)                              # (B,)
                    entry["filtered"][pname] = sorted(s.body_names[b] for b in range(len(s.body_names)) if g[b] > 1.0)
                row["cases"][case] = entry
            steps.append(row)
        verdict = {}
        for pname in PATTERNS:
            ok, reasons = True, []
            for row in steps:
                for case, entry in row["cases"].items():
                    f = entry["filtered"][pname]
                    if f is None:
                        ok, reasons = False, ["force_matrix_w is None"]
                        break
                    if set(f) != set(entry["geometric_terrain"]):
                        ok = False
                        reasons.append(f"step {row['step']} {case}: filtered {f} vs geometric {entry['geometric_terrain']}")
                    if case == "arm_zero" and set(f) & arm_bodies:
                        ok = False
                        reasons.append(f"step {row['step']} arm_zero: arm bodies with ground force {sorted(set(f) & arm_bodies)}")
            verdict[pname] = dict(pass_=ok, mismatches=reasons[:20], n_mismatches=len(reasons))
        arm_self = any(set(r["cases"]["arm_zero"]["net_contact"]) & arm_bodies for r in steps)
        side_terrain = any(set(r["cases"]["side"]["geometric_terrain"]) - {n for n in names if n.endswith("_foot")}
                           for r in steps)
        result.update(status="done", wall_s=time.perf_counter() - t0, ground_prims=ground_prims,
                      scenario_check=dict(arm_zero_self_contact_seen=arm_self, side_non_wheel_terrain_contact_seen=side_terrain),
                      verdict=verdict, any_pattern_passes=any(v["pass_"] for v in verdict.values()),
                      steps=steps, errors=errors, gpu_uuid=os.environ["ISAAC_GPU_UUID"], image_id=os.environ.get("T1_IMAGE_ID"))
    except Exception as exc:
        import traceback
        result.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    (args.out / "a17.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=1, default=str), flush=True)
    if result["status"] == "error":
        os._exit(1)       # an Isaac Sim teardown after an exception can hang
    app.close()


def _rot(q):
    from isaaclab.utils.math import matrix_from_quat
    return matrix_from_quat(q)


if __name__ == "__main__":
    main()
