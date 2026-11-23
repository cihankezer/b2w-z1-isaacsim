"""T3 acceptance test A9 (protocol v0.4 §11, D23): the extreme whole-body postures stored in the frozen T2 map
must not trigger the T3 falls F2 (tilt > 1.0 rad for >= 5 steps) or F3 (base height < 0.274 m for >= 5 steps).
GPU (Isaac Sim), no training.

Postures: every map voxel whose stored posture is an extreme of the map: deepest dh (-0.28 m), largest
nose-down and largest nose-up pitch. Per voxel: T2's A3a posture controller (eval_t2 --controller posture):
reset into the stored whole-body solution, hold POSTURE_HOLD_S = 2 s with gravity-compensated PD targets.
F2/F3 are judged conservatively from the recorded per-episode maximum tilt and minimum base height (no
debounce credit). T2 terminations stay active and are reported (a T2 termination is not an A9 failure).
Writes a9.json and episodes.csv to --out.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_t2 as E  # noqa: E402

MAX_PER_GROUP = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reachability", required=True)
    ap.add_argument("--num-envs", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        ap.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(test="A9", status="started")
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        import robot_lab.tasks  # noqa: F401
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        if str(E.OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(E.OVERLAY))
        mods = {m: importlib.import_module(f"{E.PACKAGE}.{m}") for m in ("protocol", "sampler", "metrics", "env_cfg", "wholebody")}
        mods["clearance"] = importlib.import_module("robot_lab.tasks.manager_based.manipulation.standing_ee.clearance")
        T3P = importlib.import_module("robot_lab.tasks.manager_based.manipulation.walk_reach.protocol")
        P, W = mods["protocol"], mods["wholebody"]
        cfg = mods["env_cfg"].T2ReachEvalEnvCfg()
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.commands.tcp_target.reachability_path = args.reachability
        cfg.episode_length_s = E.POSTURE_HOLD_S + 2 * P.CONTROL_DT_S
        env = gym.make("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
        with np.load(args.reachability, allow_pickle=False) as d:
            data = {k: d[k] for k in ("posture_index", "arm_q", "posture_leg_q", "leg_joint_names", "posture_dh",
                                      "posture_pitch", "posture_base_world", "G_world", "voxel_centre_G", "voxel_band",
                                      "reachable")}
        k = data["posture_index"]
        has = k >= 0
        dh, pitch = np.where(has, data["posture_dh"][np.maximum(k, 0)], np.nan), np.where(has, data["posture_pitch"][np.maximum(k, 0)], np.nan)
        groups = {"deepest_dh": has & np.isclose(dh, np.nanmin(dh)),
                  "max_nose_down": has & np.isclose(pitch, np.nanmax(pitch)),
                  "max_nose_up": has & np.isclose(pitch, np.nanmin(pitch))}
        rng = np.random.default_rng(0)
        voxels, group_of = [], []
        for name, mask in groups.items():
            v = np.flatnonzero(mask)
            v = rng.choice(v, size=min(MAX_PER_GROUP, len(v)), replace=False) if len(v) else v
            voxels += list(v); group_of += [name] * len(v)
        voxels = np.array(voxels)
        centres = data["voxel_centre_G"][voxels]
        theta = np.arctan2(centres[:, 1], centres[:, 0])
        quat = Rotation.from_matrix(W.canonical_rotation(theta, centres[:, 2])).as_quat(canonical=True)[:, [3, 0, 1, 2]]
        targets = dict(target_id=np.arange(len(voxels)), band=data["voxel_band"][voxels], voxel=voxels, position_G=centres, quat_G_wxyz=quat, azimuth_rad=theta)
        rows, t0 = [], time.perf_counter()
        csv_rows = []
        info = dict(settle_terminated=0, nominal_tcp_G=np.zeros(3), emit=csv_rows.append)
        for start in range(0, len(voxels), args.num_envs):
            sel = np.arange(start, min(start + args.num_envs, len(voxels)))
            E.run_batch(env, mods, torch, np, targets, sel, "posture", None, data, rows, info)
            print(f"batch {start // args.num_envs}: {len(rows)}/{len(voxels)}", flush=True)
        per = {}
        for g, r in zip(group_of, rows):
            f2 = r["tilt_max"] > T3P.FALL_TILT_RAD
            f3 = r["h_min"] < T3P.FALL_HEIGHT_M
            e = per.setdefault(g, dict(n=0, f2=0, f3=0, t2_terminated=0, h_min=[], tilt_max=[], dh=None, pitch=None))
            e["n"] += 1; e["f2"] += int(f2); e["f3"] += int(f3)
            e["t2_terminated"] += int(bool(r.get("walked_away") or r.get("fell")))
            e["h_min"].append(r["h_min"]); e["tilt_max"].append(r["tilt_max"])
            e["dh"], e["pitch"] = float(r["oracle_posture_dh"]), float(r["oracle_posture_pitch"])
        for e in per.values():
            e["h_min"], e["tilt_max"] = float(np.min(e["h_min"])), float(np.max(e["tilt_max"]))
        result.update(status="done", wall_s=time.perf_counter() - t0, fall_height_m=T3P.FALL_HEIGHT_M,
                      fall_tilt_rad=T3P.FALL_TILT_RAD, groups=per,
                      pass_=all(e["f2"] == 0 and e["f3"] == 0 for e in per.values()) and bool(per),
                      gpu_uuid=os.environ["ISAAC_GPU_UUID"], image_id=os.environ.get("T1_IMAGE_ID"))
        import csv
        with open(args.out / "episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group"] + list(csv_rows[0]))
            w.writeheader()
            for g, r in zip(group_of, csv_rows):
                w.writerow({"group": g, **E.json_safe(r)})
    except Exception as exc:
        import traceback
        result.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    (args.out / "a9.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str), flush=True)
    if result["status"] == "error":
        os._exit(1)       # an Isaac Sim teardown after an exception can hang
    app.close()


if __name__ == "__main__":
    main()
