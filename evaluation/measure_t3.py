"""T3 protocol Section 10 measurements (v0.3). CPU only, no SimulationApp.

Kinematic measurement on the gripper USD at the T1 nominal state (itself the final sample of a 30 s
settled standing simulation, T1 protocol draft.5), so no extra physics settling is done here.
Terrain is the plane z = 0. F_T at the nominal state: origin (x_b, y_b, h_T), yaw of the base.

Writes to --out:
  t3_measurements.json  h_T, p_home, B_body, wheel positions (F_T and body frame), knee height,
                        nominal base offset b_nom, base-feasibility table (from the frozen T2 map),
                        clearance-check timing, asset and map hashes
  t3_kinematics.json    Z1 chain (base_link -> link06) + TCP offset for the torch FK (walk_reach.kinematics)

Run: LAUNCH_TASK=t3 evaluation/diagnostics/launch.sh cpu measure 1800 evaluation/measure_t3.py --out /t1-run/artifacts
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
import usd_loader  # noqa: E402  (re-execs once with the image's USD libraries)
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

MANIP = usd_loader.ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
FROZEN_MAP = usd_loader.ROOT / ("repos/robot_lab/logs/t2/diag-reach-A1-v04r3-20261113T084127Z-1d029cc2/"
                                "artifacts/reachability_T2_protocol.npz")
LEGS = ("FR", "FL", "RR", "RL")


def module(name):
    """Import a manipulation submodule without Isaac Lab or Gym."""
    for pkg, path in (("_manip", MANIP), ("_manip.standing_ee", MANIP / "standing_ee"),
                      ("_manip.reach_height", MANIP / "reach_height"), ("_manip.walk_reach", MANIP / "walk_reach")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    return importlib.import_module("_manip." + name)


def load_model():
    from pxr import Usd
    K = module("standing_ee.kinematics")
    stage = Usd.Stage.Open(str(usd_loader.ASSET))
    root = next(p for p in stage.Traverse() if p.GetName() == "base_link").GetParent()
    return K.UsdArmKinematics(stage, str(root.GetPath()))


def ft_frame(base, h_T):
    """F_T in the world at a base transform on flat ground (z = 0)."""
    W = module("reach_height.wholebody")
    F = W.ground_frame(base)
    F[2, 3] = h_T
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=os.environ.get("T2_REACHABILITY_NPZ", str(FROZEN_MAP)))
    ap.add_argument("--timing-samples", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    T1P = module("standing_ee.protocol")
    T2P = module("reach_height.protocol")
    W = module("reach_height.wholebody")
    model = load_model()
    wb = W.WholeBody(model)
    base, joints = wb.base_nominal, wb.joints_nominal
    frames = model.forward(base, joints)

    # h_T, F_T, p_home
    h_T = float(frames["link00"][2, 3])
    F = ft_frame(base, h_T)
    Finv = np.linalg.inv(F)
    tcp_T = Finv @ frames[T2P.TCP_PARENT] @ wb.tcp
    p_home = np.r_[tcp_T[:3, 3], Rotation.from_matrix(tcp_T[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]]]

    # Nominal base offset b_nom = (dh, pitch, roll) with R_base = Rz(yaw) Ry(pitch) Rx(roll) (T2 base_pose convention)
    yaw, pitch, roll = Rotation.from_matrix(base[:3, :3]).as_euler("ZYX")
    b_nom = [float(base[2, 3] - T2P.H_NOM_M), float(pitch), float(roll)]
    rebuilt = W.base_pose(wb.G, T2P.H_NOM_M + b_nom[0], b_nom[1], b_nom[2])
    base_rebuild_error = float(np.abs(rebuilt - base).max())

    # B_body: AABB of all non-arm colliders (trunk, head, lidar, legs, wheels) in F_T, + 0.05 m per side
    body_names = sorted(n for n in model.collider_bounds if n not in W.ARM_BODIES)
    pts = []
    for n in body_names:
        for b in model.collider_bounds[n]:
            c = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
            pts.append((Finv @ frames[n] @ np.c_[c, np.ones(len(c))].T)[:3].T)
    pts = np.concatenate(pts)
    box = np.array([pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05])

    # Wheels (foot-joint anchors = wheel centres), knees (calf joint origins)
    wheels_world = np.array([wb.wheel_world[leg] for leg in LEGS])
    wheels_T = (Finv @ np.c_[wheels_world, np.ones(4)].T)[:3].T
    wheels_body = (np.linalg.inv(base) @ np.c_[wheels_world, np.ones(4)].T)[:3].T
    knees = []
    for leg in LEGS:
        chain = wb.legs[leg]
        _, origins, _ = chain.fk([joints[n] for n in chain.joint_names])
        knees.append((base @ np.r_[origins[2][0], 1.0])[:3])
    knees = np.array(knees)

    # Base-feasibility table from the frozen T2 map (T2 leg IK + clearance, all grid postures evaluated)
    map_sha = hashlib.sha256(Path(args.map).read_bytes()).hexdigest()
    with np.load(args.map, allow_pickle=True) as d:
        dh_list, pitch_list = np.asarray(d["posture_dh"]), np.asarray(d["posture_pitch"])
        feasible = np.asarray(d["posture_feasible"], dtype=bool)
        reasons = [str(r) for r in d["posture_reason"]] if "posture_reason" in d.files else None
    dh_axis = np.unique(np.round(dh_list, 6))
    pitch_axis = np.unique(np.round(pitch_list, 6))
    table = np.zeros((len(dh_axis), len(pitch_axis)), dtype=bool)
    for dh, p, f in zip(dh_list, pitch_list, feasible):
        table[np.searchsorted(dh_axis, round(dh, 6)), np.searchsorted(pitch_axis, round(p, 6))] = f

    # link00 pose in F_T over the base-offset grid (T2 base model: base x/y fixed, roll 0)
    T_base_link00 = np.linalg.inv(base) @ frames["link00"]
    link00_grid = []
    for dh in dh_axis[::4]:
        for p in pitch_axis[::2]:
            B = W.base_pose(wb.G, T2P.H_NOM_M + dh, p)
            L = Finv @ B @ T_base_link00
            link00_grid.append(dict(dh=float(dh), pitch=float(p), link00_position_T=L[:3, 3].tolist()))

    # Torch-FK export: chain base_link -> link06 (local0, axis, inv(local1) per joint edge)
    chain = W.Chain(model, "link06")
    kin = dict(chain_joint_names=chain.joint_names,
               steps=[dict(local0=l0.tolist(), axis=None if a is None else a.tolist(), local1_inv=l1.tolist())
                      for l0, a, l1 in chain.steps],
               limits_rad=chain.limits.tolist(),
               soft_limits_rad=wb.arm_limits.tolist(),
               tcp_offset_link06=list(T2P.TCP_OFFSET_M),
               T_base_link00=T_base_link00.tolist(),
               h_nom=T2P.H_NOM_M, h_T=h_T,
               nominal_arm_q=[joints[n] for n in chain.joint_names], b_nom=b_nom)
    # FK self-check of the chain against the full USD tree at the nominal state
    T_chain = chain.fk([joints[n] for n in chain.joint_names])[0][0]
    fk_check = float(np.abs(base @ T_chain - frames["link06"]).max())

    # Clearance-check timing (one pool point = leg IK + whole-body clearance)
    rng = np.random.default_rng(0)
    times, outcomes = [], {}
    for _ in range(args.timing_samples):
        q = rng.uniform(wb.arm_limits[:, 0], wb.arm_limits[:, 1])
        dh, p = rng.uniform(-0.33, 0.0), rng.uniform(-0.28, 0.28)
        t0 = time.perf_counter()
        B = W.base_pose(wb.G, T2P.H_NOM_M + dh, p)
        legs, reason = wb.solve_legs(B)
        if not reason:
            reason = W.clearance_violation(model, model.forward(B, W.full_joints(dict(joints, **legs), q)))
        times.append(time.perf_counter() - t0)
        key = (reason or "ok").split(":")[0]
        outcomes[key] = outcomes.get(key, 0) + 1

    asset_hashes = T1P.asset_hashes(usd_loader.ASSET)
    result = dict(
        protocol="T3 v0.3 Section 10", asset=str(usd_loader.ASSET.relative_to(usd_loader.ROOT)),
        asset_hashes=asset_hashes,
        asset_tree_sha256=hashlib.sha256(json.dumps(asset_hashes, sort_keys=True).encode()).hexdigest(),
        t2_map=dict(path=str(Path(args.map)), sha256=map_sha),
        method="FK at the T1 nominal state (settled 30 s standing sim, T1 draft.5); terrain z = 0",
        h_nom=T2P.H_NOM_M, h_T=h_T, F_T_yaw_rad=float(yaw),
        p_home_T=dict(position=p_home[:3].tolist(), quaternion_wxyz=p_home[3:].tolist()),
        nominal_tcp_height_m=float((frames[T2P.TCP_PARENT] @ wb.tcp)[2, 3]),
        b_nom=dict(dh=b_nom[0], pitch=b_nom[1], roll=b_nom[2], rebuild_error=base_rebuild_error),
        B_body_T=dict(min=box[0].tolist(), max=box[1].tolist(), margin_m=0.05, bodies=body_names),
        wheels=dict(order=list(LEGS), centre_T=wheels_T.tolist(), contact_T=np.c_[wheels_T[:, :2], np.full(4, -h_T)].tolist(),
                    centre_body=wheels_body.tolist(), radius_m=wheels_world[:, 2].tolist()),
        knees=dict(order=list(LEGS), height_m=knees[:, 2].tolist(), h_knee=float(knees[:, 2].max())),
        feasibility=dict(dh_axis=dh_axis.tolist(), pitch_axis=pitch_axis.tolist(), feasible=table.astype(int).tolist(),
                         n_feasible=int(table.sum()), n_total=int(table.size),
                         infeasible_reasons=sorted({r.split(":")[0] for r, f in zip(reasons or [], feasible) if not f})),
        link00_T_over_offsets=link00_grid,
        fk_chain_check_max_abs=fk_check,
        clearance_timing=dict(samples=len(times), mean_s=float(np.mean(times)), p95_s=float(np.quantile(times, 0.95)),
                              outcomes=outcomes))
    (out / "t3_measurements.json").write_text(json.dumps(result, indent=1))
    kin.update(asset_tree_sha256=result["asset_tree_sha256"], t2_map_sha256=map_sha)
    (out / "t3_kinematics.json").write_text(json.dumps(kin, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("asset_hashes", "feasibility", "link00_T_over_offsets")},
                     indent=1))
    print("feasible postures", table.sum(), "/", table.size)
    print(np.array2string(table.astype(int), max_line_width=200))


if __name__ == "__main__":
    main()
