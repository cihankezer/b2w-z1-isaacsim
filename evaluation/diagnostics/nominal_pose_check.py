"""Verify a candidate nominal arm (ready) pose on the T1 asset; CPU, T1 collider path, no physics.

Criteria (researcher, draft.5):
  * every arm joint inside the middle 80 % of its USD range;
  * self-clearance >= 15 mm at the pose and along quintic transitions to pose +/- 0.1 rad
    on each joint separately (21 samples per transition);
  * EE (link06) in front of the base (x_B > 0) and inside the targets.py workspace bounds
    (10th-90th percentile of 512 seeded collision-free arm samples, reproduced exactly);
  * wrist camera (link06/end_effector_camera, optical axis = local -Z) not facing the trunk:
    its ray misses the base_link collider boxes and points > 45 deg away from the trunk centre.
Base/legs: final sample of the given 30 s nominal trace (same as env_cfg EQUILIBRIUM_*).
Mesh colliders (gripper, calf, wheel) are unresolved without PhysX; for self-clearance they are
replaced by the exact convex hull of their USD points (flagged approximation). Pairs whose
collider boxes are >= 15.1 mm apart (OBB separating axis) are skipped: their distance is >= 15.1 mm.
    nominal_pose_check.py <out.json> <trace_artifacts_dir> q1 q2 q3 q4 q5 q6
"""
import copy
import csv
import itertools
import json
import sys
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
import numpy as np
from pxr import Usd, UsdGeom

k = usd_loader.t1_module("kinematics")
cl = usd_loader.t1_module("clearance")
g = usd_loader.t1_module("geometry")
OUT, TRACE = Path(sys.argv[1]), Path(sys.argv[2])
CANDIDATE = [float(x) for x in sys.argv[3:9]]
MIN_CLEAR, BOX_MARGIN, STEP = 0.015, 0.0151, 0.1
PRED = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

stage = Usd.Stage.Open(str(usd_loader.ASSET))
runtime_model = k.UsdArmKinematics(stage, "/Unitree_B2W")       # unresolved meshes, as at runtime
model = copy.copy(runtime_model)
model.shapes = {n: list(v) for n, v in runtime_model.shapes.items()}
cache = UsdGeom.XformCache()
approximated = []
for path, body in model.bodies.items():
    bprim = stage.GetPrimAtPath(path)
    binv = cache.GetLocalToWorldTransform(bprim).GetInverse()
    for i, shape in enumerate(model.shapes.get(body, [])):
        if shape.get("resolved", True):
            continue
        prim = stage.GetPrimAtPath(shape["path"])
        pts = []
        for m in Usd.PrimRange(prim, PRED):
            if m.IsA(UsdGeom.Mesh):
                rel = np.array(cache.GetLocalToWorldTransform(m) * binv).T
                pts.append(np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get()) @ rel[:3, :3].T + rel[:3, 3])
        hull = cl.polytope(np.vstack(pts))
        hull.update(source="usd_points_hull_approximation", path=shape["path"])
        model.shapes[body][i] = hull
        approximated.append(shape["path"].split("/Unitree_B2W/")[-1])

interface = json.load(open(TRACE / "interface.json"))
row = list(csv.DictReader(open(TRACE / "physics.csv")))[-1]
legs = dict(zip(interface["joint_order"], json.loads(row["joint_pos"])))
legs["jointGripper"] = 0.0
base = g.transform([0.0, 0.0, json.loads(row["base_position_world_xyz"])[2]], json.loads(row["base_quaternion_world_wxyz"]))
limits = model.limits
pairs = [(a, b) for a, b in itertools.combinations(sorted(model.shapes), 2)
         if ({a, b} & k.MOVING) and frozenset((a, b)) not in model.adjacent]


def frames_for(q):
    return model.forward(base, dict(legs, **dict(zip(model.arm_names, q))))


def min_self_clearance(frames):
    best = (np.inf, None)
    for a, b in pairs:
        for ia, sa in enumerate(model.shapes[a]):
            for ib, sb in enumerate(model.shapes[b]):
                # collider bounds are USD extents of the collider prims; they contain the hull stand-ins too
                if not k.boxes_overlap(model.collider_bounds[a][ia], frames[a], model.collider_bounds[b][ib], frames[b], BOX_MARGIN):
                    continue
                d = cl.surface_clearance(cl.moved_shape(sa, frames[a]), cl.moved_shape(sb, frames[b]))["clearance_m"]
                if d < best[0]:
                    best = (d, [a, b, sa["path"].split("/Unitree_B2W/")[-1], sb["path"].split("/Unitree_B2W/")[-1]])
    return best


cam_prim = stage.GetPrimAtPath("/Unitree_B2W/z1/link06/end_effector_camera")
cam_local = np.array(cache.GetLocalToWorldTransform(cam_prim) *
                     cache.GetLocalToWorldTransform(stage.GetPrimAtPath(next(p for p, n in model.bodies.items() if n == "link06"))).GetInverse()).T


def camera_check(frames):
    cam = frames["link06"] @ cam_local
    origin, axis = cam[:3, 3], -cam[:3, 2] / np.linalg.norm(cam[:3, 2])
    to_trunk = frames["base_link"][:3, 3] - origin
    angle = float(np.degrees(np.arccos(np.clip(axis @ to_trunk / np.linalg.norm(to_trunk), -1, 1))))
    hits = False
    for shape in model.shapes["base_link"]:                   # ray vs convex boxes (planes)
        s = cl.moved_shape(shape, frames["base_link"])
        lo, hi = 0.0, 10.0
        for n_, d_ in zip(s["planes"][:, :3], s["planes"][:, 3]):
            num, den = -(n_ @ origin + d_), n_ @ axis
            if abs(den) < 1e-12:
                if num < 0:
                    lo, hi = 1, 0
                continue
            t = num / den
            lo, hi = (max(lo, t), hi) if den < 0 else (lo, min(hi, t))
        hits |= lo <= hi
    return dict(angle_to_trunk_deg=angle, ray_hits_trunk=bool(hits), passed=bool(not hits and angle > 45.0),
                position_base_frame=(np.linalg.inv(frames["base_link"]) @ np.r_[origin, 1])[:3].tolist())


rng = np.random.default_rng(1101)                              # exact copy of targets.generate sampling
cloud = []
for q in rng.uniform(runtime_model.limits[:, 0] + 0.05, runtime_model.limits[:, 1] - 0.05, (512, 6)):
    fr = runtime_model.forward(base, dict(legs, **dict(zip(runtime_model.arm_names, q))))
    if runtime_model.collisions(fr) is None:
        cloud.append(fr["link06"][:3, 3])
low, high = np.percentile(cloud, [10, 90], axis=0)
print(f"workspace bounds from {len(cloud)} samples", flush=True)


def quintic(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def evaluate(q, transitions=True):
    q = np.asarray(q, float)
    span = limits[:, 1] - limits[:, 0]
    inner = (q >= limits[:, 0] + 0.1 * span) & (q <= limits[:, 1] - 0.1 * span)
    fr = frames_for(q)
    clearance, pair = min_self_clearance(fr)
    ee = fr["link06"][:3, 3]
    ee_base = (np.linalg.inv(fr["base_link"]) @ np.r_[ee, 1])[:3]
    result = dict(q=q.tolist(), joints_in_middle_80=dict(zip(model.arm_names, inner.tolist())),
                  pose_min_clearance_m=clearance, pose_min_pair=pair,
                  ee_world=g.pose(fr["link06"]).tolist(), ee_base_xyz=ee_base.tolist(),
                  ee_in_front=bool(ee_base[0] > 0), ee_in_workspace=bool(np.all((ee >= low) & (ee <= high))),
                  camera=camera_check(fr))
    if transitions:
        worst = (np.inf, None, None)
        for j, sign in itertools.product(range(6), (-1, 1)):
            end = q.copy()
            end[j] += sign * STEP
            for s in np.linspace(0, 1, 21):
                d, p = min_self_clearance(frames_for(q + (end - q) * quintic(s)))
                if d < worst[0]:
                    worst = (d, p, dict(joint=model.arm_names[j], sign=sign, s=float(s)))
        result.update(transition_min_clearance_m=worst[0], transition_min_pair=worst[1], transition_worst=worst[2],
                      transition_end_in_limits=bool(np.all((q - STEP >= limits[:, 0]) & (q + STEP <= limits[:, 1]))))
    result["passed"] = bool(inner.all() and clearance >= MIN_CLEAR and result["ee_in_front"] and result["ee_in_workspace"]
                            and result["camera"]["passed"]
                            and (not transitions or result["transition_min_clearance_m"] >= MIN_CLEAR))
    return result


out = dict(asset=str(usd_loader.ASSET), protocol_version=usd_loader.t1_module("protocol").VERSION,
           criteria=__doc__, approximated_mesh_colliders=approximated, workspace_samples=len(cloud),
           workspace_bounds_world_xyz=[low.tolist(), high.tolist()], base_pose_world=g.pose(base).tolist(),
           arm_limits_rad=dict(zip(model.arm_names, limits.tolist())))
out["candidate"] = evaluate(CANDIDATE)
print("candidate passed:", out["candidate"]["passed"], flush=True)
if not out["candidate"]["passed"]:
    grid = []
    for j2, j3, j4 in itertools.product(np.linspace(0.6, 1.4, 9), np.linspace(-1.8, -0.8, 11), np.linspace(0, 0.6, 7)):
        r = evaluate([CANDIDATE[0], j2, j3, j4, CANDIDATE[4], CANDIDATE[5]], transitions=False)
        grid.append(r)
    ranked = sorted(grid, key=lambda r: -r["pose_min_clearance_m"])
    eligible = [r for r in ranked if r["passed"]]
    out["sweep"] = dict(configs=len(grid), passing_without_transitions=len(eligible),
                        best3_by_clearance=[evaluate(r["q"]) for r in ranked[:3]])
    chosen = None
    for r in eligible[:10]:
        full = evaluate(r["q"])
        if full["passed"]:
            chosen = full
            break
    out["chosen"] = chosen
    print("sweep chosen:", None if chosen is None else chosen["q"], flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("written", OUT, flush=True)
