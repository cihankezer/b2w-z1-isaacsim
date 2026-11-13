"""Per-pair minimum collider self-clearance over a saved diagnose_t1_standing trace (CPU, T1 path).

Control-rate samples (every 4th physics tick). Pairs: non-adjacent pairs with a Z1/gripper body.
Shape pairs whose collider boxes are >= WINDOW_M apart (OBB) are not evaluated; a pair with no
evaluated shape pair in a window is reported as ">= WINDOW_M". Mesh colliders (gripper, legs)
use the exact convex hull of their USD points (flagged approximation; PhysX cooking unreadable).
    trace_selfclearance.py <out.json> <run_artifacts_dir>
"""
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
OUT, RUN = Path(sys.argv[1]), Path(sys.argv[2])
WINDOW_M = 0.05
PRED = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

stage = Usd.Stage.Open(str(usd_loader.ASSET))
model = k.UsdArmKinematics(stage, "/Unitree_B2W")
cache = UsdGeom.XformCache()
approximated = []
for path, body in model.bodies.items():
    binv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).GetInverse()
    for i, shape in enumerate(model.shapes.get(body, [])):
        if shape.get("resolved", True):
            continue
        pts = [np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get()) @ np.array(cache.GetLocalToWorldTransform(m) * binv).T[:3, :3].T
               + np.array(cache.GetLocalToWorldTransform(m) * binv).T[:3, 3]
               for m in Usd.PrimRange(stage.GetPrimAtPath(shape["path"]), PRED) if m.IsA(UsdGeom.Mesh)]
        model.shapes[body][i] = dict(cl.polytope(np.vstack(pts)), source="usd_points_hull_approximation", path=shape["path"])
        approximated.append(shape["path"].split("/Unitree_B2W/")[-1])
pairs = [(a, b) for a, b in itertools.combinations(sorted(model.shapes), 2)
         if ({a, b} & k.MOVING) and frozenset((a, b)) not in model.adjacent]

interface = json.load(open(RUN / "interface.json"))
order = interface["joint_order"]
stats = {f"{a}:{b}": {"0-2s": None, "2-30s": None} for a, b in pairs}
samples = 0
for row in csv.DictReader(open(RUN / "physics.csv")):
    tick = int(row["physics_tick"])
    if tick % 4:
        continue
    samples += 1
    t = float(row["time_s"])
    window = "0-2s" if t <= 2.0 else "2-30s"
    frames = model.forward(g.transform(json.loads(row["base_position_world_xyz"]), json.loads(row["base_quaternion_world_wxyz"])),
                           dict(zip(order, json.loads(row["joint_pos"]))))
    for a, b in pairs:
        for ia, sa in enumerate(model.shapes[a]):
            for ib, sb in enumerate(model.shapes[b]):
                if not k.boxes_overlap(model.collider_bounds[a][ia], frames[a], model.collider_bounds[b][ib], frames[b], WINDOW_M):
                    continue
                d = cl.surface_clearance(cl.moved_shape(sa, frames[a]), cl.moved_shape(sb, frames[b]))["clearance_m"]
                cur = stats[f"{a}:{b}"][window]
                if cur is None or d < cur["clearance_m"]:
                    stats[f"{a}:{b}"][window] = dict(clearance_m=d, time_s=t, shape_a=sa["path"].split("/Unitree_B2W/")[-1],
                                                     shape_b=sb["path"].split("/Unitree_B2W/")[-1])
measured = {p: v for p, v in stats.items() if v["0-2s"] or v["2-30s"]}
json.dump(dict(run=str(RUN), asset=str(usd_loader.ASSET), control_samples=samples, window_m=WINDOW_M,
               approximated_mesh_colliders=approximated, pairs=len(pairs),
               pairs_beyond_window=len(pairs) - len(measured), measured=measured),
          open(OUT, "w"), indent=1, default=float)
print("written", OUT, flush=True)
