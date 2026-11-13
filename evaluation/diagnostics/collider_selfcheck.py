"""Self-clearance on the T1 asset's PhysX colliders through the T1 code path (CPU, no physics).

Uses UsdArmKinematics shapes and clearance.surface_clearance for every shape pair of
every PhysX-eligible body pair (all pairs minus jointed parent/child minus the asset's
FilteredPairs), not only broad-phase candidates. Reports resolution per pair, the
nominal minima, and joint sweeps of selected pairs.
    collider_selfcheck.py <out.json> <trace_artifacts_dir> [physics_tick]
With a tick, only that trace sample is evaluated (no sweeps).
"""
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
import numpy as np
from pxr import Usd

k = usd_loader.t1_module("kinematics")
cl = usd_loader.t1_module("clearance")
g = usd_loader.t1_module("geometry")
OUT, TRACE = Path(sys.argv[1]), Path(sys.argv[2])
TICK = int(sys.argv[3]) if len(sys.argv) > 3 else None
MARGIN = 0.005
SWEEPS = [("link02", "link06"), ("link00", "link02"), ("link04", "link06"), ("link02", "link04"),
          ("link02", "gripperStator"), ("link02", "gripperMover")]
GRID = {2: 41, 3: 17, 4: 11, 5: 7}

stage = Usd.Stage.Open(str(usd_loader.ASSET))
model = k.UsdArmKinematics(stage, "/Unitree_B2W")
jointed = {frozenset((model.bodies[e[0]], model.bodies[e[1]])) for e in model.edges}
filtered = {frozenset((t.split("/")[-1], f["path"].split("/")[-1]))
            for f in model.physics_filtering["filtered_pairs"] for t in f["targets"]}
pairs = [(a, b) for a, b in itertools.combinations(sorted(model.shapes), 2) if frozenset((a, b)) not in jointed | filtered]


def pair_min(frames, a, b):
    best, unresolved = None, []
    for sa in model.shapes[a]:
        for sb in model.shapes[b]:
            na, nb = (s["path"].split("/Unitree_B2W/")[-1] for s in (sa, sb))
            try:
                r = cl.surface_clearance(cl.moved_shape(sa, frames[a]), cl.moved_shape(sb, frames[b]))
            except ValueError as exc:
                unresolved.append(dict(a=na, b=nb, issue=str(exc)[:120]))
                continue
            if best is None or r["clearance_m"] < best["clearance_m"]:
                best = dict(a=na, b=nb, clearance_m=r["clearance_m"], method=r["method"],
                            approximation_m=r.get("approximation_m", 0.0))
    return best, unresolved


interface = json.load(open(TRACE / "interface.json"))
physics = list(csv.DictReader(open(TRACE / "physics.csv")))
row = physics[TICK - 1] if TICK else physics[-1]
joints = dict(zip(interface["joint_order"], json.loads(row["joint_pos"])))
joints.setdefault("jointGripper", 0.0)
base = g.transform(json.loads(row["base_position_world_xyz"]), json.loads(row["base_quaternion_world_wxyz"]))
nominal = model.forward(base, joints)

started = time.time()
rows, z1_unresolved, other_unresolved = [], [], []
for a, b in pairs:
    best, unresolved = pair_min(nominal, a, b)
    z1 = bool({a, b} & k.MOVING)
    (z1_unresolved if z1 else other_unresolved).extend(dict(pair=[a, b], **u) for u in unresolved)
    rows.append(dict(pair=[a, b], z1_pair=z1, fully_resolved=not unresolved, unresolved_shape_pairs=len(unresolved),
                     best=best, classification=None if best is None else cl.classification(best["clearance_m"], MARGIN)))
rows.sort(key=lambda r: np.inf if r["best"] is None else r["best"]["clearance_m"])

limits = dict(zip(model.arm_names, model.limits.tolist()))
limits["jointGripper"] = next(e[6] for e in model.edges if e[2] == "jointGripper").tolist()
parent = {model.bodies[e[1]]: (model.bodies[e[0]], e[2], e[5] is not None) for e in model.edges}


def joints_between(a, b):
    def chain(n):
        out = [n]
        while n in parent:
            n = parent[n][0]
            out.append(n)
        return out
    ca, cb = chain(a), chain(b)
    common = next(n for n in ca if n in cb)
    return [parent[n][1] for n in ca[:ca.index(common)] + cb[:cb.index(common)] if parent[n][2]]


sweeps = {}
resolvable = {n for n, v in model.shapes.items() if any(sh.get("resolved", True) for sh in v)}
skipped_sweeps = [f"{a}-{b}" for a, b in SWEEPS if not {a, b} <= resolvable]
for a, b in ([] if TICK else [p for p in SWEEPS if set(p) <= resolvable]):
    moving = joints_between(a, b)
    worst, below, count, unresolved_configs = None, 0, 0, 0
    for values in itertools.product(*[np.linspace(*limits[j], GRID[len(moving)]) for j in moving]):
        arm = {j: float(v) for j, v in zip(moving, values)}
        best, unresolved = pair_min(model.forward(base, dict(joints, **arm)), a, b)
        count += 1
        unresolved_configs += bool(unresolved)
        if best is None:          # every shape pair unresolved: nothing measurable in this config
            continue
        below += best["clearance_m"] < MARGIN
        if worst is None or best["clearance_m"] < worst["clearance_m"]:
            worst = dict(arm=arm, **best)
    sweeps[f"{a}-{b}"] = dict(joints=moving, grid_per_joint=GRID[len(moving)], configs=count,
                              configs_below_5mm=int(below), configs_with_unresolved=int(unresolved_configs), worst=worst)
    print(f"{a}-{b}: {time.time() - started:.0f}s", flush=True)

result = dict(asset=str(usd_loader.ASSET), asset_hashes=usd_loader.t1_module("protocol").asset_hashes(str(usd_loader.ASSET)),
              protocol_version=usd_loader.t1_module("protocol").VERSION, threshold_m=MARGIN,
              state=dict(trace=str(TRACE), physics_tick=row["physics_tick"], time_s=row["time_s"]),
              pairs_checked=len(pairs), z1_pairs=sum(r["z1_pair"] for r in rows),
              z1_pairs_fully_resolved=sum(r["z1_pair"] and r["fully_resolved"] for r in rows),
              other_pairs_fully_resolved=sum(not r["z1_pair"] and r["fully_resolved"] for r in rows),
              z1_unresolved=z1_unresolved, other_unresolved_examples=other_unresolved[:10],
              shape_sources={n: sorted({s.get("source", s["kind"]) for s in v}) for n, v in model.shapes.items()},
              below_threshold=[r for r in rows if r["classification"] not in (None, "broad_phase_false_positive")],
              closest=rows[:10], sweeps=sweeps,
              skipped_sweeps_unresolved_geometry=skipped_sweeps, wall_s=time.time() - started)
json.dump(result, open(OUT, "w"), indent=1, default=float)
print("written", OUT, flush=True)
