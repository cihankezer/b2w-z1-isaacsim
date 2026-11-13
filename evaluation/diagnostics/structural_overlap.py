"""CPU-only T1 structural-overlap and unresolved-mesh audit (no SimulationApp, no physics).

Task 2: for each flagged arm pair, joints between the bodies, collider surface
clearance at the settled nominal pose and over a grid of those joints within the
USD limits; same for visual meshes (vertex-based, with a stated error bound).
Task 3: whether the OBB broad phase ever selects a leg/wheel mesh collider over
the saved 30 s trace and the task-2 sweeps; readability of those meshes.

Uses the unchanged standing_ee FK/clearance code. Writes JSON only.
    structural_overlap.py <out.json> <trace_artifacts_dir>
"""
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull, cKDTree

k = usd_loader.t1_module("kinematics")
c = usd_loader.t1_module("clearance")
g = usd_loader.t1_module("geometry")
OUT, TRACE = Path(sys.argv[1]), Path(sys.argv[2])
PRED = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
MARGIN = 0.005
PAIRS = [("link02", "link06"), ("link00", "link02"), ("link04", "link06"), ("link02", "link04")]
GRID = {2: 41, 4: 11}          # collider sweep points per joint
VISUAL_GRID = {2: 9, 4: 5}     # visual sweep points per joint
MESH_BODIES = [f"{l}_{p}" for l in ("FL", "FR", "RL", "RR") for p in ("calf", "foot")]

stage = Usd.Stage.Open(str(usd_loader.ASSET))
model = k.UsdArmKinematics(stage, "/Unitree_B2W")
path_of = {name: path for path, name in model.bodies.items()}
limits = dict(zip(model.arm_names, model.limits.tolist()))

# ---- nominal pose: final sample of the saved 30 s nominal-PD trace ----------
interface = json.load(open(TRACE / "interface.json"))
order = interface["joint_order"]
physics = list(csv.DictReader(open(TRACE / "physics.csv")))


def row_state(row):
    base = g.transform(json.loads(row["base_position_world_xyz"]), json.loads(row["base_quaternion_world_wxyz"]))
    return base, dict(zip(order, json.loads(row["joint_pos"])))


nominal_base, nominal_joints = row_state(physics[-1])


def joints_between(a, b):
    parent = {model.bodies[e[1]]: (model.bodies[e[0]], e[2], e[5] is not None) for e in model.edges}
    def chain(n):
        out = [n]
        while n in parent:
            n = parent[n][0]
            out.append(n)
        return out
    ca, cb = chain(a), chain(b)
    common = next(n for n in ca if n in cb)
    names = []
    for n in ca[:ca.index(common)] + cb[:cb.index(common)]:
        names.append(dict(joint=parent[n][1], child=n, revolute=parent[n][2]))
    return names


def collider_clearances(frames, a, b):
    rows = []
    for sa in model.shapes.get(a, []):
        for sb in model.shapes.get(b, []):
            try:
                d = c.surface_clearance(c.moved_shape(sa, frames[a]), c.moved_shape(sb, frames[b]))["clearance_m"]
            except ValueError:
                d = None
            rows.append((sa["path"].split("/collisions/")[-1], sb["path"].split("/collisions/")[-1], d))
    return rows


# ---- visual meshes (body frame) ---------------------------------------------
def under_collider(p):
    while p.IsValid():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            return True
        p = p.GetParent()
    return False


def owner(p):
    while p.IsValid() and not p.HasAPI(UsdPhysics.RigidBodyAPI):
        p = p.GetParent()
    return p


cache = UsdGeom.XformCache()


def mesh_in_body(mesh, body):
    m = UsdGeom.Mesh(mesh)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
    tris, start = [], 0
    for n in counts:                      # fan triangulation
        tris += [(idx[start], idx[start + i], idx[start + i + 1]) for i in range(1, n - 1)]
        start += n
    rel = np.array(cache.GetLocalToWorldTransform(mesh) * cache.GetLocalToWorldTransform(body).GetInverse()).T
    return pts @ rel[:3, :3].T + rel[:3, 3], np.asarray(tris)


def triangle_cover_radius(v, t):
    """Max distance from any point of each triangle to its nearest vertex."""
    a, b, cc = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (b - cc, cc - a, a - b))
    area2 = np.linalg.norm(np.cross(b - a, cc - a), axis=1)
    circum = la * lb * lc / np.maximum(2 * area2, 1e-30)
    l = np.sort(np.c_[la, lb, lc], axis=1)
    obtuse = l[:, 2] ** 2 >= l[:, 0] ** 2 + l[:, 1] ** 2
    return np.where(obtuse, l[:, 2] / 2, circum)


SAMPLE_S = 0.001     # surface sample spacing; any surface point is within S/sqrt(3) of a sample
REGION_M = 0.03      # only triangles within this box margin of the other mesh are sampled


def sample_triangles(v, t):
    """Barycentric grid on each triangle with sub-edges <= SAMPLE_S; returns points and source triangle ids."""
    a, b, cc = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    longest = np.max(np.c_[np.linalg.norm(b - a, axis=1), np.linalg.norm(cc - b, axis=1), np.linalg.norm(a - cc, axis=1)], axis=1)
    n = np.maximum(1, np.ceil(longest / SAMPLE_S)).astype(int)
    pts, ids = [], []
    for level in np.unique(n):
        sel = np.nonzero(n == level)[0]
        ij = np.array([(i, j) for i in range(level + 1) for j in range(level + 1 - i)], float) / level
        w = np.c_[1 - ij.sum(1), ij]
        pts.append(np.einsum("sw,twk->tsk", w, np.stack([a[sel], b[sel], cc[sel]], 1)).reshape(-1, 3))
        ids.append(np.repeat(sel, len(w)))
    return np.vstack(pts), np.concatenate(ids)


visual = {}
for name in {n for p in PAIRS for n in p}:
    body = stage.GetPrimAtPath(path_of[name])
    meshes = [p for p in Usd.PrimRange(body, PRED) if p.IsA(UsdGeom.Mesh) and not under_collider(p) and owner(p) == body]
    vs, ts, off = [], [], 0
    for mesh in meshes:
        v, t = mesh_in_body(mesh, body)
        vs.append(v); ts.append(t + off); off += len(v)
    v, t = np.vstack(vs), np.vstack(ts)
    uniq, inv = np.unique(np.round(v, 9), axis=0, return_inverse=True)
    t = inv.reshape(-1)[t]
    visual[name] = dict(v=uniq, t=t, h=triangle_cover_radius(uniq, t), tree=cKDTree(uniq),
                        meshes=[str(m.GetPath()) for m in meshes], triangles=len(t))


def region_samples(name, frame_to_other, other):
    """Samples of `name`'s triangles whose boxes touch the other mesh's box (+REGION_M), in the other's frame."""
    X = visual[name]
    v = X["v"] @ frame_to_other[:3, :3].T + frame_to_other[:3, 3]
    lo, hi = visual[other]["v"].min(0) - REGION_M, visual[other]["v"].max(0) + REGION_M
    tv = v[X["t"]]
    keep = np.all((tv.max(1) >= lo) & (tv.min(1) <= hi), axis=1)
    if not keep.any():
        return np.zeros((0, 3))
    return sample_triangles(v, X["t"][keep])[0]


def winding(points, v, t):
    """Generalized winding number of points w.r.t. a triangle soup (Jacobson et al. 2013)."""
    out = np.zeros(len(points))
    for s in range(0, len(points), 64):
        p = points[s:s + 64, None, :]
        a, b, cc = v[t[:, 0]] - p, v[t[:, 1]] - p, v[t[:, 2]] - p
        na, nb, nc = (np.linalg.norm(x, axis=2) for x in (a, b, cc))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, cc))
        den = na * nb * nc + np.einsum("ijk,ijk->ij", a, b) * nc + np.einsum("ijk,ijk->ij", b, cc) * na \
            + np.einsum("ijk,ijk->ij", cc, a) * nb
        out[s:s + 64] = (2 * np.arctan2(det, den)).sum(axis=1) / (4 * np.pi)
    return out


def visual_clearance(frames, a, b, inside_test):
    """Sampled-surface distance between visual meshes.

    d_sample is an upper bound of the true surface distance; the true distance is
    >= d_sample - 2*SAMPLE_S/sqrt(3) whenever d_sample <= REGION_M. Interpenetration
    is tested separately with the generalized winding number on the sampled points.
    """
    A, B = visual[a], visual[b]
    T = np.linalg.inv(frames[b]) @ frames[a]           # A body frame -> B body frame
    Ti = np.linalg.inv(T)
    sa = region_samples(a, T, b)                       # in B frame
    sb = region_samples(b, Ti, a)                      # in A frame
    if not len(sa) or not len(sb):
        return dict(sample_distance_m=None, lower_bound_m=REGION_M, note=f"no triangles within {REGION_M} m box margin")
    sb_in_b = sb @ T[:3, :3].T + T[:3, 3]
    d = float(cKDTree(sb_in_b).query(sa)[0].min())
    res = dict(sample_distance_m=d, lower_bound_m=d - 2 * SAMPLE_S / np.sqrt(3) if d <= REGION_M else REGION_M,
               samples_a=len(sa), samples_b=len(sb))
    if inside_test and res["lower_bound_m"] <= 0:
        # winding-number inside test on the 1000 samples nearest the other surface plus 500 random ones
        rng = np.random.default_rng(0)
        tree_b = cKDTree(sb_in_b)
        sa_in_a = sa @ Ti[:3, :3].T + Ti[:3, 3]

        def pick(points, dist):
            order = np.argsort(dist)
            return points[np.unique(np.r_[order[:1000], rng.permutation(len(points))[:500]])]
        cand_a = pick(sa, tree_b.query(sa)[0])                         # B frame
        cand_b = pick(sb, cKDTree(sa_in_a).query(sb)[0])               # A frame
        wa = winding(cand_a, B["v"], B["t"])
        wb = winding(cand_b, A["v"], A["t"])
        # depth proxy: farthest inside sample from the other sampled surface
        depth = [0.0]
        if (wa > 0.5).any():
            depth += list(tree_b.query(cand_a[wa > 0.5])[0])
        if (wb > 0.5).any():
            depth += list(cKDTree(sa_in_a).query(cand_b[wb > 0.5])[0])
        res.update(inside_a_in_b=int((wa > 0.5).sum()), tested_a=len(wa), inside_b_in_a=int((wb > 0.5).sum()),
                   tested_b=len(wb), approx_depth_m=float(max(depth)))
    return res


def frames_for(arm):
    return model.forward(nominal_base, dict(nominal_joints, **arm))


# ---- task 3 helper ------------------------------------------------------------
def mesh_candidates(frames):
    hits = []
    for row in model.clearance_report(frames, MARGIN):
        paths = [row["shape_a"], row["shape_b"]]
        if any(f"/{m}/collisions/" in p for p in paths for m in MESH_BODIES):
            hits.append(dict(pair=row["pair"], classification=row["classification"]))
    return hits


# ---- task 2 ---------------------------------------------------------------------
started = time.time()
result = dict(asset=str(usd_loader.ASSET), asset_sha256=usd_loader.t1_module("protocol").asset_hashes(str(usd_loader.ASSET)),
              nominal_source=dict(trace=str(TRACE), physics_tick=physics[-1]["physics_tick"],
                                  arm={n: nominal_joints[n] for n in model.arm_names}),
              arm_limits_rad=limits, margin_m=MARGIN, pairs={}, task3={})
sweep_mesh_hits = []
for a, b in PAIRS:
    between = joints_between(a, b)
    moving = [j["joint"] for j in between if j["revolute"]]
    nominal_frames = frames_for({})
    entry = dict(joints_between=between, nominal=dict(colliders=collider_clearances(nominal_frames, a, b),
                                                       visual=visual_clearance(nominal_frames, a, b, True)))
    # collider sweep
    n = GRID[len(moving)]
    axes = [np.linspace(*limits[j], n) for j in moving]
    stats = {}
    for values in itertools.product(*axes):
        arm = dict(zip(moving, values))
        frames = frames_for(arm)
        for sa, sb, d in collider_clearances(frames, a, b):
            s = stats.setdefault(f"{sa}|{sb}", dict(min=np.inf, max=-np.inf, n=0, n_pen=0, n_viol=0, n_unres=0))
            s["n"] += 1
            if d is None:
                s["n_unres"] += 1
                continue
            if d < s["min"]:
                s.update(min=d, argmin={j: float(v) for j, v in arm.items()})
            if d > s["max"]:
                s.update(max=d, argmax={j: float(v) for j, v in arm.items()})
            s["n_pen"] += d < -c.NUMERICAL_TOLERANCE_M
            s["n_viol"] += -c.NUMERICAL_TOLERANCE_M <= d < MARGIN - c.NUMERICAL_TOLERANCE_M
        hits = mesh_candidates(frames)
        if hits:
            sweep_mesh_hits.append(dict(pair=[a, b], arm=arm, hits=hits))
    for s in stats.values():
        s["min"], s["max"] = float(s["min"]), float(s["max"])
        s["config_invariant"] = bool(s["max"] - s["min"] < 1e-9)
    entry["collider_sweep"] = dict(grid_per_joint=n, joints=moving, configs=n ** len(moving), per_collider_pair=stats)
    # visual at each collider pair's argmin, and a coarse visual sweep
    entry["visual_at_collider_argmin"] = {
        key: dict(arm=s["argmin"], visual=visual_clearance(frames_for(s["argmin"]), a, b, True))
        for key, s in stats.items() if "argmin" in s}
    nv = VISUAL_GRID[len(moving)]
    vmin, worst, uncertain, inside = np.inf, None, [], []
    for values in itertools.product(*[np.linspace(*limits[j], nv) for j in moving]):
        arm = {j: float(v) for j, v in zip(moving, values)}
        r = visual_clearance(frames_for(arm), a, b, True)
        if r["lower_bound_m"] < vmin:
            vmin, worst = r["lower_bound_m"], dict(arm=arm, **r)
        if r["lower_bound_m"] <= 0:
            uncertain.append(dict(arm=arm, **r))
            if r.get("inside_a_in_b", 0) + r.get("inside_b_in_a", 0):
                inside.append(dict(arm=arm, **r))
    entry["visual_sweep"] = dict(grid_per_joint=nv, configs=nv ** len(moving), min_lower_bound_m=float(vmin),
                                 worst=worst, configs_not_certified_separate=len(uncertain),
                                 configs_with_interpenetration=len(inside), interpenetration_examples=inside[:5])
    entry["visual_meshes"] = {n: dict(paths=visual[n]["meshes"], triangles=visual[n]["triangles"]) for n in (a, b)}
    result["pairs"][f"{a}-{b}"] = entry
    print(f"{a}-{b} done at {time.time() - started:.0f}s", flush=True)

# ---- task 3 ---------------------------------------------------------------------
trace_hits, trace_pairs = [], {}
for row in physics:
    base, joints = row_state(row)
    frames = model.forward(base, joints)
    hits = mesh_candidates(frames)
    if hits:
        trace_hits.append(dict(physics_tick=row["physics_tick"], hits=hits))
    for a, b in PAIRS:      # all collider combinations, not only broad-phase selected ones
        for sa, sb, d in collider_clearances(frames, a, b):
            s = trace_pairs.setdefault(f"{a}-{b}:{sa}|{sb}", dict(min=np.inf, max=-np.inf, n_pen=0, n_viol=0))
            if d < s["min"]:
                s.update(min=d, tick=int(row["physics_tick"]), time_s=float(row["time_s"]),
                         arm={n: joints[n] for n in model.arm_names})
            s["max"] = max(s["max"], d)
            s["n_pen"] += d < -c.NUMERICAL_TOLERANCE_M
            s["n_viol"] += -c.NUMERICAL_TOLERANCE_M <= d < MARGIN - c.NUMERICAL_TOLERANCE_M
for (a, b) in PAIRS:        # visual clearance at each pair's worst trace sample
    for key, s in trace_pairs.items():
        if key.startswith(f"{a}-{b}:"):
            row = physics[s["tick"] - 1] if int(physics[s["tick"] - 1]["physics_tick"]) == s["tick"] else \
                next(r for r in physics if int(r["physics_tick"]) == s["tick"])
            base, joints = row_state(row)
            s["visual_at_min"] = visual_clearance(model.forward(base, joints), a, b, True)
result["trace_pairs"] = trace_pairs
meshes = {}
for name in MESH_BODIES:
    body = stage.GetPrimAtPath(path_of[name])
    coll = [p for p in Usd.PrimRange(body, PRED) if p.HasAPI(UsdPhysics.CollisionAPI)]
    for cp in coll:
        mesh_prims = [p for p in Usd.PrimRange(cp, PRED) if p.IsA(UsdGeom.Mesh)]
        v = np.vstack([mesh_in_body(m, body)[0] for m in mesh_prims])
        hull = ConvexHull(np.unique(np.round(v, 9), axis=0))
        api = UsdPhysics.MeshCollisionAPI(cp)
        meshes[name] = dict(collision_prim=str(cp.GetPath()), collision_prim_type=cp.GetTypeName(),
                            mesh_prims=[str(m.GetPath()) for m in mesh_prims],
                            mesh_prims_have_collision_api=[m.HasAPI(UsdPhysics.CollisionAPI) for m in mesh_prims],
                            approximation=api.GetApproximationAttr().Get() if cp.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
                            physx_attrs={a.GetName(): str(a.Get()) for a in cp.GetAttributes() if a.GetName().startswith("physx")},
                            points=len(v), exact_hull_vertices=len(hull.vertices), exact_hull_volume_m3=float(hull.volume),
                            body_frame_extent_m=[v.min(0).tolist(), v.max(0).tolist()])
result["task3"] = dict(trace_samples=len(physics), trace_samples_with_mesh_candidates=len(trace_hits),
                       trace_examples=trace_hits[:5], sweep_configs_with_mesh_candidates=len(sweep_mesh_hits),
                       sweep_examples=sweep_mesh_hits[:5], meshes=meshes)
result["wall_s"] = time.time() - started


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


json.dump(clean(result), open(OUT, "w"), indent=1)
print("written", OUT, flush=True)
