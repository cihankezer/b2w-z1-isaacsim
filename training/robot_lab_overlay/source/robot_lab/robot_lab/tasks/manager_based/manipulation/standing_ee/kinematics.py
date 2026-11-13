"""USD-joint FK/IK and physical-surface clearance auditing.

The existing composed USD is authoritative. OBBs only select candidate pairs;
physical surfaces decide penetration or a clearance below 5 mm. Unsupported
geometry is explicitly unresolved, never classified from bounding boxes alone.
Clearance policy draft.5/7: ground and arm self-clearance both use the PhysX collider
surfaces; ground rows terminate, self rows are a logged 5 mm monitor (PhysX contact
terminates). Uncookable mesh colliders use their USD-point hull; such rows carry
geometry="approx_hull" (draft.7). MOVING = Z1 links and gripper bodies; exclusions unchanged.
This module does not step physics or write the USD stage.
"""

import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import transform

MOVING = {*(f"link0{i}" for i in range(1,7)), "gripperStator", "gripperMover"}
from .clearance import (classification, ground_clearance, moved_shape, surface_clearance, usd_shapes, NUMERICAL_TOLERANCE_M)


def corners(bounds):
    return np.array(list(itertools.product(*zip(*bounds))))


def moved_bounds(bounds, matrix):
    points = corners(bounds) @ matrix[:3, :3].T + matrix[:3, 3]
    return np.array([points.min(axis=0), points.max(axis=0)])


def boxes_overlap(bounds_a, frame_a, bounds_b, frame_b, margin):
    """Conservative oriented-box test using all 15 separating axes.

    Keeps the clearance margin in metres, including on normalized cross axes.
    These boxes still enclose primitives/meshes; overlap is only a candidate.
    """
    points_a = corners(bounds_a) @ frame_a[:3, :3].T + frame_a[:3, 3]
    points_b = corners(bounds_b) @ frame_b[:3, :3].T + frame_b[:3, 3]
    axes_a, axes_b = frame_a[:3, :3].T, frame_b[:3, :3].T
    axes = [*axes_a, *axes_b, *(np.cross(a, b) for a in axes_a for b in axes_b)]
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        a, b = points_a @ (axis/norm), points_b @ (axis/norm)
        if max(a.min(), b.min()) >= min(a.max(), b.max()) + margin:
            return False
    return True


class UsdArmKinematics:
    def __init__(self, stage, root_path):
        from pxr import Usd, UsdGeom, UsdPhysics

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid() or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise ValueError("Expected an existing metre-scale robot")
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
        self.bodies = {str(p.GetPath()): p.GetName() for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        if len(set(self.bodies.values())) != len(self.bodies):
            raise ValueError("Ambiguous rigid-body names")
        self.edges, self.adjacent = [], set()
        for p in prims:
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            parents, children = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if len(parents) != 1 or len(children) != 1:
                raise ValueError(f"Joint must connect two robot bodies: {p.GetPath()}")
            parent, child = str(parents[0]), str(children[0])
            if parent not in self.bodies or child not in self.bodies:
                raise ValueError(f"External/fixed-world articulation joint: {p.GetPath()}")
            def frame(pos, quat):
                return transform(list(pos), [quat.GetReal(), *quat.GetImaginary()])
            local0 = frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
            local1 = frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
            axis = None
            limits = None
            if p.IsA(UsdPhysics.RevoluteJoint):
                r = UsdPhysics.RevoluteJoint(p)
                axis = np.eye(3)["XYZ".index(r.GetAxisAttr().Get())]
                limits = np.deg2rad([r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()])
            elif not p.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f"Unsupported joint {p.GetPath()}")
            self.edges.append((parent, child, p.GetName(), local0, local1, axis, limits))
            self.adjacent.add(frozenset((self.bodies[parent], self.bodies[child])))
        roots = set(self.bodies) - {e[1] for e in self.edges}
        if len(roots) != 1 or self.bodies[next(iter(roots))] != "base_link":
            raise ValueError("Expected a single floating base_link kinematic tree")
        self.root = next(iter(roots))
        self.arm_names = [f"joint{i}" for i in range(1, 7)]
        self.limits = np.array([next(e[6] for e in self.edges if e[2] == n) for n in self.arm_names])
        # Known intentional mount/sensor overlaps; never ignore moving arm pairs.
        self.adjacent |= {frozenset(("base_link", "link00")), frozenset(("head_Link", "lidar_link"))}
        self.physics_filtering = {"articulation_self_collisions": [], "filtered_pairs": []}
        for p in prims:
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                attr = p.GetAttribute("physxArticulation:enabledSelfCollisions")
                self.physics_filtering["articulation_self_collisions"].append({
                    "path":str(p.GetPath()),"enabled":attr.Get() if attr else None})
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                self.physics_filtering["filtered_pairs"].append({"path":str(p.GetPath()),
                    "targets":[str(t) for t in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()]})
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"],False,True)
        cache = UsdGeom.XformCache()
        self.bounds, self.collider_bounds, self.bounds_source, self.shapes = {}, {}, {}, {}
        for path,name in self.bodies.items():
            body = stage.GetPrimAtPath(path)
            geometry = []
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                owner = p
                while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                if owner == body and p.HasAPI(UsdPhysics.CollisionAPI):
                    if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False:
                        geometry.append(p)
            self.bounds_source[name] = "physical_collider" if geometry else "no_physical_collider"
            parts,shapes = [],[]
            for p in geometry:
                box = bbox.ComputeRelativeBound(p, body).ComputeAlignedRange()
                if box.IsEmpty():
                    raise ValueError(f"Physical collider has no bound: {p.GetPath()}")
                bound = np.array([list(box.GetMin()),list(box.GetMax())])
                for shape in usd_shapes(stage,p,body,cache):
                    # Cooked hulls may differ from the USD extent; broad bounds must cover them.
                    if shape["kind"] == "polytope":
                        vertices = shape["vertices"]
                        shape_bound = np.array([vertices.min(axis=0),vertices.max(axis=0)])
                    else:
                        shape_bound = bound
                    parts.append(shape_bound);shapes.append(shape)
            if parts:
                self.collider_bounds[name],self.shapes[name] = parts,shapes
                self.bounds[name] = np.array([np.min([b[0] for b in parts],axis=0),np.max([b[1] for b in parts],axis=0)])

    def forward(self, base, joints):
        frames = {self.root: base}
        pending = list(self.edges)
        while pending:
            ready = [e for e in pending if e[0] in frames]
            if not ready:
                raise ValueError("Disconnected or cyclic USD joint tree")
            for edge in ready:
                parent, child, name, local0, local1, axis, _ = edge
                motion = np.eye(4)
                if axis is not None:
                    motion[:3, :3] = Rotation.from_rotvec(axis * joints[name]).as_matrix()
                frames[child] = frames[parent] @ local0 @ motion @ np.linalg.inv(local1)
            ready_ids = {id(edge) for edge in ready}
            pending = [edge for edge in pending if id(edge) not in ready_ids]
        return {self.bodies[path]: value for path, value in frames.items()}

    def clearance_report(self, frames, margin=0.005):
        """All broad candidates, including false positives and unresolved geometry.

        Geometric exclusions are preserved independently of PhysX's global
        self-collision flag. Filtered overlaps therefore remain visible here.
        """
        bounds = {n:moved_bounds(b,frames[n]) for n,b in self.bounds.items()}
        world = {n:[moved_shape(s,frames[n]) for s in self.shapes[n]] for n in bounds}
        reports = []
        def measure(pair,a,b=None):
            row = dict(pair=pair,shape_a=a["path"],shape_b=b["path"] if b else "ground",
                       minimum_clearance_m=margin,numerical_tolerance_m=NUMERICAL_TOLERANCE_M,
                       geometry="approx_hull" if any(s.get("source") == "approx_hull" for s in (a,b) if s) else "collider")
            try:
                measured = surface_clearance(a,b) if b else ground_clearance(a)
                row.update(measured,classification=classification(measured["clearance_m"],margin))
            except ValueError as exc:
                row.update(classification="unresolved",clearance_m=None,issue=str(exc))
            reports.append(row)
        for name in bounds:
            if name.endswith("_foot"):
                continue
            for bound,shape in zip(self.collider_bounds[name],world[name]):
                if moved_bounds(bound,frames[name])[0,2] < margin:
                    measure([name,"ground"],shape)
        for a,b in itertools.combinations(bounds,2):
            if not ({a,b}&MOVING) or frozenset((a,b)) in self.adjacent:
                continue
            lo,hi = np.maximum(bounds[a][0],bounds[b][0]),np.minimum(bounds[a][1],bounds[b][1])
            if not np.all(hi+margin > lo):
                continue
            for ia,ib in itertools.product(range(len(world[a])),range(len(world[b]))):
                if boxes_overlap(self.collider_bounds[a][ia],frames[a],self.collider_bounds[b][ib],frames[b],margin):
                    measure([a,b],world[a][ia],world[b][ib])
        return reports

    def self_clearance_monitor(self, rows):
        """Per self pair: minimum collider surface clearance and its classification (monitor only)."""
        merged = {}
        for row in rows:
            if row["pair"][1] == "ground":
                continue
            key = tuple(sorted(row["pair"]))
            entry = merged.setdefault(key,dict(pair=list(key),clearance_m=None,classification=None,
                                              shape_a=None,shape_b=None,geometry=None,unresolved=0))
            d = row.get("clearance_m")
            if d is None:
                entry["unresolved"] += 1
                entry["classification"] = entry["classification"] or row["classification"]
            elif entry["clearance_m"] is None or d < entry["clearance_m"]:
                entry.update(clearance_m=d,classification=row["classification"],shape_a=row["shape_a"],shape_b=row["shape_b"],
                             geometry=row["geometry"])
        return list(merged.values())

    def collisions(self, frames, margin=0.005):
        """Feasibility rejection: any ground or self collider row below the margin, or unresolved."""
        for row in self.clearance_report(frames,margin):
            if row["classification"] != "broad_phase_false_positive":
                a,b = row["pair"]
                pair = f"ground:{a}" if b == "ground" else f"self:{a}:{b}"
                return row["classification"]+":"+pair
        return None

    def solve(self, desired, base, joints, initial=None, margin_rad=0.05):
        lo, hi = self.limits[:, 0] + margin_rad, self.limits[:, 1] - margin_rad
        q0 = np.array([joints[n] for n in self.arm_names]) if initial is None else initial
        def residual(q):
            values = dict(joints, **dict(zip(self.arm_names, q)))
            actual = self.forward(base, values)["link06"]
            return np.r_[actual[:3, 3] - desired[:3, 3],
                         Rotation.from_matrix(desired[:3, :3].T @ actual[:3, :3]).as_rotvec()]
        solved = least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=150,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        err = residual(solved.x)
        if np.linalg.norm(err[:3]) > 0.001 or np.linalg.norm(err[3:]) > 0.01:
            return None, "ik_residual"
        reason = self.collisions(self.forward(base, dict(joints, **dict(zip(self.arm_names, solved.x)))))
        return (None, reason) if reason else (solved.x, None)
