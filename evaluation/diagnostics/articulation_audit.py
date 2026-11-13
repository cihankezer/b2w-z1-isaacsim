"""Read-only articulation audit of the composed T1 robot stage (all references/payloads loaded).

    articulation_audit.py <out.json> [usd ...]   default: the T1 asset (usd_loader.ASSET)
Lists every prim with ArticulationRootAPI; every RigidBodyAPI prim outside the articulation (not
reachable from the root body through enabled, non-excluded joints); the prim(s)
named "world" with their applied schemas and both bodies of every joint touching them.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
from pxr import Usd, UsdPhysics

out = Path(sys.argv[1])
paths = [Path(a) for a in sys.argv[2:]] or [usd_loader.ASSET]


def instancing(prim):
    """Instance-proxy status and the nearest instanceable ancestor (Isaac Lab's root search skips proxies)."""
    anc = prim
    while anc.IsValid() and not anc.IsInstance():
        anc = anc.GetParent()
    return dict(is_instance_proxy=prim.IsInstanceProxy(), instance_ancestor=str(anc.GetPath()) if anc.IsValid() else None)


def audit(path):
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
    bodies = {str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints, graph = [], defaultdict(set)
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0 = [str(x) for x in j.GetBody0Rel().GetTargets()]
        b1 = [str(x) for x in j.GetBody1Rel().GetTargets()]
        enabled = j.GetJointEnabledAttr().Get()
        excluded = j.GetExcludeFromArticulationAttr().Get()
        joints.append(dict(path=str(p.GetPath()), type=p.GetTypeName(), body0=b0, body1=b1, **instancing(p),
                           enabled=enabled, exclude_from_articulation=excluded))
        if enabled is not False and not excluded and b0 and b1:
            graph[b0[0]].add(b1[0]); graph[b1[0]].add(b0[0])
    roots = []
    for p in prims:
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            root = str(p.GetPath())
            start = root if root in bodies else next((b for b in sorted(bodies) if b.startswith(root + "/")), None)
            seen, todo = set(), [start] if start else []
            while todo:
                b = todo.pop()
                if b not in seen:
                    seen.add(b); todo.extend(graph[b])
            roots.append(dict(path=root, type=p.GetTypeName(), applied_schemas=list(p.GetAppliedSchemas()), **instancing(p),
                              is_rigid_body=root in bodies, start_body=start, connected_bodies=len(seen & bodies),
                              _seen=seen))
    reach = set().union(*(r.pop("_seen") for r in roots)) if roots else set()
    outside = []
    for b in sorted(bodies):
        prim = stage.GetPrimAtPath(b)
        in_subtree = any(b == r["path"] or b.startswith(r["path"] + "/") for r in roots)
        if b not in reach:   # PhysX builds the articulation from the joint graph, not the prim subtree
            outside.append(dict(path=b, reachable_from_root=b in reach, under_root_subtree=in_subtree,
                                kinematic=prim.GetAttribute("physics:kinematicEnabled").Get(),
                                rigid_body_enabled=prim.GetAttribute("physics:rigidBodyEnabled").Get(),
                                joints=[j["path"] for j in joints if b in j["body0"] + j["body1"]]))
    world = []
    for p in prims:
        if p.GetName().lower() == "world":
            path = str(p.GetPath())
            world.append(dict(path=path, type=p.GetTypeName(), applied_schemas=list(p.GetAppliedSchemas()), **instancing(p),
                              rigid_body=p.HasAPI(UsdPhysics.RigidBodyAPI), collision=p.HasAPI(UsdPhysics.CollisionAPI),
                              mass=p.HasAPI(UsdPhysics.MassAPI), children=[c.GetName() for c in p.GetChildren()],
                              joints=[j for j in joints if path in j["body0"] + j["body1"]]))
    return dict(usd=str(path), default_prim=str(stage.GetDefaultPrim().GetPath()), prims=len(prims),
                rigid_bodies=len(bodies), joints=len(joints), articulation_roots=roots,
                rigid_bodies_outside_articulation=outside, world_prims=world,
                joints_with_one_or_no_body=[j for j in joints if not (j["body0"] and j["body1"])])


report = [audit(p) for p in paths]
out.write_text(json.dumps(report, indent=1) + "\n")
for r in report:
    print(r["usd"], "roots:", [(x["path"], x["connected_bodies"]) for x in r["articulation_roots"]],
          "outside:", [x["path"] for x in r["rigid_bodies_outside_articulation"]],
          "world:", [(w["path"], w["applied_schemas"], [(j["path"], j["body0"], j["body1"]) for j in w["joints"]])
                     for w in r["world_prims"]], flush=True)
