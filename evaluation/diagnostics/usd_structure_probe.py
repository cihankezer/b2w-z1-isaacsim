"""Read-only USD structure probe: articulation roots, joints, colliders, visual meshes."""
import json
import sys

import usd_loader  # noqa: F401  (must be first)
from pxr import Usd, UsdGeom, UsdPhysics

out = sys.argv[1]
stage = Usd.Stage.Open(str(usd_loader.ASSET))
pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
prims = list(Usd.PrimRange(stage.GetPseudoRoot(), pred))
body_of = {}
for p in prims:
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        body_of[str(p.GetPath())] = p.GetName()


def owner(p):
    while p.IsValid() and not p.HasAPI(UsdPhysics.RigidBodyAPI):
        p = p.GetParent()
    return p.GetName() if p.IsValid() else None


def attrs(p, prefix):
    return {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(prefix) and a.HasAuthoredValue()}


report = dict(default_prim=str(stage.GetDefaultPrim().GetPath()), articulation_roots=[], joints=[], colliders=[],
              visual_meshes=[], filtered_pairs=[], collision_groups=[])
for p in prims:
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        report["articulation_roots"].append(dict(path=str(p.GetPath()), physx=attrs(p, "physxArticulation")))
    if p.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(p)
        report["joints"].append(dict(name=p.GetName(), type=p.GetTypeName(),
            body0=[str(t) for t in j.GetBody0Rel().GetTargets()], body1=[str(t) for t in j.GetBody1Rel().GetTargets()],
            exclude_from_articulation=j.GetExcludeFromArticulationAttr().Get(),
            collision_enabled=j.GetCollisionEnabledAttr().Get()))
    if p.HasAPI(UsdPhysics.FilteredPairsAPI):
        report["filtered_pairs"].append(str(p.GetPath()))
    if p.IsA(UsdPhysics.CollisionGroup):
        report["collision_groups"].append(str(p.GetPath()))
    if p.HasAPI(UsdPhysics.CollisionAPI):
        report["colliders"].append(dict(path=str(p.GetPath()), body=owner(p), type=p.GetTypeName(),
            enabled=UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get(),
            schemas=list(p.GetAppliedSchemas()), physics=attrs(p, "physics:"), physx=attrs(p, "physx"),
            purpose=UsdGeom.Imageable(p).ComputePurpose() if p.IsA(UsdGeom.Imageable) else None,
            meshes=[dict(path=str(m.GetPath()), points=len(UsdGeom.Mesh(m).GetPointsAttr().Get() or []))
                    for m in Usd.PrimRange(p, pred) if m.IsA(UsdGeom.Mesh)]))
def under_collider(p):
    while p.IsValid():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            return True
        p = p.GetParent()
    return False


for p in prims:
    if p.IsA(UsdGeom.Mesh):
        if not under_collider(p):
            report["visual_meshes"].append(dict(path=str(p.GetPath()), body=owner(p),
                points=len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []),
                faces=len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []),
                purpose=UsdGeom.Imageable(p).ComputePurpose(), visible=UsdGeom.Imageable(p).ComputeVisibility()))
json.dump(report, open(out, "w"), indent=1)
print("roots", report["articulation_roots"])
print("joints", len(report["joints"]), "colliders", len(report["colliders"]), "visual meshes", len(report["visual_meshes"]))
