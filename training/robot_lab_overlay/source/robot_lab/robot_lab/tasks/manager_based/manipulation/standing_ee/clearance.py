"""Surface clearance on the unchanged USD collider primitives (protocol draft.7).

Capsule pairs use exact segment distance. Capsule/polytope queries solve a convex
quadratic problem and certify the distance with a support-plane lower bound.
Polytope/polytope pairs (boxes, cylinders, cooked hulls) use hpp-fcl GJK/EPA on the
convex hulls: signed distance, negative = penetration depth.
Cylinders are circumscribed CYLINDER_SEGMENTS-gon prisms: conservative, the reported
clearance is never larger than the true one and at most r*(1/cos(pi/N)-1) smaller.
A failed capsule/convex certificate falls back to hpp-fcl GJK/EPA (exact, signed),
never to a bounding-box distance. Mesh colliders whose cooked hull cannot be read use
the convex hull of their USD points (source "approx_hull", draft.7); anything else that
cannot be read stays unresolved.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

NUMERICAL_TOLERANCE_M = 1e-8
CYLINDER_SEGMENTS = 96


def segment_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    u, v, w = a[1]-a[0], b[1]-b[0], a[0]-b[0]
    aa, bb, cc = u@u, u@v, v@v
    dd, ee = u@w, v@w
    candidates = []
    if aa <= 1e-24 and cc <= 1e-24:
        return float(np.linalg.norm(w))
    if aa <= 1e-24:
        return float(np.linalg.norm(w-np.clip(ee/cc,0,1)*v))
    if cc <= 1e-24:
        return float(np.linalg.norm(w+np.clip(-dd/aa,0,1)*u))
    for s in (0.,1.):
        candidates.append((s,float(np.clip((ee+s*bb)/cc,0,1))))
    for t in (0.,1.):
        candidates.append((float(np.clip((t*bb-dd)/aa,0,1)),t))
    det = aa*cc-bb*bb
    if det > 1e-14*aa*cc:
        s, t = (bb*ee-cc*dd)/det, (aa*ee-bb*dd)/det
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s,t))
    return float(min(np.linalg.norm(w+s*u-t*v) for s,t in candidates))


def polytope(vertices):
    vertices = np.asarray(vertices,dtype=float)
    hull = ConvexHull(vertices)
    return dict(kind="polytope",vertices=vertices[hull.vertices],planes=hull.equations,
                source="primitive",resolved=True)


def moved_shape(shape, frame):
    result = dict(shape)
    if shape["kind"] == "capsule":
        result["ends"] = shape["ends"] @ frame[:3,:3].T + frame[:3,3]
    elif shape["kind"] == "polytope":
        result["vertices"] = shape["vertices"] @ frame[:3,:3].T + frame[:3,3]
        normals = shape["planes"][:,:3] @ frame[:3,:3].T
        result["planes"] = np.c_[normals,shape["planes"][:,3]-normals@frame[:3,3]]
    return result


def segment_polytope_distance(ends, shape):
    """Return a certified distance and whether the segment enters the solid."""
    vertices, planes = shape["vertices"], shape["planes"]
    # Work near the origin to avoid cancellation under translated environments.
    origin = vertices.mean(axis=0)
    points = vertices-origin
    ends = ends-origin
    normals, offsets = planes[:,:3], planes[:,3]+planes[:,:3]@origin
    direction = ends[1]-ends[0]
    lo, hi = 0.,1.
    for value,slope in zip(normals@ends[0]+offsets,normals@direction):
        if abs(slope) < 1e-15:
            if value > NUMERICAL_TOLERANCE_M:
                lo,hi = 1.,0.;break
        elif slope > 0:
            hi = min(hi,-value/slope)
        else:
            lo = max(lo,-value/slope)
    if lo <= hi:
        return 0.,True
    def residual(x):
        return ends[0]+x[0]*direction-x[1:]
    def jacobian(x):
        r = residual(x)
        return np.r_[2*r@direction,-2*r]
    constraints = dict(type="ineq",fun=lambda x: -(normals@x[1:]+offsets),
                       jac=lambda x: np.c_[np.zeros(len(normals)),-normals])
    solved = minimize(lambda x: float(residual(x)@residual(x)),np.r_[.5,points.mean(axis=0)],
                      jac=jacobian,bounds=[(0.,1.),(None,None),(None,None),(None,None)],
                      constraints=constraints,method="SLSQP",options={"ftol":1e-15,"maxiter":200})
    point = ends[0]+solved.x[0]*direction
    difference = point-solved.x[1:]
    upper = float(np.linalg.norm(difference))
    if not solved.success or np.max(normals@solved.x[1:]+offsets) > 1e-9 or upper <= 1e-12:
        raise ValueError("Convex distance did not converge with a feasible witness")
    axis = difference/upper
    lower = max(0.,float(np.min(ends@axis)-np.max(points@axis)))
    if upper-lower > NUMERICAL_TOLERANCE_M:
        raise ValueError(f"Convex distance certificate gap {upper-lower} m")
    return upper,False


def polytope_distance(a, b):
    """Signed convex distance (hpp-fcl GJK/EPA); built per call, no long-lived native objects."""
    import hppfcl
    origin = (a["vertices"].mean(axis=0)+b["vertices"].mean(axis=0))/2
    def convex(vertices):
        points = hppfcl.StdVec_Vec3f()
        for v in vertices-origin:
            points.append(np.asarray(v,dtype=float))
        return hppfcl.Convex.convexHull(points,False,None)
    identity = hppfcl.Transform3f()
    distance = hppfcl.distance(convex(a["vertices"]),identity,convex(b["vertices"]),identity,
                               hppfcl.DistanceRequest(),hppfcl.DistanceResult())
    return float(distance)


def capsule_polytope_distance(capsule, b):
    """Signed capsule/convex distance (hpp-fcl GJK/EPA); fallback when the certified QP fails."""
    import hppfcl
    ends = capsule["ends"]
    origin = b["vertices"].mean(axis=0)
    axis = ends[1]-ends[0]
    length = float(np.linalg.norm(axis))
    z = axis/length if length > 1e-12 else np.array([0.,0.,1.])
    helper = np.array([1.,0.,0.]) if abs(z[0]) < .9 else np.array([0.,1.,0.])
    x = np.cross(helper,z); x /= np.linalg.norm(x)
    rotation = np.c_[x,np.cross(z,x),z]
    points = hppfcl.StdVec_Vec3f()
    for v in b["vertices"]-origin:
        points.append(np.asarray(v,dtype=float))
    return float(hppfcl.distance(hppfcl.Capsule(float(capsule["radius"]),length),
                                 hppfcl.Transform3f(rotation,ends.mean(axis=0)-origin),
                                 hppfcl.Convex.convexHull(points,False,None),hppfcl.Transform3f(),
                                 hppfcl.DistanceRequest(),hppfcl.DistanceResult()))


def surface_clearance(a, b):
    if not a.get("resolved",True) or not b.get("resolved",True):
        raise ValueError("Physical collider geometry unresolved: " + str(a.get("issue") or b.get("issue")))
    if a["kind"] != "capsule":
        a,b = b,a
    if a["kind"] == "polytope" and b["kind"] == "polytope":
        return dict(clearance_m=polytope_distance(a,b),method="hppfcl_gjk_epa_convex",
                    penetration_depth_exact=True,
                    approximation_m=a.get("approximation_m",0.)+b.get("approximation_m",0.))
    if b["kind"] == "capsule":
        distance = segment_distance(a["ends"],b["ends"])
        return dict(clearance_m=distance-a["radius"]-b["radius"],method="analytic_capsule_capsule",
                    penetration_depth_exact=True)
    if b["kind"] == "polytope":
        try:
            distance,inside = segment_polytope_distance(a["ends"],b)
        except ValueError:
            return dict(clearance_m=capsule_polytope_distance(a,b),method="hppfcl_gjk_epa_capsule_convex",
                        penetration_depth_exact=True,approximation_m=b.get("approximation_m",0.))
        return dict(clearance_m=distance-a["radius"],method="certified_capsule_convex_qp",
                    penetration_depth_exact=not inside,approximation_m=b.get("approximation_m",0.))
    raise ValueError("Unsupported physical collider pair")


def ground_clearance(shape):
    if not shape.get("resolved",True):
        raise ValueError("Physical ground clearance unresolved: " + str(shape.get("issue")))
    if shape["kind"] == "capsule":
        value = np.min(shape["ends"][:,2])-shape["radius"]
    else:
        value = np.min(shape["vertices"][:,2])
    return dict(clearance_m=float(value),method="analytic_shape_plane",penetration_depth_exact=True)


def classification(distance, margin):
    if distance < -NUMERICAL_TOLERANCE_M:
        return "penetration"
    if distance < margin-NUMERICAL_TOLERANCE_M:
        return "clearance_violation"
    return "broad_phase_false_positive"


def approx_hull(stage, prim, body, cache, cooking_issue):
    """draft.7 monitor geometry for mesh colliders whose PhysX cooked hull cannot be read:
    the exact convex hull of the collider's USD mesh points (each mesh in its own frame).
    PhysX collides with its own cooked hull of the same points (convexHull approximation,
    vertex limit), so this is an approximation of the simulated surface; rows are tagged."""
    from pxr import Usd, UsdGeom
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    clouds = []
    for mesh in Usd.PrimRange(prim,Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if mesh.IsA(UsdGeom.Mesh):
            relative = np.array(cache.GetLocalToWorldTransform(mesh)*inverse).T
            clouds.append(np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get()) @ relative[:3,:3].T + relative[:3,3])
    if not clouds:
        return [dict(kind="unresolved",resolved=False,issue="No USD mesh points; "+cooking_issue)]
    shape = polytope(np.vstack(clouds))
    shape.update(source="approx_hull",cooking_issue=cooking_issue)
    return [shape]


def usd_shapes(stage, prim, body, cache):
    """Read collider surfaces. Visual meshes are never substituted for physics."""
    import itertools
    from pxr import UsdGeom
    relative = np.array(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()).T
    def points(values):
        return np.asarray(values) @ relative[:3,:3].T + relative[:3,3]
    if prim.IsA(UsdGeom.Capsule):
        cap = UsdGeom.Capsule(prim)
        scales = np.linalg.svd(relative[:3,:3],compute_uv=False)
        if not np.allclose(scales,scales.mean(),atol=1e-7,rtol=1e-6):
            return [dict(kind="unresolved",resolved=False,issue="Nonuniform capsule scale",path=str(prim.GetPath()))]
        axis = np.eye(3)["XYZ".index(cap.GetAxisAttr().Get())]
        ends = points(np.array([-axis,axis])*cap.GetHeightAttr().Get()/2)
        result = [dict(kind="capsule",ends=ends,radius=float(cap.GetRadiusAttr().Get()*scales.mean()),
                       resolved=True,source="usd_capsule")]
    elif prim.IsA(UsdGeom.Cylinder):
        cyl = UsdGeom.Cylinder(prim)
        radius, height = cyl.GetRadiusAttr().Get(), cyl.GetHeightAttr().Get()
        n = CYLINDER_SEGMENTS
        angles = np.linspace(0,2*np.pi,n,endpoint=False)
        ring = np.c_[np.cos(angles),np.sin(angles)]*radius/np.cos(np.pi/n)   # circumscribed polygon
        local = np.vstack([np.c_[ring,np.full(n,z)] for z in (-height/2,height/2)])
        axis = "XYZ".index(cyl.GetAxisAttr().Get())
        local = np.roll(local,axis-2,axis=1)      # local z -> authored axis
        scales = np.linalg.svd(relative[:3,:3],compute_uv=False)
        result = [polytope(points(local))]
        result[0].update(source="usd_cylinder_circumscribed_polygon",segments=n,
                         approximation_m=float(radius*scales.max()*(1/np.cos(np.pi/n)-1)))
    elif prim.IsA(UsdGeom.Cube):
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        result = [polytope(points(list(itertools.product(*[[-size/2,size/2]]*3))))]
        result[0]["source"] = "usd_cube"
    else:
        # Use PhysX's cooked representation, not an enclosing visual/raw-mesh hull.
        try:
            import omni.physx
            from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
            from pxr import UsdUtils, PhysicsSchemaTools
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            if stage_id < 0:
                raise ValueError("Stage is not registered with PhysX")
            from pxr import Usd
            mesh_prims = [p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
                          if p.IsA(UsdGeom.Mesh)]
            if not mesh_prims:
                raise ValueError("Collider has no queryable USD mesh")
            # Query the prim that carries the CollisionAPI (here an Xform over the mesh);
            # cooked vertices are taken in that collider prim's local frame.
            response,statuses = [],[]
            def ready(status,convexes):
                statuses.append(str(status))
                if status == PhysxCollisionRepresentationResult.RESULT_VALID:
                    response.extend([np.array([[v.x,v.y,v.z] for v in c.vertices]) for c in convexes])
            omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                run_asynchronously=False,on_result=ready)
            if not response:
                raise ValueError(f"No cooked convex representation returned: {prim.GetPath()}, status={statuses}")
            result = []
            for vertices in response:
                shape = polytope(points(vertices))
                shape.update(source="physx_cooked_convex",collider_prim=str(prim.GetPath()),
                             mesh_paths=[str(m.GetPath()) for m in mesh_prims])
                result.append(shape)
        except (ImportError,ValueError,RuntimeError) as exc:
            result = approx_hull(stage,prim,body,cache,str(exc))
    for index,shape in enumerate(result):
        shape["path"] = str(prim.GetPath())
        shape["component"] = index
    return result
