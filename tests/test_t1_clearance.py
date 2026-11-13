"""Physical-surface regression cases; no USD, simulator or GPU needed."""

import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

package = types.ModuleType('_t1_clearance_cpu')
package.__path__ = [str(Path(__file__).resolve().parents[1]/'training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee')]
sys.modules[package.__name__] = package
c = importlib.import_module(package.__name__+'.clearance')
k = importlib.import_module(package.__name__+'.kinematics')


def capsule(ends,radius):
    return dict(kind='capsule',ends=np.asarray(ends,dtype=float),radius=radius,resolved=True,path='capsule')


@pytest.mark.parametrize('distance,expected',[(.098,'penetration'),(.103,'clearance_violation'),(.105,'broad_phase_false_positive'),(.2,'broad_phase_false_positive')])
def test_capsule_surfaces_decide_not_boxes(distance,expected):
    a = capsule([[0,0,0],[0,0,1]],.05)
    b = capsule([[distance,0,.2],[distance,0,.8]],.05)
    result = c.surface_clearance(a,b)
    assert result['clearance_m'] == pytest.approx(distance-.1,abs=1e-14)
    assert c.classification(result['clearance_m'],.005) == expected


def test_overlapping_sphere_bounds_can_have_large_real_clearance():
    a = capsule([[0,0,1],[0,0,1]],.05)
    b = capsule([[.09,.09,1],[.09,.09,1]],.05)
    box = np.array([[-.05,-.05,-.05],[.05,.05,.05]])
    frame_a,frame_b = np.eye(4),np.eye(4)
    frame_b[:3,3] = [.09,.09,0]
    assert k.boxes_overlap(box,frame_a,box,frame_b,.005)
    assert c.surface_clearance(a,b)['clearance_m'] == pytest.approx(np.sqrt(2)*.09-.1)
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    model.bounds = {'base_link':box,'link06':box}
    model.collider_bounds = {'base_link':[box],'link06':[box]}
    local = capsule([[0,0,0],[0,0,0]],.05)
    model.shapes = {'base_link':[local],'link06':[local]}
    model.adjacent = set()
    frame_a[2,3] = 1;frame_b[2,3] = 1
    assert model.collisions({'base_link':frame_a,'link06':frame_b}) is None
    assert model.clearance_report({'base_link':frame_a,'link06':frame_b})[0]['classification'] == 'broad_phase_false_positive'


@pytest.mark.parametrize('ends_a,ends_b,expected',[
    ([[0,0,0],[1,0,0]],[[.5,-1,0],[.5,1,0]],0),
    ([[0,0,0],[1,0,0]],[[2,1,0],[2,2,0]],np.sqrt(2)),
    ([[0,0,0],[0,0,0]],[[1,0,0],[1,0,0]],1),
    ([[0,0,0],[1,0,0]],[[.2,.3,0],[.8,.3,0]],.3),
])
def test_segment_distance_interior_endpoints_parallel_and_degenerate(ends_a,ends_b,expected):
    assert c.segment_distance(ends_a,ends_b) == pytest.approx(expected)


def test_capsule_box_corner_distance_ignores_bounding_box_overlap():
    box = c.polytope(k.corners(np.array([[-.05]*3,[.05]*3])))
    sphere = capsule([[.095,.095,0],[.095,.095,0]],.05)
    result = c.surface_clearance(sphere,box)
    assert result['clearance_m'] == pytest.approx(np.sqrt(2)*.045-.05,abs=1e-8)
    assert c.classification(result['clearance_m'],.005) == 'broad_phase_false_positive'
    crossed = capsule([[-.2,0,0],[.2,0,0]],.01)
    inside = c.surface_clearance(crossed,box)
    assert inside['clearance_m'] < 0 and not inside['penetration_depth_exact']


def test_polytope_distance_is_translation_invariant():
    box = c.polytope(k.corners(np.array([[-.1]*3,[.1]*3])))
    cap = capsule([[.12,.12,-.2],[.12,.12,.2]],.01)
    expected = c.surface_clearance(cap,box)['clearance_m']
    frame = np.eye(4);frame[:3,3] = [100,-30,1.2]
    actual = c.surface_clearance(c.moved_shape(cap,frame),c.moved_shape(box,frame))['clearance_m']
    assert actual == pytest.approx(expected,abs=1e-8)


def test_unknown_geometry_is_unresolved_not_a_box_collision():
    unknown = dict(kind='unresolved',resolved=False,issue='Cooked mesh unavailable')
    with pytest.raises(ValueError,match='unresolved'):
        c.surface_clearance(capsule([[0,0,0],[0,0,1]],.01),unknown)


def test_ground_distance_uses_capsule_surface_and_polytope_vertices():
    cap = capsule([[0,0,.1],[.2,0,.1]],.05)
    assert c.ground_clearance(cap)['clearance_m'] == pytest.approx(.05)
    box = c.polytope(k.corners(np.array([[-.1,-.1,.01],[.1,.1,.1]])))
    assert c.ground_clearance(box)['clearance_m'] == pytest.approx(.01)


def test_failed_convex_distance_certificate_never_becomes_a_collision(monkeypatch):
    from types import SimpleNamespace
    shape = c.polytope(k.corners(np.array([[-.1]*3,[.1]*3])))
    cap = capsule([[.5,0,0],[.5,0,1]],.01)
    monkeypatch.setattr(c,'minimize',lambda *args,**kwargs:SimpleNamespace(success=True,x=np.array([.5,100.,100.,100.])))
    with pytest.raises(ValueError,match='feasible witness'):
        c.segment_polytope_distance(cap['ends'],shape)
    # draft.5: a failed QP certificate falls back to exact GJK/EPA, never to a bounding-box value
    pytest.importorskip('hppfcl')
    result = c.surface_clearance(cap,shape)
    assert result['method'] == 'hppfcl_gjk_epa_capsule_convex'
    assert result['clearance_m'] == pytest.approx(.39,abs=1e-7)


def cube(centre, half=.05):
    import itertools
    return c.polytope(np.array(list(itertools.product(*[[-half,half]]*3)))+centre)


@pytest.mark.parametrize('offset,expected',[(.13,.03),(.1,0.),(.09,-.01)])
def test_polytope_pairs_use_signed_convex_distance(offset,expected):
    pytest.importorskip('hppfcl')
    result = c.surface_clearance(cube(np.zeros(3)),cube(np.array([offset,0,0])))
    assert result['method'] == 'hppfcl_gjk_epa_convex'
    assert result['clearance_m'] == pytest.approx(expected,abs=1e-9)


def test_usd_cylinder_is_a_conservative_circumscribed_prism():
    pytest.importorskip('hppfcl')
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage,'/body').GetPrim()
    cyl = UsdGeom.Cylinder.Define(stage,'/body/cyl')
    cyl.CreateRadiusAttr(.04); cyl.CreateHeightAttr(.2); cyl.CreateAxisAttr('X')
    (shape,) = c.usd_shapes(stage,cyl.GetPrim(),body,UsdGeom.XformCache())
    assert shape['source'] == 'usd_cylinder_circumscribed_polygon'
    bound = .04*(1/np.cos(np.pi/c.CYLINDER_SEGMENTS)-1)
    assert shape['approximation_m'] == pytest.approx(bound) and bound < 4e-5
    assert np.ptp(shape['vertices'][:,0]) == pytest.approx(.2)          # axis X honoured
    # true clearance to a box 0.06 m from the axis is 0.02 m; the prism may only under-report it
    d = c.surface_clearance(shape,cube(np.array([0,.06+.05,0])))['clearance_m']
    assert .02-bound-1e-9 <= d <= .02+1e-9
    # a capsule touching the round surface
    d = c.surface_clearance(capsule([[0,0,.06],[0,0,.1]],.02),shape)['clearance_m']
    assert -bound-1e-6 <= d <= 1e-6


def test_capsule_convex_fallback_matches_certified_qp():
    pytest.importorskip('hppfcl')
    cap, box = capsule([[0,0,.2],[0,0,.4]],.03), cube(np.zeros(3))
    expected = c.surface_clearance(cap,box)['clearance_m']
    assert expected == pytest.approx(.12,abs=1e-7)
    assert c.capsule_polytope_distance(cap,box) == pytest.approx(expected,abs=1e-7)
    assert c.capsule_polytope_distance(capsule([[0,0,.0],[0,0,.4]],.03),box) < 0


def test_uncookable_mesh_collider_uses_usd_point_hull_and_tags_rows():
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import Gf, UsdGeom
    import itertools
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage,'/body').GetPrim()
    collider = UsdGeom.Xform.Define(stage,'/body/collider')
    collider.AddTranslateOp().Set(Gf.Vec3d(.2,0,0))
    mesh = UsdGeom.Mesh.Define(stage,'/body/collider/mesh')
    mesh.AddTranslateOp().Set(Gf.Vec3d(0,0,.1))
    corners = np.array(list(itertools.product(*[[-.05,.05]]*3)))
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in np.vstack([corners,[[0,0,0]]])])  # interior point dropped
    (shape,) = c.usd_shapes(stage,collider.GetPrim(),body,UsdGeom.XformCache())
    assert shape['source'] == 'approx_hull' and shape['resolved'] and shape['cooking_issue']
    assert len(shape['vertices']) == 8
    np.testing.assert_allclose(np.sort(shape['vertices'],axis=0),np.sort(corners+[.2,0,.1],axis=0),atol=1e-7)
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    box = np.array([[.15,-.05,.05],[.25,.05,.15]])
    model.bounds = {'base_link':box,'link06':box}
    model.collider_bounds = {'base_link':[box],'link06':[box]}
    model.shapes = {'base_link':[shape],'link06':[capsule([[.2,.0,.18],[.2,.0,.3]],.01)]}
    model.adjacent = set()
    rows = model.clearance_report({'base_link':np.eye(4),'link06':np.eye(4)})
    (row,) = [r for r in rows if r['pair'][1] != 'ground']
    assert row['geometry'] == 'approx_hull' and row['clearance_m'] == pytest.approx(.02)
    assert model.self_clearance_monitor(rows)[0]['geometry'] == 'approx_hull'
