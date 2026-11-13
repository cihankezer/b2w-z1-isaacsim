"""CPU-only USD tests, run with the existing Isaac image's USD libraries."""

import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Gf = Usd = UsdGeom = UsdPhysics = None

pytestmark = pytest.mark.skipif(Usd is None, reason="USD SDK is not available")

package = types.ModuleType("_t1_usd_cpu")
package.__path__ = [str(Path(__file__).resolve().parents[1] /
    "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee")]
sys.modules[package.__name__] = package
UsdArmKinematics = importlib.import_module("_t1_usd_cpu.kinematics").UsdArmKinematics


def fixture_stage(empty_link=None):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/Robot")
    for i in range(7):
        name = "base_link" if i == 0 else f"link0{i}"
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        if name != empty_link:
            cube = UsdGeom.Cube.Define(stage, f"/Robot/{name}/geometry")
            cube.CreateSizeAttr(0.1)
            cube.CreatePurposeAttr("guide")
            cube.CreateVisibilityAttr("invisible")
            if i not in (1, 5):
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if i:
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/joint{i}")
            parent = "base_link" if i == 1 else f"link0{i-1}"
            joint.CreateBody0Rel().SetTargets([f"/Robot/{parent}"])
            joint.CreateBody1Rel().SetTargets([f"/Robot/{name}"])
            joint.CreateAxisAttr("Z")
            joint.CreateLowerLimitAttr(-90.0)
            joint.CreateUpperLimitAttr(90.0)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
    return stage


def test_invisible_guide_colliders_are_physical_and_visual_only_links_are_not():
    model = UsdArmKinematics(fixture_stage(), "/Robot")
    assert model.bounds_source["base_link"] == "physical_collider"
    assert model.bounds_source["link02"] == "physical_collider"
    assert model.bounds_source["link01"] == "no_physical_collider"
    assert "link01" not in model.bounds
    assert model.bounds_source["link05"] == "no_physical_collider"
    assert "link05" not in model.bounds
    for bounds in model.bounds.values():
        np.testing.assert_allclose(bounds, [[-.05]*3, [.05]*3], atol=1e-7)


def test_visual_only_link_never_becomes_an_invented_physical_collider():
    model = UsdArmKinematics(fixture_stage(empty_link="link05"), "/Robot")
    assert model.bounds_source["link05"] == "no_physical_collider"
    assert "link05" not in model.shapes


@pytest.mark.skipif(not os.environ.get("T1_USD_ASSET"), reason="Actual robot path not provided")
def test_actual_robot_tree_and_collision_geometry():
    stage = Usd.Stage.Open(os.environ["T1_USD_ASSET"])
    model = UsdArmKinematics(stage, str(stage.GetDefaultPrim().GetPath()))
    assert sum(edge[5] is not None for edge in model.edges) == 22
    assert all(f"link0{i}" in model.bounds for i in (0,2,3,4,6))
    assert "link01" not in model.bounds and "link05" not in model.bounds
    assert model.bounds_source["link06"] == "physical_collider"
    assert np.isfinite(model.limits).all()


@pytest.mark.skipif(not os.environ.get("T1_USD_ASSET"), reason="Actual robot path not provided")
def test_actual_mount_capsules_have_unit_scale():
    stage = Usd.Stage.Open(os.environ["T1_USD_ASSET"])
    root = str(stage.GetDefaultPrim().GetPath())
    cache = UsdGeom.XformCache()
    for name in ("link00", "link02"):
        prim = stage.GetPrimAtPath(f"{root}/z1/{name}/collisions/mesh_0/capsule")
        matrix = np.array(cache.GetLocalToWorldTransform(prim))[:3,:3]
        np.testing.assert_allclose(np.linalg.svd(matrix, compute_uv=False), np.ones(3), atol=1e-6)


@pytest.mark.parametrize("name", ["link02", "link04"])
def test_body_rotation_does_not_inflate_collider_local_bounds(name):
    stage = fixture_stage()
    body = stage.GetPrimAtPath(f"/Robot/{name}")
    UsdGeom.Xformable(body).AddRotateZOp().Set(37)
    cube = UsdGeom.Cube(body.GetChild('geometry'))
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(.2,0,0))
    model = UsdArmKinematics(stage,'/Robot')
    np.testing.assert_allclose(model.bounds[name],[[.15,-.05,-.05],[.25,.05,.05]],atol=1e-7)
    assert len(model.collider_bounds[name]) == 1


@pytest.mark.skipif(not os.environ.get("T1_USD_ASSET"), reason="Actual robot path not provided")
def test_actual_mount_penetration_is_geometric_even_when_physx_filters_it():
    clearance = importlib.import_module("_t1_usd_cpu.clearance")
    stage = Usd.Stage.Open(os.environ["T1_USD_ASSET"])
    model = UsdArmKinematics(stage,str(stage.GetDefaultPrim().GetPath()))
    joints = {e[2]:0. for e in model.edges if e[5] is not None}
    joints.update(joint2=.15,joint3=-.15)
    base = np.eye(4);base[2,3]=1.2
    frames = model.forward(base,joints)
    a = clearance.moved_shape(model.shapes['link00'][0],frames['link00'])
    b = clearance.moved_shape(model.shapes['link02'][0],frames['link02'])
    assert clearance.surface_clearance(a,b)['clearance_m'] == pytest.approx(-.0125,abs=1e-6)
    assert frozenset(('link00','link02')) not in model.adjacent
    assert frozenset(('link04','link06')) not in model.adjacent
    assert model.physics_filtering['articulation_self_collisions'][0]['enabled'] is False
