"""CPU tests; load the pure T1 modules without importing Isaac Lab or Gym."""

import copy
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

MODULES = Path(__file__).resolve().parents[1] / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee"
package = types.ModuleType("_t1_cpu")
package.__path__ = [str(MODULES)]
sys.modules[package.__name__] = package
g = importlib.import_module("_t1_cpu.geometry")
p = importlib.import_module("_t1_cpu.protocol")
k = importlib.import_module("_t1_cpu.kinematics")
targets = importlib.import_module("_t1_cpu.targets")
c = importlib.import_module("_t1_cpu.clearance")


def quaternion(axis, angle):
    q = Rotation.from_rotvec(np.asarray(axis)*angle).as_quat()
    return q[[3, 0, 1, 2]]


def circle():
    return dict(target_id="circle-XY-0", family="circle", plane="XY", u=[1, 0, 0], v=[0, 1, 0],
                centre_world_xyz=[0.5, 0.1, 0.8], quaternion_world_wxyz=[1, 0, 0, 0],
                radius_m=0.1, frequency_hz=0.05, transition_s=2.0, duration_s=40.0, steady_state_start_s=20.0)


def fixed():
    return dict(target_id="fixed-00", family="fixed", position_world_xyz=[0.5, 0, 0.8],
                quaternion_world_wxyz=[1, 0, 0, 0], hold_s=5.0, steady_state_last_s=2.0)


def sampled_set():
    result = dict(protocol_version=p.VERSION, robot_variant="B2W", end_effector_frame="link06", seed=1101, settle_s=2.0,
                  sampler=p.SAMPLER, counts={"fixed": 3, "circle": 2}, targets=[])
    result["targets"] = [dict(fixed(), target_id=f"fixed-{i:02d}") for i in range(3)]
    result["targets"] += [dict(circle(), target_id=f"circle-{i:02d}-XY") for i in range(2)]
    for t in result["targets"]:
        t["sampler_audit"] = {"attempt_index": 1, "target_sha256": p.target_digest(t)}
    return result


def test_position_metric():
    actual = g.transform([0.03, 0.04, 0], [1, 0, 0, 0])
    assert g.pose_error(actual, np.eye(4))[0] == pytest.approx(0.05)


@pytest.mark.parametrize("angle", [0, 1e-10, np.pi/2, np.pi-1e-10, np.pi])
def test_rotation_metrics_and_quaternion_sign(angle):
    q = quaternion([0, 0, 1], angle)
    actual = g.transform([0, 0, 0], q)
    np.testing.assert_allclose(actual, g.transform([0, 0, 0], -q), atol=1e-14)
    dp, geo, frob = g.pose_error(actual, np.eye(4))
    assert dp == 0
    assert geo == pytest.approx(angle, abs=1e-12)
    assert frob == pytest.approx(np.sqrt(2)*angle, abs=1e-12)


@pytest.mark.parametrize("q", [[0,0,0,0], [2,0,0,0], [np.nan,0,0,0]])
def test_invalid_quaternion(q):
    with pytest.raises(ValueError):
        g.transform([0,0,0], q)


def test_world_target_invariance_and_roundtrip():
    world = g.transform([0.8, -0.2, 1.2], quaternion([1,0,0], 0.3))
    for position, angle in (([0,0,0], 0), ([3,-2,0.5], 1.1), ([-1,1,2], -2.0)):
        base = g.transform(position, quaternion([0,0,1], angle))
        body = np.linalg.inv(base) @ world
        np.testing.assert_allclose(base @ body, world, atol=1e-14)
        q = g.pose(body)
        np.testing.assert_allclose(g.transform(q[:3], q[3:]), body, atol=1e-14)


def test_circle_geometry_velocity_and_closure():
    t = circle()
    for seconds in np.linspace(0, 40, 101):
        pose, vel, acc = g.circle(t, seconds)
        offset = pose[:3, 3] - t["centre_world_xyz"]
        assert np.linalg.norm(offset) == pytest.approx(0.1)
        assert offset[2] == pytest.approx(0)
        assert np.dot(offset, vel) == pytest.approx(0, abs=1e-14)
        h = 1e-4
        finite_diff = (g.circle(t, seconds+h)[0][:3,3] - g.circle(t, seconds-h)[0][:3,3])/(2*h)
        np.testing.assert_allclose(vel, finite_diff, atol=1e-10)
    for a, b in zip(g.circle(t, 0), g.circle(t, 20)):
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_quintic_boundary_conditions():
    start = np.array([0.4, -0.1, 0.6])
    finish, vel, acc = g.circle(circle(), 0)
    for seconds, expected in ((0, (start, np.zeros(3), np.zeros(3))),
                              (2, (finish[:3,3], vel, acc))):
        for value, target in zip(g.quintic(start, finish[:3,3], vel, acc, 2, seconds), expected):
            np.testing.assert_allclose(value, target, atol=1e-14)


def test_reference_transition_and_score_boundaries():
    settled = g.transform([0.4, -0.1, 0.6], quaternion([1,0,0], 0.4))
    before = g.reference(circle(), 4, settled)
    after = g.reference(circle(), 4+1e-7, settled)
    np.testing.assert_allclose(before[0], after[0], atol=1e-8)
    assert before[1] == "transition" and not before[3]
    assert not after[3]
    assert not g.reference(circle(), 24, settled)[3]
    assert g.reference(circle(), 24.02, settled)[3]
    assert g.reference(circle(), 44, settled)[3]
    assert not g.reference(circle(), 44.02, settled)[3]


def test_exact_scored_tick_counts():
    identity = np.eye(4)
    assert sum(g.reference(fixed(), t, identity)[3] for t in np.arange(1,351)*0.02) == 250
    assert sum(g.steady_state(fixed(), t) for t in np.arange(1,351)*0.02) == 100
    assert not any(g.steady_state(circle(), t) for t in np.arange(1,2201)*0.02)
    assert g.steady_state(fixed(), 5.02) and not g.steady_state(fixed(), 5.0) and g.steady_state(fixed(), 7.0)
    assert sum(g.reference(circle(), t, identity)[3] for t in np.arange(1,2201)*0.02) == 1000


def test_fixed_target_stays_fixed():
    t = fixed()
    for seconds in np.arange(2.02, 7.01, 0.02):
        result, phase, _, _ = g.reference(t, seconds, np.eye(4))
        assert phase == "fixed"
        np.testing.assert_allclose(result[:3,3], t["position_world_xyz"])


def trial(failure=False, completed=True, tracking=None, steady=None):
    return dict(family="fixed", failure=failure, completed=completed,
                tracking=[[0.01, 0.02, 0.02*np.sqrt(2)]] if tracking is None else tracking,
                steady=[[0.01, 0.02, 0.02*np.sqrt(2)]] if steady is None else steady)


def test_failed_trials_excluded_even_with_good_steady_state_before_fall():
    result = p.summarize([trial(), trial(failure=True, completed=False, tracking=[[100,100,100]])])["families"]["fixed"]
    assert result["n_trials"] == 2 and result["n_failures"] == 1
    assert result["success_rate"] == 0.5
    assert result["n_successes"] == 1
    assert result["metrics"]["tracking_position_m_mean"] == {"mean": 0.01, "std": None, "n": 1}


def test_all_failures_and_empty_suite_export_null_not_nan():
    for trials in ([], [trial(True, False, [], [])]):
        result = p.summarize(trials)
        assert result["families"]["fixed"]["metrics"] == {}
        import json
        json.dumps(result, allow_nan=False)


def test_interruption_does_not_become_success_or_an_error_sample():
    result = p.summarize([trial(False, False)])["families"]["fixed"]
    assert result["n_incomplete"] == 1 and result["success_rate"] == 0
    assert result["n_successes"] == 0


def test_per_target_tracking_and_steady_metrics():
    m = p.target_metrics(dict(tracking=[[1, 0, 0], [3, 0, 0]], steady=[[3, 0, 0]]))
    assert m["tracking_position_m_mean"] == 2 and m["tracking_position_m_rms"] == pytest.approx(np.sqrt(5))
    assert m["tracking_position_m_max"] == 3 and "steady_position_m_max" not in m
    assert m["steady_position_m_mean"] == 3
    assert p.target_metrics(dict(tracking=[[1, 0, 0]], steady=[]))["steady_position_m_mean"] is None
    with pytest.raises(ValueError):
        p.target_metrics(dict(tracking=[[np.nan, 0, 0]], steady=[]))


def test_mean_std_over_targets_and_seeds():
    one = p.summarize([trial(tracking=[[1, 0, 0]]), trial(tracking=[[3, 0, 0]])])
    assert one["families"]["fixed"]["metrics"]["tracking_position_m_mean"] == {"mean": 2.0, "std": pytest.approx(np.sqrt(2)), "n": 2}
    two = p.summarize([trial(tracking=[[4, 0, 0]])])
    seeds = p.summarize_seeds([one, two])["families"]["fixed"]
    assert seeds["metrics"]["tracking_position_m_mean"] == {"mean": 3.0, "std": pytest.approx(np.sqrt(2)), "n": 2}
    assert seeds["success_rate"] == {"mean": 1.0, "std": 0.0, "n": 2}


def test_sampled_target_set_tampering():
    original = sampled_set()
    assert p.validate_targets(original) is original
    for mutate in (lambda m: m.update(robot_variant="B2"),
                   lambda m: m["targets"].pop(),
                   lambda m: m["targets"][0]["position_world_xyz"].__setitem__(0, 99),
                   lambda m: m["sampler"].__setitem__("fixed_count", 24),
                   lambda m: m["targets"][0].update(hold_s=10.0)):
        changed = copy.deepcopy(original)
        mutate(changed)
        with pytest.raises(ValueError):
            p.validate_targets(changed)


def test_aabb_uses_all_corners_under_rotation():
    box = np.array([[-2,-1,-0.5], [2,1,0.5]])
    rotated = k.moved_bounds(box, g.transform([0,0,1], quaternion([0,0,1], np.pi/4)))
    assert rotated[1,0] == pytest.approx(3/np.sqrt(2))
    np.testing.assert_allclose(rotated[:,2], [0.5,1.5])


def test_forward_kinematics_uses_joint_frames_and_topological_order():
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    model.root, model.bodies = "base", {"base":"base_link", "a":"a", "hand":"link06"}
    first = g.transform([1,0,0], [1,0,0,0])
    second = g.transform([0,2,0], [1,0,0,0])
    model.edges = [("a", "hand", "fixed", second, np.eye(4), None, None),
                   ("base", "a", "joint1", first, np.eye(4), np.array([0,0,1]), [-np.pi,np.pi])]
    frames = model.forward(np.eye(4), {"joint1":np.pi/2})
    np.testing.assert_allclose(frames["link06"][:3,3], [-1,0,0], atol=1e-14)


def test_collision_rejects_link_geometry_even_when_origin_is_outside():
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    model.adjacent = set()
    model.bounds = {"base_link":np.array([[-.1,-.1,-.1],[.1,.1,.1]]),
                    "link06":np.array([[-.5,-.05,-.05],[.5,.05,.05]])}
    model.collider_bounds = {n:[b] for n,b in model.bounds.items()}
    frames = {"base_link":g.transform([0,0,1],[1,0,0,0]),
              "link06":g.transform([.4,0,1],[1,0,0,0])}
    attach_shapes(model)
    assert model.collisions(frames).startswith("penetration:self:")
    frames["link06"] = g.transform([1,0,.01],[1,0,0,0])
    assert model.collisions(frames) == "penetration:ground:link06"


def test_ik_reaches_orientation_and_rejects_unreachable_position():
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    model.root = "base"
    model.bodies = {"base":"base_link", **{f"b{i}":f"link0{i}" for i in range(1,7)}}
    model.arm_names = [f"joint{i}" for i in range(1,7)]
    model.limits = np.tile([-np.pi, np.pi], (6,1))
    model.edges = []
    for i, axis in enumerate(([0,0,1], [0,1,0], [1,0,0])*2, start=1):
        model.edges.append(("base" if i == 1 else f"b{i-1}", f"b{i}", f"joint{i}",
                            np.eye(4), np.eye(4), np.array(axis), [-np.pi,np.pi]))
    model.bounds, model.adjacent = {}, set()
    joints = dict.fromkeys(model.arm_names, 0.0)
    desired = g.transform([0,0,1], quaternion([0,0,1], np.pi/2))
    base = g.transform([0,0,1], [1,0,0,0])
    solution, reason = model.solve(desired, base, joints)
    assert reason is None
    measured = model.forward(base, dict(zip(model.arm_names, solution)))["link06"]
    assert g.pose_error(measured, desired)[1] < 1e-6
    desired[0,3] = 10
    assert model.solve(desired, base, joints)[1] == "ik_residual"


class AcceptingBackend:
    """Tests the sampler contract, not physical reachability."""
    arm_names = [f"joint{i}" for i in range(1,7)]
    limits = np.tile([-1.0, 1.0], (6,1))
    edges = [("b", "g", "jointGripper", None, None, np.array([0, 0, 1.0]), np.array([-1.0, 0.0]))]
    shapes = {}

    def forward(self, base, joints):
        hand = base.copy()
        hand[:3,3] += np.array([0, 0, 1.0]) + 0.1*np.array([joints[n] for n in self.arm_names[:3]])
        return {"link06": hand, "base_link": base}

    def collisions(self, frames):
        return None

    def solve(self, desired, base, joints, initial):
        return np.array(initial), None


def test_sampler_is_seeded_counted_and_uses_soft_limits():
    model = AcceptingBackend()
    one = targets.sample(model, 1101, {"robot": "hash"}, fixed_count=4, circle_count=3)
    assert one == targets.sample(model, 1101, {"robot": "hash"}, fixed_count=4, circle_count=3)
    other = targets.sample(model, 1102, {"robot": "hash"}, fixed_count=4, circle_count=3)
    assert [t["position_world_xyz"] for t in one["targets"][:4]] != [t["position_world_xyz"] for t in other["targets"][:4]]
    assert one["counts"] == {"fixed": 4, "circle": 3} and not one["protocol_counts"]
    assert [t["plane"] for t in one["targets"][4:]] == ["XY", "XZ", "YZ"]
    assert all(np.all(np.abs(t["joint_space_sample_rad"]) <= 0.9) for t in one["targets"])
    assert one["nominal_joint_pos"]["jointGripper"] == pytest.approx(-0.05)   # closed, clamped to the soft limit
    assert all(len(t["sampler_audit"]["ik_path"]) == 421 for t in one["targets"][4:])
    assert all(t["hold_s"] == 5.0 and t["steady_state_last_s"] == 2.0 for t in one["targets"][:4])
    circle_target = one["targets"][4]
    q = circle_target["joint_space_sample_rad"]
    np.testing.assert_allclose(circle_target["centre_world_xyz"],
                               np.add(p.NOMINAL_BASE_POSITION_XYZ, [0.1*q[0], 0.1*q[1], 1.0+0.1*q[2]]), atol=1e-12)


def test_sampler_records_filter_rejections_and_caps_attempts(monkeypatch):
    class RejectingBackend(AcceptingBackend):
        def collisions(self, frames):
            return "penetration:self:link02:link06"
    monkeypatch.setitem(p.SAMPLER, "max_attempts_per_target", 3)
    with pytest.raises(ValueError, match="attempt cap"):
        targets.sample(RejectingBackend(), 1101, {"robot": "hash"}, fixed_count=2, circle_count=0)
    assert targets.category("penetration:self:link02:link06") == "self_clearance"
    assert targets.category("t=2.10:clearance_violation:ground:FL_calf") == "ground_clearance"
    assert targets.category("t=3.00:ik_branch_or_velocity") == "ik_branch_or_velocity"
    assert targets.category("ee_below_base") == "ee_below_base"


def test_csv_export_preserves_failed_trial_and_omits_its_errors(tmp_path):
    import csv
    import importlib.util
    import json
    import importlib.util
    spec = importlib.util.spec_from_file_location("_eval_t1_cpu", Path(__file__).resolve().parents[1]/"evaluation/eval_t1.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = [dict(trial(), target_id="good", diagnostic_samples=2, contact_counts=[2]*4),
            dict(trial(True, False, [[100,100,100]]), target_id="failed", diagnostic_samples=1, contact_counts=[0]*4)]
    evaluator.export_trials(tmp_path, rows, p)
    exported = list(csv.DictReader((tmp_path/"trial_summary.csv").open()))
    assert len(exported) == 2
    assert float(exported[0]["tracking_position_m_mean"]) == 0.01
    assert float(exported[0]["steady_position_m_mean"]) == 0.01
    assert exported[1]["tracking_position_m_mean"] == ""
    assert exported[1]["tracking_sample_count"] == "1"
    aggregate = json.loads((tmp_path/"aggregate_summary.json").read_text())
    assert aggregate["families"]["fixed"]["success_rate"] == 0.5


def test_oriented_boxes_reject_false_world_aabb_overlap():
    bounds = np.array([[-1,-.05,-.05], [1,.05,.05]])
    a = g.transform([0,0,1], quaternion([0,0,1], np.pi/4))
    b = a.copy()
    b[:3,3] += a[:3,1]*.2
    aa, bb = k.moved_bounds(bounds,a), k.moved_bounds(bounds,b)
    assert np.all(np.minimum(aa[1],bb[1]) > np.maximum(aa[0],bb[0]))
    assert not k.boxes_overlap(bounds,a,bounds,b,.005)
    b[:3,3] = a[:3,3] + a[:3,1]*.104
    assert k.boxes_overlap(bounds,a,bounds,b,.005)
    assert not k.boxes_overlap(bounds,a,bounds,b,0)


def test_separate_colliders_do_not_fill_empty_space_between_them():
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    left = np.array([[-1,-.1,-.1],[-.8,.1,.1]])
    right = np.array([[.8,-.1,-.1],[1,.1,.1]])
    hand = np.array([[-.1,-.1,-.1],[.1,.1,.1]])
    model.bounds = {'base_link':np.array([left[0],right[1]]),'link06':hand}
    model.collider_bounds = {'base_link':[left,right],'link06':[hand]}
    model.adjacent = set()
    frame = g.transform([0,0,1],[1,0,0,0])
    attach_shapes(model)
    assert model.collisions({'base_link':frame,'link06':frame}) is None
    other = frame.copy(); other[0,3] = .9
    assert model.collisions({'base_link':frame,'link06':other}) == 'penetration:self:base_link:link06'


def test_ground_check_uses_each_collider_without_filling_gaps():
    model = k.UsdArmKinematics.__new__(k.UsdArmKinematics)
    parts = [np.array([[-1.1,-.1,.9],[-.9,.1,1.1]]), np.array([[.9,-.1,2.9],[1.1,.1,3.1]])]
    model.collider_bounds = {'base_link':parts}
    model.bounds = {'base_link':np.array([parts[0][0],parts[1][1]])}
    model.adjacent = set()
    frame = g.transform([0,0,0],quaternion([0,1,0],np.pi/4))
    assert k.moved_bounds(model.bounds['base_link'],frame)[0,2] < 0
    attach_shapes(model)
    assert model.collisions({'base_link':frame}) is None


def attach_shapes(model):
    """Give pure kinematics fixtures physical boxes and a Z1-like capsule."""
    model.shapes = {}
    for name,parts in model.collider_bounds.items():
        shapes = []
        for i,bounds in enumerate(parts):
            if name == 'link06':
                centre = bounds.mean(axis=0)
                radius = (bounds[1,1]-bounds[0,1])/2
                ends = np.array([centre.copy(),centre.copy()])
                ends[:,0] = [bounds[0,0]+radius,bounds[1,0]-radius]
                shape = dict(kind='capsule',ends=ends,radius=radius,resolved=True)
            else:
                shape = c.polytope(k.corners(bounds))
            shape['path'] = f'{name}/{i}'
            shapes.append(shape)
        model.shapes[name] = shapes


def test_oracle_controller_reaches_sampled_configuration_and_replays_circle_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_eval_t1_cpu", Path(__file__).resolve().parents[1]/"evaluation/eval_t1.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    nominal, sample = np.zeros(6), np.arange(6)*.1
    fixed = dict(family="fixed", joint_space_sample_rad=sample.tolist())
    np.testing.assert_allclose(e.oracle_arm_q(fixed, 1.0, nominal, g.quintic), nominal)
    np.testing.assert_allclose(e.oracle_arm_q(fixed, 2.0, nominal, g.quintic), nominal, atol=1e-12)
    np.testing.assert_allclose(e.oracle_arm_q(fixed, 3.0, nominal, g.quintic), sample/2, atol=1e-12)
    for t in (4.0, 7.0):
        np.testing.assert_allclose(e.oracle_arm_q(fixed, t, nominal, g.quintic), sample, atol=1e-12)
    path = [dict(episode_time_s=2+.1*j, arm_joint_pos_rad=[j*.01]*6) for j in range(421)]
    circle = dict(family="circle", sampler_audit=dict(ik_path=path))
    np.testing.assert_allclose(e.oracle_arm_q(circle, 3.0, nominal, g.quintic), [.1]*6, atol=1e-12)
    np.testing.assert_allclose(e.oracle_arm_q(circle, 3.05, nominal, g.quintic), [.105]*6, atol=1e-12)
