"""T2 CPU tests (protocol v0.4): load the pure reach_height modules without Isaac Lab or Gym.

Synthetic-model tests run anywhere with numpy/scipy; the asset tests need the USD SDK (Isaac image).
"""

import importlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MANIP = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
for _name, _path in (("_t2m", MANIP), ("_t2m.standing_ee", MANIP / "standing_ee"), ("_t2m.reach_height", MANIP / "reach_height")):
    _m = types.ModuleType(_name)
    _m.__path__ = [str(_path)]
    sys.modules[_name] = _m
P = importlib.import_module("_t2m.reach_height.protocol")
W = importlib.import_module("_t2m.reach_height.wholebody")
R = importlib.import_module("_t2m.reach_height.reachability")
S = importlib.import_module("_t2m.reach_height.sampler")
M = importlib.import_module("_t2m.reach_height.metrics")
K = importlib.import_module("_t2m.standing_ee.kinematics")


# ---------------------------------------------------------------- protocol constants

def test_bands_and_posture_grids():
    assert P.band_index([0.0, 0.2999, 0.30, 0.7999, 0.80, 1.10, 1.1001, -0.01]).tolist() == [0, 0, 1, 1, 2, 2, -1, -1]
    np.testing.assert_allclose(P.dh_values(-0.20), -0.02 * np.arange(11), atol=1e-12)
    np.testing.assert_allclose(P.dh_values(-0.27), -0.02 * np.arange(14), atol=1e-12)
    pitch = P.pitch_values()
    assert len(pitch) == 15 and pitch[0] == -0.28 and pitch[-1] == 0.28 and 0.0 in pitch
    assert P.posture_cost(-0.20, 0.0) == pytest.approx(1.0) and P.posture_cost(0.0, -0.28) == pytest.approx(1.0)
    assert P.H_NOM_M == pytest.approx(0.6038742661476135)


# ---------------------------------------------------------------- frames and orientation

def test_approach_angle_profile():
    z = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.9, 1.1])
    expected = [np.pi / 2, 3 * np.pi / 8, np.pi / 4, np.pi / 8, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(P.approach_angle(z), expected, atol=1e-12)
    assert np.all(np.diff(P.approach_angle(np.linspace(0, 1.1, 200))) <= 1e-12)   # continuous, non-increasing


def test_canonical_orientation_is_radial_and_follows_height_profile():
    theta = np.array([0.0, 0.5, -1.0, 0.3])
    z = np.array([0.7, 1.0, 0.0, 0.30])
    Rs = W.canonical_rotation(theta, z)
    for Rm in Rs:
        np.testing.assert_allclose(Rm.T @ Rm, np.eye(3), atol=1e-12)
        assert np.linalg.det(Rm) == pytest.approx(1.0)
    np.testing.assert_allclose(Rs[0], np.eye(3), atol=1e-12)                       # z >= 0.6, theta 0 = G axes
    np.testing.assert_allclose(Rs[1][:, 0], [np.cos(0.5), np.sin(0.5), 0], atol=1e-12)  # radial tool axis
    np.testing.assert_allclose(Rs[1][:, 2], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Rs[2][:, 0], [0, 0, -1], atol=1e-12)                # z = 0: tool axis down
    np.testing.assert_allclose(Rs[2][:, 2], [np.cos(-1.0), np.sin(-1.0), 0], atol=1e-12)
    np.testing.assert_allclose(Rs[2][:, 1], [-np.sin(-1.0), np.cos(-1.0), 0], atol=1e-12)          # tangential
    c, s = np.cos(0.3), np.sin(0.3)                                                 # z = 0.30: 45 deg down
    np.testing.assert_allclose(Rs[3][:, 0], np.array([c, s, -1.0]) / np.sqrt(2), atol=1e-12)


def test_base_pose_pitch_sign_and_ground_frame():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [0.02, -0.01, 0.4]).as_matrix()
    base[:3, 3] = [1.0, 2.0, 0.6]
    G = W.ground_frame(base)
    assert G[2, 3] == 0.0 and np.allclose(G[:2, 3], [1.0, 2.0])
    np.testing.assert_allclose(G[:3, 2], [0, 0, 1], atol=1e-12)
    assert W.yaw_of(G) == pytest.approx(0.4)
    T = W.base_pose(G, 0.5, 0.2)
    assert T[2, 3] == 0.5 and np.allclose(T[:2, 3], [1.0, 2.0])
    assert (np.linalg.inv(G) @ T)[:3, :3][2, 0] == pytest.approx(-np.sin(0.2))     # positive pitch = nose down
    assert M.roll_pitch(Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])[1][0] == pytest.approx(0.2)


def test_tcp_world_round_trip():
    G = np.eye(4)
    G[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
    G[:3, 3] = [0.3, -0.2, 0.0]
    p = np.array([[0.5, 0.1, 0.3]])
    Rm = W.canonical_rotation(np.array([0.2]), np.array([0.3]))
    T = W.tcp_world(G, p, Rm)
    np.testing.assert_allclose((np.linalg.inv(G) @ T[0])[:3, 3], p[0], atol=1e-12)
    np.testing.assert_allclose((np.linalg.inv(G) @ T[0])[:3, :3], Rm[0], atol=1e-12)


# ---------------------------------------------------------------- synthetic kinematic model

def fake_model():
    bodies, edges = {"/R/base_link": "base_link"}, []

    def add(parent, child, name, offset, axis=None, limits=None):
        bodies[f"/R/{child}"] = child
        local0 = np.eye(4)
        local0[:3, 3] = offset
        local1 = np.eye(4)
        local1[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()   # non-trivial child frames
        edges.append((f"/R/{parent}", f"/R/{child}", name, local0, local1,
                      None if axis is None else np.eye(3)["XYZ".index(axis)], None if limits is None else np.array(limits)))
    add("base_link", "link00", "fixed00", (0.05, 0, 0.1))
    parent = "link00"
    for child, name, offset, axis in (("link01", "joint1", (0, 0, 0.05), "Z"), ("link02", "joint2", (0, 0, 0.05), "Y"),
                                      ("link03", "joint3", (-0.35, 0, 0), "Y"), ("link04", "joint4", (0.22, 0, 0.06), "Y"),
                                      ("link05", "joint5", (0.07, 0, 0), "Z"), ("link06", "joint6", (0.05, 0, 0), "X")):
        add(parent, child, name, offset, axis, (-2.6, 2.6))
        parent = child
    add("base_link", "FR_hip", "FR_hip_joint", (0.3, -0.08, 0), "X", (-0.8, 0.8))
    add("FR_hip", "FR_thigh", "FR_thigh_joint", (0, -0.1, 0), "Y", (-1.0, 4.0))
    add("FR_thigh", "FR_calf", "FR_calf_joint", (0, 0, -0.35), "Y", (-2.7, -0.5))
    add("FR_calf", "FR_foot", "FR_foot_joint", (0, 0, -0.35), "Y", (-np.inf, np.inf))
    return types.SimpleNamespace(bodies=bodies, edges=edges, root="/R/base_link")


def test_chain_fk_matches_t1_tree_forward_kinematics():
    model = fake_model()
    arm = W.Chain(model, "link06")
    assert arm.joint_names == [f"joint{i}" for i in range(1, 7)]
    q = np.array([0.3, 1.0, -1.2, 0.4, -0.5, 0.7])
    joints = dict(zip(arm.joint_names, q), FR_hip_joint=0.1, FR_thigh_joint=0.8, FR_calf_joint=-1.5, FR_foot_joint=0.3)
    frames = K.UsdArmKinematics.forward(model, np.eye(4), joints)
    np.testing.assert_allclose(arm.fk(q)[0][0], frames["link06"], atol=1e-12)
    leg = W.Chain(model, "FR_calf", end_joint="FR_foot_joint")
    wheel = leg.fk([0.1, 0.8, -1.5])[0][0][:3, 3]
    np.testing.assert_allclose(wheel, frames["FR_foot"][:3, 3], atol=1e-12)   # foot origin = joint anchor here


def test_jacobian_matches_finite_differences():
    arm = W.Chain(fake_model(), "link06")
    q = np.array([0.3, 1.0, -1.2, 0.4, -0.5, 0.7])
    T, origins, axes = arm.fk(q)
    J = W.point_jacobian(T[:, :3, 3], origins, axes)[0]
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T2 = arm.fk(q + dq)[0][0]
        np.testing.assert_allclose((T2[:3, 3] - T[0, :3, 3]) / eps, J[:3, i], atol=1e-5)
        w = Rotation.from_matrix(T2[:3, :3] @ T[0, :3, :3].T).as_rotvec() / eps
        np.testing.assert_allclose(w, J[3:, i], atol=1e-5)


def test_batched_arm_ik_round_trip_and_rejects_unreachable():
    arm = W.Chain(fake_model(), "link06")
    limits = np.array([[-2.5, 2.5]] * 6)
    rng = np.random.default_rng(3)
    q_nom = np.array([0.0, 1.0, -1.3, 0.3, 0.0, 0.0])
    q_true = q_nom + rng.uniform(-0.4, 0.4, size=(50, 6))
    targets = arm.fk(q_true)[0]
    far = targets[:2].copy()
    far[:, :3, 3] += [3.0, 0.0, 0.0]
    q, ok, ep, er = W.arm_ik(arm, np.concatenate([targets, far]), np.tile(q_nom, (52, 1)), limits)
    assert ok[:50].all() and not ok[50:].any()
    assert (ep[:50] <= 1e-3).all() and (er[:50] <= 1e-2).all()
    assert (q >= limits[:, 0]).all() and (q <= limits[:, 1]).all()
    reached = arm.fk(q[:50])[0]
    np.testing.assert_allclose(reached[:, :3, 3], targets[:, :3, 3], atol=1e-3)


def test_leg_ik_round_trip_and_limits():
    leg = W.Chain(fake_model(), "FR_calf", end_joint="FR_foot_joint")
    limits = np.array([[-0.8, 0.8], [-1.0, 4.0], [-2.7, -0.5]])
    q_true = np.array([0.05, 0.9, -1.6])
    target = leg.fk(q_true)[0][0][:3, 3]
    q, ok = W.leg_ik(leg, target, np.array([0.0, 0.8, -1.5]), limits)
    assert ok
    np.testing.assert_allclose(leg.fk(q)[0][0][:3, 3], target, atol=1e-6)
    _, ok = W.leg_ik(leg, target + [0, 0, -0.5], np.array([0.0, 0.8, -1.5]), limits)   # beyond leg length
    assert not ok


# ---------------------------------------------------------------- reachability bookkeeping

def test_voxel_grid_sector_and_band_alignment():
    g = R.voxel_grid()
    assert g["shape"] == (60, 80, 60)
    assert np.all(np.abs(g["theta"]) <= P.AZIMUTH_MAX_RAD + 1e-12)
    assert len(g["centres"]) < 60 * 80 * 60
    z_faces = g["axes"][2] - g["resolution"] / 2
    for _, lo, hi in P.BANDS:   # band edges lie on voxel faces
        assert np.isclose(z_faces, lo).any() and np.isclose(z_faces, hi).any()


def synthetic_reachability(radius=0.8):
    """Reachable = voxels within `radius` of a shoulder point; dh grows with lower targets."""
    g = R.voxel_grid()
    band = P.band_index(g["centres"][:, 2])
    reachable = (np.linalg.norm(g["centres"] - [0.05, 0, 0.8], axis=1) <= radius) & (band >= 0)
    volume = np.zeros(g["shape"], bool)
    volume[tuple(g["ijk"][reachable].T)] = True
    return g, band, reachable, dict(resolution=g["resolution"], grid_x=g["axes"][0], grid_y=g["axes"][1],
                                    grid_z=g["axes"][2], reachable_volume=volume, voxel_centre_G=g["centres"],
                                    voxel_band=band, reachable=reachable)


def test_band_summary_fractions():
    g, band, reachable, _ = synthetic_reachability()
    s = R.band_summary(g, band, reachable)
    for i, name in enumerate(P.BAND_NAMES):
        sel = band == i
        assert s[name]["voxels"] == sel.sum() and s[name]["reachable_voxels"] == (sel & reachable).sum()
        assert s[name]["sector_fraction"] == pytest.approx((sel & reachable).sum() / sel.sum())
        assert s[name]["sector_fraction"] <= s[name]["annulus_fraction"] <= 1.0
        assert s[name]["regions"] == 1 and s[name]["largest_region_voxels"] == s[name]["reachable_voxels"]
    assert s["low"]["z_min_m"] >= 0.0 and s["high"]["z_max_m"] <= 1.1 + 1e-9


def test_band_summary_counts_connected_regions():
    g = R.voxel_grid()
    band = P.band_index(g["centres"][:, 2])
    c = g["centres"]
    blob = lambda p: np.linalg.norm(c - p, axis=1) <= 0.05
    reachable = (blob([0.4, 0.0, 0.15]) | blob([0.8, 0.0, 0.15])) & (band == 0)   # two separate Low blobs
    s = R.band_summary(g, band, reachable)
    assert s["low"]["regions"] == 2 and s["low"]["largest_region_voxels"] <= s["low"]["reachable_voxels"]
    assert s["mid"]["regions"] == 0 and s["mid"]["reachable_voxels"] == 0


# ---------------------------------------------------------------- sampler (A2)

def test_sampler_statistics_A2():
    _, _, _, arrays = synthetic_reachability()
    rs = S.ReachableSet(arrays)
    targets = S.sample(rs, 100_000, np.random.default_rng(7))
    stats = S.statistics(targets, rs)
    assert stats["max_frequency_deviation"] < 0.01
    assert stats["all_in_reachable_set"] and stats["all_in_band"]
    assert stats["max_abs_azimuth_rad"] <= P.AZIMUTH_MAX_RAD
    q = targets["quat_G_wxyz"]
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    weighted = S.sample(rs, 100_000, np.random.default_rng(8), band_weights=(0.2, 0.5, 0.3))
    assert S.statistics(weighted, rs, (0.2, 0.5, 0.3))["max_frequency_deviation"] < 0.01


def test_sampler_uniform_over_reachable_volume():
    _, band, reachable, arrays = synthetic_reachability()
    rs = S.ReachableSet(arrays)
    t = S.sample(rs, 200_000, np.random.default_rng(1), band_weights=(0, 1, 0))
    counts = np.bincount(t["voxel"], minlength=len(reachable))[reachable & (band == 1)]
    assert 0.9 < counts.var() / counts.mean() < 1.1   # uniform multinomial over voxels: Poisson dispersion ~ 1


def test_eval_set_is_seeded_counted_and_tamper_evident(tmp_path):
    _, _, _, arrays = synthetic_reachability()
    rs = S.ReachableSet(arrays)
    a1, m1 = S.eval_target_set(rs, 2101, "sha", n_per_band=50)
    a2, m2 = S.eval_target_set(rs, 2101, "sha", n_per_band=50)
    assert m1["targets_sha256"] == m2["targets_sha256"]
    assert S.eval_target_set(rs, 2102, "sha", n_per_band=50)[1]["targets_sha256"] != m1["targets_sha256"]
    path = tmp_path / "T2_eval_targets_test.npz"
    S.save_eval_set(path, a1, m1)
    loaded, meta = S.load_eval_set(path, rs)
    assert np.bincount(loaded["band"]).tolist() == [50, 50, 50] and meta == m1
    a1["position_G"][0, 0] += 0.01
    S.save_eval_set(tmp_path / "tampered.npz", a1, m1)
    with pytest.raises(ValueError):
        S.load_eval_set(tmp_path / "tampered.npz")


def test_sampling_set_is_largest_region_per_band():
    g, band, reachable, arrays = synthetic_reachability()
    island = (np.linalg.norm(g["centres"] - [1.11, 0.01, 0.11], axis=1) < 1e-6) & (band == 0) & ~reachable
    assert island.sum() == 1
    with_island = reachable | island
    volume = arrays["reachable_volume"].copy()
    volume[tuple(g["ijk"][island].T)] = True
    rs = S.ReachableSet(dict(arrays, reachable=with_island, reachable_volume=volume))
    assert rs.dropped == [1, 0, 0]
    assert not rs.reachable[island].any() and (rs.reachable == reachable).all()
    assert not rs.contains(g["centres"][island]).any()
    targets = S.sample(rs, 20_000, np.random.default_rng(1))
    assert not np.isin(targets["voxel"], np.flatnonzero(island)).any()


def test_reachable_set_rejects_empty_band():
    _, band, reachable, arrays = synthetic_reachability()
    arrays = dict(arrays, reachable=reachable & (band != 0))
    with pytest.raises(ValueError):
        S.ReachableSet(arrays)


def reachability_npz(tmp_path, arrays, version=None, overridden=False, name="map.npz"):
    meta = dict(reachability=dict(P.REACHABILITY, version=version or P.REACHABILITY["version"]), grid_overridden=overridden)
    path = tmp_path / name
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)
    return path


def test_reachability_loader_checks_version_and_full_grid(tmp_path):
    _, _, _, arrays = synthetic_reachability()
    assert len(S.ReachableSet.load(reachability_npz(tmp_path, arrays)).by_band) == 3
    with pytest.raises(ValueError, match="version"):
        S.ReachableSet.load(reachability_npz(tmp_path, arrays, version="t2-reachability-1"))
    with pytest.raises(ValueError, match="full protocol-grid"):
        S.ReachableSet.load(reachability_npz(tmp_path, arrays, overridden=True))
    with pytest.raises(ValueError, match="metadata"):
        np.savez_compressed(tmp_path / "bare.npz", **arrays)
        S.ReachableSet.load(tmp_path / "bare.npz")
    coarse = R.voxel_grid(dict(P.REACHABILITY, grid_resolution_m=0.1))
    arrays_c = dict(arrays, resolution=0.1, grid_x=coarse["axes"][0], grid_y=coarse["axes"][1], grid_z=coarse["axes"][2])
    with pytest.raises(ValueError, match="full protocol-grid"):     # coarse grid without the override flag
        S.ReachableSet.load(reachability_npz(tmp_path, arrays_c))


def test_a1_verdict_largest_region_and_dh_fraction():
    g, band, reachable, _ = synthetic_reachability()
    dh = np.where(g["centres"][:, 2] < 0.5, -0.30, -0.10)
    s = R.band_summary(g, band, reachable, dh)
    a1 = R.a1_verdict(s, g, reachable)
    assert a1["pass"] and a1["pass_largest_region"] and a1["largest_region_fraction"]["low"] == 1.0
    assert s["low"]["dh_below_kaist_fraction"] == 1.0 and s["high"]["dh_below_kaist_fraction"] == 0.0
    assert 0.0 < s["mid"]["dh_below_kaist_fraction"] < 1.0
    # Two equal Low blobs: largest region holds 50 % < 90 % -> A1 fails although both fractions are reported.
    c = g["centres"]
    blobs = ((np.linalg.norm(c - [0.4, 0, 0.15], axis=1) <= 0.05) | (np.linalg.norm(c - [0.8, 0, 0.15], axis=1) <= 0.05)) & (band == 0)
    split = (reachable & (band != 0)) | blobs
    s2 = R.band_summary(g, band, split)
    a2 = R.a1_verdict(s2, g, split)
    assert s2["low"]["regions"] == 2 and s2["low"]["largest_region_fraction"] == pytest.approx(0.5, abs=0.05)
    assert not a2["pass_largest_region"] and not a2["pass"]
    assert s2["low"]["dh_below_kaist_fraction"] is None                     # no dh given


# ---------------------------------------------------------------- torch sampler (env command term)

try:
    import torch
    T = importlib.import_module("_t2m.reach_height.torch_targets")
except ImportError:   # pragma: no cover - the Isaac image ships torch
    torch = T = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


@needs_torch
def test_torch_orientation_matches_numpy_reference():
    rng = np.random.default_rng(3)
    theta, z = rng.uniform(-1.0, 1.0, 500), rng.uniform(-0.1, 1.2, 500)
    np.testing.assert_allclose(T.approach_angle(torch.tensor(z)).numpy(), P.approach_angle(z), atol=1e-12)
    Rt = T.canonical_matrix(torch.tensor(theta), torch.tensor(z)).numpy()
    np.testing.assert_allclose(Rt, W.canonical_rotation(theta, z), atol=1e-12)
    q = T.matrix_to_quat(torch.tensor(Rt)).numpy()
    ref = Rotation.from_matrix(Rt).as_quat(canonical=True)[:, [3, 0, 1, 2]]
    np.testing.assert_allclose(np.abs(np.sum(q * ref, axis=1)), 1.0, atol=1e-9)
    assert np.all(q[:, 0] >= 0)
    random = Rotation.random(500, random_state=4)                          # all quaternion branches
    qr = T.matrix_to_quat(torch.tensor(random.as_matrix())).numpy()
    np.testing.assert_allclose(np.abs(np.sum(qr * random.as_quat()[:, [3, 0, 1, 2]], axis=1)), 1.0, atol=1e-9)


@needs_torch
def test_torch_ground_frame_to_world_matches_numpy():
    yaw = torch.tensor([0.7, -2.0], dtype=torch.float64)
    origin = torch.tensor([[0.3, -0.2, 0.0], [5.0, 1.0, 0.0]], dtype=torch.float64)
    q_base = torch.tensor(Rotation.from_euler("xyz", [[0.1, -0.05, 0.7], [0.0, 0.2, -2.0]]).as_quat()[:, [3, 0, 1, 2]])
    np.testing.assert_allclose(T.yaw_from_quat(q_base).numpy(), yaw.numpy(), atol=1e-12)
    pos = torch.tensor([[0.5, 0.1, 0.3], [0.6, -0.2, 0.05]], dtype=torch.float64)
    rot = T.canonical_matrix(torch.atan2(pos[:, 1], pos[:, 0]), pos[:, 2])
    p_w, q_w = T.g_to_world(origin, yaw, pos, rot)
    for i in range(2):
        G = np.eye(4)
        G[:3, :3] = Rotation.from_euler("z", float(yaw[i])).as_matrix()
        G[:3, 3] = origin[i].numpy()
        ref = W.tcp_world(G, pos[i:i + 1].numpy(), rot[i:i + 1].numpy())[0]
        np.testing.assert_allclose(p_w[i].numpy(), ref[:3, 3], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q_w[i].numpy()[[1, 2, 3, 0]]).as_matrix(), ref[:3, :3], atol=1e-9)


@needs_torch
def test_torch_sampler_statistics_match_protocol():
    _, _, _, arrays = synthetic_reachability()
    rs = S.ReachableSet(arrays)
    torch.manual_seed(0)
    t = T.TorchReachableSet(rs, "cpu").sample(100_000, (0.2, 0.5, 0.3))
    pos = t["position_G"].double().numpy()
    band = t["band"].numpy()
    freq = np.bincount(band, minlength=3) / len(band)
    assert np.abs(freq - [0.2, 0.5, 0.3]).max() < 0.01
    assert rs.contains(pos).all() and (P.band_index(pos[:, 2]) == band).all()
    assert np.abs(np.arctan2(pos[:, 1], pos[:, 0])).max() <= P.AZIMUTH_MAX_RAD + 1e-6
    np.testing.assert_allclose(t["rot_G"].numpy(), W.canonical_rotation(np.arctan2(pos[:, 1], pos[:, 0]), pos[:, 2]), atol=1e-5)


# ---------------------------------------------------------------- metrics (A5)

def episode(e_p, k=500, fell=False, walked=False, pitch=0.0, h=None, contact=None):
    """Synthetic episode: TCP on the x axis at distance e_p[t] from the target at the origin."""
    e_p = np.asarray(e_p, dtype=float)[:k]
    k = len(e_p)
    q = Rotation.from_euler("y", pitch).as_quat()[[3, 0, 1, 2]]
    g = Rotation.from_euler("y", pitch).inv().apply([0, 0, -1])
    return dict(tcp_pos_w=np.c_[e_p, np.zeros(k), np.zeros(k)], target_pos_w=np.zeros(3),
                tcp_quat_w=np.tile([1.0, 0, 0, 0], (k, 1)), target_quat_w=np.array([1.0, 0, 0, 0]),
                base_pos_w=np.c_[np.zeros(k), np.zeros(k), np.full(k, P.H_NOM_M) if h is None else h],
                base_quat_w=np.tile(q, (k, 1)), projected_gravity_b=np.tile(g, (k, 1)),
                foot_contact=np.ones((k, 4), bool) if contact is None else contact,
                arm_q=np.zeros((k, 6)), arm_soft_limits=np.array([[-1.0, 1.0]] * 6),
                arm_torque=np.ones((k, 6)), leg_torque=np.full((k, 12), 2.0), reset_base_pos_w=np.zeros(3),
                fell=fell, walked_away=walked)


def test_metrics_hold_window_t_reach_and_final_error():
    e = np.r_[np.linspace(0.5, 0.0, 200), np.full(300, 0.01)]
    m = M.episode_metrics(episode(e))
    assert m["completed"] and m["success"] and m["success_kaist_strict"]
    first = np.flatnonzero(e <= 0.05)[0]
    assert m["t_reach"] == pytest.approx((first + 1) * 0.02)
    assert m["e_p_final"] == pytest.approx(0.01) and m["e_p_hold_mean"] == pytest.approx(0.01)
    never = M.episode_metrics(episode(np.full(500, 0.2)))
    assert np.isnan(never["t_reach"]) and not never["success"]


def test_hold_window_is_exactly_the_last_50_steps():
    e = np.full(500, 0.01)
    e[449] = 0.06                        # step 450 of 500 lies just outside the last 1.0 s
    m = M.episode_metrics(episode(e))
    assert m["success"] and m["success_kaist_strict"]
    e[450] = 0.06                        # first step of the hold window
    m = M.episode_metrics(episode(e))
    assert not m["success"] and m["success_kaist_strict"]
    e = np.full(500, 0.01)
    e[-1] = 0.051
    m = M.episode_metrics(episode(e))
    assert not m["success"] and not m["success_kaist_strict"]


def test_early_termination_fails_and_hold_metrics_are_nan():
    m = M.episode_metrics(episode(np.full(300, 0.01), fell=True))
    assert not m["completed"] and not m["success"] and not m["success_kaist_strict"]
    assert np.isnan(m["e_p_hold_mean"]) and np.isnan(m["tilt_rms_hold"]) and m["fell"]
    m = M.episode_metrics(episode(np.full(500, 0.01), walked=True))
    assert not m["success"] and m["walked_away"]


def test_tilt_and_pitch_from_known_rotation():
    for angle in (0.0, 0.1, 0.28, 1.2):
        Rm = Rotation.from_euler("xy", [0.3 * angle, angle])
        g = Rm.inv().apply([0, 0, -1])
        expected = np.arccos(Rm.as_matrix()[2, 2])
        assert M.tilt(g)[0] == pytest.approx(expected, abs=1e-4)
    assert M.tilt([0, 0, -9.81])[0] == pytest.approx(0.0, abs=1e-12)    # upright: tilt 0, not pi (§5.4 fix)
    q = (Rotation.from_euler("z", 2.0) * Rotation.from_euler("xy", [0.05, 0.2])).as_quat()[[3, 0, 1, 2]]
    roll, pitch = M.roll_pitch(q)
    assert pitch[0] == pytest.approx(0.2, abs=1e-9) and roll[0] == pytest.approx(0.05, abs=1e-9)  # yaw-independent


def test_posture_metrics_and_reference_band():
    h = np.full(500, P.H_NOM_M - 0.15)
    m = M.episode_metrics(episode(np.full(500, 0.01), pitch=0.2, h=h))
    assert m["dh_hold"] == pytest.approx(-0.15) and m["pitch_hold_mean"] == pytest.approx(0.2)
    assert m["tilt_rms_hold"] == pytest.approx(0.0, abs=1e-9) and not m["posture_outside_reference_band"]
    m = M.episode_metrics(episode(np.full(500, 0.01), pitch=0.35, h=h))
    assert m["posture_outside_reference_band"]
    m = M.episode_metrics(episode(np.full(500, 0.01), h=np.full(500, P.H_NOM_M + 0.02)))
    assert m["posture_outside_reference_band"] and m["success"]          # outside the band is not a failure


def test_foot_lifts_joint_limits_and_torques():
    contact = np.ones((500, 4), bool)
    contact[10:20, 0] = False
    contact[30:31, 0] = False
    contact[100:400, 3] = False
    ep = episode(np.full(500, 0.01), contact=contact)
    ep["arm_q"][:100, 2] = 0.95          # within 5 % of the soft range (0.1 rad) of the upper limit
    ep["arm_q"][100:200, 4] = -0.89      # 0.11 rad from the lower limit: not near
    m = M.episode_metrics(ep)
    assert m["n_foot_lifts"] == 3
    assert m["joint_limit_violation_frac"] == pytest.approx(0.2)
    assert m["max_joint_torque_arm"] == 1.0 and m["max_joint_torque_leg"] == 2.0


def test_mode_b_requires_orientation():
    ep = episode(np.full(500, 0.01))
    ep["tcp_quat_w"] = np.tile(Rotation.from_euler("x", 0.2).as_quat()[[3, 0, 1, 2]], (500, 1))
    assert M.episode_metrics(ep)["success"]
    assert M.episode_metrics(ep)["e_R_final"] == pytest.approx(0.2)
    assert not M.episode_metrics(ep, mode="B")["success"]


def test_aggregate_over_seeds_and_height_bins():
    rows = []
    for seed, rate in ((1, 0.5), (2, 1.0)):
        for i in range(20):
            band = P.BAND_NAMES[i % 3]
            rows.append(dict(seed=seed, band=band, target_z={"low": 0.15, "mid": 0.5, "high": 1.1}[band],
                             success=float(i < 20 * rate), success_kaist_strict=1.0, e_p_final=0.01 * i,
                             dh_hold=-0.1, pitch_hold_mean=0.1, tilt_rms_hold=0.001, fell=0.0, walked_away=0.0))
    agg = M.aggregate(rows)
    assert agg["n_seeds"] == 2
    assert agg["over_seeds"]["all"]["success"]["mean"] == pytest.approx(0.75)
    assert agg["over_seeds"]["all"]["success"]["std"] == pytest.approx(np.std([0.5, 1.0], ddof=1))
    bins = M.height_bins(rows)
    assert len(bins) == 11 and sum(b["n"] for b in bins) == len(rows)
    assert bins[-1]["n"] == sum(r["band"] == "high" for r in rows)       # z = 1.10 is in the last bin


# ---------------------------------------------------------------- real asset (USD SDK required)

try:
    from pxr import Usd
except ImportError:
    Usd = None
ASSET = Path(os.environ.get("B2W_Z1_GRIPPER_USD", str(ROOT / "robots_usd/b2w_z1_gripper.usd")))
usd = pytest.mark.skipif(Usd is None or not ASSET.exists(), reason="USD SDK or gripper asset not available")


@pytest.fixture(scope="module")
def wholebody():
    stage = Usd.Stage.Open(str(ASSET))
    root = next(p for p in stage.Traverse() if p.GetName() == "base_link").GetParent()
    return W.WholeBody(K.UsdArmKinematics(stage, str(root.GetPath())))


@usd
def test_asset_tcp_and_nominal_frames(wholebody):
    wb = wholebody
    frames = wb.model.forward(wb.base_nominal, wb.joints_nominal)
    stator = np.linalg.inv(frames["link06"]) @ frames["gripperStator"]
    np.testing.assert_allclose(stator[:3, 3], [0.051, 0, 0], atol=1e-3)          # TCP offset provenance
    np.testing.assert_allclose(stator[:3, :3], np.eye(3), atol=1e-6)
    tcp = wb.nominal_tcp_world
    np.testing.assert_allclose(tcp[:3, 3], (frames["link06"] @ np.r_[P.TCP_OFFSET_M, 1.0])[:3], atol=1e-12)
    assert wb.base_nominal[2, 3] == pytest.approx(P.H_NOM_M)
    # Nominal ready pose: tool axis close to horizontal-forward, i.e. near the canonical orientation at its height (horizontal for z >= 0.6 m).
    canon = wb.G[:3, :3] @ W.canonical_rotation(np.array([0.0]), np.array([tcp[2, 3]]))[0]
    assert Rotation.from_matrix(canon.T @ tcp[:3, :3]).magnitude() < 0.1


@usd
def test_asset_leg_ik_reproduces_nominal_legs(wholebody):
    wb = wholebody
    legs, reason = wb.solve_legs(wb.base_nominal)
    assert reason is None
    for name, value in legs.items():
        assert value == pytest.approx(wb.joints_nominal[name], abs=1e-3)


@usd
def test_asset_postures_keep_wheels_fixed(wholebody):
    wb = wholebody
    for dh, pitch in ((0.0, 0.0), (-0.1, 0.2), (-0.2, -0.2)):   # (-0.2, -0.28): hip-wheel 1.4 cm < 2 cm since v0.4 margins
        base, joints, reason = wb.posture(dh, pitch)
        assert joints is not None, reason
        assert base[2, 3] == pytest.approx(P.H_NOM_M + dh)
        for leg, chain in wb.legs.items():
            wheel = (base @ chain.fk([joints[n] for n in chain.joint_names])[0][0])[:3, 3]
            np.testing.assert_allclose(wheel, wb.wheel_world[leg], atol=1e-3)


@usd
def test_asset_nominal_passes_margined_clearance(wholebody):
    wb = wholebody
    frames = wb.model.forward(wb.base_nominal, wb.joints_nominal)
    assert W.clearance_violation(wb.model, frames) is None
    pairs = W._body_pairs(wb.model)
    assert ("FL_foot", "FL_hip") in pairs or ("FL_hip", "FL_foot") in pairs        # hip-wheel is checked


@usd
def test_asset_hip_wheel_contact_posture_is_rejected(wholebody):
    """A3 (2026-11-13): posture dh -0.26 m, pitch 0.24 rad overlaps FL_hip with FL_foot by ~2 cm."""
    wb = wholebody
    base, joints, reason = wb.posture(-0.26, 0.24)
    assert joints is None and "body_body" in reason and "_hip" in reason and "_foot" in reason
    _, joints, reason = wb.posture(-0.10, 0.0)
    assert joints is not None, reason


@usd
def test_asset_eval_hip_wheel_clearance_matches_kinematics(wholebody):
    """A3a logging (eval_t2.hip_wheel_clearance) from body poses: ~2 cm overlap at (-0.26, 0.24), >= 2 cm at (-0.10, 0)."""
    sys.path.insert(0, str(ROOT / "evaluation"))
    E = importlib.import_module("eval_t2")
    mods = dict(clearance=importlib.import_module("_t2m.standing_ee.clearance"))
    wb = wholebody
    recs = []
    for dh, pitch in ((-0.26, 0.24), (-0.10, 0.0)):
        base = W.base_pose(wb.G, P.H_NOM_M + dh, pitch)
        legs, _ = wb.solve_legs(base)
        frames = wb.model.forward(base, dict(wb.joints_nominal, **legs))
        names = [n for leg in E.LEGS for n in (f"{leg}_hip", f"{leg}_foot")]
        pos = np.array([frames[n][:3, 3] for n in names])
        quat = Rotation.from_matrix(np.array([frames[n][:3, :3] for n in names])).as_quat()[:, [3, 0, 1, 2]]
        recs.append((pos, quat))
    rec = [(np.stack([r[0] for r in recs]), np.stack([r[1] for r in recs]))]   # one step, two "envs"
    out = E.hip_wheel_clearance(np, mods, wb.model, rec, np.array([1, 1]))
    assert out[0]["min"] < 0 and out[0]["leg"] == "FL" and out[0]["first"] == out[0]["min"]
    assert out[1]["min"] >= 0.02


FROZEN_MAP = Path(os.environ.get("T2_REACHABILITY_NPZ", str(
    ROOT / "repos/robot_lab/logs/t2/diag-reach-A1-v04r3-20261113T084127Z-1d029cc2/artifacts/reachability_T2_protocol.npz")))


@pytest.mark.skipif(not FROZEN_MAP.exists(), reason="frozen reachability map not available")
def test_frozen_map_and_eval_set_digests():
    """v1.0 freeze: the map file matches env_cfg's pinned hash and the eval set regenerates to EVAL_SET_SHA256."""
    import hashlib
    import re
    pinned = re.search(r'REACHABILITY_SHA256 = "([0-9a-f]{64})"', (MANIP / "reach_height/env_cfg.py").read_text()).group(1)
    assert hashlib.sha256(FROZEN_MAP.read_bytes()).hexdigest() == pinned
    rs = S.ReachableSet.load(FROZEN_MAP)
    _, meta = S.eval_target_set(rs, P.EVAL_SEED, pinned, P.N_EVAL_PER_BAND)
    assert meta["targets_sha256"] == P.EVAL_SET_SHA256
