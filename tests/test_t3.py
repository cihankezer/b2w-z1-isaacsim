"""T3 CPU tests (protocol v0.3 §11: A1-A7, A13, A16, plus FK / pool consistency).

Loads the pure walk_reach modules without Isaac Lab or Gym. Synthetic tests need numpy, scipy and torch;
asset tests need the USD SDK and hpp-fcl (Isaac image, evaluation/diagnostics/run_pytest_t3.py); data tests
need the frozen inputs (T2 map, T1 targets, T3 pools, T3 eval set; paths via environment variables).
"""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
MANIP = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
for _name, _path in (("_t3m", MANIP), ("_t3m.standing_ee", MANIP / "standing_ee"),
                     ("_t3m.reach_height", MANIP / "reach_height"), ("_t3m.walk_reach", MANIP / "walk_reach")):
    _m = types.ModuleType(_name)
    _m.__path__ = [str(_path)]
    sys.modules[_name] = _m
P = importlib.import_module("_t3m.walk_reach.protocol")
FR = importlib.import_module("_t3m.walk_reach.frames")
KIN = importlib.import_module("_t3m.walk_reach.kinematics")
POOL = importlib.import_module("_t3m.walk_reach.pool")
GOALS = importlib.import_module("_t3m.walk_reach.goals")
MET = importlib.import_module("_t3m.walk_reach.metrics")
EVS = importlib.import_module("_t3m.walk_reach.eval_set")

LOGS = ROOT / "repos/robot_lab/logs"
T2_MAP = Path(os.environ.get("T2_REACHABILITY_NPZ", str(
    LOGS / "t2/diag-reach-A1-v04r3-20261113T084127Z-1d029cc2/artifacts/reachability_T2_protocol.npz")))
T1_TARGETS = Path(os.environ.get("T1_EVAL_TARGETS", str(
    LOGS / "t1/diag-validate-v1c-20261112T142731Z-72ac7c0a/artifacts/validation/targets.json")))
POOL_TRAIN = Path(os.environ.get("T3_POOL_TRAIN", str(LOGS / "t3" / P.POOL_FILES["train"])))
POOL_EVAL = Path(os.environ.get("T3_POOL_EVAL", str(LOGS / "t3" / P.POOL_FILES["eval"])))
EVAL_SET = Path(os.environ.get("T3_EVAL_SET", str(LOGS / "t3" / P.EVAL_SET_FILE)))
needs = lambda *paths: pytest.mark.skipif(not all(Path(p).exists() for p in paths),  # noqa: E731
                                          reason="frozen input file not available")
try:
    from pxr import Usd
    import hppfcl  # noqa: F401
except ImportError:
    Usd = None
ASSET = Path(os.environ.get("B2W_Z1_GRIPPER_USD", str(ROOT / "robots_usd/b2w_z1_gripper.usd")))
usd = pytest.mark.skipif(Usd is None or not ASSET.exists(), reason="USD SDK / hpp-fcl or gripper asset not available")
D = torch.float64


def quat(rotation):
    return torch.tensor(rotation.as_quat()[[3, 0, 1, 2]], dtype=D)


# ---------------------------------------------------------------- pins

def test_pinned_inputs():
    pinned = re.search(r'REACHABILITY_SHA256 = "([0-9a-f]{64})"', (MANIP / "reach_height/env_cfg.py").read_text()).group(1)
    assert P.T2_MAP_SHA256 == pinned
    assert P.FALL_HEIGHT_M == pytest.approx(0.224) and P.FALL_TILT_RAD == 1.0
    assert P.H_T_M > P.H_NOM_M and P.BODY_BOX_T[0][2] < -P.H_T_M < P.BODY_BOX_T[1][2]
    assert len(P.WHEEL_CONTACT_T_XY) == 4 and 0.3 < P.H_KNEE_M < 0.45


# ---------------------------------------------------------------- A1, A2 frames

def _pose_world(base_pos, base_quat, p_T, q_T):
    return FR.ft_to_world(base_pos[None], base_quat[None], p_T[None], q_T[None], P.H_T_M, torch.zeros(1, dtype=D))


def test_A1_frame_invariance_height_roll_pitch():
    p_T, q_T = torch.tensor([0.6, -0.2, -0.3], dtype=D), quat(Rotation.from_euler("xyz", [0.3, -0.2, 0.9]))
    ref = _pose_world(torch.tensor([1.0, 2.0, P.H_NOM_M], dtype=D), quat(Rotation.from_euler("z", 0.7)), p_T, q_T)
    for dz, roll, pitch in [(-0.33, 0.0, 0.0), (0.05, 0.2, -0.28), (-0.2, -0.3, 0.28)]:
        base_q = quat(Rotation.from_euler("ZYX", [0.7, pitch, roll]))
        p, q = _pose_world(torch.tensor([1.0, 2.0, P.H_NOM_M + dz], dtype=D), base_q, p_T, q_T)
        assert float((p - ref[0]).norm()) < 1e-3
        assert MET.geodesic(q.numpy(), ref[1].numpy())[0] < 1e-9


def test_A2_frame_yaw_rotation():
    base_pos = torch.tensor([0.5, -0.4, P.H_NOM_M], dtype=D)
    p_T, q_T = torch.tensor([0.7, 0.1, -0.2], dtype=D), quat(Rotation.identity())
    p0, _ = _pose_world(base_pos, quat(Rotation.from_euler("z", 0.3)), p_T, q_T)
    p1, _ = _pose_world(base_pos, quat(Rotation.from_euler("z", 0.3 + math.pi / 2)), p_T, q_T)
    o = torch.tensor([0.5, -0.4, P.H_T_M], dtype=D)
    expected = o + torch.tensor(Rotation.from_euler("z", math.pi / 2).as_matrix(), dtype=D) @ (p0[0] - o)
    assert float((p1[0] - expected).norm()) < 1e-3
    back = FR.world_to_ft(base_pos[None], quat(Rotation.from_euler("z", 0.3))[None], p0, q_T[None], P.H_T_M,
                          torch.zeros(1, dtype=D))[0]
    assert float((back[0] - p_T).norm()) < 1e-12


# ---------------------------------------------------------------- A3 interpolation

def test_A3_joint_space_interpolation():
    fk = KIN.ArmFK(dtype=D)
    rng = np.random.default_rng(0)
    lim = fk.soft_limits.numpy()
    assert lim.min() > -math.pi and lim.max() < math.pi      # no joint wraps: linear interpolation is unambiguous
    q = torch.tensor(rng.uniform(lim[:, 0], lim[:, 1], size=(2, 6)), dtype=D)
    b = torch.tensor([[-0.1, 0.1, 0.0], [-0.2, -0.05, 0.0]], dtype=D)
    ga, gb = torch.cat([q[0], b[0]]), torch.cat([q[1], b[1]])
    s = torch.linspace(-0.5, 1.5, 41, dtype=D)
    g = KIN.interpolate(ga[None].expand(41, -1), gb[None].expand(41, -1), s)
    assert torch.equal(g[s <= 0], ga[None].expand(int((s <= 0).sum()), -1))       # clamped endpoints
    assert torch.allclose(g[s >= 1], gb[None].expand(int((s >= 1).sum()), -1))
    frac = ((g - ga) / (gb - ga))                                                  # monotonic in s
    assert torch.all(torch.diff(frac[:, 0]) >= -1e-12)
    pa, qa = fk.tcp_pose(q[:1], b[:1])
    p0, q0 = fk.tcp_pose(g[s == 0][:, :6], g[s == 0][:, 6:])
    assert torch.allclose(pa, p0) and torch.allclose(qa, q0)                       # FK consistency at s = 0


def test_A3_goal_sampler_continuity_and_steps():
    arrays = _toy_pool()
    pool = GOALS.GoalPool(arrays)
    gen = torch.Generator().manual_seed(1)
    gs = GOALS.GoalSampler(pool, 256, generator=gen, fk=KIN.ArmFK())
    gs.reset(torch.arange(256))
    prev = gs.current_g().clone()
    moving = torch.zeros(256, dtype=torch.bool)
    steps = segments = 0
    for _ in range(1000):
        before_t = gs.t.clone()
        gs.update(P.CONTROL_DT_S, moving)
        new = gs.t < before_t + P.CONTROL_DT_S / 2                    # clock restarted = new segment
        # each new segment starts exactly at the previous target
        assert torch.allclose(gs.g_start[new], prev[new], atol=1e-6)
        steps += int((gs.T_traj[new] == 0).sum())
        segments += int(new.sum())
        prev = gs.current_g().clone()
    assert segments > 1500 and abs(steps / segments - P.STEP_FRACTION) < 0.05


def _toy_pool(n=12, k=4, seed=0):
    """Small synthetic pool: nominal arm with offsets (no clearance meaning; for sampler mechanics only)."""
    rng = np.random.default_rng(seed)
    q = np.array(P.KINEMATICS["nominal_arm_q"])[None] + rng.uniform(-0.3, 0.3, size=(n, 6))
    q[0] = P.KINEMATICS["nominal_arm_q"]
    b = np.zeros((n, 3))
    b[0] = P.B_NOM
    mode = np.r_[0, 1 + np.arange(n - 1) % 3]
    succ = np.array([rng.choice([j for j in range(1, n) if j != i], size=k, replace=False) for i in range(n)])
    return dict(q_arm=q, b=b, mode=mode, successors=succ)


# ---------------------------------------------------------------- A6 velocity frame (CPU kinematic)

def test_A6_heading_velocity_pitched_base():
    yaw = 0.8
    for pitch in (-0.28, 0.28):
        q = quat(Rotation.from_euler("ZYX", [yaw, pitch, 0.0]))[None]
        # pure pitching about the base origin: zero translation, angular velocity about the body y-axis
        w_world = torch.tensor(Rotation.from_euler("ZYX", [yaw, pitch, 0.0]).as_matrix() @ [0, 0.5, 0], dtype=D)[None]
        vx, vy, wz = FR.heading_velocity(q, torch.zeros(1, 3, dtype=D), w_world)
        assert abs(float(vx)) < 0.01 and abs(float(vy)) < 0.01 and abs(float(wz)) < 1e-9
        # forward heading motion with a vertical component: v_H unaffected by pitch and v_z
        v_world = torch.tensor([0.6 * math.cos(yaw), 0.6 * math.sin(yaw), 0.3], dtype=D)[None]
        vx, vy, _ = FR.heading_velocity(q, v_world, torch.zeros(1, 3, dtype=D))
        assert float(vx) == pytest.approx(0.6, abs=1e-9) and abs(float(vy)) < 1e-9
        nvx, nvy, _ = MET.heading_velocity(q.numpy(), v_world.numpy(), np.zeros((1, 3)))
        assert nvx[0] == pytest.approx(0.6, abs=1e-9) and abs(nvy[0]) < 1e-9


# ---------------------------------------------------------------- A7 metric units

def _episode(k=500, offset=(0.03, 0.0, 0.0), angle=0.0999, v=(0.3, 0.0, 0.0), w=0.0):
    I = np.tile([1.0, 0, 0, 0], (k, 1))
    rot = np.tile(Rotation.from_rotvec([0, 0, angle]).as_quat()[[3, 0, 1, 2]], (k, 1))
    tgt = np.tile([1.0, 0.2, 0.9], (k, 1))
    return dict(tcp_pos_w=tgt + offset, tcp_quat_w=rot, target_pos_w=tgt, target_quat_w=I,
                tcp_pos_T=tgt + offset, target_pos_T=tgt, base_quat_w=I, base_lin_vel_w=np.tile(v, (k, 1)),
                base_ang_vel_w=np.tile([0, 0, w], (k, 1)), base_lin_vel_b=np.tile(v, (k, 1)),
                base_height=np.full(k, 0.6), wheel_pos_body_xy=np.tile(np.asarray(P.WHEEL_CENTRE_BODY)[:, :2], (k, 1, 1)),
                joint_pos=np.zeros((k, 18)), joint_limits=np.tile([-1.0, 1.0], (18, 1)), joint_vel=np.ones((k, 22)),
                torque=np.full((k, 22), 2.0), effort_limit=np.full(22, 100.0), termination_id=None)


def test_A7_metric_units():
    m = MET.episode_metrics(_episode(), "S1", (0.3, 0.0))
    assert m["e_pos_mean_all"] == pytest.approx(0.03, abs=1e-4) and m["e_pos_mean_hold"] == pytest.approx(0.03, abs=1e-4)
    assert m["e_ori_mean_all"] == pytest.approx(0.1, abs=1e-4)
    assert m["e_frob_mean_all"] == pytest.approx(math.sqrt(2) * 0.1, abs=1e-3)
    assert MET.episode_metrics(_episode(angle=0.12), "S1", (0.3, 0.0))["hold_success_rate"] == 0.0
    assert m["hold_success_rate"] == 1.0 and m["n_holds_completed"] == 2         # 3 cm, 0.1 rad
    assert m["e_vx_mean"] == pytest.approx(0.0, abs=1e-9) and m["vel_settle_time"] == 0.0
    assert m["power_mean"] == pytest.approx(44.0) and m["torque_sat_frac"] == 0.0 and m["nominal_deviation"] == 0.0
    m = MET.episode_metrics(_episode(v=(0.0, 0.0, 0.0)), "S3", (0.3, 0.0))          # robot does not move
    assert m["e_vx_mean"] == pytest.approx(0.3, abs=1e-4) and math.isnan(m["vel_settle_time"])
    assert m["t2_rule_success"] is True and m["n_holds_completed"] == 1
    ep = _episode(k=300)
    ep.update(termination_id="F2", t_termination=6.0)
    m = MET.episode_metrics(ep, "S1", (0.3, 0.0))
    assert m["termination_class"] == "fall" and m["n_holds_completed"] == 1 and m["n_alive_steps"] == 300
    assert MET.wilson(0, 64)[1] == pytest.approx(0.0566, abs=1e-4)
    assert MET.bootstrap_ci([0.02] * 50) == pytest.approx((0.02, 0.02))
    hid = MET.hold_ids("S1")
    assert [(hid == k).sum() for k in (0, 1)] == [100, 100] and hid[149] == -1 and hid[150] == 0 and hid[499] == 1
    assert (MET.hold_ids("S3") == 0).sum() == 50 and (MET.hold_ids("S4") >= 0).sum() == 0


# ---------------------------------------------------------------- feasibility table

def test_feasibility_table_sampling():
    f = POOL.Feasibility()
    b = f.sample(20000, np.random.default_rng(0))
    assert f.contains(b[:, 0], b[:, 1]).all() and np.all(b[:, 2] == 0)
    assert b[:, 0].min() >= -0.28 - 1e-9      # deeper postures of the frozen map are infeasible
    assert not f.contains(np.array([-0.31]), np.array([0.0]))[0]


# ---------------------------------------------------------------- asset: torch FK vs USD

@pytest.fixture(scope="module")
def checker():
    K = importlib.import_module("_t3m.standing_ee.kinematics")
    stage = Usd.Stage.Open(str(ASSET))
    root = next(p for p in stage.Traverse() if p.GetName() == "base_link").GetParent()
    return POOL.Checker(K.UsdArmKinematics(stage, str(root.GetPath())))


@usd
def test_torch_fk_matches_usd(checker):
    fk = KIN.ArmFK(dtype=D)
    rng = np.random.default_rng(3)
    lim = fk.soft_limits.numpy()
    q = rng.uniform(lim[:, 0], lim[:, 1], size=(200, 6))
    b = np.c_[rng.uniform(-0.3, 0.0, 200), rng.uniform(-0.28, 0.28, 200), rng.uniform(-0.05, 0.05, 200)]
    T = fk.tcp_matrix(torch.tensor(q, dtype=D), torch.tensor(b, dtype=D)).numpy()
    ref = np.array([checker.tcp_T(qi, bi) for qi, bi in zip(q, b)])
    assert np.abs(T - ref).max() < 1e-9
    p, qh = fk.tcp_pose(fk.nominal_q[None], fk.b_nom[None])
    assert np.allclose(p[0].numpy(), P.P_HOME_T[:3], atol=1e-9)
    assert MET.geodesic(qh[0].numpy(), np.asarray(P.P_HOME_T[3:])) < 1e-6


@needs(T1_TARGETS)
def test_t1_targets_to_tcp_frame_match_fk():
    """M2 consistency: T1 stored joint angles + nominal base offset reproduce the T1 target pose in F_T."""
    data = json.loads(T1_TARGETS.read_bytes())
    ids, poses = EVS.t1_targets_T(data)
    q = np.array([t["joint_space_sample_rad"] for t in data["targets"] if t["family"] == "fixed"])
    fk = KIN.ArmFK(dtype=D)
    p, qq = fk.tcp_pose(torch.tensor(q, dtype=D), torch.tensor([P.B_NOM] * len(q), dtype=D))
    assert np.abs(p.numpy() - poses[:, :3]).max() < 1e-6
    assert MET.geodesic(qq.numpy(), poses[:, 3:]).max() < 1e-6


# ---------------------------------------------------------------- A4, A5, A16 on the frozen pools

@pytest.fixture(scope="module")
def train_pool():
    return POOL.load(POOL_TRAIN, P.TRAIN_POOL_SHA256)


@needs(POOL_TRAIN)
def test_pool_goals_match_fk_and_modes(train_pool):
    arrays, meta = train_pool
    fk = KIN.ArmFK(dtype=D)
    p, q = fk.tcp_pose(torch.tensor(arrays["q_arm"], dtype=D), torch.tensor(arrays["b"], dtype=D))
    assert np.abs(p.numpy() - arrays["tcp_pos_T"]).max() < 1e-9
    counts = [int((arrays["mode"] == m).sum()) for m in (1, 2, 3)]
    assert counts == [P.POOL_SIZES["train"][k] for k in ("M1", "M2", "M3")]
    assert meta["stats"]["segments"]["goals_without_successor"] == 0


@needs(POOL_TRAIN)
def test_A4_million_goals_valid(train_pool):
    """10^6 drawn segments: checks 1-3 of §6.3 at the 20 samples (step: goal only for check 3); check 4
    (clearance, legs) is offline per stored segment (re-checked on a subset in test_pool_segments_recheck)."""
    arrays, _ = train_pool
    pool = GOALS.GoalPool(arrays)
    n_env, fk = 50_000, KIN.ArmFK()
    gs = GOALS.GoalSampler(pool, n_env, generator=torch.Generator().manual_seed(7), fk=fk)
    gs.reset(torch.arange(n_env))
    gen = torch.Generator().manual_seed(8)
    wheels = torch.tensor(P.WHEEL_CONTACT_T_XY)
    s = torch.linspace(0, 1, P.SEGMENT_SAMPLES)
    lo, hi = torch.tensor(P.BODY_BOX_T[0]), torch.tensor(P.BODY_BOX_T[1])
    total, bad = 0, 0
    while total < 1_000_000:
        moving = torch.rand(n_env, generator=gen) < 0.8
        gs.T_traj.zero_(); gs.hold.zero_()                      # every env finishes its segment now
        gs.update(P.CONTROL_DT_S, moving)
        idx = (~gs.pending).nonzero(as_tuple=True)[0]
        pts = KIN.interpolate(gs.g_start[idx, None].expand(-1, 20, -1), gs.g_goal[idx, None].expand(-1, 20, -1),
                              s[None].expand(len(idx), -1))
        p, _ = fk.tcp_pose(pts.reshape(-1, 9)[:, :6], pts.reshape(-1, 9)[:, 6:])
        p = p.reshape(len(idx), 20, 3)
        step = gs.T_traj[idx] == 0
        sent = torch.ones(len(idx), 20, dtype=torch.bool)
        sent[step, :-1] = False                                  # a step sends only its goal
        in_box = ((p > lo) & (p < hi)).all(dim=-1)
        low = p[..., 2] + P.H_T_M < P.Z_MIN_M - 1e-6
        mv = GOALS.moving_ok_positions(p.reshape(-1, 3), wheels).reshape(len(idx), 20) | ~moving[idx, None]
        viol = sent & (in_box | low | ~mv)
        bad += int(viol.any(dim=1).sum())
        total += len(idx)
    assert bad == 0, f"{bad} of {total} segments violate §6.3"


@needs(POOL_TRAIN, T2_MAP, EVAL_SET)
def test_A5_height_coverage(train_pool):
    arrays, _ = train_pool
    z_m3 = arrays["tcp_pos_T"][arrays["mode"] == 3, 2] + P.H_T_M
    ev = np.load(EVAL_SET)
    s3 = ev["cell_S"] == "S3"
    z_s3 = ev["target_T"][s3, 0, 2] + P.H_T_M
    S = importlib.import_module("_t3m.reach_height.sampler")
    T2P = importlib.import_module("_t3m.reach_height.protocol")
    t2, _ = S.eval_target_set(S.ReachableSet.load(T2_MAP), T2P.EVAL_SEED, P.T2_MAP_SHA256)
    z_t2 = t2["position_G"][:, 2]
    edges = np.arange(P.Z_MIN_M, 1.10 + 1e-9, 0.10)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        if np.any((z_t2 >= lo_) & (z_t2 < hi_)):
            assert np.any((z_m3 >= lo_) & (z_m3 < hi_)), f"M3 pool misses T2 heights [{lo_:.2f}, {hi_:.2f})"
    for name, lo_, hi_ in T2P.BANDS:
        if name in P.S3_BANDS:
            band_z = z_t2[(z_t2 >= max(lo_, P.Z_MIN_M)) & (z_t2 < hi_)]
            band_s3 = z_s3[(z_s3 >= lo_) & (z_s3 < hi_)]
            assert len(band_s3) and band_s3.min() <= band_z.min() + 0.10 and band_s3.max() >= band_z.max() - 0.10


@needs(POOL_TRAIN)
def test_A16_coverage_report(train_pool):
    arrays, meta = train_pool
    cov = POOL.coverage(arrays["tcp_pos_T"][arrays["mode"] > 0])
    assert cov == meta["stats"]["coverage_vbc"] and 0.0 < cov["fraction"] <= 1.0


@usd
@needs(POOL_TRAIN)
def test_pool_segments_recheck(train_pool, checker):
    """Re-check 40 random stored segments with the USD collider model (check 4 and checks 1-2)."""
    arrays, _ = train_pool
    g = np.c_[arrays["q_arm"], arrays["b"]]
    rng = np.random.default_rng(11)
    pairs = [(i, j) for i in rng.choice(len(g), 40, replace=False) for j in arrays["successors"][i][:1] if j >= 0]
    for i, j in pairs:
        assert checker.point(g[i, :6], g[i, 6:]) is None
        assert checker.segment(g[i], g[j])[0] is None


# ---------------------------------------------------------------- A13 evaluation set

@needs(EVAL_SET, POOL_EVAL, T1_TARGETS, T2_MAP)
def test_A13_eval_set_regenerates():
    sys.path.insert(0, str(ROOT / "evaluation"))
    build = importlib.import_module("build_t3_eval_set")
    arrays, meta = EVS.build(*build.inputs(T1_TARGETS, T2_MAP, POOL_EVAL))
    assert meta["eval_set_sha256"] == P.EVAL_SET_SHA256
    stored = np.load(EVAL_SET)
    assert POOL.digest({k: stored[k] for k in stored.files if k != "metadata"}) == P.EVAL_SET_SHA256
    n_cells = (len(P.PROFILES) + len(P.STRESS_PROFILES)) * len(P.SCENARIOS)
    assert len(arrays["cell_V"]) == n_cells * P.EPISODES_PER_CELL
    s4 = arrays["cell_S"] == "S4"
    total = np.nansum(np.nan_to_num(arrays["s4_T_traj"][s4]) + np.nan_to_num(arrays["s4_hold"][s4]), axis=1)
    assert np.all(total >= P.TEST_S - 1e-9)
