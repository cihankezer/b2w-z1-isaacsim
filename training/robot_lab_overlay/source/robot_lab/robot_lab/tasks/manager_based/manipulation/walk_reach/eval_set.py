"""Fixed T3 evaluation set (protocol v0.3 §7). CPU: numpy + torch (FK on CPU). No Isaac Lab import.

Inputs (all pinned by hash): the T1 evaluation target set (seed 1101), the frozen T2 map (the T2 evaluation
set is regenerated from it with seed 2026 and must match T2's EVAL_SET_SHA256), the T3 evaluation goal pool
(scenario S4). One episode = one cell (profile, scenario) and an episode index; 64 per cell.

Scenario contents (targets in F_T, TCP frame, step targets as in the source task, decision D15):
  S1  2 T1 fixed targets, 5 s each, last 2 s scored. First points cycle through a seeded permutation of the
      valid T1 targets (so (V0, S1) first points are T1 targets with T1's schedule, A10); second point random.
  S3  1 T2 evaluation target, 10 s, last 1 s scored; episodes alternate low / high band.
  S4  goal-node chain on the evaluation pool from home: T_traj ~ U[1, 3] s (20 % steps), hold ~ U[0.5, 2] s,
      §6.3 check 3 online as in training; filled to the 10 s test phase.
Filters (v0.4, D28): S1 uses T1's evaluation targets unfiltered except by the moving-cell rules (§6.3 check 3);
S3 uses the T2 targets with the 0.05 m minimum and the moving-cell rules (no body box). Counts in the metadata.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..reach_height import protocol as T2P
from . import protocol as P
from .goals import moving_ok_positions, moving_reason
from .kinematics import ArmFK, interpolate
from .pool import digest, target_reason

S4_MAX_SEGMENTS = 16


def cells():
    """(profile id, (vx, wz), scenario) for core then stress cells."""
    profiles = list(P.PROFILES.items()) + list(P.STRESS_PROFILES.items())
    return [(v, cmd, s) for v, cmd in profiles for s in P.SCENARIOS]


def nominal_ft():
    """F_T of the T1 nominal state in the env frame (flat ground z = 0)."""
    from ..standing_ee.geometry import transform
    from ..standing_ee.protocol import NOMINAL_BASE_POSITION_XYZ, NOMINAL_BASE_QUAT_WXYZ
    base = transform(NOMINAL_BASE_POSITION_XYZ, NOMINAL_BASE_QUAT_WXYZ)
    F = np.eye(4)
    psi = np.arctan2(base[1, 0], base[0, 0])
    F[:3, :3] = Rotation.from_euler("z", psi).as_matrix()
    F[:2, 3], F[2, 3] = base[:2, 3], P.H_T_M
    return F


def t1_targets_T(data):
    """T1 2.0.0 fixed targets (TCP world poses at the nominal state, ee_frame tcp) -> TCP poses in F_T: ids, (n, 7)."""
    if data["ee_frame"] != "tcp" or tuple(data["tcp_offset_m"]) != tuple(P.TCP_OFFSET_M):
        raise ValueError("T1 target file must be T1 2.0.0 (ee_frame tcp, same TCP offset)")
    F_inv = np.linalg.inv(nominal_ft())
    ids, poses = [], []
    for t in data["targets"]:
        if t["family"] != "fixed":
            continue
        T = np.eye(4)
        q = t["quaternion_world_wxyz"]
        T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = t["position_world_xyz"]
        X = F_inv @ T
        ids.append(t["target_id"])
        poses.append(np.r_[X[:3, 3], Rotation.from_matrix(X[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]]])
    return ids, np.array(poses)


def t2_targets_T(t2_arrays):
    """T2 evaluation targets (frame G = F_T shifted down by h_T, same yaw) -> TCP poses in F_T (n, 7)."""
    p = np.asarray(t2_arrays["position_G"], float) - np.array([0.0, 0.0, P.H_T_M])
    return np.c_[p, np.asarray(t2_arrays["quat_G_wxyz"], float)]


def valid_positions(p_T, moving, z_min=True):
    ok = np.array([r is None for r in target_reason(p_T)]) if z_min else np.ones(len(p_T), dtype=bool)
    if moving:
        wheels = torch.tensor(P.WHEEL_CONTACT_T_XY, dtype=torch.float32)
        ok &= moving_ok_positions(torch.as_tensor(p_T, dtype=torch.float32), wheels).numpy()
    return ok


def s4_chain(pool_arrays, rng, moving, fk, wheels):
    """Goal-node chain from home filling TEST_S. Returns nodes, T_traj, hold (padded to S4_MAX_SEGMENTS)."""
    g_all = torch.as_tensor(np.c_[pool_arrays["q_arm"], pool_arrays["b"]], dtype=torch.float32)
    succ = pool_arrays["successors"]
    s = torch.linspace(0, 1, P.SEGMENT_SAMPLES)
    nodes, T, H = [], [], []
    node, t = 0, 0.0
    while t < P.TEST_S - 1e-9 and len(nodes) < S4_MAX_SEGMENTS:
        cur_bad = moving and not bool(moving_reason(fk, g_all[node][None], wheels)[0])
        for _ in range(P.MAX_TRIES):
            options = succ[node][succ[node] >= 0]
            j = int(rng.choice(options))
            step = cur_bad or rng.random() < P.STEP_FRACTION
            if moving:
                pts = interpolate(g_all[node][None].expand(P.SEGMENT_SAMPLES, -1), g_all[j][None].expand(P.SEGMENT_SAMPLES, -1), s)
                ok = moving_reason(fk, pts, wheels)
                if not bool(ok[-1] if step else ok.all()):
                    continue
            break
        else:                                  # no valid draw: hold the current goal for one more hold time
            j, step = node, True
        T_traj = 0.0 if step else float(rng.uniform(*P.T_TRAJ_S))
        hold = float(rng.uniform(*P.HOLD_S))
        nodes.append(j); T.append(T_traj); H.append(hold)
        node, t = j, t + T_traj + hold
    pad = S4_MAX_SEGMENTS - len(nodes)
    return np.r_[nodes, [-1] * pad], np.r_[T, [np.nan] * pad], np.r_[H, [np.nan] * pad]


def build(t1_data, t2_arrays, pool_arrays, seed=P.EVAL_SET_SEED, n=P.EPISODES_PER_CELL):
    """Returns (arrays, meta). Arrays have one row per episode."""
    fk = ArmFK("cpu")
    wheels = torch.tensor(P.WHEEL_CONTACT_T_XY, dtype=torch.float32)
    t1_ids, t1_poses = t1_targets_T(t1_data)
    t2_poses = t2_targets_T(t2_arrays)
    s3_bands = [T2P.BAND_NAMES.index(b) for b in P.S3_BANDS]
    rows = dict(cell_V=[], cell_S=[], episode_index=[], cmd=[], target_T=[], source_id=[], band=[],
                s4_nodes=[], s4_T_traj=[], s4_hold=[])
    excluded = {}
    for c, (v, cmd, scen) in enumerate(cells()):
        rng = np.random.default_rng([seed, c])
        moving = cmd != (0.0, 0.0)
        for e in range(n):
            target = np.full((2, 7), np.nan)
            src, band = [-1, -1], -1
            nodes, T, H = np.full(S4_MAX_SEGMENTS, -1), np.full(S4_MAX_SEGMENTS, np.nan), np.full(S4_MAX_SEGMENTS, np.nan)
            if scen == "S1":
                if e == 0:
                    ok = valid_positions(t1_poses[:, :3], moving, z_min=False)
                    pool1 = np.flatnonzero(ok)
                    excluded[f"{v}-S1"] = int((~ok).sum())
                    first = np.resize(rng.permutation(pool1), n)
                second = int(rng.choice(pool1[pool1 != first[e]]))
                target[0], target[1] = t1_poses[first[e]], t1_poses[second]
                src = [int(first[e]), second]
            elif scen == "S3":
                if e == 0:
                    ok = valid_positions(t2_poses[:, :3], moving)
                    lists = {b: rng.permutation(np.flatnonzero(ok & (np.asarray(t2_arrays["band"]) == b))) for b in s3_bands}
                    excluded[f"{v}-S3"] = {P.S3_BANDS[i]: int((~ok & (np.asarray(t2_arrays["band"]) == b)).sum())
                                           for i, b in enumerate(s3_bands)}
                b = s3_bands[e % 2]
                k = int(lists[b][e // 2])
                target[0], src, band = t2_poses[k], [k, -1], b
            else:
                nodes, T, H = s4_chain(pool_arrays, rng, moving, fk, wheels)
            rows["cell_V"].append(v); rows["cell_S"].append(scen); rows["episode_index"].append(e)
            rows["cmd"].append(cmd); rows["target_T"].append(target); rows["source_id"].append(src)
            rows["band"].append(band); rows["s4_nodes"].append(nodes); rows["s4_T_traj"].append(T); rows["s4_hold"].append(H)
    arrays = {k: np.asarray(v) if k not in ("cell_V", "cell_S") else np.asarray(v, dtype="U2") for k, v in rows.items()}
    arrays["cmd"] = arrays["cmd"].astype(float)
    arrays["s4_nodes"] = arrays["s4_nodes"].astype(np.int64)
    meta = dict(benchmark=P.BENCHMARK, protocol_version=P.VERSION, version="t3_eval_set_v5", seed=int(seed),
                episodes_per_cell=int(n), cells=[[v, list(cmd), s] for v, cmd, s in cells()],
                p_home_T=list(P.P_HOME_T), settle_s=P.SETTLE_S, test_s=P.TEST_S,
                s1_schedule=P.S1_SCHEDULE, s3_schedule=P.S3_SCHEDULE, frame="F_T", tcp_offset_m=list(P.TCP_OFFSET_M),
                t1_targets_sha256=P.T1_EVAL_TARGETS_SHA256, t1_target_ids=t1_ids,
                t2_eval_set_sha256=P.T2_EVAL_SET_SHA256, t2_map_sha256=P.T2_MAP_SHA256,
                eval_pool_sha256=digest(pool_arrays), excluded_by_validity=excluded)
    meta["eval_set_sha256"] = digest(arrays)
    return arrays, meta
