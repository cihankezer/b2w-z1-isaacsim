"""Collision-checked FK goal pool (protocol v0.3 §6.2-6.4). CPU, numpy; building needs the USD model.

Not an IK reachability bank (D4): every goal is a forward-kinematics sample g = (q_arm, b) and its TCP
pose is FK(g). Each goal and each stored segment g_i -> g_j is checked offline at all SEGMENT_SAMPLES
joint-space interpolation points (s = k / (SEGMENT_SAMPLES - 1)):
  legs: T2 leg IK with the wheel centres fixed, within soft limits (base offset b);
  clearance (v0.4, inherited from the source task): M1/M3 T2 per-class margins (ground 5 mm, arm-arm 5 mm,
  arm-body 2 cm, body-body 2 cm); M2 (T1-derived) T1 margins (5 mm; body-body 2 cm, legs move with b);
  a segment uses T1 margins if either end is an M2 goal; unresolved geometry fails;
  TCP: >= Z_MIN_M above the terrain. No body box (v0.4: goals are checked with the real colliders).
Velocity-dependent checks (§6.3 check 3) are done online (goals.py).

Goal modes: 0 = home (nominal state, index 0), 1 = M1 random reach, 2 = M2 T1 rule, 3 = M3 T2 voxel.
"""

import hashlib
import json
import multiprocessing as mp
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..reach_height.protocol import AZIMUTH_MAX_RAD
from . import protocol as P

MODE_HOME, MODE_M1, MODE_M2, MODE_M3 = 0, 1, 2, 3
_STATE = {}


# ---------------------------------------------------------------- base feasibility (decision D12)

class Feasibility:
    """Base offsets (dh, pitch) allowed for sampling: inside a grid cell of the frozen T2 map's posture table
    whose four corner postures are all feasible (T2 leg IK + clearance). Roll is 0."""

    def __init__(self, table=P.FEASIBILITY):
        self.dh = np.asarray(table["dh_axis"], dtype=float)
        self.pitch = np.asarray(table["pitch_axis"], dtype=float)
        f = np.asarray(table["feasible"], dtype=bool)
        self.cells = f[:-1, :-1] & f[1:, :-1] & f[:-1, 1:] & f[1:, 1:]

    def contains(self, dh, pitch):
        dh, pitch = np.atleast_1d(dh), np.atleast_1d(pitch)
        i = np.searchsorted(self.dh, dh, side="right") - 1
        j = np.searchsorted(self.pitch, pitch, side="right") - 1
        i = np.where(np.isclose(dh, self.dh[-1]), len(self.dh) - 2, i)
        j = np.where(np.isclose(pitch, self.pitch[-1]), len(self.pitch) - 2, j)
        ok = (i >= 0) & (i < self.cells.shape[0]) & (j >= 0) & (j < self.cells.shape[1])
        out = np.zeros(len(dh), dtype=bool)
        out[ok] = self.cells[i[ok], j[ok]]
        return out

    def sample(self, n, rng):
        """n offsets (n, 3) uniform over the feasible cells (cells have equal area)."""
        idx = np.argwhere(self.cells)
        pick = idx[rng.integers(len(idx), size=n)]
        u = rng.random((n, 2))
        dh = self.dh[pick[:, 0]] + u[:, 0] * np.diff(self.dh)[pick[:, 0]]
        pitch = self.pitch[pick[:, 1]] + u[:, 1] * np.diff(self.pitch)[pick[:, 1]]
        return np.c_[dh, pitch, np.zeros(n)]


# ---------------------------------------------------------------- per-point whole-body check (needs USD model)

class Checker:
    """Whole-body check of one goal point g = (q_arm, b) on the T2 robot model (reach_height.wholebody)."""

    def __init__(self, model):
        from ..reach_height import wholebody as W
        self.W, self.model = W, model
        self.wb = W.WholeBody(model)
        self.F = self.wb.G.copy()
        self.F[2, 3] = P.H_T_M
        self.F_inv = np.linalg.inv(self.F)

    def base(self, b):
        return self.W.base_pose(self.wb.G, P.H_NOM_M + b[0], b[1], b[2])

    def tcp_T(self, q, b):
        """TCP transform in F_T (4, 4) from the USD tree (reference for the torch FK)."""
        joints = self.W.full_joints(self.wb.joints_nominal, q)
        frames = self.model.forward(self.base(b), joints)
        return self.F_inv @ frames[P.TCP_PARENT] @ self.wb.tcp

    def point(self, q, b, margins=P.CLEARANCE_MARGINS_M):
        """None if valid, else a rejection reason."""
        base = self.base(b)
        legs, reason = self.wb.solve_legs(base)
        if reason:
            return reason
        frames = self.model.forward(base, self.W.full_joints(dict(self.wb.joints_nominal, **legs), q))
        tcp = self.F_inv @ frames[P.TCP_PARENT] @ self.wb.tcp
        reason = target_reason(tcp[:3, 3][None])[0]
        return reason or self.W.clearance_violation(self.model, frames, margins)

    def segment(self, ga, gb, margins=P.CLEARANCE_MARGINS_M):
        """Check interior points of the joint-space segment ga -> gb (endpoints are checked goals).
        Midpoint-first order to fail fast. Returns (None | reason, points checked)."""
        k = P.SEGMENT_SAMPLES - 1
        order = sorted(range(1, k), key=lambda i: abs(i - k / 2))
        for n, i in enumerate(order, 1):
            g = ga + (i / k) * (gb - ga)
            reason = self.point(g[:6], g[6:], margins)
            if reason:
                return f"s={i}/{k}:{reason}", n
        return None, len(order)


def target_reason(p_T):
    """Offline TCP check of §6.3 (v0.4: height only) for positions in F_T (N, 3): None or reason per point."""
    low = p_T[:, 2] + P.H_T_M < P.Z_MIN_M
    return ["tcp_below_z_min" if b else None for b in low]


def height_histogram(z_world, width=P.M3_HEIGHT_BIN_M, top=1.6):
    edges = np.arange(0.0, top + 1e-9, width)
    h, _ = np.histogram(z_world, bins=edges)
    return {f"{a:.1f}-{b:.1f}": int(c) for a, b, c in zip(edges[:-1], edges[1:], h)}


# ---------------------------------------------------------------- stratification and coverage (§6.2 step 4, §6.4)

def spherical_np(p):
    p = np.atleast_2d(p)
    return np.linalg.norm(p, axis=1), np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=1)), np.arctan2(p[:, 1], p[:, 0])


def bin_ids(p_T, bins=P.STRATIFICATION_BINS):
    l, pitch, yaw = spherical_np(p_T)
    return np.stack([np.floor(l / bins[0]), np.floor(pitch / bins[1]), np.floor(yaw / bins[2])], axis=1).astype(int)


def stratify(p_T, n, rng):
    """Indices of n points chosen round-robin over occupied (l, pitch, yaw) bins (uniform within a bin)."""
    keys = [tuple(k) for k in bin_ids(p_T)]
    groups = {}
    for i in rng.permutation(len(keys)):
        groups.setdefault(keys[i], []).append(int(i))
    queues = [groups[k] for k in sorted(groups)]
    chosen = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop())
    if len(chosen) < n:
        raise ValueError(f"stratify: only {len(chosen)} of {n} candidates")
    return np.array(chosen)


def coverage(p_T, region=P.VBC_REGION, bins=P.STRATIFICATION_BINS):
    """Fraction of the VBC (l, pitch, yaw) region's bins that contain at least one point (A16)."""
    edges = [np.arange(lo, hi + 1e-9, w) for (lo, hi), w in zip((region["l"], region["pitch"], region["yaw"]), bins)]
    edges = [e if np.isclose(e[-1], hi) else np.r_[e, hi] for e, (_, hi) in zip(edges, (region["l"], region["pitch"], region["yaw"]))]
    l, pitch, yaw = spherical_np(p_T)
    inside = ((l >= edges[0][0]) & (l <= edges[0][-1]) & (pitch >= edges[1][0]) & (pitch <= edges[1][-1])
              & (yaw >= edges[2][0]) & (yaw <= edges[2][-1]))
    H, _ = np.histogramdd(np.c_[l, pitch, yaw][inside], bins=edges)
    return dict(fraction=float((H > 0).mean()), occupied=int((H > 0).sum()), bins=int(H.size),
                points_inside=int(inside.sum()), points=int(len(l)))


# ---------------------------------------------------------------- pool build

def _point_task(items):
    c = _STATE["checker"]
    return [c.point(q, b) for q, b in items]


def _t1_task(items):
    """T1 fixed-target rule (TASK_T1 1.0.0 §5.1 filters) on the nominal state, then the T3 point check."""
    from ..standing_ee.targets import check
    c = _STATE["checker"]
    out = []
    for q in items:
        _, reason = check(c.model, c.wb.base_nominal, c.wb.joints_nominal, q)
        out.append("t1_rule:" + reason if reason else c.point(q, np.asarray(P.B_NOM), P.T1_CLEARANCE_MARGINS_M))
    return out


def _segment_task(args):
    c = _STATE["checker"]
    i, ga, candidates, successors = args
    kept, tried, rejected, points = [], 0, {}, 0
    for j, gb, margins in candidates:
        tried += 1
        reason, n = c.segment(ga, gb, margins)
        points += n
        if reason:
            key = reason.split(":", 1)[1].split(":")[0]
            rejected[key] = rejected.get(key, 0) + 1
        else:
            kept.append(int(j))
            if len(kept) == successors:
                break
    return i, kept, tried, rejected, points


def _chunks(seq, n):
    return [seq[k::n] for k in range(n) if seq[k::n]]


def _parallel(pool, fn, items, workers):
    parts = _chunks(list(range(len(items))), workers * 4)
    out = [None] * len(items)
    for idx, res in zip(parts, pool.map(fn, [[items[i] for i in p] for p in parts])):
        for i, r in zip(idx, res):
            out[i] = r
    return out


def _count(reasons):
    c = {}
    for r in reasons:
        if r:
            k = r.split(":")[0] if not r.startswith("t1_rule:") else "t1_rule:" + r.split(":")[1]
            c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items()))


def build(model, reach_set, seed, sizes, workers=8, log=print, oversample=4):
    """Build a goal pool. reach_set: reach_height.sampler.ReachableSet of the frozen T2 map (with stored
    postures in reach_set.raw: dh, pitch, arm_q). Returns (arrays, meta)."""
    rng = np.random.default_rng(seed)
    checker = Checker(model)
    wb = checker.wb
    feas = Feasibility()
    q_nom = np.array([wb.joints_nominal[n] for n in wb.arm.joint_names])
    lo, hi = wb.arm_limits[:, 0], wb.arm_limits[:, 1]
    stats, t0 = {}, time.time()
    _STATE["checker"] = checker
    goals_q, goals_b, modes, bands, sources = [q_nom], [np.asarray(P.B_NOM)], [MODE_HOME], [-1], [-1]
    with mp.get_context("fork").Pool(workers) as pool:
        home = checker.point(q_nom, np.asarray(P.B_NOM))
        if home:
            raise ValueError(f"home goal invalid: {home}")
        # M1: q uniform in the soft limits, b uniform over feasible cells; stratified selection.
        if sizes["M1"]:
            n_c = oversample * sizes["M1"]
            q = rng.uniform(lo, hi, size=(n_c, 6))
            b = feas.sample(n_c, rng)
            reasons = _parallel(pool, _point_task, list(zip(q, b)), workers)
            ok = np.array([r is None for r in reasons])
            p = np.array([checker.tcp_T(qi, bi)[:3, 3] for qi, bi in zip(q[ok], b[ok])])
            pick = stratify(p, sizes["M1"], rng)
            stats["M1"] = dict(candidates=n_c, valid=int(ok.sum()), rejections=_count(reasons),
                               occupied_bins=len({tuple(k) for k in bin_ids(p)}))
            goals_q += list(q[ok][pick]); goals_b += list(b[ok][pick])
            modes += [MODE_M1] * len(pick); bands += [-1] * len(pick); sources += list(np.flatnonzero(ok)[pick])
            log(f"M1: {ok.sum()}/{n_c} valid, {len(pick)} kept ({time.time() - t0:.0f} s)")
        # M2: T1 fixed-target rule with the training seed (b = nominal base pose).
        if sizes["M2"]:
            kept, tried, reasons_all = [], 0, []
            while len(kept) < sizes["M2"]:
                q = rng.uniform(lo, hi, size=(2 * (sizes["M2"] - len(kept)), 6))
                reasons = _parallel(pool, _t1_task, list(q), workers)
                tried += len(q)
                reasons_all += reasons
                kept += [(tried - len(q) + k, qi) for k, (qi, r) in enumerate(zip(q, reasons)) if r is None]
            kept = kept[:sizes["M2"]]
            stats["M2"] = dict(candidates=tried, valid=int(sum(r is None for r in reasons_all)),
                               rejections=_count(reasons_all))
            goals_q += [k[1] for k in kept]; goals_b += [np.asarray(P.B_NOM)] * len(kept)
            modes += [MODE_M2] * len(kept); bands += [-1] * len(kept); sources += [k[0] for k in kept]
            log(f"M2: {len(kept)} kept of {tried} ({time.time() - t0:.0f} s)")
        # M3: T2 sampling-set voxels (largest region, centre azimuth within the sector), stored posture + arm pose,
        # equal count per band, stratified round-robin over 0.10 m height bins within each band (v0.4, A5);
        # TCP pose = FK of the stored posture.
        if sizes["M3"]:
            raw = reach_set.raw
            theta = np.arctan2(reach_set.centres[:, 1], reach_set.centres[:, 0])
            per_band = [np.flatnonzero(reach_set.reachable & (reach_set.band == k) & (np.abs(theta) <= AZIMUTH_MAX_RAD))
                        for k in range(3)]
            gb_all = np.c_[raw["dh"], raw["pitch"], np.zeros(len(raw["dh"]))]
            n_b = sizes["M3"] // 3
            chosen, checked, reasons_all, valid_per_band = [], [], [], []
            for k in range(3):
                z = reach_set.centres[per_band[k], 2]
                hb = np.floor(z / P.M3_HEIGHT_BIN_M + 1e-9).astype(int)
                queues = [list(rng.permutation(per_band[k][hb == h])) for h in np.unique(hb)]
                kept_bins = [[] for _ in queues]
                start = 0
                while sum(map(len, kept_bins)) < n_b and any(queues):
                    # round-robin: each open bin offers voxels in proportion to its remaining deficit
                    need = n_b - sum(map(len, kept_bins))
                    open_ = [i for i, q in enumerate(queues) if q]
                    share = max(1, -(-2 * need // len(open_)))
                    batch, owner = [], []
                    for i in open_:
                        take = [queues[i].pop() for _ in range(min(share, len(queues[i])))]
                        batch += take; owner += [i] * len(take)
                    start += len(batch)
                    reasons = _parallel(pool, _point_task, [(raw["arm_q"][v], gb_all[v]) for v in batch], workers)
                    reasons_all += reasons
                    checked += list(batch)
                    for v, i, r in zip(batch, owner, reasons):
                        if r is None:
                            kept_bins[i].append(v)
                pick = []                                              # equal count per bin, capped by availability
                while len(pick) < n_b and any(kept_bins):
                    for kb in kept_bins:
                        if kb and len(pick) < n_b:
                            pick.append(kb.pop(0))
                valid_per_band.append((len(pick), start))
                chosen += pick
            valid = [v for v, r in zip(checked, reasons_all) if r is None]
            cand = checked
            stats["M3"] = dict(candidates=len(cand), valid=len(valid), rejections=_count(reasons_all),
                               per_band_valid_of_checked=[[int(a), int(b)] for a, b in valid_per_band],
                               per_band_sampling_set=[len(v) for v in per_band])
            goals_q += [raw["arm_q"][v] for v in chosen]; goals_b += [gb_all[v] for v in chosen]
            modes += [MODE_M3] * len(chosen); bands += [int(reach_set.band[v]) for v in chosen]; sources += chosen
            log(f"M3: {len(valid)}/{len(cand)} valid, {len(chosen)} kept ({time.time() - t0:.0f} s)")

        g = np.c_[np.array(goals_q), np.array(goals_b)]
        modes = np.array(modes)
        # Successor graph: candidates drawn by the mode mixture; keep the first `successors` valid segments.
        by_mode = {m: np.flatnonzero(modes == m) for m in (MODE_M1, MODE_M2, MODE_M3)}
        weights = np.array([P.MODE_WEIGHTS[m - 1] if len(by_mode[m]) else 0.0 for m in by_mode])
        weights /= weights.sum()
        tasks = []
        for i in range(len(g)):
            ms = rng.choice(list(by_mode), size=sizes["max_candidates"], p=weights)
            cands = dict.fromkeys(int(rng.choice(by_mode[m])) for m in ms)
            margins = lambda j: (P.T1_CLEARANCE_MARGINS_M if MODE_M2 in (modes[i], modes[j])  # noqa: E731
                                 else P.CLEARANCE_MARGINS_M)
            tasks.append((i, g[i], [(j, g[j], margins(j)) for j in cands if j != i], sizes["successors"]))
        succ = np.full((len(g), sizes["successors"]), -1, dtype=np.int64)
        seg = dict(tried=0, rejected={}, points=0)
        for done, (i, kept, tried, rejected, points) in enumerate(pool.imap_unordered(_segment_task, tasks, chunksize=4), 1):
            succ[i, :len(kept)] = kept
            seg["tried"] += tried
            seg["points"] += points
            for k, v in rejected.items():
                seg["rejected"][k] = seg["rejected"].get(k, 0) + v
            if done % 256 == 0:
                log(f"segments: {done}/{len(g)} goals ({time.time() - t0:.0f} s)")
    _STATE.clear()
    n_succ = (succ >= 0).sum(axis=1)
    seg.update(kept=int(n_succ.sum()), goals_without_successor=int((n_succ == 0).sum()),
               successors_per_goal=dict(zip(*[a.tolist() for a in np.unique(n_succ, return_counts=True)])))
    stats["segments"] = seg
    tcp = np.array([checker.tcp_T(gi[:6], gi[6:]) for gi in g])
    quat = Rotation.from_matrix(tcp[:, :3, :3]).as_quat(canonical=True)[:, [3, 0, 1, 2]]
    arrays = dict(q_arm=g[:, :6], b=g[:, 6:], mode=modes, band=np.array(bands), source=np.array(sources),
                  successors=succ, tcp_pos_T=tcp[:, :3, 3], tcp_quat_T=quat)
    stats["coverage_vbc"] = coverage(arrays["tcp_pos_T"][modes != MODE_HOME])
    z_world = arrays["tcp_pos_T"][:, 2] + P.H_T_M
    stats["height_histogram"] = {name: height_histogram(z_world[modes == m])
                                 for name, m in (("M1", MODE_M1), ("M2", MODE_M2), ("M3", MODE_M3)) if (modes == m).any()}
    m1 = modes == MODE_M1
    stats["M1_below_moving_min"] = dict(n=int((z_world[m1] < P.Z_MIN_MOVING_M).sum()), of=int(m1.sum()))
    stats["build_s"] = time.time() - t0
    meta = dict(benchmark=P.BENCHMARK, protocol_version=P.VERSION, seed=int(seed), sizes=sizes,
                segment_samples=P.SEGMENT_SAMPLES, clearance_margins_m=P.CLEARANCE_MARGINS_M,
                t1_clearance_margins_m=P.T1_CLEARANCE_MARGINS_M, m3_height_bin_m=P.M3_HEIGHT_BIN_M,
                asset_tree_sha256=P.ASSET_TREE_SHA256, t2_map_sha256=P.T2_MAP_SHA256,
                measurements_sha256=P.MEASUREMENTS_SHA256, stats=stats)
    meta["pool_sha256"] = digest(arrays)
    return arrays, meta


def digest(arrays):
    """SHA-256 over sorted keys and array bytes (as T2 sampler.digest)."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(np.ascontiguousarray(arrays[key]).tobytes())
    return h.hexdigest()


def save(path, arrays, meta):
    np.savez_compressed(path, metadata=json.dumps(meta, default=float), **arrays)


def load(path, expected_sha256=None):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata"]))
        arrays = {k: d[k] for k in d.files if k != "metadata"}
    if digest(arrays) != meta["pool_sha256"] or (expected_sha256 and meta["pool_sha256"] != expected_sha256):
        raise ValueError("Goal pool digest mismatch")
    return arrays, meta
