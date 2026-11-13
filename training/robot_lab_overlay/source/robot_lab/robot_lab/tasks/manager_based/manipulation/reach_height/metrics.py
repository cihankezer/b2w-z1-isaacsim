"""T2 per-episode metrics (protocol v0.4 §5) and aggregation (§5.6). Pure numpy, no simulator.

An episode is a sequence of K recorded control steps at t_k = (k + 1) * dt (post-step states).
Hold window = the last round(HOLD_S / dt) steps of an episode that reached the horizon; for an
episode that terminated early, success is False and every hold-window metric is NaN.
"""

import numpy as np

from . import protocol as P


def tilt(projected_gravity):
    """Angle between base z-axis and world up [rad] from the projected gravity direction in B."""
    g = np.atleast_2d(np.asarray(projected_gravity, dtype=float))
    return np.arccos(np.clip(-g[:, 2] / np.linalg.norm(g, axis=1), -1.0, 1.0))


def roll_pitch(quat_wxyz):
    """Signed roll and pitch [rad] of base quaternions (wxyz), Isaac Lab euler_xyz convention (yaw-independent)."""
    q = np.atleast_2d(np.asarray(quat_wxyz, dtype=float))
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def orientation_error(q_actual, q_target):
    """Geodesic angle [rad] between quaternions (wxyz), sign-invariant."""
    a, b = np.atleast_2d(q_actual), np.atleast_2d(q_target)
    dot = np.abs(np.sum(a * b, axis=1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 2 * np.arccos(np.clip(dot, -1.0, 1.0))


def contact_losses(contact):
    """Number of foot-contact loss events (True -> False transitions), summed over feet."""
    c = np.asarray(contact, dtype=bool)
    return int(np.sum(c[:-1] & ~c[1:])) if len(c) > 1 else 0


def near_limit_fraction(q, soft_limits, margin=P.JOINT_LIMIT_MARGIN_FRACTION):
    """Fraction of steps with any joint within margin * (soft range) of a soft limit (D11)."""
    q, lim = np.atleast_2d(q), np.asarray(soft_limits, dtype=float)
    band = margin * (lim[:, 1] - lim[:, 0])
    near = (q - lim[:, 0] <= band) | (lim[:, 1] - q <= band)
    return float(near.any(axis=1).mean())


def episode_metrics(ep, dt=P.CONTROL_DT_S, horizon_s=P.EPISODE_S, hold_s=P.HOLD_S, h_nom=P.H_NOM_M,
                    threshold=P.SUCCESS_POSITION_M, orientation_threshold=P.SUCCESS_ORIENTATION_RAD, mode="A"):
    """Section 5 metrics for one episode.

    ep keys (K recorded steps): tcp_pos_w (K,3), target_pos_w (3,), tcp_quat_w (K,4), target_quat_w (4,),
    base_pos_w (K,3) [z above ground], base_quat_w (K,4), projected_gravity_b (K,3), foot_contact (K,4),
    arm_q (K,6), arm_soft_limits (6,2), arm_torque (K,6), leg_torque (K,12), reset_base_pos_w (3,),
    fell (bool), walked_away (bool).
    """
    e_p = np.linalg.norm(np.asarray(ep["tcp_pos_w"]) - np.asarray(ep["target_pos_w"]), axis=1)
    e_r = orientation_error(ep["tcp_quat_w"], np.asarray(ep["target_quat_w"])[None])
    k = len(e_p)
    horizon = int(round(horizon_s / dt))
    completed = bool(k >= horizon and not ep["fell"] and not ep["walked_away"])
    n_hold = int(round(hold_s / dt))
    hold = slice(k - n_hold, k) if completed else None
    t = (np.arange(k) + 1) * dt
    h = np.asarray(ep["base_pos_w"])[:, 2]
    tl = tilt(ep["projected_gravity_b"])
    roll, pitch = roll_pitch(ep["base_quat_w"])
    reached = np.flatnonzero(e_p <= threshold)
    displacement = np.linalg.norm(np.asarray(ep["base_pos_w"])[:, :2] - np.asarray(ep["reset_base_pos_w"])[:2], axis=1)
    nan = float("nan")

    def over_hold(values, fn=np.mean):
        return float(fn(values[hold])) if hold is not None else nan

    hold_ok = bool(hold is not None and np.all(e_p[hold] <= threshold)
                   and (mode == "A" or np.all(e_r[hold] <= orientation_threshold)))
    final_ok = bool(completed and e_p[-1] <= threshold and (mode == "A" or e_r[-1] <= orientation_threshold))
    tilt_hold_mean = over_hold(tl)
    h_hold_mean = over_hold(h)
    pitch_hold = over_hold(pitch)
    out = dict(
        completed=completed, success=hold_ok, success_kaist_strict=final_ok,
        e_p_final=float(e_p[-1]), e_R_final=float(e_r[-1]), e_p_hold_mean=over_hold(e_p),
        t_reach=float(t[reached[0]]) if len(reached) else nan,
        h_final=float(h[-1]), h_hold_mean=h_hold_mean, dh_hold=h_hold_mean - h_nom, h_min=float(h.min()),
        tilt_hold_mean=tilt_hold_mean, tilt_max=float(tl.max()),
        pitch_hold_mean=pitch_hold, roll_hold_mean=over_hold(roll),
        tilt_rms_hold=float(np.sqrt(np.mean((tl[hold] - tilt_hold_mean) ** 2))) if hold is not None else nan,
        fell=bool(ep["fell"]), walked_away=bool(ep["walked_away"]), max_base_displacement=float(displacement.max()),
        n_foot_lifts=contact_losses(ep["foot_contact"]),
        joint_limit_violation_frac=near_limit_fraction(ep["arm_q"], ep["arm_soft_limits"]),
        max_joint_torque_arm=float(np.abs(ep["arm_torque"]).max()),
        max_joint_torque_leg=float(np.abs(ep["leg_torque"]).max()),
        n_steps=k)
    lo, hi = P.REFERENCE_DH_BAND_M
    out["posture_outside_reference_band"] = bool(completed and (not lo <= out["dh_hold"] <= hi
                                                                or abs(pitch_hold) > P.REFERENCE_PITCH_RAD))
    return out


def mean_std(values):
    """Mean and sample std (ddof=1, None for n < 2) over seeds, as in T1."""
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))


def _group(rows):
    def col(key):
        return np.array([r[key] for r in rows], dtype=float)
    e = col("e_p_final")
    return dict(n=len(rows), success=float(col("success").mean()), success_kaist_strict=float(col("success_kaist_strict").mean()),
                e_p_final_median=float(np.median(e)), e_p_final_p90=float(np.percentile(e, 90)),
                dh_hold_mean=float(np.nanmean(col("dh_hold"))) if np.isfinite(col("dh_hold")).any() else None,
                pitch_hold_mean=float(np.nanmean(col("pitch_hold_mean"))) if np.isfinite(col("pitch_hold_mean")).any() else None,
                tilt_rms_hold_mean=float(np.nanmean(col("tilt_rms_hold"))) if np.isfinite(col("tilt_rms_hold")).any() else None,
                fall_rate=float(col("fell").mean()), walk_away_rate=float(col("walked_away").mean()))


def aggregate(rows):
    """§5.6: per seed and band (plus 'all'), then mean ± std over seeds. rows need 'seed' and 'band' (name)."""
    seeds = sorted({r["seed"] for r in rows})
    groups = list(P.BAND_NAMES) + ["all"]
    per_seed = {s: {g: _group([r for r in rows if r["seed"] == s and (g == "all" or r["band"] == g)]) for g in groups}
                for s in seeds}
    keys = [k for k in per_seed[seeds[0]]["all"] if k != "n"] if seeds else []
    return dict(protocol_version=P.VERSION, n_seeds=len(seeds), per_seed=per_seed,
                over_seeds={g: {k: mean_std([per_seed[s][g][k] for s in seeds]) for k in keys} for g in groups})


def height_bins(rows, width=0.1, z_max=P.BANDS[-1][2]):
    """Mandatory figure data: success rate and median final error per target-height bin (pooled over seeds)."""
    edges = np.round(np.arange(0.0, z_max + width / 2, width), 9)
    z = np.array([r["target_z"] for r in rows])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, zi in zip(rows, z) if lo <= zi < hi or (hi == edges[-1] and zi == hi)]
        out.append(dict(z_lo=float(lo), z_hi=float(hi), n=len(sel),
                        success=float(np.mean([r["success"] for r in sel])) if sel else None,
                        e_p_final_median=float(np.median([r["e_p_final"] for r in sel])) if sel else None))
    return out
