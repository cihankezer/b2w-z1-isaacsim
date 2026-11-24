"""T3 metrics (protocol v0.3 §8-9). Pure numpy, no simulator.

An episode's test phase has K_max = TEST_S / dt control steps at t_k = (k + 1) dt after the test start;
an episode that terminates early has K < K_max recorded (alive) steps. All per-step means are over alive
test-phase steps. Hold windows come from the scenario schedule (inherited from T1 / T2, decision D15).
"""

import numpy as np

from ..reach_height import protocol as T2P
from . import protocol as P

TERMINATION_CLASS = {"F1": "fall", "F2": "fall", "F3": "fall", "S1": "safety", "S2": "safety", "S3": "safety", "S4": "safety",
                     "D": "drift", "invalid": "invalid", None: "none"}   # invalid = non-finite simulator state
# Per-height-band reporting (v0.5): T2's bands by target height above the terrain; targets outside [0, 1.10] m.
BAND_NAMES = T2P.BAND_NAMES + ("outside",)


# ---------------------------------------------------------------- per-step errors

def position_error(p, p_target):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(p_target, float), axis=-1)


def geodesic(q, q_target):
    """Geodesic angle 2 arccos |<q, q*>| [rad] (primary orientation error, T1/T2)."""
    q, qt = np.asarray(q, float), np.asarray(q_target, float)
    d = np.abs(np.sum(q * qt, axis=-1)) / (np.linalg.norm(q, axis=-1) * np.linalg.norm(qt, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def frobenius(q, q_target):
    """Jiang et al. d_theta = ||log(R* R^T)||_F = sqrt(2) x geodesic (secondary)."""
    return np.sqrt(2.0) * geodesic(q, q_target)


def yaw(quat_wxyz):
    w, x, y, z = np.moveaxis(np.asarray(quat_wxyz, float), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_velocity(base_quat_w, lin_vel_w, ang_vel_w):
    """(v_H,x, v_H,y, omega_z) (§5.2)."""
    psi = yaw(base_quat_w)
    v = np.asarray(lin_vel_w, float)
    c, s = np.cos(psi), np.sin(psi)
    return c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], np.asarray(ang_vel_w, float)[:, 2]


def tilt(base_quat_w):
    """Angle between base z-axis and world z [rad]: R[2,2] = 1 - 2(x^2 + y^2)."""
    w, x, y, z = np.moveaxis(np.asarray(base_quat_w, float), -1, 0)
    return np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))


def spherical(p):
    p = np.atleast_2d(np.asarray(p, float))
    return np.stack([np.linalg.norm(p, axis=1), np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=1)),
                     np.arctan2(p[:, 1], p[:, 0])], axis=1)


def legacy_ee_sphere_l1(p_T, p_target_T, region=P.VBC_REGION):
    """DeepWBC-style normalized spherical L1 with the VBC range widths (cross-reference only)."""
    d = np.abs(spherical(p_T) - spherical(p_target_T))
    d[:, 2] = np.abs((d[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    widths = np.array([np.diff(region[k])[0] for k in ("l", "pitch", "yaw")])
    return (d / widths).sum(axis=1)


# ---------------------------------------------------------------- schedules (decision D15)

def hold_ids(scenario, dt=P.CONTROL_DT_S, test_s=P.TEST_S):
    """Per test-phase step: index of the scored hold window it belongs to, or -1 (S4: none)."""
    n = int(round(test_s / dt))
    out = np.full(n, -1, dtype=int)
    if scenario == "S4":
        return out
    sch = P.S1_SCHEDULE if scenario == "S1" else P.S3_SCHEDULE
    per, scored = int(round(sch["hold_s"] / dt)), int(round(sch["scored_last_s"] / dt))
    for k in range(sch["points"]):
        out[(k + 1) * per - scored:(k + 1) * per] = k
    return out


def settle_time(e_vx, e_w, dt=P.CONTROL_DT_S, tol=P.SETTLE_TOLERANCE):
    """First time [s] after the command step at which e_vx < tol and e_w < tol hold for duration_s; NaN if never."""
    ok = (np.asarray(e_vx) < tol["vx"]) & (np.asarray(e_w) < tol["wz"])
    need = int(round(tol["duration_s"] / dt))
    run = 0
    for k, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= need:
            return float((k - need + 1) * dt)
    return float("nan")


# ---------------------------------------------------------------- episode metrics (§8, §9.1)

def episode_metrics(ep, scenario, cmd, dt=P.CONTROL_DT_S, test_s=P.TEST_S):
    """One episode. cmd = (vx*, wz*). ep keys over K alive test-phase steps:
    tcp_pos_w, tcp_quat_w, target_pos_w, target_quat_w (K,3/4); tcp_pos_T, target_pos_T (K,3);
    base_quat_w (K,4), base_lin_vel_w, base_ang_vel_w, base_lin_vel_b (K,3); base_height (K,);
    wheel_pos_body_xy (K,4,2); joint_pos (K,J), joint_limits (J,2) [non-wheel joints];
    joint_vel, torque (K,A), effort_limit (A,) [all actuated joints]; target_height (K,) above the terrain;
    termination_id (None | F1..F3 | S1..S3 | D), t_termination (s after test start or NaN)."""
    k = len(ep["tcp_pos_w"])
    tid = ep.get("termination_id")
    n_max = int(round(test_s / dt))
    e_pos = position_error(ep["tcp_pos_w"], ep["target_pos_w"])
    e_ori = geodesic(ep["tcp_quat_w"], ep["target_quat_w"])
    vx, vy, w = heading_velocity(ep["base_quat_w"], ep["base_lin_vel_w"], ep["base_ang_vel_w"])
    e_vx, e_vy, e_w = np.abs(vx - cmd[0]), np.abs(vy), np.abs(w - cmd[1])
    steady = np.arange(k) >= int(round(P.VELOCITY_TRANSIENT_S / dt))
    hid = hold_ids(scenario, dt, test_s)[:k]
    nan = float("nan")
    mean = lambda x, m=None: float(np.mean(x if m is None else x[m])) if (k if m is None else int(np.sum(m))) else nan  # noqa: E731

    band = T2P.band_index(np.asarray(ep["target_height"], float))   # (K,), -1 = outside
    per_band = {}
    for i, name in enumerate(BAND_NAMES):
        m = band == (i if i < len(T2P.BAND_NAMES) else -1)
        per_band.update({f"e_pos_mean_{name}": mean(e_pos, m), f"e_ori_mean_{name}": mean(e_ori, m),
                         f"holds_{name}": 0, f"holds_ok_{name}": 0})

    holds, hold_pos, hold_ori = [], [], []
    full = hold_ids(scenario, dt, test_s)
    for h in range(full.max() + 1):
        idx = np.flatnonzero(full == h)
        if idx[-1] < k:                       # hold window fully alive
            mp, mo = float(e_pos[idx].mean()), float(e_ori[idx].mean())
            holds.append(mp <= P.SUCCESS_POSITION_M and mo <= P.SUCCESS_ORIENTATION_RAD)
            hold_pos.append(mp)
            hold_ori.append(mo)
            name = BAND_NAMES[band[idx[0]]]   # the target is fixed during a hold; -1 indexes "outside"
            per_band[f"holds_{name}"] += 1
            per_band[f"holds_ok_{name}"] += int(holds[-1])
    first_hold = np.flatnonzero(full == 0)
    # Primary success = source task's rule (D28). T1 (S1): a started point succeeds if it completes without a
    # termination (TASK_T1 summarize: success = completed without failure / started). T2 (S3): the episode
    # completes and every scored step is within 5 cm (position only; terminated = failure).
    if scenario == "S1":
        per = int(round(P.S1_SCHEDULE["hold_s"] / dt))
        started = [j for j in range(P.S1_SCHEDULE["points"]) if j * per < k or j == 0]
        primary = float(np.mean([(j + 1) * per <= k for j in started]))
    elif scenario == "S3":
        primary = float(k >= n_max and tid is None and bool(np.all(e_pos[full == 0] <= P.SUCCESS_POSITION_M)))
    else:
        primary = nan

    q, lim = np.asarray(ep["joint_pos"], float), np.asarray(ep["joint_limits"], float)
    band = P.JOINT_LIMIT_FRACTION * (lim[:, 1] - lim[:, 0])
    near = ((q - lim[:, 0] <= band) | (lim[:, 1] - q <= band)).any(axis=1)
    tau, qd = np.asarray(ep["torque"], float), np.asarray(ep["joint_vel"], float)
    sat = (np.abs(tau) >= np.asarray(ep["effort_limit"], float) * (1 - 1e-3)).any(axis=1)
    wheel_dev = np.linalg.norm(np.asarray(ep["wheel_pos_body_xy"], float) - np.asarray(P.WHEEL_CENTRE_BODY)[None, :, :2],
                               axis=2).sum(axis=1)
    tl = tilt(ep["base_quat_w"])
    h = np.asarray(ep["base_height"], float)
    return dict(
        termination_class=TERMINATION_CLASS[tid], termination_id=tid or "",
        t_termination=float(ep.get("t_termination", nan)) if tid else nan,
        n_alive_steps=k, completed=bool(k >= n_max and tid is None),
        e_pos_mean_all=mean(e_pos), e_pos_p95=float(np.quantile(e_pos, 0.95)) if k else nan,
        e_pos_mean_hold=float(np.mean(hold_pos)) if hold_pos else nan,
        e_ori_mean_all=mean(e_ori), e_ori_mean_hold=float(np.mean(hold_ori)) if hold_ori else nan,
        e_frob_mean_all=mean(frobenius(ep["tcp_quat_w"], ep["target_quat_w"])),
        primary_success_rate=primary,
        hold_success_rate=float(np.mean(holds)) if holds else nan, n_holds_completed=len(holds),
        e_pos_mean_first_hold=float(e_pos[first_hold].mean()) if len(first_hold) and first_hold[-1] < k else nan,
        e_ori_mean_first_hold=float(e_ori[first_hold].mean()) if len(first_hold) and first_hold[-1] < k else nan,
        e_vx_mean=mean(e_vx, steady), e_vy_mean=mean(e_vy, steady), e_w_mean=mean(e_w, steady),
        vel_settle_time=settle_time(e_vx, e_w, dt),
        base_h_mean=mean(h), base_h_min=float(h.min()) if k else nan,
        tilt_mean=mean(tl), tilt_max=float(tl.max()) if k else nan,
        nominal_deviation=mean(wheel_dev),
        joint_limit_frac=mean(near.astype(float)), torque_sat_frac=mean(sat.astype(float)),
        power_mean=mean(np.abs(tau * qd).sum(axis=1)),
        legacy_ee_sphere_l1=mean(legacy_ee_sphere_l1(ep["tcp_pos_T"], ep["target_pos_T"])),
        legacy_vx_l1=mean(np.abs(np.asarray(ep["base_lin_vel_b"], float)[:, 0] - cmd[0])), **per_band)


# ---------------------------------------------------------------- aggregation (§9)

def wilson(successes, n, z=1.959963984540054):
    """Wilson score interval (lower, upper) for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def bootstrap_ci(values, n_resamples=10_000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean over episodes (NaNs dropped)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(len(v), size=(n_resamples, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def termination_rates(rows):
    """Fall / safety / drift rates with Wilson 95 % intervals and mean time to termination per class."""
    n = len(rows)
    out = {}
    for cls in ("fall", "safety", "drift"):
        hit = [r for r in rows if r["termination_class"] == cls]
        out[cls] = dict(rate=len(hit) / n if n else float("nan"), wilson95=wilson(len(hit), n), n=len(hit),
                        t_mean=float(np.mean([r["t_termination"] for r in hit])) if hit else float("nan"))
    return out


PRIMARY = ("e_pos_mean_all", "e_pos_mean_hold", "e_ori_mean_all", "e_ori_mean_hold", "primary_success_rate",
           "hold_success_rate",
           "e_vx_mean", "e_w_mean")


def band_table(rows):
    """Per T2 height band: e_pos and e_ori (per-seed mean of episode band means, then mean +- std over seeds)
    and hold success pooled over holds with a Wilson 95 % interval."""
    seeds = sorted({r["policy_seed"] for r in rows})
    out = {}
    for name in BAND_NAMES:
        e = {}
        for key in ("e_pos_mean", "e_ori_mean"):
            per_seed = [np.nanmean([r[f"{key}_{name}"] for r in rows if r["policy_seed"] == s]
                                   or [float("nan")]) for s in seeds]
            e[key] = dict(mean=float(np.nanmean(per_seed)) if np.isfinite(per_seed).any() else float("nan"),
                          std=float(np.nanstd(per_seed, ddof=1)) if np.isfinite(per_seed).sum() > 1 else float("nan"))
        n, ok = sum(r[f"holds_{name}"] for r in rows), sum(r[f"holds_ok_{name}"] for r in rows)
        out[name] = dict(**e, hold_success=ok / n if n else float("nan"), wilson95=wilson(ok, n), n_holds=n)
    return out


def aggregate(rows, keys=PRIMARY, seed=0):
    """Per method: per policy seed mean over episodes, then mean +- std (ddof 1) over seeds; bootstrap CI
    over all episodes; termination rates pooled over episodes; per-height-band table."""
    seeds = sorted({r["policy_seed"] for r in rows})
    out = dict(n_episodes=len(rows), n_seeds=len(seeds), termination=termination_rates(rows), metrics={},
               bands=band_table(rows))
    for key in keys:
        per_seed = [np.nanmean([r[key] for r in rows if r["policy_seed"] == s]) for s in seeds]
        out["metrics"][key] = dict(mean=float(np.mean(per_seed)),
                                   std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
                                   ci95=bootstrap_ci([r[key] for r in rows], seed=seed))
    return out
