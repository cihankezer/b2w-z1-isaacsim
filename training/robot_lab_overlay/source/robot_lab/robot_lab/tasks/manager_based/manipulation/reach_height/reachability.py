"""T2 reachability map (protocol v0.4 §4.3). CPU only, deterministic for a given seed.

For every voxel centre of the G-frame grid inside the ±60° sector, search the base postures
(dh, pitch) in increasing posture cost; a voxel is reachable at the first posture for which the legs
reach the fixed wheel centres, the arm reaches the TCP target with the Mode A canonical orientation
(6D, within soft limits) and the whole configuration passes the T1 clearance check (ground and arm
self-clearance >= 5 mm on the collider geometry). The recorded (dh, pitch) is that least-cost posture.
"""

import json
import multiprocessing as mp

import numpy as np
from scipy import ndimage

from . import protocol as P
from .wholebody import WholeBody, arm_ik, canonical_rotation, clearance_violation, full_joints, tcp_world

_STATE = {}   # inherited by forked workers: {"wb": WholeBody}


def voxel_grid(spec=P.REACHABILITY):
    """Voxel centres (G frame) of the protocol grid whose azimuth lies inside the sector."""
    res = spec["grid_resolution_m"]
    axes = [np.round(np.arange(lo + res / 2, hi - 1e-9, res), 9)
            for lo, hi in (spec["grid_x_m"], spec["grid_y_m"], spec["grid_z_m"])]
    shape = tuple(len(a) for a in axes)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    centres = np.stack([axes[d][ijk[:, d]] for d in range(3)], axis=1)
    theta = np.arctan2(centres[:, 1], centres[:, 0])
    keep = np.abs(theta) <= P.AZIMUTH_MAX_RAD + 1e-12
    return dict(axes=axes, shape=shape, ijk=ijk[keep], centres=centres[keep], theta=theta[keep], resolution=res)


def crouch_limit(wb, step=P.CROUCH_SEARCH_STEP_M, floor=-0.6):
    """Deepest dh (multiple of `step`) at pitch 0 for which every shallower posture is feasible."""
    last, dh = None, 0.0
    while dh >= floor - 1e-9:
        _, joints, reason = wb.posture(dh, 0.0)
        if joints is None:
            if last is None:
                raise ValueError(f"Nominal posture infeasible: {reason}")
            return last, f"dh={dh:.2f}:{reason}"
        last, dh = dh, round(dh - step, 6)
    return last, "search_floor"


def _ik_task(args):
    targets, q0 = args
    q, ok, _, _ = arm_ik(_STATE["wb"].arm, targets, np.repeat(q0[None], len(targets), axis=0), _STATE["wb"].arm_limits)
    return q, ok


def _check_task(args):
    base, legs, items = args
    model, out = _STATE["wb"].model, []
    for voxel, solutions in items:
        reasons = []
        for q in solutions:
            reason = clearance_violation(model, model.forward(base, full_joints(legs, q)))
            if reason is None:
                out.append((voxel, q, reasons))
                break
            reasons.append(reason)
        else:
            out.append((voxel, None, reasons))
    return out


def _chunks(n, parts):
    return [c for c in np.array_split(np.arange(n), max(1, parts)) if len(c)]


def analyse(model, dh_min, workers=8, starts=4, seed=0, spec=P.REACHABILITY, log=print):
    """Run the reachability search. Returns a dict of arrays and summaries (see save())."""
    wb = WholeBody(model)
    grid = voxel_grid(spec)
    band = P.band_index(grid["centres"][:, 2])
    rotations = canonical_rotation(grid["theta"], grid["centres"][:, 2])
    tcp_T = tcp_world(wb.G, grid["centres"], rotations)
    tcp, link06 = tcp_T[:, :3, 3], wb.tcp_to_link06(tcp_T)
    reach = wb.reach_radius(seed=seed)
    rng = np.random.default_rng(seed)
    q_starts = np.vstack([[wb.joints_nominal[n] for n in wb.arm.joint_names],
                          rng.uniform(wb.arm_limits[:, 0], wb.arm_limits[:, 1], size=(starts - 1, 6))])

    dhs, pitches = P.dh_values(dh_min, spec["dh_step_m"]), P.pitch_values(spec["pitch_max_rad"], spec["pitch_step_rad"])
    table = sorted(((P.posture_cost(dh, p), -dh, abs(p), p, dh) for dh in dhs for p in pitches))
    postures = dict(dh=np.array([t[4] for t in table]), pitch=np.array([t[3] for t in table]),
                    cost=np.array([t[0] for t in table]))
    n_post, n_vox = len(table), len(grid["centres"])
    postures.update(feasible=np.zeros(n_post, bool), reason=np.array([""] * n_post, dtype=object),
                    leg_q=np.full((n_post, 12), np.nan), base_world=np.zeros((n_post, 4, 4)),
                    newly_reachable=np.zeros(n_post, int), ik_converged=np.zeros(n_post, int),
                    collision_rejected=np.zeros(n_post, int))
    leg_names = [n for leg in wb.legs.values() for n in leg.joint_names]
    result = dict(posture_index=np.full(n_vox, -1), arm_q=np.full((n_vox, 6), np.nan))
    ik_any = np.zeros(n_vox, bool)
    rejection = {}

    _STATE["wb"] = wb
    with mp.get_context("fork").Pool(workers) as pool:
        for k in range(n_post):
            base, joints, reason = wb.posture(postures["dh"][k], postures["pitch"][k])
            postures["base_world"][k] = base
            postures["reason"][k] = reason or ""
            if joints is None:
                log(f"posture {k:3d} dh={postures['dh'][k]:+.2f} pitch={postures['pitch'][k]:+.2f} infeasible: {reason}")
                continue
            postures["feasible"][k] = True
            postures["leg_q"][k] = [joints[n] for n in leg_names]
            shoulder = (base @ np.r_[wb.shoulder_base, 1.0])[:3]
            open_ = np.flatnonzero((result["posture_index"] < 0) & (np.linalg.norm(tcp - shoulder, axis=1) <= reach))
            if not len(open_):
                continue
            targets = np.linalg.inv(base)[None] @ link06[open_]
            solutions = [[] for _ in open_]
            for q0 in q_starts:
                parts = _chunks(len(open_), workers * 2)
                for c, (q, ok) in zip(parts, pool.map(_ik_task, [(targets[c], q0) for c in parts])):
                    for i, qi, oki in zip(c, q, ok):
                        if oki and not any(np.allclose(qi, s, atol=1e-3) for s in solutions[i]):
                            solutions[i].append(qi)
            items = [(int(open_[i]), s) for i, s in enumerate(solutions) if s]
            ik_any[[v for v, _ in items]] = True
            postures["ik_converged"][k] = len(items)
            parts = _chunks(len(items), workers * 4)
            for out in pool.map(_check_task, [(base, joints, [items[j] for j in c]) for c in parts]):
                for voxel, q, reasons in out:
                    for r in reasons:
                        key = r.split(":")[0] + ":" + ":".join(r.split(":")[1:3])
                        rejection[key] = rejection.get(key, 0) + 1
                    if q is not None:
                        result["posture_index"][voxel] = k
                        result["arm_q"][voxel] = q
                        postures["newly_reachable"][k] += 1
                    else:
                        postures["collision_rejected"][k] += 1
            log(f"posture {k:3d} dh={postures['dh'][k]:+.2f} pitch={postures['pitch'][k]:+.2f}: "
                f"{len(open_)} open, {len(items)} IK, {postures['newly_reachable'][k]} reachable "
                f"(total {(result['posture_index'] >= 0).sum()})")
    _STATE.clear()

    idx = result["posture_index"]
    reachable = idx >= 0
    return dict(grid=grid, band=band, reachable=reachable, posture_index=idx,
                dh=np.where(reachable, postures["dh"][idx], np.nan),
                pitch=np.where(reachable, postures["pitch"][idx], np.nan),
                arm_q=result["arm_q"], ik_any=ik_any, postures=postures, leg_joint_names=leg_names,
                G=wb.G, reach_radius=reach, q_starts=q_starts, dh_min=float(dh_min), rejections=rejection,
                summary=band_summary(grid, band, reachable, np.where(reachable, postures["dh"][idx], np.nan)))


def band_summary(grid, band, reachable, dh=None):
    """Per band: reachable voxel count, two volume fractions and connected regions (protocol v0.4 §4.3).
    sector: of all sector voxels of the band; annulus (normative for A1): of the band's sector voxels with
    r in [r_min, r_max] of its reachable voxels (r = horizontal distance from the G origin).
    regions: connected components of the band's reachable voxels (6-neighbour faces).
    dh_below_kaist_fraction: share of the band's reachable voxels whose least-cost posture needs
    dh < REACHABILITY["dh_min_kaist_m"] (only if per-voxel dh is given)."""
    r = np.linalg.norm(grid["centres"][:, :2], axis=1)
    out = {}
    for i, name in enumerate(P.BAND_NAMES):
        in_band = band == i
        hit = in_band & reachable
        entry = dict(voxels=int(in_band.sum()), reachable_voxels=int(hit.sum()),
                     sector_fraction=float(hit.sum() / in_band.sum()) if in_band.any() else 0.0,
                     r_min_m=None, r_max_m=None, annulus_fraction=0.0, z_min_m=None, z_max_m=None,
                     regions=0, largest_region_voxels=0, largest_region_fraction=0.0, dh_below_kaist_fraction=None)
        if hit.any():
            lo, hi = r[hit].min(), r[hit].max()
            annulus = in_band & (r >= lo - 1e-9) & (r <= hi + 1e-9)
            z = grid["centres"][hit, 2]
            half = grid["resolution"] / 2
            entry.update(r_min_m=float(lo), r_max_m=float(hi), annulus_fraction=float(hit.sum() / annulus.sum()),
                         z_min_m=float(z.min() - half), z_max_m=float(z.max() + half))
            volume = np.zeros(grid["shape"], dtype=bool)
            volume[tuple(grid["ijk"][hit].T)] = True
            labels, n = ndimage.label(volume)
            largest = int(np.bincount(labels.ravel())[1:].max())
            entry.update(regions=int(n), largest_region_voxels=largest, largest_region_fraction=largest / int(hit.sum()))
            if dh is not None:
                entry["dh_below_kaist_fraction"] = float(np.mean(dh[hit] < P.REACHABILITY["dh_min_kaist_m"] - 1e-9))
        out[name] = entry
    return out


def a1_verdict(summary, grid, reachable, spec=P.REACHABILITY):
    """Acceptance test A1 (v0.4): per band annulus fraction >= 20 % and largest connected region >= 90 % of
    the band's reachable voxels. Sector fraction and empty z layers are information only."""
    names = list(summary)
    a1 = dict(
        criterion=f"v0.4: every band annulus fraction >= {spec['min_band_reachable_fraction']:.0%} and largest "
                  f"connected region (6-neighbour) >= {spec['min_largest_region_fraction']:.0%} of its reachable voxels",
        pass_annulus_fraction=all(summary[k]["annulus_fraction"] >= spec["min_band_reachable_fraction"] for k in names),
        pass_largest_region=all(summary[k]["reachable_voxels"] > 0 and
                                summary[k]["largest_region_fraction"] >= spec["min_largest_region_fraction"] for k in names),
        reachable_voxels={k: summary[k]["reachable_voxels"] for k in names},
        regions={k: summary[k]["regions"] for k in names},
        largest_region_fraction={k: summary[k]["largest_region_fraction"] for k in names},
        dh_below_kaist_fraction={k: summary[k]["dh_below_kaist_fraction"] for k in names},
        info_sector_fraction_ge_min=all(summary[k]["sector_fraction"] >= spec["min_band_reachable_fraction"] for k in names))
    a1["pass"] = a1["pass_annulus_fraction"] and a1["pass_largest_region"]
    z_axis, res = grid["axes"][2], grid["resolution"]
    hit_layers = set(np.unique(grid["ijk"][reachable, 2]).tolist())
    a1["empty_z_layers_m"] = [[round(z - res / 2, 4), round(z + res / 2, 4)] for k, z in enumerate(z_axis)
                              if z < P.BANDS[-1][2] and k not in hit_layers]
    return a1


def grid_from_npz(d):
    """Rebuild the grid dict used by band_summary / a1_verdict from a saved reachability npz."""
    axes = [np.asarray(d[k], dtype=float) for k in ("grid_x", "grid_y", "grid_z")]
    return dict(axes=axes, resolution=float(d["resolution"]), shape=tuple(len(a) for a in axes),
                ijk=np.asarray(d["voxel_ijk"]), centres=np.asarray(d["voxel_centre_G"]), theta=np.asarray(d["voxel_theta"]))


def save(path, data, metadata):
    """Write the reachability npz (reachability_T2*.npz)."""
    g, p = data["grid"], data["postures"]
    volume = np.zeros(g["shape"], dtype=bool)
    volume[tuple(g["ijk"][data["reachable"]].T)] = True
    np.savez_compressed(
        path, grid_x=g["axes"][0], grid_y=g["axes"][1], grid_z=g["axes"][2], resolution=g["resolution"],
        reachable_volume=volume, voxel_ijk=g["ijk"], voxel_centre_G=g["centres"], voxel_theta=g["theta"],
        voxel_band=data["band"], reachable=data["reachable"], ik_converged_any=data["ik_any"],
        posture_index=data["posture_index"], dh=data["dh"], pitch=data["pitch"], arm_q=data["arm_q"],
        posture_dh=p["dh"], posture_pitch=p["pitch"], posture_cost=p["cost"], posture_feasible=p["feasible"],
        posture_reason=p["reason"].astype(str), posture_leg_q=p["leg_q"], posture_base_world=p["base_world"],
        leg_joint_names=np.array(data["leg_joint_names"]), G_world=data["G"], tcp_offset=np.array(P.TCP_OFFSET_M),
        h_nom=P.H_NOM_M, dh_min=data["dh_min"], reach_radius=data["reach_radius"], ik_starts=data["q_starts"],
        metadata=np.array(json.dumps(metadata, sort_keys=True)))
