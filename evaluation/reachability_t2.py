"""T2 acceptance test A1: whole-body reachability map (protocol v0.4 §4.3). CPU only, no SimulationApp.

Run inside the Isaac image via evaluation/diagnostics/launch.sh (LAUNCH_TASK=t2, mode cpu):
  --dh-min protocol  dh in [DH_MIN_M, 0] = [-0.33, 0] m (v0.4 training and reachability range)
  --dh-min kaist   dh in [-0.20, 0] m (KAIST command range, reference)
  --dh-min crouch  dh down to the physical crouch limit (deepest feasible dh at pitch 0, legs within soft
                   limits, non-foot colliders >= 5 mm above ground)
Writes reachability_T2_<mode>.npz, reachability_T2_<mode>.json and reachability_T2_<mode>.png to --out.

  --rescore NPZ    no IK: recompute band summary and the current A1 rule from an existing npz and write
                   <npz stem>.rescored.json to --out (the npz itself is not modified)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
import usd_loader  # noqa: E402  (re-execs once with the image's USD libraries)
import numpy as np  # noqa: E402

MANIP = usd_loader.ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"


def t2_module(name):
    """Import a reach_height module (and its standing_ee dependencies) without Isaac Lab or Gym."""
    for pkg, path in (("_manip", MANIP), ("_manip.standing_ee", MANIP / "standing_ee"),
                      ("_manip.reach_height", MANIP / "reach_height")):
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = [str(path)]
            sys.modules[pkg] = module
    return importlib.import_module("_manip." + name)


def side_view(path, data, P, nominal_tcp, title):
    """Side view (least-cost posture over y). Dash-dot contour/colourbar line = KAIST reference dh = -0.20 m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = data["grid"]
    nx, _, nz = g["shape"]
    cost = np.where(data["reachable"], data["postures"]["cost"][np.maximum(data["posture_index"], 0)], np.inf)
    images = {k: np.full((nz, nx), np.nan) for k in ("dh", "pitch")}
    best = np.full((nz, nx), np.inf)
    for v in np.flatnonzero(data["reachable"]):
        i, _, k = g["ijk"][v]
        if cost[v] < best[k, i]:
            best[k, i] = cost[v]
            images["dh"][k, i], images["pitch"][k, i] = data["dh"][v], data["pitch"][v]
    extent = [g["axes"][0][0] - g["resolution"] / 2, g["axes"][0][-1] + g["resolution"] / 2,
              g["axes"][2][0] - g["resolution"] / 2, g["axes"][2][-1] + g["resolution"] / 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, key, label, cmap in ((axes[0], "dh", "required Δh [m]", "viridis"), (axes[1], "pitch", "required pitch [rad] (+ = nose down)", "coolwarm")):
        lim = (np.nanmin(images["dh"]), 0.0) if key == "dh" else (-P.REACHABILITY["pitch_max_rad"], P.REACHABILITY["pitch_max_rad"])
        im = ax.imshow(images[key], origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="equal")
        bar = fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        if key == "dh":
            kaist = P.REACHABILITY["dh_min_kaist_m"]
            if lim[0] < kaist:
                bar.ax.axhline(kaist, color="r", lw=1.5, ls="-.")
                ax.contour(images["dh"], levels=[kaist], origin="lower", extent=extent, colors="r",
                           linewidths=1.0, linestyles="-.")
            ax.plot([], [], "r-.", label=f"KAIST reference Δh = {kaist:+.2f} m")
        for _, lo, hi in P.BANDS:
            ax.axhline(hi, color="k", lw=0.6, ls="--")
        ax.plot(nominal_tcp[0], nominal_tcp[2], "r*", ms=10, label="nominal TCP")
        ax.set_xlabel("x in G [m] (forward)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("TCP height z [m]")
    fig.suptitle(title + "\nside view, least-cost posture over y; dashed = band edges 0.30 / 0.80 / 1.10 m", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def rescore(args, P, R):
    """Re-score an existing npz with the current band summary and A1 rule (protocol v0.4)."""
    with np.load(args.rescore, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    meta = json.loads(str(data["metadata"]))
    grid = R.grid_from_npz(data)
    band = np.asarray(data["voxel_band"])
    reachable = np.asarray(data["reachable"], dtype=bool)
    spec = dict(meta["reachability"])
    spec.update(min_largest_region_fraction=P.REACHABILITY["min_largest_region_fraction"],
                min_band_reachable_fraction=P.REACHABILITY["min_band_reachable_fraction"])
    summary = R.band_summary(grid, band, reachable, np.asarray(data["dh"], dtype=float))
    a1 = R.a1_verdict(summary, grid, reachable, spec)
    try:   # would the environment accept this file? (version + full-grid check of the env loader)
        t2_module("reach_height.sampler").ReachableSet.load(args.rescore, require_full_grid=True)
        env_loadable = True
    except ValueError as exc:
        env_loadable = f"refused: {exc}"
    report = dict(env_loadable=env_loadable, source_npz=str(args.rescore), source_npz_sha256=hashlib.sha256(args.rescore.read_bytes()).hexdigest(),
                  source_protocol_version=meta.get("protocol_version"), source_reachability_version=spec.get("version"),
                  rescored_with_protocol_version=P.VERSION, dh_mode=meta.get("dh_mode"), dh_min_m=meta.get("dh_min_m"),
                  grid_overridden=meta.get("grid_overridden"), bands=summary, A1=a1)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.rescore.stem}.rescored.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(bands=summary, A1=a1, env_loadable=env_loadable), indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dh-min", choices=("protocol", "kaist", "crouch"))
    parser.add_argument("--rescore", type=Path, default=None, help="existing reachability npz to re-score (no IK)")
    parser.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--starts", type=int, default=4, help="IK starts: nominal arm pose + (starts-1) seeded random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=None, help="override the protocol grid (flagged in output)")
    args = parser.parse_args()
    if (args.rescore is None) == (args.dh_min is None):
        parser.error("give exactly one of --dh-min or --rescore")
    P = t2_module("reach_height.protocol")
    R = t2_module("reach_height.reachability")
    if args.rescore is not None:
        return rescore(args, P, R)

    from pxr import Usd
    W = t2_module("reach_height.wholebody")
    K = t2_module("standing_ee.kinematics")
    T1P = t2_module("standing_ee.protocol")

    start = time.time()
    stage = Usd.Stage.Open(str(usd_loader.ASSET))
    root = next(p for p in stage.Traverse() if p.GetName() == "base_link").GetParent()
    model = K.UsdArmKinematics(stage, str(root.GetPath()))
    wb = W.WholeBody(model)
    crouch = None
    if args.dh_min == "protocol":
        dh_min = P.DH_MIN_M
    elif args.dh_min == "kaist":
        dh_min = P.REACHABILITY["dh_min_kaist_m"]
    else:
        dh_min, stop = R.crouch_limit(wb)
        crouch = dict(dh_min_m=dh_min, base_height_m=P.H_NOM_M + dh_min, first_infeasible=stop,
                      search_step_m=P.CROUCH_SEARCH_STEP_M)
        print(f"crouch limit: dh = {dh_min:+.2f} m (base {P.H_NOM_M + dh_min:.3f} m); next step: {stop}", flush=True)
    spec = dict(P.REACHABILITY)
    if args.resolution:
        spec["grid_resolution_m"] = args.resolution

    data = R.analyse(model, dh_min, workers=args.workers, starts=args.starts, seed=args.seed, spec=spec,
                     log=lambda s: print(s, flush=True))
    summary = data["summary"]
    a1 = R.a1_verdict(summary, data["grid"], data["reachable"], spec)

    tag = f"reachability_T2_{args.dh_min}"
    args.out.mkdir(parents=True, exist_ok=True)
    nominal_tcp_G = (np.linalg.inv(wb.G) @ wb.nominal_tcp_world)[:3, 3]
    metadata = dict(benchmark=P.BENCHMARK, protocol_version=P.VERSION, reachability=spec, dh_mode=args.dh_min,
                    dh_min_m=dh_min, crouch=crouch, grid_overridden=args.resolution is not None, seed=args.seed,
                    ik_starts=args.starts, asset=str(usd_loader.ASSET), asset_hashes=T1P.asset_hashes(usd_loader.ASSET),
                    h_nom_m=P.H_NOM_M, tcp_parent=P.TCP_PARENT, tcp_offset_m=list(P.TCP_OFFSET_M), approach_profile=[list(x) for x in P.APPROACH_PROFILE],
                    nominal_tcp_G=nominal_tcp_G.tolist(), image_id=os.environ.get("T1_IMAGE_ID"))
    R.save(args.out / f"{tag}.npz", data, metadata)
    npz_sha = hashlib.sha256((args.out / f"{tag}.npz").read_bytes()).hexdigest()
    p = data["postures"]
    report = dict(metadata, npz=f"{tag}.npz", npz_sha256=npz_sha, bands=summary, A1=a1,
                  reach_radius_m=data["reach_radius"], voxels_in_sector=int(len(data["reachable"])),
                  voxels_reachable=int(data["reachable"].sum()), voxels_ik_but_collision=int((data["ik_any"] & ~data["reachable"]).sum()),
                  postures=dict(total=int(len(p["dh"])), feasible=int(p["feasible"].sum()),
                                infeasible={f"{d:+.2f}/{q:+.2f}": r for d, q, r, f in zip(p["dh"], p["pitch"], p["reason"], p["feasible"]) if not f}),
                  rejections=data["rejections"], wall_s=round(time.time() - start, 1))
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    side_view(args.out / f"{tag}.png", data, P, nominal_tcp_G,
              f"T2 reachability ({args.dh_min}: dh >= {dh_min:+.2f} m, |pitch| <= {spec['pitch_max_rad']} rad)")
    print(json.dumps(dict(bands=summary, A1=a1, wall_s=report["wall_s"]), indent=1), flush=True)
    (args.out / "completion.json").write_text(json.dumps(dict(completed=True, tag=tag)) + "\n")


if __name__ == "__main__":
    main()
