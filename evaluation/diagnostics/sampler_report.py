"""Draft.7 sampler report: counts, filter rejections, EE distribution (CPU, T1 path, no physics).

    sampler_report.py <out.json> --sample SEED [--sample SEED ...]   draw on CPU from the T1 asset
    sampler_report.py <out.json> --targets PATH [--targets PATH ...] summarise existing targets.json
Distribution: position ranges (fixed targets; circle centres), 3 equal-width bins over the observed
range of z and of the orientation deviation (geodesic) from the nominal link06 orientation.
"""
import argparse
import json
import time
from pathlib import Path

import usd_loader  # noqa: F401  (must be first)
import numpy as np
from scipy.spatial.transform import Rotation

p = usd_loader.t1_module("protocol")


def histogram(values):
    counts, edges = np.histogram(values, bins=3)
    return dict(edges=edges.tolist(), counts=counts.tolist())


def summarise(data, wall_s=None):
    nominal = Rotation.from_quat(np.roll(data["nominal_ee_pose_env"][3:], -1))
    out = dict(seed=data["seed"], counts=data["counts"], protocol_counts=data.get("protocol_counts"),
               attempts=data["attempts"], acceptance_rate={f: data["counts"][f] / data["attempts"][f]
                                                           for f in data["counts"] if data["attempts"][f]},
               rejections=data["rejections"], top_rejection_details=dict(list(data["rejection_details"].items())[:10]),
               approx_hull_colliders=len(data["approx_hull_colliders"]), sampling_wall_s=wall_s,
               nominal_ee_position_env=data["nominal_ee_pose_env"][:3], distribution={})
    for family, key in (("fixed", "position_world_xyz"), ("circle", "centre_world_xyz")):
        targets = [t for t in data["targets"] if t["family"] == family]
        if not targets:
            continue
        xyz = np.array([t[key] for t in targets])
        deviation = np.degrees([(nominal.inv() * Rotation.from_quat(np.roll(t["quaternion_world_wxyz"], -1))).magnitude()
                                for t in targets])
        out["distribution"][family] = dict(
            position=key, min_xyz=xyz.min(axis=0).tolist(), max_xyz=xyz.max(axis=0).tolist(),
            z_histogram_m=histogram(xyz[:, 2]), orientation_deviation_deg=histogram(deviation),
            orientation_deviation_deg_min_median_max=[float(deviation.min()), float(np.median(deviation)), float(deviation.max())])
    return out


parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--sample", type=int, action="append", default=[])
parser.add_argument("--targets", type=Path, action="append", default=[])
args = parser.parse_args()
reports = []
if args.sample:
    from pxr import Usd
    k, t = usd_loader.t1_module("kinematics"), usd_loader.t1_module("targets")
    model = k.UsdArmKinematics(Usd.Stage.Open(str(usd_loader.ASSET)), "/Unitree_B2W")
    hashes = p.asset_hashes(usd_loader.ASSET)
    for seed in args.sample:
        start = time.monotonic()
        data = t.sample(model, seed, hashes)
        wall = time.monotonic() - start
        (args.out.parent / f"targets-cpu-{seed}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        reports.append(summarise(data, wall))
        print("seed", seed, "sampled in", round(wall, 1), "s", flush=True)
for path in args.targets:
    reports.append(dict(summarise(p.load_targets(path)), source=str(path)))
args.out.write_text(json.dumps(dict(protocol_version=p.VERSION, sampler=p.SAMPLER, asset=str(usd_loader.ASSET),
                                    reports=reports), indent=1) + "\n")
print("written", args.out, flush=True)
