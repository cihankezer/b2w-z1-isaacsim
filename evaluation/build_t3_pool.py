"""Build the T3 collision-checked FK goal pool (protocol v0.3 §6.2-6.4). CPU only, no SimulationApp.

  --which train   POOL_SIZES["train"], TRAIN_POOL_SEED (M1 + M2 + M3)
  --which eval    POOL_SIZES["eval"], EVAL_POOL_SEED (M1 only, for scenario S4)
  --scale F       multiply the goal counts by F (smoke runs; not a protocol pool)
Writes t3_pool_<which>.npz (+ .json with metadata and statistics) to --out.

Run: LAUNCH_TASK=t3 evaluation/diagnostics/launch.sh cpu pool-train 14400 evaluation/build_t3_pool.py \
       --which train --workers 12 --out /t1-run/artifacts
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_t3 import FROZEN_MAP, load_model, module  # noqa: E402  (imports usd_loader first)
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("train", "eval"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=os.environ.get("T2_REACHABILITY_NPZ", str(FROZEN_MAP)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    P = module("walk_reach.protocol")
    pool = module("walk_reach.pool")
    S = module("reach_height.sampler")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.map, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    S.check_reachability_metadata(data, require_full_grid=True)
    rs = S.ReachableSet(data)
    rs.raw = dict(dh=data["dh"], pitch=data["pitch"], arm_q=data["arm_q"])

    sizes = dict(P.POOL_SIZES[args.which])
    for k in ("M1", "M2", "M3"):
        sizes[k] = int(round(sizes[k] * args.scale))
    seed = P.TRAIN_POOL_SEED if args.which == "train" else P.EVAL_POOL_SEED
    arrays, meta = pool.build(load_model(), rs, seed, sizes, workers=args.workers,
                              log=lambda m: print(m, flush=True))
    meta.update(which=args.which, scale=args.scale, protocol_pool=args.scale == 1.0)
    name = f"t3_pool_{args.which}" + ("" if args.scale == 1.0 else f"_x{args.scale:g}")
    pool.save(out / f"{name}.npz", arrays, meta)
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta["stats"], indent=1, default=float))
    print("pool_sha256", meta["pool_sha256"])


if __name__ == "__main__":
    main()
