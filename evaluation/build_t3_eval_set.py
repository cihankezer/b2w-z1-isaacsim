"""Build the fixed T3 evaluation set (protocol v0.3 §7.4). CPU only, no SimulationApp.

Inputs (hash-checked): T1 evaluation targets (seed 1101), frozen T2 map (T2 eval set regenerated, seed 2026),
T3 evaluation goal pool. Writes t3_eval_set_v5.npz and .json to --out.

Run: LAUNCH_TASK=t3 evaluation/diagnostics/launch.sh cpu eval-set 1800 evaluation/build_t3_eval_set.py \
       --pool /workspace/<run>/artifacts/t3_pool_eval.npz --out /t1-run/artifacts
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_t3 import FROZEN_MAP, module  # noqa: E402  (imports usd_loader first)
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T1_TARGETS = ROOT / "repos/robot_lab/logs/t3/diag-t1v2-validate-20261123T204134Z-26af8a39/artifacts/validation/targets.json"


def inputs(t1_path, map_path, pool_path):
    """Load and hash-check the three inputs. Returns (t1_data, t2_arrays, pool_arrays)."""
    P = module("walk_reach.protocol")
    S = module("reach_height.sampler")
    pool = module("walk_reach.pool")
    raw = Path(t1_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != P.T1_EVAL_TARGETS_SHA256:
        raise ValueError("T1 evaluation target file hash differs")
    t1 = json.loads(raw)
    if t1["seed"] != P.T1_EVAL_SEED:
        raise ValueError("T1 target set seed differs")
    if hashlib.sha256(Path(map_path).read_bytes()).hexdigest() != P.T2_MAP_SHA256:
        raise ValueError("T2 map hash differs")
    rs = S.ReachableSet.load(map_path)
    t2, meta = S.eval_target_set(rs, P.T2_EVAL_SEED, P.T2_MAP_SHA256)
    if meta["targets_sha256"] != P.T2_EVAL_SET_SHA256:
        raise ValueError("Regenerated T2 evaluation set differs from T2 EVAL_SET_SHA256")
    pool_arrays, _ = pool.load(pool_path, P.EVAL_POOL_SHA256)
    return t1, t2, pool_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=os.environ.get("T3_POOL_EVAL", ""))
    ap.add_argument("--t1-targets", default=str(T1_TARGETS))
    ap.add_argument("--map", default=os.environ.get("T2_REACHABILITY_NPZ", str(FROZEN_MAP)))
    args = ap.parse_args()
    E = module("walk_reach.eval_set")
    pool = module("walk_reach.pool")
    arrays, meta = E.build(*inputs(args.t1_targets, args.map, args.pool))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "t3_eval_set_v5.npz", metadata=json.dumps(meta), **arrays)
    (out / "t3_eval_set_v5.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "t1_target_ids"}, indent=1))
    assert pool.digest(arrays) == meta["eval_set_sha256"]


if __name__ == "__main__":
    main()
