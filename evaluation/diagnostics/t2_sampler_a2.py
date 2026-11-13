"""T2 A2 (protocol §11) on the frozen map (CPU): 10^5 targets from sampler.sample, statistics vs. the config.

Pass: band frequencies within 1 % of the configured weights, all targets inside the sampling set and their
band, |theta| <= 60 deg, unit quaternions. Writes /t1-run/artifacts/a2.json.
Usage: t2_sampler_a2.py NPZ
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from reachability_t2 import t2_module  # noqa: E402

P = t2_module("reach_height.protocol")
S = t2_module("reach_height.sampler")
rs = S.ReachableSet.load(sys.argv[1])
targets = S.sample(rs, 100_000, np.random.default_rng(7))
stats = S.statistics(targets, rs)
q = np.linalg.norm(targets["quat_G_wxyz"], axis=1)
stats = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}
stats["max_quat_norm_error"] = float(np.abs(q - 1).max())
stats["pass_"] = bool(stats["max_frequency_deviation"] < 0.01 and stats["all_in_reachable_set"] and stats["all_in_band"]
                      and stats["max_abs_azimuth_rad"] <= P.AZIMUTH_MAX_RAD + 1e-9 and stats["max_quat_norm_error"] < 1e-9)
out = Path("/t1-run/artifacts")
out.mkdir(parents=True, exist_ok=True)
(out / "a2.json").write_text(json.dumps(stats, indent=1, default=float) + "\n")
print(json.dumps(stats, indent=1, default=float))
raise SystemExit(0 if stats["pass_"] else 1)
