"""Goal-pool statistics of a training preset restricted to some goal modes (D42). CPU, no simulator.

For the restricted pool (GoalPool(modes=...)): masked (dead-end) goals by mode, goals of the drawn modes that
remain usable (not masked and reachable from home through stored successors of the drawn modes), and the TCP
height histogram (above the terrain, 0.1 m bins) of the usable goals next to all goals of the drawn modes.
With --fallback (D43): the full successor graph is kept (only the D34 mask) and a goal without a successor of the
drawn mode steps to any unmasked goal of that mode, so every unmasked goal of the drawn modes is usable.
Usage: t3_preset_goal_stats.py OUT_JSON [--modes 2] [--fallback] [--pool PATH]
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_t3 import module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--modes", default="2")
    ap.add_argument("--pool", default="")
    ap.add_argument("--fallback", action="store_true")
    args = ap.parse_args()
    P, POOL, GOALS = module("walk_reach.protocol"), module("walk_reach.pool"), module("walk_reach.goals")
    torch = importlib.import_module("torch")
    path = args.pool or str(ROOT / "repos/robot_lab/logs/t3" / P.POOL_FILES["train"])
    arrays, _ = POOL.load(path, P.TRAIN_POOL_SHA256)
    modes = tuple(int(m) for m in args.modes.split(","))
    pool = GOALS.GoalPool(arrays, modes=(1, 2, 3) if args.fallback else modes)
    mode = np.asarray(arrays["mode"])
    dead = pool.dead_end.numpy()
    succ = pool.successors.numpy()
    reach = np.zeros(len(mode), dtype=bool)          # BFS from home over the remaining (unmasked) edges
    reach[0], frontier = True, [0]
    while frontier:
        nxt = succ[frontier].ravel()
        nxt = np.unique(nxt[nxt >= 0])
        nxt = nxt[~reach[nxt]]
        reach[nxt] = True
        frontier = nxt.tolist()
    drawn = np.isin(mode, modes)
    usable = drawn & ~dead & (True if args.fallback else reach)
    z = np.asarray(arrays["tcp_pos_T"])[:, 2] + P.H_T_M
    edges = np.round(np.arange(0.0, max(1.6, z.max() + 0.1), 0.1), 2)
    hist = lambda m: np.histogram(z[m], bins=edges)[0].tolist()  # noqa: E731
    out = dict(pool=path, modes=list(modes), fallback_step=args.fallback, goals_total=int(len(mode)),
               masked_total=int(dead.sum()),
               masked_by_mode={str(m): int((dead & (mode == m)).sum()) for m in (0, 1, 2, 3)},
               drawn_mode_goals=int(drawn.sum()), drawn_masked=int((drawn & dead).sum()),
               drawn_unreachable_unmasked=int((drawn & ~dead & ~reach).sum()),
               drawn_usable=int(usable.sum()), drawn_lost_fraction=float(1 - usable.sum() / drawn.sum()),
               height_bin_edges_m=edges.tolist(), height_hist_all_drawn=hist(drawn), height_hist_usable=hist(usable))
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
