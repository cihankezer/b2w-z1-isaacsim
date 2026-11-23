"""T3 goal-sampler rollout on the frozen training pool (protocol v0.4 §6.3 logging). CPU, no simulator.

Emulates the training base-velocity command (Isaac Lab uniform velocity command: resampled every 3 s and at
episode reset, 20 % standing, vx ~ U[-0.6, 0.9], wz ~ U[-1, 1]; moving = command non-zero) and 20 s episodes,
and reports the stranding-rule counters per 1000 transitions, how often the 0.10 m moving minimum binds
for M1 candidates and the dead-end mask (D34). Writes sampler_rollout.json to --out.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
MANIP = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"


def module(name):
    for pkg, path in (("_manip", MANIP), ("_manip.standing_ee", MANIP / "standing_ee"),
                      ("_manip.reach_height", MANIP / "reach_height"), ("_manip.walk_reach", MANIP / "walk_reach")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    return importlib.import_module("_manip." + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    P, POOL, GOALS = module("walk_reach.protocol"), module("walk_reach.pool"), module("walk_reach.goals")
    arrays, meta = POOL.load(args.pool, P.TRAIN_POOL_SHA256)
    gen = torch.Generator().manual_seed(args.seed)
    n = args.envs
    pool = GOALS.GoalPool(arrays)
    gs = GOALS.GoalSampler(pool, n, generator=torch.Generator().manual_seed(args.seed + 1))
    gs.reset(torch.arange(n))
    steps_ep = int(round(P.TRAIN_EPISODE_S / P.CONTROL_DT_S))
    steps_cmd = int(round(P.VELOCITY_RESAMPLE_S / P.CONTROL_DT_S))
    t_ep = torch.randint(steps_ep, (n,), generator=gen)                   # staggered episodes
    moving = torch.zeros(n, dtype=torch.bool)
    t_cmd = torch.zeros(n, dtype=torch.long)

    def resample(ids):
        stand = torch.rand(len(ids), generator=gen) < P.STANDING_FRACTION
        moving[ids] = ~stand          # U[-0.6, 0.9] x U[-1, 1] is non-zero with probability 1
        t_cmd[ids] = 0

    resample(torch.arange(n))
    for _ in range(int(round(args.seconds / P.CONTROL_DT_S))):
        t_ep += 1
        t_cmd += 1
        done = (t_ep >= steps_ep).nonzero(as_tuple=True)[0]
        if len(done):
            t_ep[done] = 0
            gs.reset(done)
            resample(done)
        due = (t_cmd >= steps_cmd).nonzero(as_tuple=True)[0]
        if len(due):
            resample(due)
        gs.update(P.CONTROL_DT_S, moving)
    r = gs.rates()
    c = r["counts"]
    r["m1_min_height_binding_fraction"] = c["m1_rejected_min_height_moving"] / max(1, c["m1_candidates_moving"])
    r["m1_wheel_binding_fraction"] = c["m1_rejected_wheel_moving"] / max(1, c["m1_candidates_moving"])
    dead = pool.dead_end
    r["dead_end_mask"] = dict(masked_goals=int(dead.sum()), without_stored_successor=int((arrays["successors"] < 0).all(1).sum()),
                              by_mode={int(m): int((dead & (pool.mode == m)).sum()) for m in pool.mode.unique()},
                              drawn_masked_goals=int(dead[gs.goal_node].sum()))
    r.update(pool_sha256=meta["pool_sha256"], envs=n, seconds=args.seconds, seed=args.seed,
             note="command emulated on CPU; repeat in the Isaac training env (run-order step 5)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sampler_rollout.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
