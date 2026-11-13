"""Native-shutdown bisection: the clean unstepped control plus one env.step(zeros(1,22)).

Variants (nested, because StepRecorder reads the failure term and both read the
contact sensors -- runtime.py:189-190,255,285):
  base     : unchanged T1 env
  c        : recorders=None
  a        : terminations.failure=None + recorders=None
  d        : a + contacts/arm_contacts removed
  unstepped: base without env.step (same-day control)
  s50      : base with 50 env.step calls (1 s)
  s50p     : s50 + the 30 s diagnostic's two filtered pair contact sensors (contact points, friction)
  s50p2    : s50p with num_envs=2
  u2       : unstepped with num_envs=2 (draft.7 side task)
lifecycle.json is rewritten after every stage; judge outcomes by it, not exit code.
"""
import argparse
import atexit
import importlib
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["base", "a", "c", "d", "unstepped", "s50", "s50p", "s50p2", "u2"], required=True)
parser.add_argument("--out", default="/t1-run/artifacts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
stages = []


def mark(stage):
    stages.append(dict(stage=stage, t=time.time()))
    path = os.path.join(args.out, "lifecycle.json")
    with open(path + ".tmp", "w") as f:
        json.dump(dict(variant=args.variant, stages=stages), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    print("LIFECYCLE:", stage, flush=True)


atexit.register(mark, "python_atexit")
launcher = AppLauncher(args, fast_shutdown=False)
app = launcher.app
mark("app_started")


def main():
    import gymnasium as gym
    import torch
    import robot_lab.tasks  # noqa: F401
    parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
    parent.__path__.insert(0, "/workspace/training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation")
    pkg = "robot_lab.tasks.manager_based.manipulation.standing_ee"
    importlib.import_module(pkg)
    cfg = importlib.import_module(pkg + ".env_cfg").StandingEnvCfg()
    cfg.sim.device = args.device
    if args.variant in ("c", "a", "d"):
        cfg.recorders = None
    if args.variant in ("a", "d"):
        cfg.terminations.failure = None
    if args.variant in ("s50p", "s50p2"):
        from isaaclab.sensors import ContactSensorCfg
        for name, source, target in (("pair_00_02", "link00", "link02"), ("pair_02_00", "link02", "link00")):
            setattr(cfg.scene, name, ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/" + source,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/z1/" + target], update_period=.005, history_length=4,
                track_contact_points=True, track_friction_forces=True, max_contact_data_count_per_prim=128))
    if args.variant in ("s50p2", "u2"):
        cfg.scene.num_envs = 2
    if args.variant == "d":
        cfg.scene.contacts = None
        cfg.scene.arm_contacts = None
    env = gym.make("RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
    mark("env_made")
    env.reset(seed=1101)
    mark("reset_done")
    if args.variant not in ("unstepped", "u2"):
        for _ in range(50 if args.variant.startswith("s50") else 1):
            env.step(torch.zeros(env.num_envs, 22, device=env.device))
        mark("step_done")
    env.close()
    mark("env_close_returned")


main()
mark("main_returned")
app.close()
mark("app_close_returned")
