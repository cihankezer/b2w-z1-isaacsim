"""Open/close lifecycle probe for a registered task: gym.make, reset, close; no stepping.

    open_close.py --task <gym id> [--num_envs 2] [--out /t1-run/artifacts]
--fast_exit 1 uses Isaac Sim's default immediate process exit at app.close(). The robot USD is chosen by the task config (for the robot_lab B2W+Z1 configs, the B2W_Z1_USD /
B2W_Z1_GRIPPER_USD environment variables). After gym.make the live stage of env_0 is scanned for
ArticulationRootAPI prims (instance proxies included). lifecycle.json is rewritten after every
stage; judge outcomes by it, not by the exit code.
"""
import argparse
import atexit
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--out", default="/t1-run/artifacts")
parser.add_argument("--fast_exit", type=int, default=0, help="SimulationApp fast_shutdown (Isaac Sim default 1)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
stages = []
facts = {}


def mark(stage):
    stages.append(dict(stage=stage, t=time.time()))
    path = os.path.join(args.out, "lifecycle.json")
    with open(path + ".tmp", "w") as f:
        json.dump(dict(task=args.task, num_envs=args.num_envs, stages=stages, facts=facts,
                       usd_env={k: v for k, v in os.environ.items() if k.startswith("B2W_Z1")}), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    print("LIFECYCLE:", stage, flush=True)


atexit.register(mark, "python_atexit")
facts["fast_shutdown"] = bool(args.fast_exit)
launcher = AppLauncher(args, fast_shutdown=bool(args.fast_exit))
app = launcher.app
mark("app_started")


def main():
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import robot_lab.tasks  # noqa: F401
    import omni.usd
    from isaaclab_tasks.utils import parse_env_cfg
    from pxr import Usd, UsdPhysics
    if "T1-Standing" in args.task:   # overlay package (as in eval_t1.py)
        import importlib
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        parent.__path__.insert(0, "/workspace/training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation")
        importlib.import_module("robot_lab.tasks.manager_based.manipulation.standing_ee")
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    stage = omni.usd.get_context().get_stage()
    prims = Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0"), Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    facts["articulation_roots_env_0"] = [dict(path=str(p.GetPath()), type=p.GetTypeName(), instance_proxy=p.IsInstanceProxy())
                                         for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    facts["articulations"] = {name: dict(root_prim=a.root_physx_view.prim_paths[0] if a.root_physx_view.prim_paths else None,
                                         num_bodies=a.num_bodies, is_fixed_base=bool(a.is_fixed_base))
                              for name, a in env.scene.articulations.items()}
    mark("env_made")
    env.reset(seed=1101)
    mark("reset_done")
    env.close()
    mark("env_close_returned")


main()
mark("main_returned")
app.close()
mark("app_close_returned")
