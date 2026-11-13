"""Run robot_lab's unchanged rsl_rl train.py (the run_train.sh entrypoint) with lifecycle markers.

    train_lifecycle.py <train.py args...>     e.g. --task <id> --num_envs 2 --max_iterations 1
Working directory /t1-run, so RSL-RL logs/checkpoints and Hydra outputs land in this run only
(run_train.sh itself would use host networking, shared caches and chmod the shared logs/).
lifecycle.json is rewritten after every stage; judge outcomes by it, not by the exit code.
"""
import atexit
import json
import os
import runpy
import sys
import time

SCRIPT = "/workspace/repos/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py"
OUT = "/t1-run/artifacts"
stages = []


def mark(stage):
    stages.append(dict(stage=stage, t=time.time()))
    with open(OUT + "/lifecycle.json.tmp", "w") as f:
        json.dump(dict(script=SCRIPT, argv=sys.argv[1:], stages=stages), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(OUT + "/lifecycle.json.tmp", OUT + "/lifecycle.json")
    print("LIFECYCLE:", stage, flush=True)


atexit.register(mark, "python_atexit")
os.chdir("/t1-run")
sys.path.insert(0, os.path.dirname(SCRIPT))
sys.argv = [SCRIPT, *sys.argv[1:]]
from rsl_rl.runners import OnPolicyRunner  # noqa: E402  (pure torch; patched before train.py imports it)

_learn = OnPolicyRunner.learn


def learn(self, *a, **k):
    mark("learn_started")
    result = _learn(self, *a, **k)
    mark("learn_returned")
    return result


OnPolicyRunner.learn = learn
mark("wrapper_started")
runpy.run_path(SCRIPT, run_name="__main__")   # main() then simulation_app.close()
mark("train_py_returned")
