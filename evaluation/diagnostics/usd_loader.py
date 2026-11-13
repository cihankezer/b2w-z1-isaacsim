"""CPU-only USD SDK bootstrap for the Isaac image (no SimulationApp, no GPU).

Import first: re-execs the interpreter once with the image's USD libraries on
LD_LIBRARY_PATH, then exposes `pxr` and a loader for the read-only T1 package.
"""
import glob
import importlib
import os
import sys
import types
from pathlib import Path

_PATHS = glob.glob("/isaac-sim/extscache/omni.usd.libs*")
if not os.environ.get("T1_USD_LOADER"):
    libdirs = sorted({str(p.parent) for root in _PATHS for p in Path(root).rglob("libusd_tf.so")})
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["T1_USD_LOADER"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
for _p in _PATHS:
    sys.path.insert(0, _p)

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee"
# T1 asset (protocol draft.4): same switch as the T1 robot config (fetch_env_cfg.B2W_Z1_GRIPPER_USD).
ASSET = Path(os.environ.get("B2W_Z1_GRIPPER_USD", str(ROOT / "robots_usd/b2w_z1_gripper.usd")))


def t1_module(name):
    """Import an unchanged standing_ee module without importing Isaac Lab."""
    if "_t1" not in sys.modules:
        pkg = types.ModuleType("_t1")
        pkg.__path__ = [str(TASK)]
        sys.modules["_t1"] = pkg
    return importlib.import_module("_t1." + name)
