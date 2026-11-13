"""Run the T1 CPU test suite inside the Isaac image with its USD SDK and hpp-fcl (no SimulationApp)."""
import os
import sys

import usd_loader  # noqa: F401  (must be first)
import pytest

os.environ["T1_USD_ASSET"] = str(usd_loader.ROOT / "robots_usd/b2w_z1_rl.usd")  # these tests pin RL-asset facts
tests = [str(usd_loader.ROOT / "tests" / n) for n in
         ("test_t1.py", "test_t1_clearance.py", "test_t1_usd.py", "test_validate_t1.py")]
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", *tests, *sys.argv[1:]]))
