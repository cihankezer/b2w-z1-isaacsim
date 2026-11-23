"""Run the T3 CPU test suite inside the Isaac image (USD SDK, hpp-fcl, torch; no SimulationApp)."""
import sys

import usd_loader  # noqa: F401  (must be first)
import pytest

raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", str(usd_loader.ROOT / "tests/test_t3.py"), *sys.argv[1:]]))
