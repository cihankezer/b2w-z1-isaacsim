"""CPU check of the T1 model on the selected asset (B2W_Z1_USD): bodies, colliders, nominal clearance."""
import json
import sys
from collections import Counter

import usd_loader  # noqa: F401  (must be first)
import numpy as np
from pxr import Usd

k = usd_loader.t1_module("kinematics")
g = usd_loader.t1_module("geometry")
stage = Usd.Stage.Open(str(usd_loader.ASSET))
out = dict(asset=str(usd_loader.ASSET), hashes=usd_loader.t1_module("protocol").asset_hashes(str(usd_loader.ASSET)))
try:
    model = k.UsdArmKinematics(stage, "/Unitree_B2W")
except Exception as exc:
    out["model_error"] = f"{type(exc).__name__}: {exc}"
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(out["model_error"])
    raise SystemExit(0)
out.update(physics_filtering=model.physics_filtering, adjacent=sorted(sorted(p) for p in model.adjacent),
           revolute=[e[2] for e in model.edges if e[5] is not None],
           shapes={n: dict(Counter(s.get("source", s["kind"]) for s in v)) for n, v in model.shapes.items()},
           unresolved=sorted({s["path"].split("/Unitree_B2W/")[-1] + " :: " + str(s.get("issue"))[:80]
                              for v in model.shapes.values() for s in v if not s.get("resolved", True)}),
           )
stow = dict(dict.fromkeys(out["revolute"], 0.0), joint2=0.15, joint3=-0.15)
frames = model.forward(g.transform([0, 0, 0.6], [1, 0, 0, 0]), stow)
caps = model.clearance_report(frames)
out["nominal_capsule_rows_not_false_positive"] = [
    dict(pair=r["pair"], a=r["shape_a"].split("/collisions/")[-1], b=r["shape_b"].split("/collisions/")[-1],
         cls=r["classification"], d=r.get("clearance_m")) for r in caps if r["classification"] != "broad_phase_false_positive"]
out["nominal_self_monitor"] = model.self_clearance_monitor(caps)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok")
