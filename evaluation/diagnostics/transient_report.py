"""Settling-transient report from a saved diagnose_t1_standing run (stdlib only, no simulator).

Init joint positions vs PD targets, arm joint 2-6 and base height over the first
`window` seconds, first foot touchdown, first arm contact.
    transient_report.py <run_artifacts_dir> [window_s=0.3] [> out.json]
"""
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
window = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
interface = json.load(open(run / "interface.json"))
order, action_order = interface["joint_order"], interface["action_order"]
physics = list(csv.DictReader(open(run / "physics.csv")))
control = list(csv.DictReader(open(run / "control.csv")))
arm = [f"joint{i}" for i in range(1, 7)] + (["jointGripper"] if "jointGripper" in order else [])

first = dict(zip(order, json.loads(physics[0]["joint_pos"])))
targets = dict(zip(action_order, json.loads(control[0]["joint_targets"])))
touchdown = next((r for r in physics if max(sum(f * f for f in v) ** .5 for v in json.loads(r["foot_contact_forces_n"])) > 1.0), None)
arm_hit = next((r for r in physics if max(json.loads(r["arm_contact_forces_n"])) > 1.0), None)
trajectory = [dict(tick=int(r["physics_tick"]), t=float(r["time_s"]),
                   base_z=json.loads(r["base_position_world_xyz"])[2],
                   **{j: dict(zip(order, json.loads(r["joint_pos"])))[j] for j in arm[1:6]})
              for r in physics if float(r["time_s"]) <= window + 1e-9]
report = dict(run=str(run), physics_samples=len(physics), last_time_s=float(physics[-1]["time_s"]),
              init_vs_target={j: dict(init_rad=first.get(j), pd_target_rad=targets.get(j),
                                      diff_rad=None if targets.get(j) is None else first[j] - targets[j]) for j in arm},
              first_tick_base_z_m=json.loads(physics[0]["base_position_world_xyz"])[2],
              first_foot_touchdown=dict(tick=int(touchdown["physics_tick"]), t=float(touchdown["time_s"])) if touchdown else None,
              first_arm_contact=dict(tick=int(arm_hit["physics_tick"]), t=float(arm_hit["time_s"])) if arm_hit else None,
              trajectory=trajectory[::2])
json.dump(report, sys.stdout, indent=1)
