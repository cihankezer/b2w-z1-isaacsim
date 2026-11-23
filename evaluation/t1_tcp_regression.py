"""T1 2.0.0 TCP-switch regression check (T3 protocol D33). Stdlib only; runs on the host.

Compares metrics.csv of the T1 1.0.0 reference (link06 errors) and a T1 2.0.0 validation run (TCP errors),
per step of every T1 trial (rows without a validation_case; fault-injection rows are excluded):
  1. orientation_error_geodesic_rad identical (the TCP is a rigid translation of link06);
  2. |e_pos_tcp - e_pos_flange| <= 0.171 * theta_geo + 1e-6.
Rows are matched by (stage, repetition, env_id, target_id, episode_time_s); the row sets must be equal.
Usage: t1_tcp_regression.py OUT_JSON REF_VALIDATION_DIR NEW_VALIDATION_DIR
"""

import csv
import json
import math
from pathlib import Path
import sys

csv.field_size_limit(1 << 30)
OFFSET_M = 0.171
TOL = 1e-6
KEY = ("stage", "repetition", "env_id", "target_id", "episode_time_s")


def rows(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["validation_case"]:
                continue
            k = tuple(r[c] for c in KEY)
            if k in out:
                raise ValueError(f"duplicate row {k} in {path}")
            out[k] = (r["position_error_m"], r["orientation_error_geodesic_rad"], r["failure_reason"])
    return out


def main():
    out, ref, new = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    a, b = rows(ref / "metrics.csv"), rows(new / "metrics.csv")
    result = dict(reference=str(ref), new=str(new), rows_ref=len(a), rows_new=len(b),
                  rows_only_ref=len(a.keys() - b.keys()), rows_only_new=len(b.keys() - a.keys()),
                  orientation_mismatch=0, orientation_max_abs_delta=0.0, failure_reason_mismatch=0,
                  bound_violations=0, max_violation_m=-math.inf, max_abs_dpos_m=0.0, rows_compared=0, examples=[])
    for k in sorted(a.keys() & b.keys()):
        (pa, oa, fa), (pb, ob, fb) = a[k], b[k]
        result["failure_reason_mismatch"] += fa != fb
        if pa == "" or pb == "":                  # invalid-state rows carry no error
            result["orientation_mismatch"] += (oa != ob)
            continue
        result["rows_compared"] += 1
        if oa != ob:
            result["orientation_mismatch"] += 1
            result["orientation_max_abs_delta"] = max(result["orientation_max_abs_delta"], abs(float(oa) - float(ob)))
        dpos = abs(float(pb) - float(pa))
        violation = dpos - OFFSET_M * float(oa)
        result["max_abs_dpos_m"] = max(result["max_abs_dpos_m"], dpos)
        result["max_violation_m"] = max(result["max_violation_m"], violation)
        if violation > TOL:
            result["bound_violations"] += 1
            if len(result["examples"]) < 10:
                result["examples"].append(dict(key=k, e_pos_flange=pa, e_pos_tcp=pb, theta=oa))
    result["pass_"] = (result["rows_only_ref"] == 0 and result["rows_only_new"] == 0 and result["rows_compared"] > 0
                       and result["orientation_mismatch"] == 0 and result["bound_violations"] == 0
                       and result["failure_reason_mismatch"] == 0)
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
