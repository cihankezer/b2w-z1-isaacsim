"""T2 A7 determinism check (protocol §11): compare two eval_t2.py run directories (stdlib only).

Pass: identical eval-set digest; per target identical categorical columns (EXACT); |delta| <= 1e-4 m for
TOL_M columns. Max |delta| of every other numeric column is reported. Writes <out>/a7_compare.json.
Usage: t2_compare_runs.py RUN_A RUN_B OUT_DIR
"""

import csv
import json
import math
from pathlib import Path
import sys

EXACT = ("outcome", "success", "success_kaist_strict", "completed", "fell", "terminations", "contact_bodies")
TOL_M = {"e_p_final": 1e-4, "e_p_hold_mean": 1e-4}


def load(run):
    art = Path(run) / "artifacts"
    rows = {r["target_id"]: r for r in csv.DictReader(open(art / "episodes.csv"))}
    return rows, json.load(open(art / "summary.json"))


def num(v):
    try:
        x = float(v)
    except ValueError:
        return None
    return x


def main():
    a, b, out = sys.argv[1:4]
    ra, sa = load(a)
    rb, sb = load(b)
    res = dict(run_a=a, run_b=b, n_a=len(ra), n_b=len(rb), same_targets=sorted(ra) == sorted(rb),
               eval_set_sha256_a=sa.get("eval_set_sha256"), eval_set_sha256_b=sb.get("eval_set_sha256"))
    res["same_eval_set"] = res["eval_set_sha256_a"] == res["eval_set_sha256_b"]
    mismatch = {c: [] for c in EXACT}
    maxdiff, nan_mismatch = {}, {}
    for tid in sorted(set(ra) & set(rb), key=int):
        x, y = ra[tid], rb[tid]
        for c in EXACT:
            if x[c] != y[c]:
                mismatch[c].append(dict(target_id=int(tid), a=x[c], b=y[c]))
        for c in x:
            if c in EXACT or c == "target_id":
                continue
            u, v = num(x[c]), num(y[c])
            if u is None or v is None:
                if x[c] != y[c]:
                    nan_mismatch.setdefault(c, 0)
                    nan_mismatch[c] += 1
                continue
            if math.isnan(u) or math.isnan(v):
                if math.isnan(u) != math.isnan(v):
                    nan_mismatch[c] = nan_mismatch.get(c, 0) + 1
                continue
            maxdiff[c] = max(maxdiff.get(c, 0.0), abs(u - v))
    res["exact_mismatches"] = {c: dict(count=len(v), first=v[:5]) for c, v in mismatch.items()}
    res["tolerance_m"] = TOL_M
    res["max_abs_diff"] = dict(sorted(maxdiff.items()))
    res["nan_or_text_mismatch"] = nan_mismatch
    res["bitwise_identical_numeric"] = all(v == 0.0 for v in maxdiff.values()) and not nan_mismatch
    res["pass_"] = bool(res["same_targets"] and res["same_eval_set"] and not any(mismatch.values())
                        and all(maxdiff.get(c, 0.0) <= t for c, t in TOL_M.items())
                        and not any(c in nan_mismatch for c in TOL_M))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "a7_compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "max_abs_diff"}, indent=1))
    print("max_abs_diff (nonzero):", {k: v for k, v in res["max_abs_diff"].items() if v})
    raise SystemExit(0 if res["pass_"] else 1)


if __name__ == "__main__":
    main()
