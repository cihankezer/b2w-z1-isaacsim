"""Exact run comparison for T3 regression tests (A15, A14). Stdlib only; runs on the host.

Compares the CSV outputs of a reference run and a new run cell by cell. A column is "exact" if every cell is
byte-identical; for numeric columns the max |delta| is also reported. Columns that are expected to differ
between runs (wall-clock times, paths, run ids) are listed in IGNORE and reported separately.
Usage: t3_compare_exact.py OUT_JSON REF_ARTIFACTS NEW_ARTIFACTS FILE [FILE ...]
"""

import csv
import json
import math
from pathlib import Path
import sys

csv.field_size_limit(1 << 30)
IGNORE = {"wall_s", "wall_time_s", "elapsed_s", "timestamp", "run_id", "path", "out", "image_id", "gpu_uuid"}


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def compare(ref, new):
    with open(ref, newline="") as fa, open(new, newline="") as fb:
        ra, rb = csv.reader(fa), csv.reader(fb)
        ha, hb = next(ra), next(rb)
        out = dict(rows_ref=0, rows_new=0, header_equal=ha == hb, columns={}, ignored=sorted(set(ha) & IGNORE))
        cols = [c for c in ha if c in hb and c not in IGNORE]
        ia, ib = {c: ha.index(c) for c in cols}, {c: hb.index(c) for c in cols}
        stats = {c: dict(mismatch=0, max_abs_delta=0.0) for c in cols}
        for a, b in zip(ra, rb):
            out["rows_ref"] += 1
            out["rows_new"] += 1
            for c in cols:
                x, y = a[ia[c]], b[ib[c]]
                if x != y:
                    s = stats[c]
                    s["mismatch"] += 1
                    nx, ny = num(x), num(y)
                    if nx is not None and ny is not None:
                        s["max_abs_delta"] = max(s["max_abs_delta"], abs(nx - ny))
                    else:
                        s["max_abs_delta"] = float("inf")
                        s.setdefault("example", [x[:80], y[:80]])
        out["rows_ref"] += sum(1 for _ in ra)
        out["rows_new"] += sum(1 for _ in rb)
    out["columns"] = {c: s for c, s in stats.items() if s["mismatch"]}
    out["exact"] = out["header_equal"] and out["rows_ref"] == out["rows_new"] and not out["columns"]
    return out


def main():
    out, ref, new, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    result = {f: compare(ref / f, new / f) for f in files}
    result = dict(reference=str(ref), new=str(new), files=result, exact=all(r["exact"] for r in result.values()))
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(dict(exact=result["exact"], files={f: dict(exact=r["exact"], rows=(r["rows_ref"], r["rows_new"]),
                                                               mismatched_columns=list(r["columns"])[:10])
                                                       for f, r in result["files"].items()}), indent=1))


if __name__ == "__main__":
    main()
