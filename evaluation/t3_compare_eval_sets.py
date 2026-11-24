"""Compare T3 evaluation set v5 against v4 (S1 source switched from T1 v1c to T1 2.0.0 targets).

Every S1 target must equal v4 within 1e-9 (position and quaternion, element-wise); every other array must be
identical. Usage: t3_compare_eval_sets.py OUT_JSON V4_NPZ V5_NPZ
"""

import json
from pathlib import Path
import sys

import numpy as np

TOL = 1e-9


def main():
    out, a, b = sys.argv[1], np.load(sys.argv[2]), np.load(sys.argv[3])
    keys = sorted(set(a.files) - {"metadata"})
    res = dict(v4=sys.argv[2], v5=sys.argv[3], keys_equal=keys == sorted(set(b.files) - {"metadata"}), other_arrays_differ=[])
    s1 = a["cell_S"] == "S1"
    res["s1_episodes"] = int(s1.sum())
    res["s1_mask_equal"] = bool(np.array_equal(s1, b["cell_S"] == "S1"))
    for k in keys:
        if k == "target_T":
            continue
        if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"):
            res["other_arrays_differ"].append(k)
    ta, tb = a["target_T"], b["target_T"]
    res["target_shape_equal"] = ta.shape == tb.shape
    non_s1 = ~s1
    res["non_s1_targets_identical"] = bool(np.array_equal(ta[non_s1], tb[non_s1], equal_nan=True))
    da, db = ta[s1], tb[s1]
    res["s1_nan_pattern_equal"] = bool(np.array_equal(np.isnan(da), np.isnan(db)))
    d = np.abs(np.nan_to_num(da) - np.nan_to_num(db))
    res["s1_max_pos_diff_m"] = float(d[..., :3].max())
    res["s1_max_quat_diff"] = float(d[..., 3:].max())
    res["s1_targets_bitwise_identical"] = bool(np.array_equal(da, db, equal_nan=True))
    res["v4_metadata_version"] = json.loads(str(a["metadata"]))["version"]
    res["v5_metadata_version"] = json.loads(str(b["metadata"]))["version"]
    res["pass_"] = (res["keys_equal"] and res["s1_mask_equal"] and not res["other_arrays_differ"]
                    and res["target_shape_equal"] and res["non_s1_targets_identical"] and res["s1_nan_pattern_equal"]
                    and res["s1_max_pos_diff_m"] <= TOL and res["s1_max_quat_diff"] <= TOL)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
