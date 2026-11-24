"""T3 protocol constants (docs/protocols/T3_walk_and_reach_protocol.md, v0.8).

Units: metres, radians, seconds. Quaternions wxyz. Frames: F_T (target frame: origin at base x/y and
terrain + h_T, yaw of the base, z up), F_H (heading frame for velocities), B (base_link).
Base pose offset b = (dh, pitch, roll): base at h_nom + dh, R_base = Rz(yaw) Ry(pitch) Rx(roll), base x/y
and wheel centres fixed (T2 base model). Pitch positive = nose down (T2 convention).

Measured values (Section 10) come from data/measurements.json, written by evaluation/measure_t3.py
(run diag-measure-20261123T121813Z-2df42c5d) and pinned here by hash.
"""

import hashlib
import json
from pathlib import Path

from ..reach_height import protocol as T2P

VERSION = "0.8.0"   # protocol v0.8 (benchmark-suite-v1; final except open item 12)
BENCHMARK = "T3-WALK-REACH"
ROBOT_VARIANT = "B2W"

DATA = Path(__file__).resolve().parent / "data"
MEASUREMENTS_SHA256 = "7952d3b15f5be05c1f1014231f25bb7834415fe010fc85085780ce4201d1159b"
KINEMATICS_SHA256 = "95648caa690e4f7620fea9d11fceef268f177035615236855821e7c89261fb8e"
ASSET_TREE_SHA256 = "4e55713b49b42d37347ed31ce407f92571c08bdb9a72f785a7e75fa1cce1f1b4"
T2_MAP_SHA256 = "22b9e665baa11bced9832107ca17c154ba57751fb55f54c3dbf67d9e1b865404"   # = reach_height.env_cfg pin


def _load(name, sha):
    raw = (DATA / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{name} does not match the pinned hash")
    return json.loads(raw)


MEASURED = _load("measurements.json", MEASUREMENTS_SHA256)
KINEMATICS = _load("kinematics.json", KINEMATICS_SHA256)
if MEASURED["asset_tree_sha256"] != ASSET_TREE_SHA256 or MEASURED["t2_map"]["sha256"] != T2_MAP_SHA256:
    raise ValueError("Measurements were taken on a different asset or T2 map")

# §3 / §10 Platform (measured)
H_NOM_M = T2P.H_NOM_M
H_T_M = MEASURED["h_T"]
TCP_PARENT, TCP_OFFSET_M = T2P.TCP_PARENT, T2P.TCP_OFFSET_M
P_HOME_T = tuple(MEASURED["p_home_T"]["position"]) + tuple(MEASURED["p_home_T"]["quaternion_wxyz"])
B_NOM = (MEASURED["b_nom"]["dh"], MEASURED["b_nom"]["pitch"], MEASURED["b_nom"]["roll"])
BODY_BOX_T = (tuple(MEASURED["B_body_T"]["min"]), tuple(MEASURED["B_body_T"]["max"]))   # v0.4: measured, not used
WHEEL_CONTACT_T_XY = tuple(tuple(c[:2]) for c in MEASURED["wheels"]["contact_T"])   # FR, FL, RR, RL
WHEEL_CENTRE_BODY = tuple(tuple(c) for c in MEASURED["wheels"]["centre_body"])
H_KNEE_M = MEASURED["knees"]["h_knee"]
FEASIBILITY = MEASURED["feasibility"]     # dh_axis, pitch_axis, feasible[dh][pitch] from the frozen T2 map

# §3 / D5, D26 T2 limits and fall thresholds. v0.4: lowest base height = deepest feasible map posture
# (dh = -0.28 m) -> 0.6039 - 0.28 = 0.324 m; h_fall = min(0.5 h_nom, h_T2_min - 0.05) = 0.274 m.
H_T2_MIN_M = round(T2P.H_NOM_M - 0.28, 3)
THETA_T2_MAX_RAD = 0.28
FALL_TILT_RAD = max(1.0, THETA_T2_MAX_RAD + 0.2)
FALL_HEIGHT_M = min(0.5 * H_NOM_M, H_T2_MIN_M - 0.05)
DEBOUNCE_STEPS = 5
DRIFT_M = 0.30
CONTACT_N = 1.0
GROUND_CLEARANCE_M = 0.005
CLEARANCE_MARGINS_M = T2P.CLEARANCE_MARGINS_M   # pool checks for M1/M3: T2 per-class margins
# M2 (T1-derived) goals: T1's 5 mm clearance (TASK_T1 CLEARANCE_POLICY); body-body (leg) pairs keep 2 cm because
# T1 has no moving legs and therefore no leg-pair margin of its own (D29).
T1_CLEARANCE_MARGINS_M = dict(ground=0.005, arm_arm=0.005, arm_body=0.005, body_body=0.02)

# §4 Simulation and action
CONTROL_DT_S = 0.02
WHEEL_ACTION_SCALE_RAD_S = 10.0
TRAIN_EPISODE_S = 20.0

# §6.1 Base velocity command (training)
VX_RANGE = (-0.6, 0.9)
VY_RANGE = (0.0, 0.0)
WZ_RANGE = (-1.0, 1.0)
STANDING_FRACTION = 0.20
VELOCITY_RESAMPLE_S = 3.0

# §6.2 EE target command (training)
MODE_NAMES = ("M1", "M2", "M3")
MODE_WEIGHTS = (0.50, 0.25, 0.25)
T_TRAJ_S = (1.0, 3.0)
HOLD_S = (0.5, 2.0)
STEP_FRACTION = 0.20
ARM_SOFT_LIMIT_FACTOR = 0.9            # 5 % margin at each limit = Isaac Lab soft limits (as the T1 sampler)
STRATIFICATION_BINS = (0.1, 0.2, 0.2)  # (l [m], pitch [rad], yaw [rad]) of the TCP position in F_T
M3_HEIGHT_BIN_M = 0.10                 # M3 stratification by TCP height within each band (D27)

# §6.3 Validity
SEGMENT_SAMPLES = 20
MAX_TRIES = 20
Z_MIN_M = 0.05
Z_MIN_MOVING_M = 0.10
WHEEL_EXCLUSION_M = 0.15

# §6.4 Coverage reference region (VBC sampling ranges, B1 + Z1)
VBC_REGION = dict(l=(0.40, 0.95), pitch=(-3.141592653589793 / 2.5, 3.141592653589793 / 3), yaw=(-1.2, 1.2))

# §7 Evaluation
SETTLE_S = 2.0
TEST_S = 10.0
EPISODES_PER_CELL = 64
PROFILES = {"V0": (0.0, 0.0), "V1": (0.3, 0.0), "V2": (0.6, 0.0), "V3": (0.9, 0.0), "V4": (-0.3, 0.0),
            "V5": (0.0, 0.6), "V6": (0.0, -0.6), "V7": (0.5, 0.5)}
STRESS_PROFILES = {"X1": (1.0, 0.0), "X2": (0.0, 1.0), "X3": (0.0, -1.0)}
SCENARIOS = ("S1", "S3", "S4")
# Schedules inherited from the source task (decision D15): step target, hold, scored window at the end.
S1_SCHEDULE = dict(points=2, hold_s=5.0, scored_last_s=2.0)      # T1 fixed targets (TASK_T1 1.0.0)
S3_SCHEDULE = dict(points=1, hold_s=10.0, scored_last_s=1.0)     # T2 (EPISODE_S = 10, HOLD_S = 1.0)
S3_BANDS = ("low", "high")
VELOCITY_TRANSIENT_S = 1.0
SETTLE_TOLERANCE = dict(vx=0.1, wz=0.1, duration_s=0.5)
SUCCESS_POSITION_M = 0.05
SUCCESS_ORIENTATION_RAD = 0.10       # geodesic angle (T1/T2 primary)
JOINT_LIMIT_FRACTION = 0.01

# Seeds and frozen inputs
T1_EVAL_SEED = 1101
T1_EVAL_TARGETS_SHA256 = "908f0ad64c5a2d24ca7acfd3effceb688927014a68bd910a2fd1dd5807ba3c3d"   # T1 2.0.0 reference targets.json (v1c: 505128a1…f190, archived)
T2_EVAL_SEED = T2P.EVAL_SEED
T2_EVAL_SET_SHA256 = T2P.EVAL_SET_SHA256
TRAIN_POOL_SEED = 3101   # M1 candidates + segment draws; M2 uses the T1 rule with this seed (T1 used 1101, 1102)
EVAL_POOL_SEED = 3901    # S4 goals (disjoint from training)
EVAL_SET_SEED = 3301
POOL_SIZES = dict(train=dict(M1=15000, M2=7500, M3=7500, successors=8, max_candidates=32),
                  eval=dict(M1=512, M2=0, M3=0, successors=8, max_candidates=32))

# Frozen artifacts (relative to repos/robot_lab/logs/t3; pinned by digest, A13). v0.4 rebuilds.
# All artifacts (v0.3 and v0.4) are listed with path, size and SHA-256 in data/artifacts_manifest.json.
POOL_FILES = dict(train="diag-pool-v4-train-20261123T130844Z-d65b4fdb/artifacts/t3_pool_train.npz",
                  eval="diag-pool-v4-eval-20261123T130847Z-ff4cfa08/artifacts/t3_pool_eval.npz")
EVAL_SET_FILE = "diag-eval-set-v5-20261124T073730Z-cf80791b/artifacts/t3_eval_set_v5.npz"   # v5: S1 from T1 2.0.0 (v4 superseded)
TRAIN_POOL_SHA256 = "6e4a18f7fc531e1cb9799efa99c0dee9f63ff9e222a766f0a3cc5118fc270c70"
EVAL_POOL_SHA256 = "86b45c40638009b40250d92595b6f3697bfe55543c7bebf365d59f923001d911"
EVAL_SET_SHA256 = "0a7dd22d8d7bf3fcee335b70e658d12b990d52399c0c8f07c0fd0bf6cd0d4151"
