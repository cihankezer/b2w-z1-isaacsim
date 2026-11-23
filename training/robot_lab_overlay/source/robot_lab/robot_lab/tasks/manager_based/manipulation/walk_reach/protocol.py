"""T3 protocol constants (docs/protocols/T3_walk_and_reach_protocol.md, v0.3).

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

VERSION = "0.3.0"
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
BODY_BOX_T = (tuple(MEASURED["B_body_T"]["min"]), tuple(MEASURED["B_body_T"]["max"]))
WHEEL_CONTACT_T_XY = tuple(tuple(c[:2]) for c in MEASURED["wheels"]["contact_T"])   # FR, FL, RR, RL
WHEEL_CENTRE_BODY = tuple(tuple(c) for c in MEASURED["wheels"]["centre_body"])
H_KNEE_M = MEASURED["knees"]["h_knee"]
FEASIBILITY = MEASURED["feasibility"]     # dh_axis, pitch_axis, feasible[dh][pitch] from the frozen T2 map

# §3 / D5 T2 limits (design limits of the T2 map) and fall thresholds
H_T2_MIN_M = 0.274
THETA_T2_MAX_RAD = 0.28
FALL_TILT_RAD = max(1.0, THETA_T2_MAX_RAD + 0.2)
FALL_HEIGHT_M = min(0.5 * H_NOM_M, H_T2_MIN_M - 0.05)
DEBOUNCE_STEPS = 5
DRIFT_M = 0.30
CONTACT_N = 1.0
GROUND_CLEARANCE_M = 0.005
CLEARANCE_MARGINS_M = T2P.CLEARANCE_MARGINS_M   # pool checks: T2 per-class margins

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
T1_EVAL_TARGETS_SHA256 = "505128a11752916727de056b049b87555cfdeb11981fcff27768eda2cadaf190"   # validate-v1c targets.json
T2_EVAL_SEED = T2P.EVAL_SEED
T2_EVAL_SET_SHA256 = T2P.EVAL_SET_SHA256
TRAIN_POOL_SEED = 3101   # M1 candidates + segment draws; M2 uses the T1 rule with this seed (T1 used 1101, 1102)
EVAL_POOL_SEED = 3901    # S4 goals (disjoint from training)
EVAL_SET_SEED = 3301
POOL_SIZES = dict(train=dict(M1=1536, M2=768, M3=768, successors=8, max_candidates=32),
                  eval=dict(M1=512, M2=0, M3=0, successors=8, max_candidates=32))

# Frozen artifacts (relative to repos/robot_lab/logs/t3; pinned by digest, A13)
POOL_FILES = dict(train="diag-pool-train-20261123T122622Z-0b82b3db/artifacts/t3_pool_train.npz",
                  eval="diag-pool-eval-20261123T122627Z-762eb62d/artifacts/t3_pool_eval.npz")
EVAL_SET_FILE = "diag-eval-set-20261123T123507Z-ea4e9782/artifacts/t3_eval_set_v3.npz"
TRAIN_POOL_SHA256 = "93529dbdbf3c97e640d9a0e1aa5d3cb8c00f624bb1ae7ec6aef037f578c61da5"
EVAL_POOL_SHA256 = "58742edddd9b0a8718ff065c61ef1bb6f9a447cc6f0d186e767de44bb583b5ee"
EVAL_SET_SHA256 = "77d5bd4e0091d3c4f9916971e249f9987c2da064231eabbbe013dd6bb36054d7"
