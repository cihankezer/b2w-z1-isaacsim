"""T2 protocol constants (docs/protocols/T2_reach_low_high_protocol.md, v0.4).

Units: metres, radians, seconds. Frames: G = ground frame of an episode (origin = base origin at
reset projected onto the ground, axes = base yaw at reset); targets are stored in G and in W.
Pitch sign: positive = nose down (rotation about +y with x forward, z up; Isaac Lab convention).
"""

import numpy as np

from ..standing_ee.protocol import NOMINAL_BASE_POSITION_XYZ

VERSION = "1.0.0"
BENCHMARK = "T2-REACH-LOW-HIGH"
ROBOT_VARIANT = "B2W"

# §3 Robot. h_nom = T1 nominal base height (D1); EE = TCP at a fixed offset from link06 (D2):
# link06 -> gripperStator is +0.051 m on the tool axis (+x of link06, verified in the asset) and the
# fetch pipeline's grasp point is +0.12 m further along it (fetch_env_cfg.GRASP_POINT, x component).
H_NOM_M = float(NOMINAL_BASE_POSITION_XYZ[2])
TCP_PARENT = "link06"
TCP_OFFSET_M = (0.171, 0.0, 0.0)
TOOL_AXIS = "+x of link06"

# §4.1 Bands (z = TCP height above ground) and azimuth sector in G.
BANDS = (("low", 0.00, 0.30), ("mid", 0.30, 0.80), ("high", 0.80, 1.10))
BAND_NAMES = tuple(b[0] for b in BANDS)
AZIMUTH_MAX_RAD = float(np.deg2rad(60.0))

# §4.2 Mode A canonical orientation (D6, v0.4): radial approach (tool axis in the vertical plane through
# the target azimuth, pointing away from the base), pitched below horizontal by a continuous function of
# the target height z: vertical (pi/2) at z = 0, 45 deg at z = 0.30 m, horizontal at z >= 0.60 m, linear
# in between. Replaces the v0.3 per-band switch (top-down Low, horizontal Mid/High).
ORIENTATION_MODE = "A"
APPROACH_PROFILE = ((0.00, float(np.pi / 2)), (0.30, float(np.pi / 4)), (0.60, 0.0))   # (z [m], angle below horizontal [rad])

# §4.3 Reachability (base x/y fixed over the wheels, option (c) rejected). v0.4: training and reachability
# use dh in [DH_MIN_M, 0] (physical crouch limit found by A1 run 2); KAIST -0.20 m is a reference only.
DH_MIN_M = -0.33
REACHABILITY = {
    "version": "t2-reachability-3",
    "grid_resolution_m": 0.02,
    "grid_x_m": (0.0, 1.2), "grid_y_m": (-0.8, 0.8), "grid_z_m": (0.0, 1.2),
    "dh_step_m": 0.02, "dh_max_m": 0.0, "dh_min_kaist_m": -0.20,
    "pitch_max_rad": 0.28, "pitch_step_rad": 0.04,
    "base_roll_rad": 0.0, "base_xy": "fixed at the nominal base position",
    "legs": "wheel centres (foot-joint anchors) fixed at their nominal world positions; leg IK within soft limits",
    "arm": "6D IK of link06 for the TCP target with the canonical orientation (APPROACH_PROFILE), within soft limits",
    "joint_limit_factor": 0.9,
    "ik_tolerance": {"position_m": 0.001, "orientation_rad": 0.01},
    "clearance": "T1 collider geometry; per-class margins CLEARANCE_MARGINS_M (wholebody.clearance_violation): non-wheel "
                 "bodies to ground, arm-arm, arm-body, body-body incl. legs and wheels (non-adjacent, not USD-filtered)",
    "clearance_margins_m": None,   # filled below from CLEARANCE_MARGINS_M
    "posture_cost": "(dh/0.20)^2 + (pitch/0.28)^2; the least-cost feasible posture is recorded per voxel",
    "min_band_reachable_fraction": 0.20,   # A1 (v0.4): annulus fraction of every band
    "connectivity": "6-neighbour voxel faces",
    "min_largest_region_fraction": 0.90,   # A1 (v0.4): largest connected region / band's reachable voxels
}
# v0.4 (2026-11-13): per-class clearance margins of the reachability check (and oracle path check).
# arm_arm = two Z1 bodies (link00 mount, link01-06, gripper); arm_body = Z1 body vs. any other body;
# body_body = two non-arm bodies (legs, wheels, base, head, lidar) whose distance can change, i.e. leg
# self-collision (pairs welded to base_link by fixed joints are constant and skipped). body_body = 2 cm: A3 hip-wheel contacts occurred in
# simulation at 4-9 mm kinematic clearance.
CLEARANCE_MARGINS_M = {"ground": 0.005, "arm_arm": 0.005, "arm_body": 0.02, "body_body": 0.02}
REACHABILITY["clearance_margins_m"] = dict(CLEARANCE_MARGINS_M)
CROUCH_SEARCH_STEP_M = 0.01   # D4 run 2: deepest feasible dh at pitch 0, searched in these steps

# §4.5 / §5 Episode and success.
CONTROL_DT_S = 0.02
EPISODE_S = 10.0
HOLD_S = 1.0
SUCCESS_POSITION_M = 0.05
SUCCESS_ORIENTATION_RAD = 0.10          # Mode B only
REFERENCE_DH_BAND_M = (REACHABILITY["dh_min_kaist_m"], 0.0)   # KAIST command range, reporting band only
REFERENCE_PITCH_RAD = 0.28              # KAIST command range, reporting band only
JOINT_LIMIT_MARGIN_FRACTION = 0.05      # D11: of the soft-limit range, measured from a soft limit

# §7 Terminations (D10).
WALK_AWAY_M = 0.30
FALL_TILT_RAD = 1.0
CLEARANCE_M = 0.005

# §8 Evaluation.
N_EVAL_PER_BAND = 2000
EVAL_SEED = 2026
# Frozen v1.0 evaluation target set: sampler.eval_target_set(seed EVAL_SEED, N_EVAL_PER_BAND per band) on the
# pinned t2-reachability-3 map (env_cfg.REACHABILITY_SHA256); sampler.digest of the target arrays (A7 runs).
EVAL_SET_SHA256 = "6260df589d63db4d8633959e2244e9ec0320ab4c7a716bab3279b8a7a4f5937e"
N_POLICY_SEEDS = 5


def approach_angle(z):
    """Mode A approach angle below horizontal [rad] for target heights z (piecewise linear, APPROACH_PROFILE)."""
    zs, angles = zip(*APPROACH_PROFILE)
    return np.interp(np.asarray(z, dtype=float), zs, angles)


def band_index(z):
    """Band id (0 low, 1 mid, 2 high) for TCP heights; -1 outside [0, 1.10]. Lower edges inclusive."""
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, -1, dtype=int)
    for i, (_, lo, hi) in enumerate(BANDS):
        out[(z >= lo) & (z < hi)] = i
    out[np.isclose(z, BANDS[-1][2])] = len(BANDS) - 1
    return out


def dh_values(dh_min, step=REACHABILITY["dh_step_m"]):
    """0, -step, ... down to the last multiple of step >= dh_min (dh_min itself if it is a multiple)."""
    n = int(np.floor(-dh_min / step + 1e-9))
    return np.round(-step * np.arange(n + 1), 6)


def pitch_values(limit=REACHABILITY["pitch_max_rad"], step=REACHABILITY["pitch_step_rad"]):
    n = int(round(limit / step))
    return np.round(step * np.arange(-n, n + 1), 6)


def posture_cost(dh, pitch):
    return (np.asarray(dh) / 0.20) ** 2 + (np.asarray(pitch) / 0.28) ** 2
