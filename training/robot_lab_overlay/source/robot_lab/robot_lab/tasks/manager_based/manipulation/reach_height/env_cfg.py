"""T2 reach-low-and-high environment (protocol v0.4), derived from the T1 standing env.

Diff from T1 (StandingEnvCfg): command term (ReachTargetCommand from the frozen reachability map),
policy observations in B with TCP terms, Section 7 terminations, 10 s episodes, no T1 step recorder.
Same robot, scene, sensors, 22-dim action, physics and control rates as T1. No rewards, critic group,
domain randomisation or PPO runner yet (training configuration, protocol §9, is a later step).

T2ReachEnvCfg      training defaults: 4096 envs, reset joints ±0.05 rad and base yaw ±π (§4.4)
T2ReachEvalEnvCfg  benchmark: deterministic T1 nominal reset (§4.4), few envs
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils import configclass

from ..standing_ee.env_cfg import EventsCfg as T1EventsCfg
from ..standing_ee.env_cfg import ActionsCfg, SceneCfg, StandingEnvCfg
from ..standing_ee.runtime import JOINTS, zero_velocity_goal
from . import protocol as P
from .runtime import (COMMAND, GroundClearance, IllegalContact, ReachTargetCommandCfg, invalid_state,
                      tcp_pose_body, tcp_pose_error_body, walk_away)

_JOINTS = SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)


# Frozen v0.4 map, reachability version t2-reachability-3 (A1 passed, protocol §12):
# diag-reach-A1-v04r3-20261113T084127Z-1d029cc2/artifacts/reachability_T2_protocol.npz. Path via ReachTargetCommandCfg.reachability_path or T2_REACHABILITY_NPZ.
REACHABILITY_SHA256 = "22b9e665baa11bced9832107ca17c154ba57751fb55f54c3dbf67d9e1b865404"


@configclass
class CommandsCfg:
    tcp_target = ReachTargetCommandCfg(reachability_sha256=REACHABILITY_SHA256)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # §6 order; frame B, positions + quaternions wxyz (T1 encoding, decided in step 3).
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS})
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": _JOINTS})
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        previous_action = ObservationTermCfg(func=mdp.last_action)
        base_velocity_goal = ObservationTermCfg(func=zero_velocity_goal)
        tcp_goal_body = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": COMMAND})
        tcp_pose_body = ObservationTermCfg(func=tcp_pose_body)
        tcp_pose_error_body = ObservationTermCfg(func=tcp_pose_error_body)
        target_height_band = None   # flag obs_target_height_band: ObservationTermCfg(func=runtime.band_one_hot)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class TrainingEventsCfg(T1EventsCfg):
    """§4.4 training reset: joint positions ±0.05 rad, base yaw ±π; no height or tilt randomisation."""

    def __post_init__(self):
        self.root.params["pose_range"] = {**{k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch")},
                                          "yaw": (-math.pi, math.pi)}
        self.joints = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
            "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)})


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fall_tilt = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": P.FALL_TILT_RAD})
    # Any non-wheel body with net contact force > 1 N (T1 collision_contact semantics; includes ground).
    body_contact = TerminationTermCfg(func=IllegalContact, params={
        "threshold": 1.0, "sensor_cfg": SceneEntityCfg("contacts", body_names="(?!.*_foot$).*")})
    arm_contact = TerminationTermCfg(func=IllegalContact, params={
        "threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_contacts", body_names=".*")})
    invalid_state = TerminationTermCfg(func=invalid_state)
    walk_away = TerminationTermCfg(func=walk_away, params={"max_displacement_m": P.WALK_AWAY_M})
    ground_clearance = TerminationTermCfg(func=GroundClearance, params={"margin_m": P.CLEARANCE_M})


@configclass
class T2ReachEnvCfg(StandingEnvCfg):
    scene = SceneCfg(num_envs=4096, env_spacing=3.0)
    commands = CommandsCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = TrainingEventsCfg()
    terminations = TerminationsCfg()
    recorders = RecorderManagerBaseCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = P.EPISODE_S
        assert abs(self.sim.dt * self.decimation - P.CONTROL_DT_S) < 1e-12


@configclass
class T2ReachEvalEnvCfg(T2ReachEnvCfg):
    scene = SceneCfg(num_envs=16, env_spacing=3.0)
    events = T1EventsCfg()
