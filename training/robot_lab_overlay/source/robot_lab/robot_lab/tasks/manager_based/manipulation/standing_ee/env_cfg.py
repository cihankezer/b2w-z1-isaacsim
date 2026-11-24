"""Nominal B2W+Z1 T1 task. No PPO runner, rewards, curriculum or randomization."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RecorderTermCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg

from ...locomotion.velocity.config.wheeled.unitree_b2w.fetch_env_cfg import (B2W_Z1_GRIPPER_USD, GRIP_KD, GRIP_KP,
                                                                              GRIPPER_CLOSED)
from ..reach_b2w_z1.reach_env_cfg import UNITREE_B2W_Z1_STANDING_CFG

from .protocol import NOMINAL_ARM_POSE, NOMINAL_BASE_POSITION_XYZ, NOMINAL_BASE_QUAT_WXYZ, NOMINAL_LEG_JOINT_POS

# Nominal initial state (protocol draft.5; constants and provenance in protocol.py). Deterministic:
# PD targets equal these positions (StandingAction targets default_joint_pos + action). jointGripper
# stays closed (Isaac Lab clamps it to its 90 % soft limit, about -0.0785 rad).

# T1 robot (protocol draft.4/5): the gripper asset b2w_z1_gripper.usd with the standing actuators.
# Its PhysX self-collision is used as authored (enabled; asset filters head/lidar and arm base/body).
# jointGripper is held closed by its own drive and is not part of the 22-dim T1 action space.
T1_ROBOT_CFG = UNITREE_B2W_Z1_STANDING_CFG.replace(
    spawn=UNITREE_B2W_Z1_STANDING_CFG.spawn.replace(
        usd_path=B2W_Z1_GRIPPER_USD,
        articulation_props=UNITREE_B2W_Z1_STANDING_CFG.spawn.articulation_props.replace(enabled_self_collisions=True),
    ),
    init_state=UNITREE_B2W_Z1_STANDING_CFG.init_state.replace(
        pos=NOMINAL_BASE_POSITION_XYZ, rot=NOMINAL_BASE_QUAT_WXYZ, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0),
        joint_pos={**NOMINAL_LEG_JOINT_POS, **NOMINAL_ARM_POSE, "jointGripper": GRIPPER_CLOSED}, joint_vel={".*": 0.0}),
    actuators={**UNITREE_B2W_Z1_STANDING_CFG.actuators,
               "gripper": ImplicitActuatorCfg(joint_names_expr=["jointGripper"], stiffness=GRIP_KP, damping=GRIP_KD)},
)
from .observations import SuitePolicyCfg, tcp_pose_body, tcp_pose_error_body
from .runtime import (StandingActionCfg, StandingCommandCfg, StandingFailure,
                      StepRecorder, time_out)
from .terminations import DEBOUNCE_STEPS, DRIFT_M, FALL_HEIGHT_M, FALL_TILT_RAD


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
    robot = T1_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, update_period=0.005)
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/.*", history_length=4, update_period=0.005)


@configclass
class CommandsCfg:
    ee_pose = StandingCommandCfg()


@configclass
class ActionsCfg:
    joints = StandingActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(SuitePolicyCfg):
        # 2.1.0: shared 95-dim suite observation (T3 D40), velocity-goal slot zero; T1's goal command is `ee_pose`.
        tcp_goal_body = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        tcp_pose_body = ObservationTermCfg(func=tcp_pose_body, params={"command_name": "ee_pose"})
        tcp_pose_error_body = ObservationTermCfg(func=tcp_pose_error_body, params={"command_name": "ee_pose"})

    policy = PolicyCfg()


@configclass
class EventsCfg:
    nominal_material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (1.0, 1.0),
        "dynamic_friction_range": (1.0, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    root = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}})
    joints = EventTermCfg(func=mdp.reset_joints_by_scale, mode="reset", params={
        "position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})


@configclass
class TerminationsCfg:
    failure = TerminationTermCfg(func=StandingFailure, params={
        "min_height_m": FALL_HEIGHT_M, "max_tilt_rad": FALL_TILT_RAD, "contact_force_n": 1.0,
        "enforce_stance": False, "debounce_steps": DEBOUNCE_STEPS, "drift_m": DRIFT_M})
    time_out = TerminationTermCfg(func=time_out, time_out=True)


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    samples = RecorderTermCfg(class_type=StepRecorder)


@configclass
class StandingEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=3.0)
    commands = CommandsCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    recorders = RecordersCfg()
    rewards: dict = {}
    curriculum: dict = {}

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.decimation = 4
        self.sim.render_interval = 4
        self.episode_length_s = 60.0  # actual per-target timeout is handled by time_out
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        self.seed = 1101
