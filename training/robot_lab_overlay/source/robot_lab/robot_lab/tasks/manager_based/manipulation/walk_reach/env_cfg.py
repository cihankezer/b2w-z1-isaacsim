"""T3 walk-and-reach environment (protocol v0.6), derived from the T2 environment.

Diff from T2 (T2ReachEnvCfg): commands base_velocity (§6.1) and tcp_target (F_T, goal pool, §6.2-6.3);
the policy observation keeps T2's layout and order (95 dims) with the 3-dim velocity-goal slot filled by the
velocity command (no T3-only dims); contact terminations report F1 / S1 / S2 (D21); standing drift only while
the velocity command is zero (D6). Same robot, scene, sensors, 22-dim action, physics and control rates as
T1/T2. No rewards, critic group, domain randomisation or PPO runner (WP1).

T3WalkReachEnvCfg      training: 4096 envs, T2 training reset, 20 s episodes (= preset train-t3)
T3WalkReachEvalEnvCfg  benchmark: deterministic T1 nominal reset, settle 2 s + test 10 s (§7.4)

Training presets (D42). The T3 environment is the single training environment of the suite; the T1 and T2
environments are evaluation-only. All presets share the action, the 95-dim observation and the termination set;
they differ only in the standing fraction of the velocity command and the goal-mode weights:
  TrainT1EnvCfg    train-t1    standing fraction 1.0, goal modes M2 only
  TrainT1T2EnvCfg  train-t1t2  standing fraction 1.0, goal modes M1 + M2 + M3 (protocol weights)
  TrainT3EnvCfg    train-t3    full T3 distribution (protocol standing fraction and weights)
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from ..reach_height.env_cfg import ObservationsCfg as T2ObservationsCfg
from ..reach_height.env_cfg import T2ReachEnvCfg
from ..reach_height.env_cfg import TerminationsCfg as T2TerminationsCfg
from ..standing_ee.env_cfg import EventsCfg as T1EventsCfg
from ..standing_ee.env_cfg import SceneCfg
from ..standing_ee.terminations import StandingDrift
from . import protocol as P
from .runtime import VELOCITY, ClassifiedContact, WalkReachTargetCommandCfg, WalkVelocityCommandCfg


@configclass
class CommandsCfg:
    base_velocity = WalkVelocityCommandCfg()      # computed before tcp_target (moving flag for §6.3 check 3)
    tcp_target = WalkReachTargetCommandCfg()


@configclass
class ObservationsCfg(T2ObservationsCfg):
    @configclass
    class PolicyCfg(T2ObservationsCfg.PolicyCfg):
        # Same slot and position as T1/T2 (zero there): (vx*, vy*, wz*) of the velocity command.
        base_velocity_goal = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": VELOCITY})

    policy = PolicyCfg()


@configclass
class TerminationsCfg(T2TerminationsCfg):
    body_contact = TerminationTermCfg(func=ClassifiedContact, params={
        "threshold": 1.0, "sensor_cfg": SceneEntityCfg("contacts", body_names="(?!.*_foot$).*")})
    arm_contact = TerminationTermCfg(func=ClassifiedContact, params={
        "threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_contacts", body_names=".*")})
    walk_away = TerminationTermCfg(func=StandingDrift, params={"limit_m": P.DRIFT_M, "velocity_command": VELOCITY})


@configclass
class T3WalkReachEnvCfg(T2ReachEnvCfg):
    commands = CommandsCfg()
    observations = ObservationsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = P.TRAIN_EPISODE_S


@configclass
class T3WalkReachEvalEnvCfg(T3WalkReachEnvCfg):
    scene = SceneCfg(num_envs=16, env_spacing=3.0)
    events = T1EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Settle + test = 12 s = 600 steps. Two spare steps keep the time-out from resetting the 600th state
        # before it is recorded; the evaluation runner stops after 600 steps.
        self.episode_length_s = P.SETTLE_S + P.TEST_S + 2 * P.CONTROL_DT_S


def _preset(cfg, standing_fraction, mode_weights):
    cfg.commands.base_velocity.rel_standing_envs = standing_fraction
    cfg.commands.tcp_target.mode_weights = tuple(mode_weights)


@configclass
class TrainT1EnvCfg(T3WalkReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _preset(self, 1.0, (0.0, 1.0, 0.0))


@configclass
class TrainT1T2EnvCfg(T3WalkReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _preset(self, 1.0, P.MODE_WEIGHTS)


@configclass
class TrainT3EnvCfg(T3WalkReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _preset(self, P.STANDING_FRACTION, P.MODE_WEIGHTS)


PRESETS = {"train-t1": TrainT1EnvCfg, "train-t1t2": TrainT1T2EnvCfg, "train-t3": TrainT3EnvCfg}
