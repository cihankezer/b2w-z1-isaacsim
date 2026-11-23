"""Isaac Lab adapters for the no-learning T1 benchmark."""

import numpy as np
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, RecorderTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms

from .geometry import pose, pose_error, reference, steady_state, transform
from . import terminations as TR

LEGS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
WHEELS = [f"{leg}_foot_joint" for leg in ("FR", "FL", "RR", "RL")]
ARM = [f"joint{i}" for i in range(1, 7)]
JOINTS = LEGS + WHEELS + ARM
# Shared T1/T2/T3 action interface (T3 protocol D1, D22): wheel channels are velocity targets a_w * s_w.
WHEEL_ACTION_SCALE_RAD_S = 10.0


class StandingAction(ActionTerm):
    """Radians relative to nominal pose (18 position channels); four wheel channels are velocity targets
    a_w * WHEEL_ACTION_SCALE_RAD_S (rad/s). Zero wheel action = 0 rad/s, as before (T3 D1, D22)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, names = self._asset.find_joints(JOINTS, preserve_order=True)
        # draft.4: the gripper asset adds jointGripper, held closed by its own drive, outside the action space
        extra = sorted(set(self._asset.joint_names) - set(JOINTS))
        if names != JOINTS or extra not in ([], ["jointGripper"]):
            raise ValueError(f"T1 requires the named 22 B2W+Z1 joints plus at most jointGripper; extra={extra}")
        self.position_ids = self.joint_ids[:12] + self.joint_ids[16:]
        self.wheel_ids = self.joint_ids[12:16]
        self._raw = torch.zeros(env.num_envs, 22, device=env.device)
        self._processed = self._raw.clone()
        # Compatibility with training/inspect_task.py's action-order audit.
        self._joint_ids = self.joint_ids

    @property
    def action_dim(self):
        return 22

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        if actions.shape != self._raw.shape or not bool(torch.isfinite(actions).all()):
            raise ValueError("T1 actions must be finite [num_envs, 22]: nominal offsets in radians, wheel velocity commands")
        self._raw[:] = actions
        limits = self._asset.data.soft_joint_pos_limits[:, self.joint_ids]
        targets = self._asset.data.default_joint_pos[:, self.joint_ids] + actions
        self._processed[:] = targets.clamp(limits[..., 0], limits[..., 1])
        self._processed[:, 12:16] = actions[:, 12:16] * WHEEL_ACTION_SCALE_RAD_S

    def apply_actions(self):
        targets = torch.cat([self._processed[:, :12], self._processed[:, 16:]], dim=1)
        self._asset.set_joint_position_target(targets, joint_ids=self.position_ids)
        self._asset.set_joint_velocity_target(self._processed[:, 12:16], joint_ids=self.wheel_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._processed[ids] = self._asset.data.default_joint_pos[ids][:, self.joint_ids]
        self._processed[ids, 12:16] = 0.0
        self.apply_actions()


@configclass
class StandingActionCfg(ActionTermCfg):
    class_type: type = StandingAction
    asset_name: str = "robot"


class StandingCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.hand = self.robot.find_bodies(["link06"])[0][0]
        self.targets = [None] * env.num_envs   # None = nominal hold; set per env with assign()
        self.settled = [None] * env.num_envs
        self.initial_base = [None] * env.num_envs
        self.initial_feet = [None] * env.num_envs
        self.initial_wheels = [None] * env.num_envs
        self.previous_wheels = [None] * env.num_envs
        self.wheel_travel = np.zeros((env.num_envs, 4))
        self.foot_ids = self.robot.find_bodies([f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")],
                                                preserve_order=True)[0]
        self.wheel_ids = self.robot.find_joints(WHEELS, preserve_order=True)[0]
        self._command = torch.zeros(env.num_envs, 7, device=env.device)

    @property
    def command(self):
        return self._command

    def assign(self, i, target):
        """Give env i a sampled target (env-frame poses shifted by the env origin); None = nominal hold."""
        if target is not None:
            import copy
            target = copy.deepcopy(target)
            key = "position_world_xyz" if target["family"] == "fixed" else "centre_world_xyz"
            target[key] = (np.array(target[key]) + self._env.scene.env_origins[i].cpu().numpy()).tolist()
        self.targets[i] = target

    def evaluate(self, i):
        t = float(self._env.episode_length_buf[i]) * self._env.step_dt
        d = self.robot.data
        measured = transform(d.body_pos_w[i, self.hand].cpu().numpy(), d.body_quat_w[i, self.hand].cpu().numpy())
        if self.settled[i] is None or t <= 2.0 + 1e-9:
            self.settled[i] = measured
            self.initial_base[i] = d.root_pos_w[i].cpu().numpy().copy()
            self.initial_feet[i] = d.body_pos_w[i, self.foot_ids].cpu().numpy().copy()
            self.initial_wheels[i] = d.joint_pos[i, self.wheel_ids].cpu().numpy().copy()
            self.previous_wheels[i] = self.initial_wheels[i].copy()
            self.wheel_travel[i] = 0.0
        if self.targets[i] is None:
            return (measured, "settle", t, False) if t <= 2.0 else (measured, "hold", t-2.0, False)
        return reference(self.targets[i], t, self.settled[i])

    def duration(self, i):
        t = self.targets[i]
        return 2.0 + (t["hold_s"] if t["family"] == "fixed" else t["transition_s"] + t["duration_s"]) if t else 60.0

    def _resample_command(self, env_ids):
        for i in env_ids:
            self.settled[int(i)] = None
        self._update_command()

    def _update_metrics(self):
        pass  # Recorder computes errors at matching reference/state timestamps.

    def _update_command(self):
        values = []
        for i in range(self.num_envs):
            try:
                values.append(pose(self.evaluate(i)[0]))
            except ValueError:
                values.append([0, 0, 0, 1, 0, 0, 0])  # invalid-state termination records failure
        world = torch.as_tensor(np.array(values), device=self.device, dtype=torch.float32)
        p, q = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w,
                                         world[:, :3], world[:, 3:])
        self._command[:] = torch.cat([p, q], dim=1)


@configclass
class StandingCommandCfg(CommandTermCfg):
    class_type: type = StandingCommand
    resampling_time_range: tuple = (1e9, 1e9)


def measured_ee(env):
    robot = env.scene["robot"]
    hand = robot.find_bodies(["link06"])[0][0]
    p, q = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                    robot.data.body_pos_w[:, hand], robot.data.body_quat_w[:, hand])
    return torch.cat([p, q], dim=1)


def zero_velocity_goal(env):
    return torch.zeros(env.num_envs, 3, device=env.device)


def time_out(env):
    cmd = env.command_manager.get_term("ee_pose")
    return torch.tensor([float(env.episode_length_buf[i])*env.step_dt >= cmd.duration(i)-1e-9
                         for i in range(env.num_envs)], device=env.device)


class StandingFailure(ManagerTermBase):
    """Evaluate named reasons together; retain them before automatic reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.model = None
        self.reasons = [[] for _ in range(env.num_envs)]
        self.diagnostics = [{} for _ in range(env.num_envs)]
        self.self_clearance_episode = [{} for _ in range(env.num_envs)]  # per-episode counts, monitor only
        # Shared termination set (T3 protocol D6, D20): debounce counters for F2/F3, drift latch for D.
        self.tilt_steps = np.zeros(env.num_envs, dtype=np.int64)
        self.height_steps = np.zeros(env.num_envs, dtype=np.int64)
        self.drift_latch = torch.zeros(env.num_envs, 2, device=env.device)
        self.drift_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = range(len(self.self_clearance_episode)) if env_ids is None else \
            (env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids))
        for i in ids:
            self.self_clearance_episode[i] = {}
            self.tilt_steps[i] = self.height_steps[i] = 0
            self.drift_latched[i] = False

    def _count_self_clearance(self, i, rows):
        counts = self.self_clearance_episode[i]
        for row in rows:
            key = ":".join(row["pair"])
            c = counts.setdefault(key, dict(control_steps=0, below_threshold_steps=0, penetration_steps=0,
                                            unresolved_steps=0, min_clearance_m=None))
            c["control_steps"] += 1
            c["below_threshold_steps"] += row["classification"] in ("clearance_violation", "penetration")
            c["penetration_steps"] += row["classification"] == "penetration"
            c["unresolved_steps"] += row["classification"] == "unresolved"
            if row["clearance_m"] is not None and (c["min_clearance_m"] is None or row["clearance_m"] < c["min_clearance_m"]):
                c["min_clearance_m"] = row["clearance_m"]
        return {k: dict(v) for k, v in counts.items()}

    def __call__(self, env, min_height_m=TR.FALL_HEIGHT_M, max_tilt_rad=TR.FALL_TILT_RAD, contact_force_n=1.0,
                 enforce_stance=False, debounce_steps=TR.DEBOUNCE_STEPS, drift_m=TR.DRIFT_M):
        cmd, robot = env.command_manager.get_term("ee_pose"), env.scene["robot"]
        d = robot.data
        if self.model is None:
            import omni.usd
            from .kinematics import UsdArmKinematics
            self.model = UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
        contact = env.scene["contacts"].data.net_forces_w_history.norm(dim=-1).amax(dim=1).cpu().numpy()
        sensor_names = env.scene["contacts"].body_names
        # T1 has no velocity command: every step is standing (D latch = position at the first step after reset).
        drift_hit = TR.drift_update(self.drift_latch, self.drift_latched,
                                    torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                                    d.root_pos_w[:, :2], drift_m).tolist()
        results = []
        for i in range(env.num_envs):
            reasons = []
            arrays = [d.root_state_w[i], d.joint_pos[i], d.joint_vel[i], d.body_state_w[i]]
            invalid = not all(bool(torch.isfinite(v).all()) for v in arrays)
            invalid |= bool((d.body_quat_w[i].norm(dim=-1) - 1).abs().max() > 1e-3)
            diag = dict(base_height_m=None, tilt_rad=None, foot_drift_m=None,
                        base_xy_displacement_m=None, foot_contact_flags=None,
                        wheel_rotation_rad=None, wheel_travel_rad=None, base_roll_pitch_rad=None,
                        standing_violation=True, min_height_m=None, collision_geometry=[],
                        self_clearance=[], self_clearance_episode=dict(self.self_clearance_episode[i]))
            if invalid:
                reasons.append("invalid_state")
            else:
                cmd.evaluate(i)
                base = d.root_pos_w[i].cpu().numpy()
                height = float(base[2] - env.scene.env_origins[i, 2].item())
                tilt = float(torch.acos((-d.projected_gravity_b[i, 2]).clamp(-1, 1)))
                drift = np.linalg.norm(d.body_pos_w[i, cmd.foot_ids, :2].cpu().numpy() - cmd.initial_feet[i][:, :2], axis=1)
                foot_contact = [bool(contact[i, sensor_names.index(robot.body_names[j])] > contact_force_n)
                                for j in cmd.foot_ids]
                wheels = d.joint_pos[i, cmd.wheel_ids].cpu().numpy() - cmd.initial_wheels[i]
                current_wheels = d.joint_pos[i, cmd.wheel_ids].cpu().numpy()
                cmd.wheel_travel[i] += np.abs(current_wheels-cmd.previous_wheels[i])
                cmd.previous_wheels[i] = current_wheels.copy()
                angles = euler_xyz_from_quat(d.root_quat_w[i:i+1])[:2]
                roll_pitch = [float(torch.atan2(a.sin(), a.cos())[0]) for a in angles]
                stance = bool(max(drift) > 0.05 or not all(foot_contact) or max(abs(wheels)) > 0.05)
                self.height_steps[i], fell_height = TR.debounce(self.height_steps[i], height < min_height_m, debounce_steps)
                self.tilt_steps[i], fell_tilt = TR.debounce(self.tilt_steps[i], tilt > max_tilt_rad, debounce_steps)
                if fell_height:
                    reasons.append("fall_height")
                if fell_tilt:
                    reasons.append("fall_tilt")
                if drift_hit[i]:
                    reasons.append("standing_drift")
                if any(force > contact_force_n and not name.endswith("_foot")
                       for force, name in zip(contact[i], sensor_names)):
                    reasons.append("collision_contact")
                if bool(env.scene["arm_contacts"].data.net_forces_w_history[i].norm(dim=-1).max() > contact_force_n):
                    reasons.append("collision_arm_contact")
                frames = {name: transform(d.body_pos_w[i, j].cpu().numpy(), d.body_quat_w[i, j].cpu().numpy())
                          for j, name in enumerate(robot.body_names)
                          if name in self.model.bounds}
                # Policy draft.5: collider ground rows terminate; collider self rows are a logged
                # 5 mm monitor (physical self-contact terminates via collision_arm_contact).
                geometry_report = self.model.clearance_report(frames)
                for row in geometry_report:
                    row["gating"] = row["pair"][1] == "ground"
                    kind = row["classification"]
                    if row["gating"] and kind != "broad_phase_false_positive":
                        reasons.append("collision_geometry:"+kind+":"+":".join(row["pair"]))
                self_rows = self.model.self_clearance_monitor(geometry_report)
                episode_counts = self._count_self_clearance(i, self_rows)
                if enforce_stance and stance and float(env.episode_length_buf[i])*env.step_dt > 2.0:
                    reasons.append("standing_violation")
                diag = dict(base_height_m=height, tilt_rad=tilt, foot_drift_m=drift.tolist(),
                            base_xy_displacement_m=float(np.linalg.norm(base[:2] - cmd.initial_base[i][:2])),
                            foot_contact_flags=foot_contact, wheel_rotation_rad=wheels.tolist(),
                            wheel_travel_rad=cmd.wheel_travel[i].tolist(), base_roll_pitch_rad=roll_pitch,
                            standing_violation=stance, min_height_m=min_height_m,
                            collision_geometry=geometry_report, self_clearance=self_rows,
                            self_clearance_episode=episode_counts)
            self.reasons[i], self.diagnostics[i] = reasons, diag
            results.append(bool(reasons))
        return torch.tensor(results, dtype=torch.bool, device=env.device)


class StepRecorder(RecorderTerm):
    def record_post_step(self):
        env = self._env
        cmd, robot = env.command_manager.get_term("ee_pose"), env.scene["robot"]
        monitor = env.termination_manager.get_term_cfg("failure").func
        action = env.action_manager.get_term("joints")
        d = robot.data
        samples = []
        for i in range(env.num_envs):
            goal = actual = body_goal = None
            phase, phase_time, scored, errors = "invalid", 0.0, False, (None, None, None)
            target = cmd.targets[i]
            if "invalid_state" not in monitor.reasons[i]:
                goal, phase, phase_time, scored = cmd.evaluate(i)
                actual = transform(d.body_pos_w[i, cmd.hand].cpu().numpy(), d.body_quat_w[i, cmd.hand].cpu().numpy())
                body = transform(d.root_pos_w[i].cpu().numpy(), d.root_quat_w[i].cpu().numpy())
                body_goal = pose(np.linalg.inv(body) @ goal)
                errors = pose_error(actual, goal)
            goal_pose = pose(goal) if goal is not None else [None]*7
            steady = bool(phase != "invalid" and steady_state(target, float(env.episode_length_buf[i])*env.step_dt))
            samples.append(dict(env_id=i, target_id=target["target_id"] if target else "nominal_hold",
                family=target["family"] if target else "hold", sim_time_s=env.common_step_counter*env.step_dt,
                episode_time_s=float(env.episode_length_buf[i])*env.step_dt,
                phase=phase, phase_time_s=phase_time, scored=scored, steady=steady,
                base_pos_world_xyz=d.root_pos_w[i].tolist(), base_quat_world_wxyz=d.root_quat_w[i].tolist(),
                base_lin_vel_world_xyz=d.root_lin_vel_w[i].tolist(), base_ang_vel_body_xyz=d.root_ang_vel_b[i].tolist(),
                projected_gravity_body_xyz=d.projected_gravity_b[i].tolist(),
                ee_pos_world_xyz=d.body_pos_w[i, cmd.hand].tolist(), ee_quat_world_wxyz=d.body_quat_w[i, cmd.hand].tolist(),
                ee_goal_pos_world_xyz=list(goal_pose[:3]), ee_goal_quat_world_wxyz=list(goal_pose[3:]),
                ee_goal_pos_body_xyz=list(body_goal[:3]) if body_goal is not None else None,
                ee_goal_quat_body_wxyz=list(body_goal[3:]) if body_goal is not None else None,
                position_error_m=errors[0], orientation_error_geodesic_rad=errors[1],
                orientation_error_paper_frobenius=errors[2],
                joint_pos=d.joint_pos[i, action.joint_ids].tolist(), joint_vel=d.joint_vel[i, action.joint_ids].tolist(),
                action_command=action.raw_actions[i].tolist(), applied_joint_targets=action.processed_actions[i].tolist(),
                foot_contact_forces=env.scene["contacts"].data.net_forces_w[i, [env.scene["contacts"].body_names.index(robot.body_names[j]) for j in cmd.foot_ids]].tolist(),
                foot_positions_world_xyz=d.body_pos_w[i, cmd.foot_ids].tolist(),
                terminated=bool(env.reset_terminated[i]), truncated=bool(env.reset_time_outs[i]),
                failure=bool(monitor.reasons[i]), failure_reason=monitor.reasons[i], **monitor.diagnostics[i]))
        env.t1_samples = samples  # copied physical states, before any automatic reset
        return None, None
