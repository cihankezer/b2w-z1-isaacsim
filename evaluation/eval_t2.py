"""T2 evaluation on the fixed seeded target set (protocol v0.4 §5, §8, §10 step 6; acceptance tests A3, A4).

Controllers (no policy yet):
  hold    zero action = hold the T1 nominal pose (A4)
  posture A3a gate: after the deterministic reset, the robot state is written directly into the map's
          stored whole-body solution of the target's voxel (base pose mapped from the map's G to the env's
          G, leg and arm joints, zero velocities) and held for POSTURE_HOLD_S. Static-feasibility test
          (protocol §11): no termination and TCP error < 5 cm at the last step; PD targets are gravity-compensated, q_target = q_posture + tau_static / K, with tau_static from
          the static equilibrium at the written state (static_torques). Pass per episode additionally
          requires |tau_static| <= effort limit for every joint (max torque ratio per joint logged).
          Also logged per episode: minimum same-leg hip-wheel collider clearance over the hold (sim body
          poses, T1 collider model) and the maximum leg-joint tracking error |q - q_posture| over the hold.
  oracle  naive scripted baseline, informational only (A3b dropped, out of scope of WP1); joint space,
          to the least-cost posture (legs) and arm solution
          stored in the reachability map for the target's voxel (voxel centre: <= 1.7 cm offset at 2 cm):
          1. 0 - LEG_S: legs smoothstep to the posture, arm at nominal (the posture the map checked);
          2. LEG_S - LEG_S+ARM_S: arm along the first candidate joint path (PATH_CANDIDATES) that passes
             wholebody.clearance_violation at the target posture in PATH_SAMPLES samples per segment, first
             with the arm-body margin + 1 cm, then with the protocol margins; if none passes, the direct
             path is used and flagged (oracle_path = "none_free");
          integral action on the 18 position targets while moving, I += KI_MOVE * (q_ref - q), |I| <= I_MAX
          (arm PD stiffness 100 N m/rad sags without it); final approach (--final-approach): "pd" freezes
          the integral after the motion, "integral" keeps integrating with KI_HOLD.

Per batch of --num-envs targets: reset (deterministic eval reset), one zero-action harness step (§4.4,
outside the episode), assign the targets, reset the episode counters, then 500 recorded control steps
(t_k = (k+1) dt). An env that terminates is not recorded further. Metrics: metrics.episode_metrics.

Outputs in --out: eval_set.npz (+ digest), episodes.csv (appended per episode as it ends), summary.json (per band and acceptance verdicts).
Run inside the Isaac image via evaluation/diagnostics/launch.sh (LAUNCH_TASK=t2, mode gpu).
"""

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.reach_height"
FALL_TERMS = ("fall_tilt", "fall_height", "body_contact")
LEG_S, ARM_S, KI_MOVE, KI_HOLD, I_MAX, PATH_SAMPLES = 4.0, 4.0, 0.02, 0.1, 0.5, 20
PATH_EXTRA_ARM_BODY_M = (0.01, 0.0)   # planning: arm-body margin + 1 cm first (tracking lag), then protocol
POSTURE_HOLD_S = 2.0
PATH_CANDIDATES = ("direct", "shoulder_last", "elbow_first", "via_stretch", "via_stretch_high",
                   "via_stretch_shoulder_last")


def smooth(np, s):
    s = np.clip(s, 0.0, 1.0)
    return 3 * s * s - 2 * s ** 3


class OraclePlanner:
    """CPU arm-path selection with the T1 collider model (reach_height.wholebody.WholeBody)."""

    def __init__(self, np, mods, model):
        W = mods["wholebody"]
        self.np, self.W, self.P = np, W, mods["protocol"]
        self.wb = W.WholeBody(model)
        self.q0 = np.array([self.wb.joints_nominal[n] for n in W.ARM])
        # Forward-stretched arm waypoints: link06 ahead of the shoulder, tool axis forward.
        self.stretch = self._ik(np.array([0.55, 0.0, 0.0]))
        self.stretch_high = self._ik(np.array([0.45, 0.0, 0.15]))

    def _ik(self, offset):
        target = self.np.eye(4)
        target[:3, 3] = self.wb.shoulder_base + offset
        q, ok, _, _ = self.W.arm_ik(self.wb.arm, target[None], self.q0[None], self.wb.arm_limits)
        return q[0] if ok[0] else None

    def free(self, base, legs, path, margins):
        np, W = self.np, self.W
        for a, b in zip(path[:-1], path[1:]):
            for s in np.linspace(0.0, 1.0, PATH_SAMPLES):
                q = (1 - s) * a + s * b
                if W.clearance_violation(self.wb.model, self.wb.model.forward(base, W.full_joints(legs, q)), margins):
                    return False
        return True

    def plan(self, base, leg_q, leg_names, goal):
        """Arm waypoints (list of (6,)) from nominal to goal at the target posture, and the chosen candidate."""
        legs = dict(self.wb.joints_nominal, **dict(zip(leg_names, leg_q)))
        shoulder_last = goal.copy()
        shoulder_last[1] = self.q0[1]                # goal with the nominal shoulder pitch (joint2)
        elbow_first = self.q0.copy()
        elbow_first[2] = goal[2]                     # nominal with the goal's elbow (joint3)
        candidates = [("direct", [self.q0, goal]), ("shoulder_last", [self.q0, shoulder_last, goal]),
                      ("elbow_first", [self.q0, elbow_first, goal])]
        if self.stretch is not None:
            candidates.append(("via_stretch", [self.q0, self.stretch, goal]))
        if self.stretch_high is not None:
            candidates.append(("via_stretch_high", [self.q0, self.stretch_high, goal]))
        if self.stretch is not None:
            mid = goal.copy()
            mid[1] = self.stretch[1]                 # goal with the stretched shoulder pitch (joint2)
            candidates.append(("via_stretch_shoulder_last", [self.q0, self.stretch, mid, goal]))
        for extra in PATH_EXTRA_ARM_BODY_M:
            margins = dict(self.P.CLEARANCE_MARGINS_M)
            margins["arm_body"] += extra
            for name, path in candidates:
                if self.free(base, legs, path, margins):
                    return path, f"{name}@+{extra * 100:.0f}cm"
        return candidates[0][1], "none_free"


def json_safe(v):
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=1, default=float) + "\n")


def oracle_targets(np, data, voxels, joint_names, default):
    """Absolute joint targets (n, 22) in T1 action order from the map's posture and arm solution per voxel."""
    k = data["posture_index"][voxels]
    if (k < 0).any():
        raise ValueError("oracle: target voxel without a stored solution")
    q = np.tile(default, (len(voxels), 1))
    legs = [str(x) for x in data["leg_joint_names"]]
    for c, name in enumerate(legs):
        q[:, joint_names.index(name)] = data["posture_leg_q"][k, c]
    for c in range(6):
        q[:, joint_names.index(f"joint{c + 1}")] = data["arm_q"][voxels, c]
    q[:, 12:16] = 0.0
    return q, k


def run_batch(env, mods, torch, np, targets, sel, controller, planner, data, result_rows, failures, final_approach="pd"):
    P, M = mods["protocol"], mods["metrics"]
    n_env = env.num_envs
    robot, cmd = env.scene["robot"], env.command_manager.get_term("tcp_target")
    action_term = env.action_manager.get_term("joints")
    jid = action_term.joint_ids
    joint_names = [robot.joint_names[j] for j in jid]
    default = robot.data.default_joint_pos[0, jid].cpu().numpy()
    soft = robot.data.soft_joint_pos_limits[0, jid].cpu().numpy()
    arm_cols, leg_cols = list(range(16, 22)), list(range(12))
    contacts = env.scene["contacts"]
    foot_ids = [contacts.body_names.index(f"{leg}_foot") for leg in ("FR", "FL", "RR", "RL")]
    tm = env.termination_manager
    terms = list(tm.active_terms)
    n = len(sel)
    zero = torch.zeros(n_env, 22, device=env.device)

    env.reset()
    env.step(zero)                                            # §4.4 harness step, not recorded
    settle_terminated = int(env.reset_buf.sum())
    ids = torch.arange(n, device=env.device)
    cmd.assign(ids, targets["position_G"][sel], targets["quat_G_wxyz"][sel], targets["band"][sel])
    env.episode_length_buf[:] = 0
    target_pos_w, target_quat_w = cmd.position_w[:n].cpu().numpy().copy(), cmd.quat_w[:n].cpu().numpy().copy()
    reset_base = robot.data.root_pos_w[:n].cpu().numpy().copy()
    oracle, posture = controller == "oracle", controller == "posture"
    if oracle or posture:
        q_goal, posture_k = oracle_targets(np, data, targets["voxel"][sel], joint_names, default)
        q_goal_raw = q_goal.copy()
        q_goal = np.clip(q_goal, soft[:, 0], soft[:, 1])
    if posture:
        from scipy.spatial.transform import Rotation
        G_env = np.tile(np.eye(4), (n, 1, 1))
        G_env[:, :3, :3] = Rotation.from_euler("z", cmd.yaw[:n].cpu().numpy()).as_matrix()
        G_env[:, :3, 3] = cmd.origin_w[:n].cpu().numpy()
        base = G_env @ np.linalg.inv(data["G_world"]) @ data["posture_base_world"][posture_k]
        pose = np.concatenate([base[:, :3, 3], Rotation.from_matrix(base[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]], axis=1)
        q_full = robot.data.default_joint_pos[:n].clone()
        q_full[:, jid] = torch.as_tensor(q_goal, dtype=q_full.dtype, device=env.device)
        robot.write_root_pose_to_sim(torch.as_tensor(pose, dtype=torch.float32, device=env.device), env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
        robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full), env_ids=ids)
        env.sim.physics_sim_view.update_articulations_kinematic()     # link poses for the Jacobians
        tau_static, statics = static_torques(env, robot, torch, np, n, jid, foot_body_ids(robot))
        stiffness = robot.data.joint_stiffness[:n, jid].cpu().numpy()
        effort = robot.data.joint_effort_limits[:n, jid].cpu().numpy()
        pos_cols = leg_cols + arm_cols
        q_hold = q_goal.copy()
        q_hold[:, pos_cols] += tau_static[:, pos_cols] / stiffness[:, pos_cols]
        hold_clamped = (q_hold < soft[:, 0]) | (q_hold > soft[:, 1])
        hold_clamped[:, 12:16] = False
        torque_ratio = np.abs(tau_static) / effort
        torque_ratio[:, 12:16] = 0.0          # wheels: penalised to ~0 in the solve (wheel_torque_abs_max logged)
        hold_action = torch.as_tensor(q_hold - default, dtype=torch.float32, device=env.device)
        hold_action[:, 12:16] = 0.0
        reset_base = robot.data.root_pos_w[:n].cpu().numpy().copy()
        path_names = ["reset_into_posture"] * n
        cmd_last = np.zeros((n, 22))
        hip_wheel_ids = [(robot.body_names.index(f"{leg}_hip"), robot.body_names.index(f"{leg}_foot")) for leg in LEGS]
        hw_rec, leg_q_rec = [], []
        robot_model = tm.get_term_cfg("ground_clearance").func.model     # T1 collider model of env_0
    if oracle:
        leg_names = [str(x) for x in data["leg_joint_names"]]
        paths, path_names = [], []
        for i in range(n):
            path, name = planner.plan(data["posture_base_world"][posture_k[i]], data["posture_leg_q"][posture_k[i]],
                                      leg_names, q_goal[i, arm_cols])
            paths.append(np.array(path))
            path_names.append(name)
        integral = np.zeros((n, 22))
        cmd_last = np.zeros((n, 22))
        pos_cols = leg_cols + arm_cols
    horizon_s = POSTURE_HOLD_S if posture else P.EPISODE_S
    horizon = int(round(horizon_s / P.CONTROL_DT_S))
    rec = {k: [] for k in ("tcp_pos", "tcp_quat", "base_pos", "base_quat", "grav", "foot", "arm_q", "arm_tau", "leg_tau")}
    alive = np.ones(n, dtype=bool)
    length = np.zeros(n, dtype=int)
    reasons = [[] for _ in range(n)]
    clearance_reasons = [[] for _ in range(n)]
    contact_bodies = [[] for _ in range(n)]
    contact_terms = [tm.get_term_cfg(t).func for t in ("body_contact", "arm_contact")]
    q_last = np.zeros((n, 22))
    for k in range(horizon):
        act = zero.clone()
        if posture:
            act[:n] = hold_action
        if oracle:
            t = (k + 1) * P.CONTROL_DT_S
            q_ref = np.tile(default, (n, 1))
            q_ref[:, leg_cols] = default[leg_cols] + smooth(np, t / LEG_S) * (q_goal[:, leg_cols] - default[leg_cols])
            u = np.clip((t - LEG_S) / ARM_S, 0.0, 1.0)
            for i in range(n):
                m = len(paths[i]) - 1
                seg = min(int(u * m), m - 1)
                w = smooth(np, u * m - seg)
                q_ref[i, arm_cols] = (1 - w) * paths[i][seg] + w * paths[i][seg + 1]
            ki = KI_MOVE if t <= LEG_S + ARM_S else (KI_HOLD if final_approach == "integral" else 0.0)
            q_meas = robot.data.joint_pos[:n][:, jid].cpu().numpy()
            integral[:, pos_cols] = np.clip(integral[:, pos_cols] + ki * (q_ref[:, pos_cols] - q_meas[:, pos_cols]),
                                            -I_MAX, I_MAX)
            q_raw = q_ref + integral
            q_cmd = np.clip(q_raw, soft[:, 0], soft[:, 1])
            q_cmd[:, 12:16] = 0.0
            clamped_now = np.abs(q_raw - q_cmd) > 1e-6
            clamped_now[:, 12:16] = False
            cmd_last[alive] = clamped_now[alive]
            act[:n] = torch.as_tensor(q_cmd - default, dtype=torch.float32, device=env.device)
            act[:n, 12:16] = 0.0
        # One step; recorded states are post-step and pre-reset for terminated envs (captured below).
        env.step(act)
        d = robot.data
        tcp_p, tcp_q = cmd.tcp_pose_w()
        # Isaac Lab resets terminated envs inside step(); the termination terms keep the pre-reset values,
        # but the robot state is already reset, so a terminated env's last state is not recorded.
        done = env.reset_buf[:n].cpu().numpy().astype(bool)
        rec["tcp_pos"].append(tcp_p[:n].cpu().numpy())
        rec["tcp_quat"].append(tcp_q[:n].cpu().numpy())
        rec["base_pos"].append((d.root_pos_w[:n] - env.scene.env_origins[:n] * torch.tensor([0, 0, 1.0], device=env.device)).cpu().numpy())
        rec["base_quat"].append(d.root_quat_w[:n].cpu().numpy())
        rec["grav"].append(d.projected_gravity_b[:n].cpu().numpy())
        rec["foot"].append((contacts.data.net_forces_w[:n][:, foot_ids].norm(dim=-1) > 1.0).cpu().numpy())
        rec["arm_q"].append(d.joint_pos[:n][:, [jid[c] for c in arm_cols]].cpu().numpy())
        rec["arm_tau"].append(d.applied_torque[:n][:, [jid[c] for c in arm_cols]].cpu().numpy())
        rec["leg_tau"].append(d.applied_torque[:n][:, [jid[c] for c in leg_cols]].cpu().numpy())
        if posture:
            flat = [b for pair in hip_wheel_ids for b in pair]
            hw_rec.append((d.body_pos_w[:n][:, flat].cpu().numpy(), d.body_quat_w[:n][:, flat].cpu().numpy()))
            leg_q_rec.append(d.joint_pos[:n][:, [jid[c] for c in leg_cols]].cpu().numpy())
        values = {t: tm.get_term(t)[:n].cpu().numpy().astype(bool) for t in terms}
        gc = tm.get_term_cfg("ground_clearance").func.reasons
        for i in np.flatnonzero(alive & done):
            reasons[i] = [t for t in terms if values[t][i]]
            clearance_reasons[i] = list(gc[i])
            contact_bodies[i] = sorted({b for term in contact_terms for b in term.reasons[i]})
            length[i] = k            # steps recorded before the terminating step (its state is post-reset)
            alive[i] = False
        q_now = d.joint_pos[:n][:, jid].cpu().numpy()
        q_last[alive] = q_now[alive]          # last state of an env before it terminates or the episode ends
        length[alive] = k + 1
        if not alive.any():
            break
    arrays = {key: np.stack(v, axis=1) for key, v in rec.items()}   # (n, K, ...)
    if posture:
        hip_wheel = hip_wheel_clearance(np, mods, robot_model, hw_rec, length)
        leg_q = np.stack(leg_q_rec, axis=1)
    for i in range(n):
        K = max(int(length[i]), 1)
        timeout_only = reasons[i] == ["time_out"] or (not reasons[i] and length[i] >= horizon)
        fell = bool(set(reasons[i]) & set(FALL_TERMS))
        terminated = bool(reasons[i]) and not timeout_only
        # walked_away=terminated: any non-time-out termination makes the episode incomplete in metrics.py.
        ep = dict(tcp_pos_w=arrays["tcp_pos"][i, :K], target_pos_w=target_pos_w[i],
                  tcp_quat_w=arrays["tcp_quat"][i, :K], target_quat_w=target_quat_w[i],
                  base_pos_w=arrays["base_pos"][i, :K], base_quat_w=arrays["base_quat"][i, :K],
                  projected_gravity_b=arrays["grav"][i, :K], foot_contact=arrays["foot"][i, :K],
                  arm_q=arrays["arm_q"][i, :K], arm_soft_limits=soft[arm_cols],
                  arm_torque=arrays["arm_tau"][i, :K], leg_torque=arrays["leg_tau"][i, :K],
                  reset_base_pos_w=reset_base[i], fell=fell, walked_away=terminated)
        m = M.episode_metrics(ep, horizon_s=horizon_s, hold_s=P.CONTROL_DT_S if posture else P.HOLD_S)
        m["walked_away"] = "walk_away" in reasons[i]
        if not m["completed"]:
            cause = ("fall" if fell else "clearance" if "ground_clearance" in reasons[i]
                     else "other_termination")
        else:
            cause = "success" if m["success"] else "did_not_reach"
        row = dict(target_id=int(targets["target_id"][sel[i]]), band=P.BAND_NAMES[int(targets["band"][sel[i]])],
                   target_x_G=float(targets["position_G"][sel[i], 0]), target_y_G=float(targets["position_G"][sel[i], 1]),
                   target_z=float(targets["position_G"][sel[i], 2]),
                   target_dist_nominal_tcp=float(np.linalg.norm(targets["position_G"][sel[i]] - failures["nominal_tcp_G"])),
                   outcome=cause, terminations="|".join(t for t in reasons[i] if t != "time_out"),
                   clearance_reasons="|".join(clearance_reasons[i]), contact_bodies="|".join(contact_bodies[i]), **m)
        if oracle or posture:
            err = np.abs(q_last[i] - q_goal[i])
            row.update(oracle_arm_joint_err_max=float(err[arm_cols].max()), oracle_leg_joint_err_max=float(err[leg_cols].max()),
                       oracle_arm_clamped=bool(np.any(np.abs(q_goal_raw[i] - q_goal[i]) > 1e-6)),
                       oracle_arm_joint_err="|".join(f"{e:.3f}" for e in err[arm_cols]),
                       oracle_cmd_at_soft_limit="|".join(joint_names[j] for j in np.flatnonzero(cmd_last[i])),
                       oracle_path=path_names[i], oracle_posture_dh=float(data["posture_dh"][posture_k[i]]),
                       oracle_posture_pitch=float(data["posture_pitch"][posture_k[i]]))
        if posture:   # A3a diagnostics over the recorded hold steps (protocol §12, map t2-reachability-3)
            K = max(int(length[i]), 1)
            hw = hip_wheel[i]
            row.update(hip_wheel_clearance_min=float(hw["min"]), hip_wheel_clearance_first=float(hw["first"]),
                       hip_wheel_clearance_leg=hw["leg"],
                       leg_joint_track_err_max=float(np.abs(leg_q[i, :K] - q_goal[i, leg_cols]).max()))
            ratio = torque_ratio[i]
            applied = np.zeros(22)
            applied[leg_cols] = np.abs(arrays["leg_tau"][i, :K]).max(axis=0)
            applied[arm_cols] = np.abs(arrays["arm_tau"][i, :K]).max(axis=0)
            row.update(torque_ratio_max=float(ratio.max()), torque_ratio_argmax=joint_names[int(ratio.argmax())],
                       applied_torque_ratio_max=float((applied / effort[i]).max()),
                       hold_target_clamped="|".join(joint_names[j] for j in np.flatnonzero(hold_clamped[i])),
                       **{f"static_{k}": float(v[i]) for k, v in statics.items()},
                       **{f"torque_ratio_{joint_names[j]}": float(ratio[j]) for j in pos_cols})
            row["static_feasible"] = bool(row["success"] and ratio.max() <= 1.0)
            if row["outcome"] == "success" and ratio.max() > 1.0:
                row["outcome"] = "torque_limit"
        result_rows.append(row)
        failures["emit"](row)
    failures["settle_terminated"] += settle_terminated


LEGS = ("FR", "FL", "RR", "RL")


def foot_body_ids(robot):
    return [robot.body_names.index(f"{leg}_foot") for leg in LEGS]


def static_torques(env, robot, torch, np, n, jid, feet):
    """Static joint torques holding the written state (A3a, protocol §11), from PhysX at that state.

    Equilibrium S tau + Jc^T F = g, with g = get_gravity_compensation_forces (floating base: 6 base + dof
    coordinates; base moment rows about the base COM). Jc = point Jacobians of the 4 wheel-ground contacts.
    PhysX Jacobians give link-COM velocities, and the contact points lie below the wheel COMs at the
    ground height of the env's G. Unknowns: F (4 x 3, world frame). Hard constraints are the 6 base rows
    (the contacts carry the robot). F is statically indeterminate (6 internal DOF), so this picks the
    distribution minimising sum (tau_j / limit_j)^2 over legs and wheels. Wheels get a 1 N m pseudo-limit,
    because velocity-controlled wheels with damping only hold no static torque. A wheel-torque equality
    would be linearly dependent on the base force rows for parallel wheels (rank loss). Validated against
    the settled nominal pose by diagnostics/t2_static_check.py. Returns tau (n, 22) in action order and
    per-env diagnostics.
    """
    view = robot.root_physx_view
    g = view.get_gravity_compensation_forces()[:n].double()
    J = view.get_jacobians()[:n].double()                       # (n, bodies, 6, 6 + dof)
    dof = robot.num_joints
    if g.shape[1] != dof + 6 or J.shape[-1] != dof + 6:
        raise RuntimeError(f"expected floating-base PhysX shapes, got g {tuple(g.shape)}, J {tuple(J.shape)}")
    com = robot.data.body_com_pos_w[:n][:, feet].double()        # (n, 4, 3), reference point of J
    contact = com.clone()
    contact[..., 2] = env.command_manager.get_term("tcp_target").origin_w[:n, 2:3].double()
    r = contact - com
    Jv, Jw = J[:, feet, :3, :], J[:, feet, 3:, :]
    rx = torch.zeros(n, 4, 3, 3, dtype=r.dtype, device=r.device)   # skew(r)
    rx[..., 0, 1], rx[..., 0, 2], rx[..., 1, 2] = -r[..., 2], r[..., 1], -r[..., 0]
    rx = rx - rx.transpose(-1, -2)
    Jc = (Jv - rx @ Jw).reshape(n, 12, dof + 6)                  # v_contact = v_com + w x r
    A = Jc[:, :, :6].transpose(1, 2)                              # (n, 6, 12) base rows
    jidx = torch.as_tensor(list(jid), device=g.device)
    B = Jc[:, :, 6:].transpose(1, 2)[:, jidx]                     # (n, 22, 12): tau = g_j - B F
    gj = g[:, 6:][:, jidx]
    limits = robot.data.joint_effort_limits[:n, jidx].double().clone()
    limits[:, 12:16] = 1.0
    w2 = torch.zeros_like(limits)
    w2[:, :16] = 1.0 / limits[:, :16] ** 2                        # legs and wheels; arm torques do not depend on F
    H = 2 * (B.transpose(1, 2) * w2[:, None, :]) @ B + 1e-8 * torch.eye(12, dtype=g.dtype, device=g.device)
    h = 2 * ((B.transpose(1, 2) * w2[:, None, :]) @ gj[..., None])[..., 0]
    K = torch.zeros(n, 18, 18, dtype=g.dtype, device=g.device)
    K[:, :12, :12], K[:, :12, 12:], K[:, 12:, :12] = H, A.transpose(1, 2), A
    F = torch.linalg.solve(K, torch.cat([h, g[:, :6]], dim=1))[:, :12]
    residual = (A @ F[..., None])[..., 0] - g[:, :6]
    tau = (gj - (B @ F[..., None])[..., 0]).cpu().numpy()
    Fc = F.reshape(n, 4, 3).cpu().numpy()
    weight = float(robot.data.default_mass[0].sum()) * 9.81
    statics = dict(contact_fz_total_over_weight=Fc[..., 2].sum(axis=1) / weight,
                   contact_fz_min=Fc[..., 2].min(axis=1),
                   contact_tangential_over_normal_max=(np.linalg.norm(Fc[..., :2], axis=-1)
                                                       / np.maximum(Fc[..., 2], 1e-9)).max(axis=1),
                   equilibrium_residual=residual.norm(dim=1).cpu().numpy(),
                   wheel_torque_abs_max=np.abs(tau[:, 12:16]).max(axis=1))
    if np.median(statics["contact_fz_total_over_weight"]) < 0:
        raise RuntimeError("gravity force sign convention: contact forces pull, not push (check PhysX convention)")
    return tau, statics


def hip_wheel_clearance(np, mods, model, rec, length):
    """Per env: minimum same-leg hip-wheel collider clearance (T1 collider model, hpp-fcl) over the recorded steps
    (sim body poses, steps before a termination only), its value at the first recorded step and the leg of the minimum."""
    from scipy.spatial.transform import Rotation
    C = mods["clearance"]
    pos = np.stack([r[0] for r in rec], axis=1)    # (n, K, 2 * legs, 3)
    quat = np.stack([r[1] for r in rec], axis=1)
    out = []
    for i in range(pos.shape[0]):
        K = max(int(length[i]), 1)
        best, first, leg_best = np.inf, None, None
        for k in range(K):
            frames = np.tile(np.eye(4), (pos.shape[2], 1, 1))
            frames[:, :3, :3] = Rotation.from_quat(quat[i, k][:, [1, 2, 3, 0]]).as_matrix()
            frames[:, :3, 3] = pos[i, k]
            step_min = np.inf
            for l, leg in enumerate(LEGS):
                d = min(C.surface_clearance(C.moved_shape(a, frames[2 * l]), C.moved_shape(b, frames[2 * l + 1]))["clearance_m"]
                        for a in model.shapes[f"{leg}_hip"] for b in model.shapes[f"{leg}_foot"])
                step_min = min(step_min, d)
                if d < best:
                    best, leg_best = d, leg
            if first is None:
                first = step_min
        out.append(dict(min=best, first=first, leg=leg_best))
    return out


def quantiles(np, values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(len(v)), **{k: float(x) for k, x in zip(("min", "p5", "p25", "median", "p75", "p95", "max"), q)})


def height_figure(path, bins, controller):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = [(b["z_lo"] + b["z_hi"]) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(z, [b["success"] if b["success"] is not None else 0 for b in bins], width=0.08, color="#4c72b0")
    ax2 = ax.twinx()
    ax2.plot(z, [b["e_p_final_median"] if b["e_p_final_median"] is not None else float("nan") for b in bins], "k.-")
    ax.set_xlabel("target TCP height z [m]")
    ax.set_ylabel("hold-window success")
    ax2.set_ylabel("median final error [m]")
    ax.set_ylim(0, 1)
    for edge in (0.3, 0.8):
        ax.axvline(edge, color="grey", lw=0.6, ls="--")
    ax.set_title(f"T2 {controller}: success and final error vs. target height", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def band_report(np, rows, P, controller, nominal_frac):
    out = {}
    for b in P.BAND_NAMES:
        r = [x for x in rows if x["band"] == b]
        e = np.array([x["e_p_final"] for x in r])
        causes = {c: sum(x["outcome"] == c for x in r) for c in ("success", "torque_limit", "did_not_reach", "fall", "clearance", "other_termination")}
        term_counts = {}
        for x in r:
            for t in filter(None, x["terminations"].split("|")):
                term_counts[t] = term_counts.get(t, 0) + 1
        tilt = [x["tilt_rms_hold"] for x in r if x["tilt_rms_hold"] is not None and np.isfinite(x["tilt_rms_hold"])]
        out[b] = dict(n=len(r), success=float(np.mean([x["success"] for x in r])),
                      success_kaist_strict=float(np.mean([x["success_kaist_strict"] for x in r])),
                      e_p_final_median=float(np.median(e)), e_p_final_p90=float(np.percentile(e, 90)),
                      e_p_final_mean=float(e.mean()), outcomes=causes, termination_terms=term_counts,
                      falls=sum(x["fell"] for x in r), tilt_rms_hold_mean=float(np.mean(tilt)) if tilt else None,
                      tilt_rms_hold_max=float(np.max(tilt)) if tilt else None,
                      dh_hold_mean=float(np.nanmean([x["dh_hold"] for x in r])) if any(np.isfinite(x["dh_hold"]) for x in r) else None,
                      targets_within_5cm_of_nominal_tcp=nominal_frac[b])
        if r and "oracle_path" in r[0]:
            out[b]["oracle_paths"] = {p: sum(x["oracle_path"] == p for x in r) for p in sorted({x["oracle_path"] for x in r})}
            out[b]["oracle_outcome_by_path"] = {p: {c: sum(x["oracle_path"] == p and x["outcome"] == c for x in r) for c in causes}
                                                for p in out[b]["oracle_paths"]}
        if r and "torque_ratio_max" in r[0]:
            out[b]["static_feasible"] = float(np.mean([x["static_feasible"] for x in r]))
            names = [k[len("torque_ratio_"):] for k in r[0] if k.startswith("torque_ratio_") and k not in
                     ("torque_ratio_max", "torque_ratio_argmax")]
            out[b]["torque_ratio_per_joint"] = {j: dict(quantiles(np, [x[f"torque_ratio_{j}"] for x in r]),
                                                        above_1=sum(x[f"torque_ratio_{j}"] > 1.0 for x in r)) for j in names}
            out[b]["torque_ratio_max"] = quantiles(np, [x["torque_ratio_max"] for x in r])
            out[b]["applied_torque_ratio_max"] = quantiles(np, [x["applied_torque_ratio_max"] for x in r])
            out[b]["hold_target_clamped_episodes"] = sum(bool(x["hold_target_clamped"]) for x in r)
            for k in ("contact_fz_total_over_weight", "contact_fz_min", "contact_tangential_over_normal_max",
                      "equilibrium_residual", "wheel_torque_abs_max"):
                out[b][f"static_{k}"] = quantiles(np, [x[f"static_{k}"] for x in r])
        if r and "hip_wheel_clearance_min" in r[0]:
            ok = [x for x in r if x["outcome"] == "success"]
            bad = [x for x in r if x["outcome"] != "success"]
            out[b]["hip_wheel_clearance_min_m"] = dict(
                all=quantiles(np, [x["hip_wheel_clearance_min"] for x in r]),
                success=quantiles(np, [x["hip_wheel_clearance_min"] for x in ok]),
                failure=quantiles(np, [x["hip_wheel_clearance_min"] for x in bad]),
                first_step=quantiles(np, [x["hip_wheel_clearance_first"] for x in r]),
                below_0=sum(x["hip_wheel_clearance_min"] < 0 for x in r),
                below_5mm=sum(x["hip_wheel_clearance_min"] < 0.005 for x in r),
                below_1cm=sum(x["hip_wheel_clearance_min"] < 0.01 for x in r),
                below_2cm=sum(x["hip_wheel_clearance_min"] < 0.02 for x in r))
            out[b]["leg_joint_track_err_max_rad"] = dict(
                all=quantiles(np, [x["leg_joint_track_err_max"] for x in r]),
                success=quantiles(np, [x["leg_joint_track_err_max"] for x in ok]),
                failure=quantiles(np, [x["leg_joint_track_err_max"] for x in bad]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=("hold", "posture", "oracle"), required=True)
    parser.add_argument("--final-approach", choices=("pd", "integral"), default="pd", help="oracle (A3b) only")
    parser.add_argument("--reachability", required=True, help="container path of the frozen v0.4 npz")
    parser.add_argument("--eval-seed", type=int, default=None, help="override protocol EVAL_SEED (flagged)")
    parser.add_argument("--n-per-band", type=int, default=None, help="override protocol N_EVAL_PER_BAND (flagged)")
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--profile-clearance", action="store_true", help="time the ground-clearance term phases")
    parser.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(controller=args.controller, eval_seed=args.eval_seed, n_per_band_override=args.n_per_band,
                   num_envs=args.num_envs,
                   oracle=dict(leg_s=LEG_S, arm_s=ARM_S, ki_move=KI_MOVE, ki_hold=KI_HOLD, i_max=I_MAX, final_approach=args.final_approach,
                               path_samples=PATH_SAMPLES, path_extra_arm_body_m=list(PATH_EXTRA_ARM_BODY_M),
                               path_candidates=list(PATH_CANDIDATES)),
                   posture_hold_s=POSTURE_HOLD_S if args.controller == "posture" else None if args.controller == "oracle" else None,
                   gpu_uuid=os.environ["ISAAC_GPU_UUID"], image_id=os.environ.get("T1_IMAGE_ID"),
                   reachability=args.reachability, status="started")
    write_json(args.out / "summary.json", summary)
    app = env = None
    try:
        app = AppLauncher(args).app
        import gymnasium as gym
        import numpy as np
        import torch
        import robot_lab.tasks  # noqa: F401
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        if str(OVERLAY) not in parent.__path__:
            parent.__path__.insert(0, str(OVERLAY))
        importlib.import_module(PACKAGE)
        mods = {m: importlib.import_module(f"{PACKAGE}.{m}") for m in ("protocol", "sampler", "metrics", "env_cfg", "wholebody")}
        mods["clearance"] = importlib.import_module("robot_lab.tasks.manager_based.manipulation.standing_ee.clearance")
        P, S = mods["protocol"], mods["sampler"]
        cfg = mods["env_cfg"].T2ReachEvalEnvCfg()
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.commands.tcp_target.reachability_path = args.reachability
        # The harness records exactly EPISODE_S / dt steps itself; the env time-out is set 2 control steps
        # later so that it never resets an env before its last recorded state (terminations unchanged).
        cfg.episode_length_s = P.EPISODE_S + 2 * P.CONTROL_DT_S
        env = gym.make("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
        cmd = env.command_manager.get_term("tcp_target")
        clearance = env.termination_manager.get_term_cfg("ground_clearance").func
        clearance.profile = args.profile_clearance

        rs = S.ReachableSet.load(args.reachability)
        n_per_band = args.n_per_band or P.N_EVAL_PER_BAND
        eval_seed = P.EVAL_SEED if args.eval_seed is None else args.eval_seed
        targets, meta = S.eval_target_set(rs, eval_seed, cmd.reachability_sha256, n_per_band)
        frozen_set = eval_seed == P.EVAL_SEED and n_per_band == P.N_EVAL_PER_BAND
        if frozen_set and meta["targets_sha256"] != P.EVAL_SET_SHA256:
            raise RuntimeError(f"evaluation set digest {meta['targets_sha256']} != pinned EVAL_SET_SHA256")
        summary.update(eval_seed=eval_seed, frozen_eval_set=frozen_set)
        S.save_eval_set(args.out / "eval_set.npz", targets, meta)
        targets, meta = S.load_eval_set(args.out / "eval_set.npz", rs)       # verify round trip
        with np.load(args.reachability, allow_pickle=False) as d:
            data = {k: d[k] for k in ("posture_index", "arm_q", "posture_leg_q", "leg_joint_names", "posture_dh", "posture_pitch",
                                      "posture_base_world", "G_world")}
            map_meta = json.loads(str(d["metadata"]))
        nominal_tcp_G = np.array(map_meta["nominal_tcp_G"])
        dist = np.linalg.norm(targets["position_G"] - nominal_tcp_G, axis=1)
        nominal_frac = {b: float(np.mean(dist[targets["band"] == i] <= P.SUCCESS_POSITION_M)) for i, b in enumerate(P.BAND_NAMES)}
        summary.update(protocol_version=P.VERSION, eval_set_sha256=meta["targets_sha256"], n_targets=int(len(targets["band"])),
                       reachability_sha256=cmd.reachability_sha256, nominal_tcp_G=nominal_tcp_G.tolist(),
                       torch=torch.__version__, device=torch.cuda.get_device_name(0))

        planner = OraclePlanner(np, mods, clearance.model) if args.controller == "oracle" else None
        if planner is not None:
            summary["oracle"]["stretch_waypoints_found"] = [planner.stretch is not None, planner.stretch_high is not None]
        episodes_csv = open(args.out / "episodes.csv", "w", newline="")
        writer = []

        def emit(row):            # one CSV line per episode as soon as it ends (survives an interrupted run)
            if not writer:
                writer.append(csv.DictWriter(episodes_csv, fieldnames=list(row)))
                writer[0].writeheader()
            writer[0].writerow(json_safe(row))
            episodes_csv.flush()

        rows, info = [], dict(settle_terminated=0, nominal_tcp_G=nominal_tcp_G, emit=emit)
        order = np.arange(len(targets["band"]))
        t0 = time.perf_counter()
        for start in range(0, len(order), args.num_envs):
            sel = order[start:start + args.num_envs]
            run_batch(env, mods, torch, np, targets, sel, args.controller, planner, data, rows, info, args.final_approach)
            print(f"batch {start // args.num_envs}: {len(rows)}/{len(order)} episodes, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
        summary["wall_s"] = time.perf_counter() - t0
        episodes_csv.close()
        bands = band_report(np, rows, P, args.controller, nominal_frac)
        summary.update(bands=bands, harness_settle_step_terminations=info["settle_terminated"],
                       height_bins=mods["metrics"].height_bins(rows))
        height_figure(args.out / "success_vs_height.png", summary["height_bins"], args.controller)
        if clearance.profile:
            st = clearance.stats
            summary["clearance"] = dict(steps=st["calls"], flagged_envs_per_step=st["flagged_envs"] / max(st["calls"], 1),
                                        flagged_max_per_step=st["flagged_max"], exact_checks=st["narrow_checks"],
                                        broad_ms_per_step=st["broad_s"] / max(st["calls"], 1) * 1e3,
                                        exact_ms_per_step=st["narrow_s"] / max(st["calls"], 1) * 1e3,
                                        exact_ms_per_check=st["narrow_s"] / max(st["narrow_checks"], 1) * 1e3,
                                        exact_total_s=st["narrow_s"])
        if args.controller == "posture":  # A3a gate (protocol §11)
            summary["A3a"] = dict(criterion=f"static feasibility, per band >= 95 %: gravity-compensated hold of {POSTURE_HOLD_S} s "
                                            "with no fall, no illegal contact (no termination), TCP error < 5 cm at the end, "
                                            "static joint torques within effort limits",
                                  per_band={b: bands[b]["static_feasible"] >= 0.95 for b in P.BAND_NAMES})
            summary["A3a"]["pass"] = all(summary["A3a"]["per_band"].values())
        elif args.controller == "oracle":  # naive scripted baseline (A3b dropped), informational
            summary["A3b"] = dict(note="naive scripted baseline, informational only; A3b dropped (out of scope of WP1)",
                                  success={b: bands[b]["success"] for b in P.BAND_NAMES})
        else:                             # A4 (protocol §11, D5)
            falls = sum(r["fell"] for r in rows)
            tilt = [r["tilt_rms_hold"] for r in rows if r["tilt_rms_hold"] is not None and np.isfinite(r["tilt_rms_hold"])]
            per_band = {b: bands[b]["success"] <= nominal_frac[b] + 1e-12 for b in P.BAND_NAMES}
            summary["A4"] = dict(criterion="0 falls; tilt_rms_hold < 0.01 rad; per band success <= fraction of targets within 5 cm of nominal TCP",
                                 episodes=len(rows), falls=int(falls), tilt_rms_hold_max=float(max(tilt)) if tilt else None,
                                 success_le_nominal_fraction=per_band)
            summary["A4"]["pass"] = bool(falls == 0 and tilt and max(tilt) < 0.01 and all(per_band.values()))
        summary["status"] = "completed"
    except Exception:
        summary.update(status="error", error=traceback.format_exc())
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        write_json(args.out / "summary.json", summary)
        print(json.dumps(json_safe({k: v for k, v in summary.items() if k not in ("error", "bands")}), indent=1, default=float), flush=True)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    raise SystemExit(0 if summary["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
