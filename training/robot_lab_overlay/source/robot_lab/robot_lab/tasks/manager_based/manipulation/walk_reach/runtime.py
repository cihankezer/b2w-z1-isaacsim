"""Isaac Lab terms for T3 (protocol §4.3, §5, §6, §7.4). Vectorised (torch).

Commands (computed in this order every control step, after physics and terminations):
  base_velocity  stock uniform velocity command (§6.1); evaluation: fixed profile after the settle phase.
  tcp_target     TCP target in F_T (§5.1). Training: GoalSampler over the frozen training pool (§6.2-6.3).
                 Evaluation: the schedule of one evaluation-set episode per env (§7.3-7.4).
The target is held in F_T and mapped to W with the current base pose every step; the command value (policy
observation) is the target in the base frame B, position + quaternion wxyz, as in T2. The command is named
`tcp_target` (protocol: `ee_target`) so that T2's observation terms apply unchanged (same layout, D19).
"""

import os
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from ..reach_height.runtime import IllegalContact
from ..standing_ee import terminations as TR
from . import goals as GOALS
from . import pool as POOL
from . import protocol as P
from .frames import ft_to_world
from .kinematics import ArmFK, interpolate

COMMAND = "tcp_target"
VELOCITY = "base_velocity"
LOGS = Path(__file__).resolve().parents[9] / "repos/robot_lab/logs/t3"
SETTLE_STEPS = int(round(P.SETTLE_S / P.CONTROL_DT_S))


# ---------------------------------------------------------------- base velocity (§6.1, §7.2)

class WalkVelocityCommand(UniformVelocityCommand):
    """Stock uniform velocity command. Evaluation hook assign(): fixed (vx*, wz*) after the settle phase
    (V0 during settle), per env; NaN profile = training behaviour."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.profile = torch.full((self.num_envs, 2), float("nan"), device=self.device)

    def assign(self, env_ids, cmd):
        self.profile[torch.as_tensor(env_ids, device=self.device)] = torch.as_tensor(
            cmd, dtype=torch.float32, device=self.device)
        self._update_command()

    def _update_command(self):
        super()._update_command()
        fixed = ~torch.isnan(self.profile[:, 0])
        if bool(fixed.any()):
            test = (self._env.episode_length_buf > SETTLE_STEPS) & fixed
            v = torch.zeros_like(self.vel_command_b)
            v[:, 0], v[:, 2] = self.profile[:, 0], self.profile[:, 1]
            self.vel_command_b[fixed] = torch.where(test[:, None], v, torch.zeros_like(v))[fixed]


@configclass
class WalkVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = WalkVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple = (P.VELOCITY_RESAMPLE_S, P.VELOCITY_RESAMPLE_S)
    rel_standing_envs: float = P.STANDING_FRACTION
    rel_heading_envs: float = 0.0
    heading_command: bool = False
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=P.VX_RANGE, lin_vel_y=P.VY_RANGE, ang_vel_z=P.WZ_RANGE)


# ---------------------------------------------------------------- TCP target (§5.1, §6.2-6.3, §7.3)

class WalkReachTargetCommand(CommandTerm):
    """T3 TCP target. State per env: target in F_T (position_T, quat_T) and in W (position_w, quat_w)."""

    cfg: "WalkReachTargetCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.robot = env.scene[cfg.asset_name]
        self.link06 = self.robot.find_bodies([P.TCP_PARENT])[0][0]
        self.tcp_offset = torch.tensor(P.TCP_OFFSET_M, device=dev).repeat(n, 1)
        path = cfg.pool_path or os.environ.get("T3_POOL_TRAIN", "") or str(LOGS / P.POOL_FILES["train"])
        arrays, meta = POOL.load(path, cfg.pool_sha256)
        self.pool_path, self.pool_sha256 = path, meta["pool_sha256"]
        self.fk = ArmFK(dev)
        self.pool = GOALS.GoalPool(arrays, dev)
        self.generator = torch.Generator(device=dev)
        self.generator.manual_seed(cfg.seed)
        self.sampler = GOALS.GoalSampler(self.pool, n, dev, generator=self.generator, fk=self.fk)
        home = self.pool.g[self.pool.home][None]
        self.home_T = [x[0] for x in self.fk.tcp_pose(home[:, :6], home[:, 6:])]
        # Evaluation schedule per env (kind 0 = training sampler, 1 = step points S1/S3, 2 = goal chain S4).
        self.kind = torch.zeros(n, dtype=torch.long, device=dev)
        self.points = torch.zeros(n, 2, 7, device=dev)
        self.n_points = torch.ones(n, dtype=torch.long, device=dev)
        self.point_steps = torch.ones(n, dtype=torch.long, device=dev)
        self.chain_g = torch.zeros(n, 1, 9, device=dev)
        self.chain_start = torch.zeros(n, 0, dtype=torch.float64, device=dev)
        self.chain_T = torch.zeros(n, 0, dtype=torch.float64, device=dev)
        self.position_T = self.home_T[0].repeat(n, 1)
        self.quat_T = self.home_T[1].repeat(n, 1)
        self.position_w = torch.zeros(n, 3, device=dev)
        self.quat_w = torch.zeros(n, 4, device=dev)
        self.quat_w[:, 0] = 1.0
        self._command = torch.zeros(n, 7, device=dev)
        self.marker_pose_w = None         # last pose handed to the debug marker (A18)
        self.profile = False              # time the target generator (adds CUDA syncs)
        self.stats = dict(calls=0, generator_s=0.0)
        self.metrics["position_error"] = torch.zeros(n, device=dev)

    @property
    def command(self):
        return self._command

    def tcp_pose_w(self):
        d = self.robot.data
        q = d.body_quat_w[:, self.link06]
        return d.body_pos_w[:, self.link06] + quat_apply(q, self.tcp_offset), q

    def target_height(self):
        """Target height above the terrain (flat: env origin z)."""
        return self.position_w[:, 2] - self._env.scene.env_origins[:, 2]

    # evaluation hook --------------------------------------------------------
    def assign(self, env_ids, scenario, target_T=None, chain_g=None, chain_T=None, chain_H=None):
        """Give env_ids one evaluation episode each. S1/S3: target_T (n, 2, 7) step points in F_T (NaN rows
        unused). S4: chain_g (n, S + 1, 9) goals starting at home, chain_T / chain_H (n, S) in s (NaN = unused)."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if scenario in ("S1", "S3"):
            sch = P.S1_SCHEDULE if scenario == "S1" else P.S3_SCHEDULE
            pts = torch.as_tensor(target_T, dtype=torch.float32, device=self.device)
            self.kind[ids] = 1
            self.points[ids] = torch.nan_to_num(pts)
            self.n_points[ids] = sch["points"]
            self.point_steps[ids] = int(round(sch["hold_s"] / P.CONTROL_DT_S))
        else:
            T = torch.as_tensor(np.nan_to_num(chain_T), dtype=torch.float64, device=self.device)
            H = torch.as_tensor(np.nan_to_num(chain_H), dtype=torch.float64, device=self.device)
            S = T.shape[1]
            if self.chain_g.shape[1] != S + 1:
                n = self.num_envs
                self.chain_g = torch.zeros(n, S + 1, 9, device=self.device)
                self.chain_start = torch.full((n, S), float("inf"), dtype=torch.float64, device=self.device)
                self.chain_T = torch.zeros(n, S, dtype=torch.float64, device=self.device)
            start = torch.cat([torch.zeros(len(ids), 1, dtype=torch.float64, device=self.device),
                               (T + H).cumsum(dim=1)[:, :-1]], dim=1)
            unused = torch.as_tensor(np.isnan(np.asarray(chain_T)), device=self.device)
            self.kind[ids] = 2
            self.chain_g[ids] = torch.as_tensor(chain_g, dtype=torch.float32, device=self.device)
            self.chain_start[ids] = torch.where(unused, torch.full_like(start, float("inf")), start)
            self.chain_T[ids] = T
        self._refresh()

    def _schedule(self, ids):
        """Evaluation targets in F_T at the current step for env ids (kind 1 or 2)."""
        buf = self._env.episode_length_buf[ids]
        pos, quat = self.home_T[0].repeat(len(ids), 1), self.home_T[1].repeat(len(ids), 1)
        test = buf > SETTLE_STEPS
        k1 = ids[(self.kind[ids] == 1) & test]
        if len(k1):
            j = ((self._env.episode_length_buf[k1] - SETTLE_STEPS - 1) // self.point_steps[k1]).clamp(
                torch.zeros_like(self.n_points[k1]), self.n_points[k1] - 1)
            p = self.points[k1, j]
            m = (self.kind[ids] == 1) & test
            pos[m], quat[m] = p[:, :3], p[:, 3:]
        k2 = ids[(self.kind[ids] == 2) & test]
        if len(k2):
            tau = (self._env.episode_length_buf[k2] - SETTLE_STEPS).double() * P.CONTROL_DT_S
            seg = ((self.chain_start[k2] < tau[:, None]).sum(dim=1) - 1).clamp(min=0)
            T = self.chain_T[k2, seg]
            s = torch.where(T > 0, (tau - self.chain_start[k2, seg]) / T.clamp(min=1e-9), torch.ones_like(T))
            g = interpolate(self.chain_g[k2, seg], self.chain_g[k2, seg + 1], s.float().clamp(0, 1))
            p, q = self.fk.tcp_pose(g[:, :6], g[:, 6:])
            m = (self.kind[ids] == 2) & test
            pos[m], quat[m] = p, q
        return pos, quat

    # CommandTerm ------------------------------------------------------------
    def _resample_command(self, env_ids):
        """Episode start: the target is p_home (training: the sampler draws from home at the next step)."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.sampler.reset(ids)
        self.position_T[ids], self.quat_T[ids] = self.home_T[0], self.home_T[1]

    def _update_command(self):
        if self.profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        train = (self.kind == 0).nonzero(as_tuple=True)[0]
        if len(train):
            moving = (self._env.command_manager.get_command(VELOCITY) != 0).any(dim=1)
            self.sampler.update(self._env.step_dt, moving)
            p, q = self.sampler.target()
            self.position_T[train], self.quat_T[train] = p[train], q[train]
        ev = (self.kind > 0).nonzero(as_tuple=True)[0]
        if len(ev):
            self.position_T[ev], self.quat_T[ev] = self._schedule(ev)
        if self.profile:
            torch.cuda.synchronize()
            self.stats["calls"] += 1
            self.stats["generator_s"] += time.perf_counter() - t0
        self._refresh()

    def _refresh(self):
        """F_T -> W with the current base pose, then W -> B (command value)."""
        d = self.robot.data
        self.position_w[:], self.quat_w[:] = ft_to_world(d.root_pos_w, d.root_quat_w, self.position_T, self.quat_T,
                                                         P.H_T_M, self._env.scene.env_origins[:, 2])
        p, q = subtract_frame_transforms(d.root_pos_w, d.root_quat_w, self.position_w, self.quat_w)
        self._command[:] = torch.cat([p, q], dim=1)

    def _update_metrics(self):
        self.metrics["position_error"][:] = (self.tcp_pose_w()[0] - self.position_w).norm(dim=1)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "marker"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG
                self.marker = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/tcp_target"))
            self.marker.set_visibility(True)
        elif hasattr(self, "marker"):
            self.marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.marker_pose_w = (self.position_w.clone(), self.quat_w.clone())
        self.marker.visualize(self.marker_pose_w[0], self.marker_pose_w[1])


@configclass
class WalkReachTargetCommandCfg(CommandTermCfg):
    class_type: type = WalkReachTargetCommand
    asset_name: str = "robot"
    resampling_time_range: tuple = (1e9, 1e9)   # goals are drawn by the sampler; resampled at reset only
    pool_path: str = ""                          # empty: T3_POOL_TRAIN or the pinned path under logs/t3
    pool_sha256: str = P.TRAIN_POOL_SHA256
    seed: int = P.TRAIN_POOL_SEED


# ---------------------------------------------------------------- termination reasons (§4.3, D21)

class ClassifiedContact(IllegalContact):
    """F1 / S1 / S2 / S4 contact termination (> 1 N net force). A17 found no ground-filtered multi-body sensor, so
    each contacting body is classified by geometry at the terminating step (before the reset): terrain if one of
    its colliders lies within TERRAIN_BAND_M of the ground (exact T1 clearance, broad-phase false positives
    dropped), else self-contact. Arm bodies (the arm_contacts sensor): S1 terrain / S2 self. Trunk and legs:
    F1 terrain / S4 self (S2 when an arm body is in self-contact at the same step, D39). The body poses of the hit envs are kept (GPU) at the call; the geometric check
    runs only when an ID is requested (classify), so training pays nothing for the labels."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.arm = set(env.scene.sensors["arm_contacts"].body_names)
        self.model = None
        self._rows, self._pos, self._quat, self._ids = {}, None, None, {}

    def __call__(self, env, threshold, sensor_cfg):
        out = super().__call__(env, threshold, sensor_cfg)
        hit = out.nonzero(as_tuple=True)[0]
        self._ids = {}
        self._rows = {int(i): r for r, i in enumerate(hit.tolist())}
        if self._rows:
            d = env.scene["robot"].data
            self._pos = d.body_pos_w[hit] - env.scene.env_origins[hit][:, None]   # flat ground at the env origin
            self._quat = d.body_quat_w[hit].clone()
        return out

    def classify(self, env, i):
        """IDs (F1 / S1 / S2 / S4) of env i at the last call; [] if it had no contact termination."""
        if i not in self._rows:
            return []
        if i not in self._ids:
            if self.model is None:
                import omni.usd
                from ..standing_ee.kinematics import UsdArmKinematics
                self.model = UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
            from ..standing_ee.geometry import transform
            r, names = self._rows[i], env.scene["robot"].body_names
            pos, quat = self._pos[r].cpu().numpy(), self._quat[r].cpu().numpy()
            frames = {name: transform(pos[j], quat[j]) for j, name in enumerate(names) if name in self.model.bounds}
            terrain = {row["pair"][0] for row in self.model.clearance_report(frames, TR.TERRAIN_BAND_M)
                       if row["pair"][1] == "ground" and row["classification"] != "broad_phase_false_positive"}
            arm_self = any(b in self.arm and b not in terrain for b in self.reasons[i])
            # S4 (body self-contact, D39): trunk/leg body in contact, not with the terrain, and no arm body in
            # self-contact at this step (otherwise the trunk/leg side of an arm contact is attributed to S2).
            self._ids[i] = sorted({("S1" if b in terrain else "S2") if b in self.arm else
                                   ("F1" if b in terrain else ("S2" if arm_self else "S4")) for b in self.reasons[i]})
        return self._ids[i]


TERM_ID = dict(fall_tilt="F2", fall_height="F3", ground_clearance="S3", walk_away="D", invalid_state="invalid")
ID_ORDER = ("invalid", "F1", "F2", "F3", "S1", "S2", "S3", "S4", "D")


def termination_ids(env, i):
    """All termination IDs of env i at the last termination compute (time-out excluded), in ID_ORDER."""
    tm, out = env.termination_manager, set()
    for name in tm.active_terms:
        if name == "time_out" or not bool(tm.get_term(name)[i]):
            continue
        func = tm.get_term_cfg(name).func
        out |= set(func.classify(env, i)) if isinstance(func, ClassifiedContact) else {TERM_ID[name]}
    return [k for k in ID_ORDER if k in out]
