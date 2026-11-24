"""Isaac Lab terms for T2 (protocol v0.4 §4, §6, §7). Vectorised (torch) for training.

Command `tcp_target`: one TCP target per episode, sampled in the ground frame G at reset from the frozen
reachability map (full protocol grid, reachability version checked) and frozen in W. The command value
(policy observation) is the target TCP pose in the base frame B: position + quaternion wxyz (T1 encoding).
"""

import hashlib
import os

import numpy as np
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, subtract_frame_transforms

from . import protocol as P
from .sampler import ReachableSet
from .torch_targets import TorchReachableSet, g_to_world, yaw_from_quat

COMMAND = "tcp_target"


class ReachTargetCommand(CommandTerm):
    """T2 TCP target (Mode A). Exposes G (origin_w, yaw), the target in G and W, and band_id per env."""

    cfg: "ReachTargetCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.orientation_mode != "A":
            raise NotImplementedError("T2 implements orientation Mode A only (protocol 4.2); Mode B is not built")
        path = cfg.reachability_path or os.environ.get("T2_REACHABILITY_NPZ", "")
        if not path:
            raise ValueError("Set ReachTargetCommandCfg.reachability_path or T2_REACHABILITY_NPZ to the frozen "
                             f"full-grid reachability npz (version {P.REACHABILITY['version']})")
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if cfg.reachability_sha256 and sha != cfg.reachability_sha256:
            raise ValueError(f"Reachability file hash {sha} != configured {cfg.reachability_sha256}")
        self.reachability_path, self.reachability_sha256 = path, sha
        self.reachable = TorchReachableSet(ReachableSet.load(path, require_full_grid=True), self.device)
        self.band_weights = tuple(cfg.band_weights)

        self.robot = env.scene[cfg.asset_name]
        self.link06 = self.robot.find_bodies([P.TCP_PARENT])[0][0]
        n, dev = self.num_envs, self.device
        self.tcp_offset = torch.tensor(P.TCP_OFFSET_M, device=dev).repeat(n, 1)
        self.origin_w = torch.zeros(n, 3, device=dev)        # G origin in W (ground below base at reset)
        self.yaw = torch.zeros(n, device=dev)                # G yaw in W
        self.band_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.position_G = torch.zeros(n, 3, device=dev)
        self.quat_G = torch.zeros(n, 4, device=dev)
        self.position_w = torch.zeros(n, 3, device=dev)
        self.quat_w = torch.zeros(n, 4, device=dev)
        self.quat_w[:, 0] = 1.0
        self._command = torch.zeros(n, 7, device=dev)
        self.metrics["position_error"] = torch.zeros(n, device=dev)

    @property
    def command(self):
        return self._command

    def tcp_pose_w(self):
        d = self.robot.data
        q = d.body_quat_w[:, self.link06]
        return d.body_pos_w[:, self.link06] + quat_apply(q, self.tcp_offset), q

    def _set_ground_frame(self, env_ids):
        d = self.robot.data
        self.origin_w[env_ids, :2] = d.root_pos_w[env_ids, :2]
        self.origin_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2]   # flat ground at the env origin
        self.yaw[env_ids] = yaw_from_quat(d.root_quat_w[env_ids])

    def assign(self, env_ids, position_G, quat_G_wxyz, band):
        """Evaluation hook: replace the sampled targets of env_ids by given G-frame targets (G kept)."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pos = torch.as_tensor(position_G, dtype=torch.float32, device=self.device)
        quat = torch.as_tensor(quat_G_wxyz, dtype=torch.float32, device=self.device)
        self.position_G[env_ids], self.quat_G[env_ids] = pos, quat
        self.band_id[env_ids] = torch.as_tensor(band, dtype=torch.long, device=self.device)
        self.position_w[env_ids], self.quat_w[env_ids] = g_to_world(
            self.origin_w[env_ids], self.yaw[env_ids], pos, matrix_from_quat(quat))
        self._update_command()

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._set_ground_frame(env_ids)
        t = self.reachable.sample(len(env_ids), self.band_weights)
        self.band_id[env_ids] = t["band"]
        self.position_G[env_ids], self.quat_G[env_ids] = t["position_G"], t["quat_G"]
        self.position_w[env_ids], self.quat_w[env_ids] = g_to_world(
            self.origin_w[env_ids], self.yaw[env_ids], t["position_G"], t["rot_G"])

    def _update_command(self):
        d = self.robot.data
        p, q = subtract_frame_transforms(d.root_pos_w, d.root_quat_w, self.position_w, self.quat_w)
        self._command[:] = torch.cat([p, q], dim=1)

    def _update_metrics(self):
        # Last value before reset = final TCP position error (CommandTerm.reset logs the mean over env_ids).
        self.metrics["position_error"][:] = (self.tcp_pose_w()[0] - self.position_w).norm(dim=1)


@configclass
class ReachTargetCommandCfg(CommandTermCfg):
    class_type: type = ReachTargetCommand
    asset_name: str = "robot"
    resampling_time_range: tuple = (1e9, 1e9)       # one target per episode (resampled at reset only)
    reachability_path: str = ""                       # empty: environment variable T2_REACHABILITY_NPZ
    reachability_sha256: str | None = None            # optional freeze: refuse any other file
    band_weights: tuple = (1.0, 1.0, 1.0)             # curriculum hook (§9)
    orientation_mode: str = P.ORIENTATION_MODE


# ---------------------------------------------------------------- observations (policy, frame B)

# tcp_pose_body, tcp_pose_error_body: shared suite observation (standing_ee.observations, T3 D40).


def band_one_hot(env):
    """Diagnostic posture hint (flag obs_target_height_band, default off)."""
    band = env.command_manager.get_term(COMMAND).band_id
    return torch.nn.functional.one_hot(band, len(P.BANDS)).float()


# ---------------------------------------------------------------- terminations (§7)

def invalid_state(env):
    d = env.scene["robot"].data
    finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for v in (d.root_state_w, d.joint_pos, d.joint_vel, d.body_state_w):
        finite &= torch.isfinite(v.reshape(env.num_envs, -1)).all(dim=1)
    unit = ((d.body_quat_w.norm(dim=-1) - 1).abs() <= 1e-3).all(dim=1)
    return ~(finite & unit)


class IllegalContact(ManagerTermBase):
    """mdp.illegal_contact (net contact force > threshold on the selected bodies), keeping the names of the
    bodies that triggered it per env (`reasons`) for failure analysis."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reasons = [[] for _ in range(env.num_envs)]
        self._with_reasons = set()

    def __call__(self, env, threshold, sensor_cfg):
        sensor = env.scene.sensors[sensor_cfg.name]
        force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > threshold
        out = force.any(dim=1)
        for i in self._with_reasons:
            self.reasons[i] = []
        self._with_reasons.clear()
        hit = out.nonzero(as_tuple=True)[0].tolist()
        if hit:
            names = [sensor.body_names[j] for j in (sensor_cfg.body_ids if not isinstance(sensor_cfg.body_ids, slice)
                                                    else range(len(sensor.body_names)))]
            rows = force[hit].cpu().numpy()
            for i, row in zip(hit, rows):
                self.reasons[i] = [n for n, f in zip(names, row) if f]
                self._with_reasons.add(i)
        return out


class GroundClearance(ManagerTermBase):
    """D10: any non-wheel collider < margin above the ground terminates.

    Broad phase on all envs in torch: lowest corner of each collider's local AABB in W (the same gate T1
    uses before measuring). Exact narrow phase (T1 UsdArmKinematics.clearance_report, hpp-fcl, CPU) only
    for flagged envs; rows classified broad_phase_false_positive do not terminate (as T1).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        import omni.usd
        from ..standing_ee.kinematics import UsdArmKinematics, corners
        self.model = UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
        robot = env.scene["robot"]
        body, pts = [], []
        for name, bounds in self.model.collider_bounds.items():
            if name.endswith("_foot") or name not in robot.body_names:
                continue
            for b in bounds:
                c = corners(np.asarray(b))
                pts.append(c)
                body += [robot.body_names.index(name)] * len(c)
        self.body = torch.tensor(body, device=env.device, dtype=torch.long)
        self.points = torch.as_tensor(np.concatenate(pts), dtype=torch.float32, device=env.device)
        self.reasons = [[] for _ in range(env.num_envs)]
        # Diagnostics (smoke test): set profile = True to time both phases (adds CUDA syncs).
        self.profile = False
        self.stats = dict(calls=0, flagged_envs=0, flagged_max=0, narrow_checks=0, broad_s=0.0, narrow_s=0.0)
        self._with_reasons = set()   # envs whose reasons are non-empty (cleared lazily)

    def broad_phase(self, env, margin_m):
        """Envs whose lowest collider-AABB corner (non-wheel bodies) is below margin_m above the ground."""
        d = env.scene["robot"].data
        pos, quat = d.body_pos_w[:, self.body], d.body_quat_w[:, self.body]           # (N, K, 3/4)
        z = (pos + quat_apply(quat.reshape(-1, 4), self.points.repeat(env.num_envs, 1)).reshape(pos.shape))[..., 2]
        return (z - env.scene.env_origins[:, 2:3]).min(dim=1).values < margin_m

    def __call__(self, env, margin_m=P.CLEARANCE_M):
        import time
        from ..standing_ee.geometry import transform
        if self.profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        flagged = self.broad_phase(env, margin_m).nonzero(as_tuple=True)[0].tolist()
        if self.profile:
            t1 = time.perf_counter()
        for i in self._with_reasons:
            self.reasons[i] = []
        self._with_reasons.clear()
        out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if not flagged:
            if self.profile:
                self._count(0, t0, t1)
            return out
        d = env.scene["robot"].data
        names = env.scene["robot"].body_names
        for i in flagged:
            frames = {}
            for j, name in enumerate(names):
                if name in self.model.bounds:
                    p = d.body_pos_w[i, j].cpu().numpy() - np.r_[0.0, 0.0, env.scene.env_origins[i, 2].item()]
                    frames[name] = transform(p, d.body_quat_w[i, j].cpu().numpy())
            rows = [r for r in self.model.clearance_report(frames, margin_m)
                    if r["pair"][1] == "ground" and r["classification"] != "broad_phase_false_positive"]
            self.reasons[i] = [r["classification"] + ":" + r["pair"][0] for r in rows]
            if rows:
                self._with_reasons.add(i)
                out[i] = True
        if self.profile:
            self._count(len(flagged), t0, t1)
        return out

    def _count(self, flagged, t0, t1):
        import time
        s = self.stats
        s["calls"] += 1
        s["flagged_envs"] += flagged
        s["flagged_max"] = max(s["flagged_max"], flagged)
        s["narrow_checks"] += flagged
        s["broad_s"] += t1 - t0
        s["narrow_s"] += time.perf_counter() - t1
