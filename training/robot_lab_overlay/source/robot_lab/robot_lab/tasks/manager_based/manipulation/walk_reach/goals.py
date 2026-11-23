"""Training EE target generator over the goal pool (protocol v0.3 §6.2-6.3), torch, per env. No Isaac Lab import.

Per env: the active segment g_start -> g_goal (joint space, goal = pool node), its duration T_traj (0 = step)
and hold time. Each new goal: mode ~ MODE_WEIGHTS, a random stored successor of the current node with that
mode (offline-checked segment), then the online velocity-dependent check 3 of §6.3 if the base command is
non-zero. Up to MAX_TRIES draws; if none passes, the previous goal is kept and the draw is repeated next step.

Rules added in v0.3 (§6.3):
  * a step transition (T_traj = 0) sends only its goal, so check 3 applies to the goal only;
  * if the current target itself fails check 3 (command became non-zero while at a low target), the next
    transition is a forced step to a successor of the current node;
  * when the command changes from zero to non-zero, the rest of the active segment is checked again;
    if it fails, a forced step to a successor of the segment's start node is drawn at once (that pool
    segment is clearance-checked; only the step's goal is sent).
"""

import torch

from . import protocol as P
from .kinematics import ArmFK, interpolate


class GoalPool:
    """Pool arrays (pool.build) on a device."""

    def __init__(self, arrays, device="cpu"):
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        self.g = torch.cat([t(arrays["q_arm"]), t(arrays["b"])], dim=1)       # (N, 9)
        self.mode = t(arrays["mode"], torch.long)                              # 0 home, 1 M1, 2 M2, 3 M3
        self.successors = t(arrays["successors"], torch.long)                  # (N, K), -1 = none
        self.succ_mode = torch.where(self.successors >= 0, self.mode[self.successors.clamp(min=0)],
                                     torch.full_like(self.successors, -1))
        self.home = 0
        self.device = device


def moving_ok_positions(p_T, wheels_xy, h_knee=P.H_KNEE_M):
    """§6.3 check 3 for TCP positions in F_T (N, 3) on flat terrain: True where valid while moving."""
    z_world = p_T[:, 2] + P.H_T_M
    near = (p_T[:, None, :2] - wheels_xy[None]).norm(dim=-1).min(dim=1).values < P.WHEEL_EXCLUSION_M
    return (z_world >= P.Z_MIN_MOVING_M) & ~((z_world < h_knee) & near)


def moving_reason(fk, g, wheels_xy, h_knee=P.H_KNEE_M):
    """§6.3 check 3 for goal points g (..., 9): True where valid while the base command is non-zero."""
    p, _ = fk.tcp_pose(g.reshape(-1, 9)[:, :6], g.reshape(-1, 9)[:, 6:])
    return moving_ok_positions(p, wheels_xy, h_knee).reshape(g.shape[:-1])


class GoalSampler:
    def __init__(self, pool, num_envs, device="cpu", generator=None, fk=None):
        self.pool, self.n, self.device, self.gen = pool, num_envs, device, generator
        self.fk = fk or ArmFK(device)
        self.wheels_xy = torch.tensor(P.WHEEL_CONTACT_T_XY, dtype=torch.float32, device=device)
        self.weights = torch.tensor(P.MODE_WEIGHTS, device=device)
        z = lambda dt=torch.float32: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
        self.start_node, self.goal_node = z(torch.long), z(torch.long)
        self.g_start = pool.g[self.goal_node].clone()
        self.g_goal = self.g_start.clone()
        self.t, self.T_traj, self.hold = z(), z(), z()
        self.moving = z(torch.bool)
        self.pending = z(torch.bool)          # keep the goal, retry the draw next step
        self.draw_from = z(torch.long)        # node whose stored successors the next draw uses
        self.draw_step = z(torch.bool)        # next transition is a forced step
        self.n_failed_draws = 0
        s = torch.linspace(0.0, 1.0, P.SEGMENT_SAMPLES, device=device)
        self.s_samples = s

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def reset(self, env_ids):
        """Episode start at the home node (nominal state); a new goal is drawn at the first update."""
        self.start_node[env_ids] = self.pool.home
        self.goal_node[env_ids] = self.pool.home
        self.g_start[env_ids] = self.pool.g[self.pool.home]
        self.g_goal[env_ids] = self.pool.g[self.pool.home]
        self.t[env_ids] = 0.0
        self.T_traj[env_ids] = 0.0
        self.hold[env_ids] = 0.0
        self.pending[env_ids] = True
        self.draw_from[env_ids] = self.pool.home
        self.draw_step[env_ids] = False

    def s(self):
        return torch.where(self.T_traj > 0, self.t / self.T_traj.clamp(min=1e-9), torch.ones_like(self.t)).clamp(0, 1)

    def current_g(self):
        return interpolate(self.g_start, self.g_goal, self.s())

    def target(self):
        """Current TCP target in F_T: position (N, 3), quaternion wxyz (N, 4)."""
        g = self.current_g()
        return self.fk.tcp_pose(g[:, :6], g[:, 6:])

    def _segment_ok(self, ids, g_from, g_to, step, moving):
        """Online check 3 for candidate segments; always True while not moving."""
        ok = torch.ones(len(ids), dtype=torch.bool, device=self.device)
        if not bool(moving.any()):
            return ok
        pts = interpolate(g_from[:, None, :].expand(-1, P.SEGMENT_SAMPLES, -1),
                          g_to[:, None, :].expand(-1, P.SEGMENT_SAMPLES, -1),
                          self.s_samples[None].expand(len(ids), -1))
        valid = moving_reason(self.fk, pts, self.wheels_xy)              # (n, S)
        seg_ok = torch.where(step, valid[:, -1], valid.all(dim=1))
        return torch.where(moving, seg_ok, ok)

    def draw(self, env_ids):
        """New goals for env_ids (long): successors of draw_from, segment start = current target.
        Envs without a valid draw keep their goal and stay pending (draw_from/draw_step retained)."""
        if len(env_ids) == 0:
            return
        node = self.draw_from[env_ids]
        g_from = self.current_g()[env_ids]
        moving = self.moving[env_ids]
        force_step = self.draw_step[env_ids]
        # If the current target fails check 3 while moving, only a step can leave it.
        cur_bad = moving & ~moving_reason(self.fk, g_from, self.wheels_xy)
        force_step = force_step | cur_bad
        todo = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        for _ in range(P.MAX_TRIES):
            idx = todo.nonzero(as_tuple=True)[0]
            if not len(idx):
                break
            mode = 1 + torch.multinomial(self.weights, len(idx), replacement=True, generator=self.gen)
            succ = self.pool.successors[node[idx]]                       # (n, K)
            match = (self.pool.succ_mode[node[idx]] == mode[:, None]) & (succ >= 0)
            has = match.any(dim=1)
            pick = torch.multinomial(match.float() + (~has[:, None]).float(), 1, generator=self.gen)[:, 0]
            cand = succ[torch.arange(len(idx), device=self.device), pick]
            step = force_step[idx] | (self._rand(len(idx)) < P.STEP_FRACTION)
            ok = has & self._segment_ok(idx, g_from[idx], self.pool.g[cand.clamp(min=0)], step, moving[idx])
            acc = idx[ok]
            e = env_ids[acc]
            self.start_node[e] = node[acc]
            self.goal_node[e] = cand[ok]
            self.g_start[e] = g_from[acc]
            self.g_goal[e] = self.pool.g[cand[ok]]
            lo, hi = P.T_TRAJ_S
            self.T_traj[e] = torch.where(step[ok], torch.zeros_like(cand[ok], dtype=torch.float32),
                                         lo + (hi - lo) * self._rand(len(acc)))
            lo, hi = P.HOLD_S
            self.hold[e] = lo + (hi - lo) * self._rand(len(acc))
            self.t[e] = 0.0
            self.pending[e] = False
            todo[acc] = False
        failed = env_ids[todo]
        self.n_failed_draws += int(len(failed))
        if len(failed):   # keep the previous goal: hold the current target, retry next step
            g = self.current_g()[failed]
            self.g_start[failed], self.g_goal[failed] = g, g
            self.T_traj[failed] = 0.0
            self.pending[failed] = True

    def update(self, dt, moving):
        """Advance one control step. moving (N,) bool: base velocity command non-zero."""
        moving = moving.to(self.device)
        switched = moving & ~self.moving
        self.moving = moving.clone()
        self.t += dt
        if bool(switched.any()):
            ids = switched.nonzero(as_tuple=True)[0]
            rest = torch.linspace(0.0, 1.0, P.SEGMENT_SAMPLES, device=self.device)
            s0 = self.s()[ids]
            ss = s0[:, None] + (1 - s0[:, None]) * rest[None]
            pts = interpolate(self.g_start[ids, None].expand(-1, P.SEGMENT_SAMPLES, -1),
                              self.g_goal[ids, None].expand(-1, P.SEGMENT_SAMPLES, -1), ss)
            bad = ids[~moving_reason(self.fk, pts, self.wheels_xy).all(dim=1)]
            self.draw_from[bad] = self.start_node[bad]
            self.draw_step[bad] = True
            self.pending[bad] = True
        finished = (self.t >= self.T_traj + self.hold) & ~self.pending
        self.draw_from[finished] = self.goal_node[finished]
        self.draw_step[finished] = False
        self.draw((finished | self.pending).nonzero(as_tuple=True)[0])
