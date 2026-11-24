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
    """Pool arrays (pool.build) on a device. modes: goal modes the sampler may draw (training preset, D42);
    stored successors of other modes are removed before the dead-end mask (D34 then applies per preset); edges to
    home (mode 0) are kept as before (the frozen pools store none)."""

    def __init__(self, arrays, device="cpu", modes=(1, 2, 3)):
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        self.g = torch.cat([t(arrays["q_arm"]), t(arrays["b"])], dim=1)       # (N, 9)
        self.mode = t(arrays["mode"], torch.long)                              # 0 home, 1 M1, 2 M2, 3 M3
        self.successors = t(arrays["successors"], torch.long)                  # (N, K), -1 = none
        allowed = torch.zeros(4, dtype=torch.bool, device=device)
        allowed[[0, *modes]] = True
        keep = (self.successors >= 0) & allowed[self.mode[self.successors.clamp(min=0)]]
        self.successors = torch.where(keep, self.successors, torch.full_like(self.successors, -1))
        # Dead-end mask (protocol v0.5 D34): a goal without a stored successor is never drawn. Repeated until no
        # goal is left whose successors are all masked; the pool file is unchanged.
        self.dead_end = torch.zeros(len(self.mode), dtype=torch.bool, device=device)
        while True:
            valid = (self.successors >= 0) & ~self.dead_end[self.successors.clamp(min=0)]
            dead = ~valid.any(dim=1)
            if bool((dead == self.dead_end).all()):
                break
            self.dead_end = dead
        self.successors = torch.where(valid, self.successors, torch.full_like(self.successors, -1))
        assert not bool(self.dead_end[0]), "home node is a dead end"
        self.succ_mode = torch.where(self.successors >= 0, self.mode[self.successors.clamp(min=0)],
                                     torch.full_like(self.successors, -1))
        self.home = 0
        self.device = device


def moving_ok_positions(p_T, wheels_xy, h_knee=P.H_KNEE_M, parts=False):
    """§6.3 check 3 for TCP positions in F_T (N, 3) on flat terrain: True where valid while moving.
    parts=True also returns (height_ok, wheel_ok)."""
    z_world = p_T[:, 2] + P.H_T_M
    near = (p_T[:, None, :2] - wheels_xy[None]).norm(dim=-1).min(dim=1).values < P.WHEEL_EXCLUSION_M
    z_ok, wheel_ok = z_world >= P.Z_MIN_MOVING_M, ~((z_world < h_knee) & near)
    return (z_ok & wheel_ok, z_ok, wheel_ok) if parts else z_ok & wheel_ok


def moving_reason(fk, g, wheels_xy, h_knee=P.H_KNEE_M, parts=False):
    """§6.3 check 3 for goal points g (..., 9): True where valid while the base command is non-zero."""
    p, _ = fk.tcp_pose(g.reshape(-1, 9)[:, :6], g.reshape(-1, 9)[:, 6:])
    out = moving_ok_positions(p, wheels_xy, h_knee, parts)
    return tuple(o.reshape(g.shape[:-1]) for o in out) if parts else out.reshape(g.shape[:-1])


COUNTERS = ("transitions", "steps", "rule_step_goal_only", "rule_forced_step_current_bad", "rule_recheck_forced_step",
            "kept_goal", "m1_candidates_moving", "m1_rejected_min_height_moving", "m1_rejected_wheel_moving",
            "mode_M1", "mode_M2", "mode_M3", "fallback_step")


class GoalSampler:
    def __init__(self, pool, num_envs, device="cpu", generator=None, fk=None, mode_weights=P.MODE_WEIGHTS,
                 fallback_step=False):
        self.pool, self.n, self.device, self.gen = pool, num_envs, device, generator
        self.fk = fk or ArmFK(device)
        self.wheels_xy = torch.tensor(P.WHEEL_CONTACT_T_XY, dtype=torch.float32, device=device)
        self.weights = torch.tensor(mode_weights, dtype=torch.float32, device=device)   # M1, M2, M3 (D42)
        # D43 (train-t1): if the current goal has no stored successor of the drawn mode, step (T_traj = 0) to a
        # random goal of that mode instead (a step needs no stored segment). Off for the other presets.
        self.fallback = fallback_step
        self.mode_goals = {m: ((pool.mode == m) & ~pool.dead_end).nonzero(as_tuple=True)[0] for m in (1, 2, 3)}
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
        self.counts = dict.fromkeys(COUNTERS, 0)   # stranding-rule logging (v0.4): see rates()
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

    def rates(self):
        """Counters per 1000 accepted transitions (and the raw counts)."""
        n = max(1, self.counts["transitions"])
        return dict(counts=dict(self.counts), per_1000_transitions={k: 1000.0 * v / n for k, v in self.counts.items()
                                                                     if k != "transitions"})

    def _segment_ok(self, ids, g_from, g_to, step, moving, cand_mode=None):
        """Online check 3 for candidate segments; always True while not moving.
        Returns (ok, full_ok): full_ok = all 20 points pass (what an interpolated segment needs)."""
        ok = torch.ones(len(ids), dtype=torch.bool, device=self.device)
        if not bool(moving.any()):
            return ok, ok
        pts = interpolate(g_from[:, None, :].expand(-1, P.SEGMENT_SAMPLES, -1),
                          g_to[:, None, :].expand(-1, P.SEGMENT_SAMPLES, -1),
                          self.s_samples[None].expand(len(ids), -1))
        valid, z_ok, wheel_ok = moving_reason(self.fk, pts, self.wheels_xy, parts=True)   # (n, S)
        sent = torch.ones_like(valid)
        sent[step, :-1] = False
        seg_ok = (valid | ~sent).all(dim=1)
        if cand_mode is not None:
            m1 = moving & (cand_mode == 1)
            self.counts["m1_candidates_moving"] += int(m1.sum())
            self.counts["m1_rejected_min_height_moving"] += int((m1 & ~(z_ok | ~sent).all(dim=1)).sum())
            self.counts["m1_rejected_wheel_moving"] += int((m1 & ~(wheel_ok | ~sent).all(dim=1)).sum())
        return torch.where(moving, seg_ok, ok), torch.where(moving, valid.all(dim=1), ok)

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
            fb = torch.zeros_like(has)
            if self.fallback and not bool(has.all()):
                fb = ~has
                for m in (1, 2, 3):
                    rows = fb & (mode == m)
                    goals = self.mode_goals[m]
                    if bool(rows.any()) and len(goals):
                        cand[rows] = goals[torch.randint(len(goals), (int(rows.sum()),), device=self.device,
                                                         generator=self.gen)]
                fb = fb & (cand >= 0)
                has = has | fb
            step = force_step[idx] | (self._rand(len(idx)) < P.STEP_FRACTION) | fb
            seg_ok, full_ok = self._segment_ok(idx, g_from[idx], self.pool.g[cand.clamp(min=0)], step, moving[idx],
                                               torch.where(has, self.pool.mode[cand.clamp(min=0)], -1))
            ok = has & seg_ok
            acc = idx[ok]
            c = self.counts
            c["transitions"] += int(ok.sum())
            c["steps"] += int((ok & step).sum())
            c["rule_step_goal_only"] += int((ok & step & moving[idx] & ~full_ok).sum())
            c["rule_forced_step_current_bad"] += int((ok & cur_bad[idx]).sum())
            c["fallback_step"] += int((ok & fb).sum())
            for m in (1, 2, 3):
                c[f"mode_M{m}"] += int((ok & (self.pool.mode[cand.clamp(min=0)] == m)).sum())
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
        self.counts["kept_goal"] += int(len(failed))
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
            self.counts["rule_recheck_forced_step"] += int(len(bad))
            self.draw_from[bad] = self.start_node[bad]
            self.draw_step[bad] = True
            self.pending[bad] = True
        finished = (self.t >= self.T_traj + self.hold) & ~self.pending
        self.draw_from[finished] = self.goal_node[finished]
        self.draw_step[finished] = False
        self.draw((finished | self.pending).nonzero(as_tuple=True)[0])
