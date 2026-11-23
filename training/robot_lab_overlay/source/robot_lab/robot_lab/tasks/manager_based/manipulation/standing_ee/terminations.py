"""Shared termination set for T1, T2 and T3 (T3 protocol v0.4 §4.3, D6, D20, D21, D26).

| ID | Condition | Class |
| F1 | net contact > 1 N on a non-wheel trunk/leg body (terrain or self; terrain decided by geometry, A17) | fall |
| F2 | tilt > 1.0 rad for >= 5 consecutive control steps | fall |
| F3 | base height above terrain < 0.274 m for >= 5 consecutive control steps | fall |
| S1 | net contact > 1 N on an arm/gripper body, classified as terrain by geometry | safety |
| S2 | net contact > 1 N on an arm/gripper body otherwise (self-contact incl. arm-arm) | safety |
| S3 | any non-wheel collider < 5 mm above the ground (T2 GroundClearance) | safety |
| D  | standing drift: command zero and base x/y > 0.30 m from the latched position | drift |

A17 (2026-11-23) showed that a ground-filtered multi-body contact sensor reports nothing, so terrain versus
self-contact is classified by geometry: a body in contact counts as terrain contact if its lowest collider-AABB
corner is within TERRAIN_BAND_M of the terrain. The split only changes the reported reason (D21).

T1 uses these rules inside StandingFailure (reasons fall_tilt, fall_height, standing_drift); T2 uses the
manager terms below (term names fall_tilt, fall_height, walk_away are kept for the T2 evaluation tools).
"""

import torch

from isaaclab.managers import ManagerTermBase

FALL_TILT_RAD = 1.0
FALL_HEIGHT_M = 0.274          # min(0.5 h_nom, h_T2_min - 0.05), h_T2_min = 0.324 m (D26)
DEBOUNCE_STEPS = 5
CONTACT_N = 1.0
DRIFT_M = 0.30
GROUND_CLEARANCE_M = 0.005
TERRAIN_BAND_M = 0.01
CLASS = {"F1": "fall", "F2": "fall", "F3": "fall", "S1": "safety", "S2": "safety", "S3": "safety", "D": "drift"}


def debounce(counter, condition, steps=DEBOUNCE_STEPS):
    """Consecutive-step counter update (torch or numpy). Returns (new counter, triggered)."""
    counter = (counter + 1) * condition
    return counter, counter >= steps


def drift_update(latch, latched, standing, pos_xy, limit=DRIFT_M):
    """Standing-drift rule. latch (N,2) and latched (N,) bool are updated in place; returns triggered (N,).
    The latch is (re)set at the first standing step after a moving step or reset; moving clears it."""
    new = standing & ~latched
    latch[new] = pos_xy[new]
    latched[:] = standing
    return standing & ((pos_xy - latch).norm(dim=-1) > limit)


class _Debounced(ManagerTermBase):
    """Base for a condition that must hold for `steps` consecutive control steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self.count[slice(None) if env_ids is None else env_ids] = 0

    def _update(self, condition, steps):
        self.count, hit = debounce(self.count, condition.long(), steps)
        return hit


class DebouncedTilt(_Debounced):
    """F2: angle between base z and world z > limit for >= steps control steps."""

    def __call__(self, env, limit_rad=FALL_TILT_RAD, steps=DEBOUNCE_STEPS):
        g = env.scene["robot"].data.projected_gravity_b
        return self._update(torch.acos((-g[:, 2]).clamp(-1, 1)) > limit_rad, steps)


class DebouncedHeight(_Debounced):
    """F3: base height above the (flat) terrain at the env origin < limit for >= steps control steps."""

    def __call__(self, env, limit_m=FALL_HEIGHT_M, steps=DEBOUNCE_STEPS):
        h = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return self._update(h < limit_m, steps)


class StandingDrift(ManagerTermBase):
    """D: base x/y > limit from the position latched at the first standing step. T1/T2 have no velocity
    command, so every step is standing and the latch is the position at the first step after reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.latch = torch.zeros(env.num_envs, 2, device=env.device)
        self.latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.latched[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, limit_m=DRIFT_M):
        standing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        return drift_update(self.latch, self.latched, standing, env.scene["robot"].data.root_pos_w[:, :2], limit_m)
