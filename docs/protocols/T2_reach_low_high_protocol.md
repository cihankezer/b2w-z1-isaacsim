# T2 — Reach Low and High Targets While Standing
## Simulation Environment Specification (Isaac Lab, Unitree B2W + Z1)

Status: **v1.0, frozen 2026-11-13** (acceptance tests A1–A8 passed, Section 11; A3b dropped as out of scope) · 2026-11-13 · Work package 1 · Scope: environment only, no policy training yet
Depends on: T1 environment (`docs/protocols/TASK_T1.md`, protocol 1.0.0; 6D end-effector tracking while standing)
Code: `training/robot_lab_overlay/.../manipulation/reach_height/` (constants in `protocol.py`, `VERSION = "1.1.0"`; 1.1.0 changes the interface only, results identical to 1.0.0, see the 2026-11-23 entry in Section 12)

Frozen as v1.0 (2026-11-13). Frozen artefacts:
- Reachability map `t2-reachability-3`, sha256 `22b9e665baa11bced9832107ca17c154ba57751fb55f54c3dbf67d9e1b865404` (`env_cfg.REACHABILITY_SHA256`).
- Evaluation target set: seed 2026, 2 000 targets per band, digest `6260df589d63db4d8633959e2244e9ec0320ab4c7a716bab3279b8a7a4f5937e` (`protocol.EVAL_SEED`, `protocol.EVAL_SET_SHA256`; `eval_t2.py` refuses a regenerated default set with another digest).
- Image `isaac-lab:2.3.2` (`sha256:b52a5210…`).

Changes from v0.4 to v1.0: leg/body self-clearance in the map (`t2-reachability-3`, per-class margins); A3 split, then A3a redefined as static feasibility and A3b dropped; A7 tolerance fixed; evaluation set pinned. Any change to the map, the evaluation set, the terminations or the success criterion now needs a new protocol version.

Changes from v0.3 to v0.4 (2026-11-13): A1 judged on the annulus fraction plus a largest connected reachable region holding ≥ 90 % of the band's reachable voxels, with absolute voxel counts and the share needing Δh < −0.20 m reported; Δh ∈ [−0.33, 0] m (physical crouch limit) for training and reachability, KAIST −0.20 m kept as a reference; the per-band orientation switch replaced by a continuous approach-angle profile (4.2). v0.3 results stay in Section 12 for provenance.

Changes from v0.2 to v0.3: nominal height from the T1 asset (0.604 m, not 0.75 m); end-effector = TCP at a fixed offset from `link06`; 22-dim action space and closed gripper as in T1; radial approach direction; ground frame `G` defined; deterministic evaluation reset; terminations merged with T1; tilt formula sign fixed; joint-limit metric on soft limits; reachability run twice (KAIST and physical-crouch height range); KAIST provenance corrected (Section 2, 13).

---

## 1. Purpose

T2 tests whether a whole-body controller can reach end-effector targets over a large vertical range — from near the ground up to 1.1 m — by using the legs (lowering, pitching the base) and not only the arm. T1 covers targets inside the arm workspace at nominal standing height. T2 covers targets that need base posture changes.

Metrics: success rate, final end-effector error, base height, base tilt.

This document specifies the environment so that it can be implemented, validated and frozen before any training. Every numeric choice is either adapted from a reference paper (Section 2) or marked as a thesis decision.

---

## 2. References and what is taken from each

| Item | Value in T2 | Source | Note |
|---|---|---|---|
| Vertical range | 0.0 – 1.1 m TCP height (bands, 4.1) | **Adapted** from KAIST dynamic pick-and-place (Jung et al., RA-L 2026), Table I and Sec. IV-C | KAIST values are **table heights** (0.0 – 1.3 m training, 0 m ↔ 1.1 m in the real "Large Height Gap" scenario) with a 6 – 10 cm object standing on the table. The comparable gripper height is therefore about 0.03 – 1.15 m, not exactly 0.0 – 1.1 m. |
| Position success threshold | ≤ 5 cm, 3D TCP error | **Adapted** from KAIST, Sec. IV-B | KAIST: the *placed object* must be upright and within 5 cm of the place-table centre (horizontal), and the arm must have retreated ≥ 10 cm. T2 applies the number to the 3D TCP position error. Also the WP1 T1 target. |
| Episode horizon | 10 s | KAIST, Sec. IV-B | In KAIST the 10 s cover a full walk–grasp–transport–place–retreat sequence; for a standing reach it is generous. |
| Base height range | Δh ∈ [−0.33, 0] m (T2); KAIST −0.2 m as reference | KAIST, Sec. III-C (reference) | KAIST's −0.2 m is the operative limit of its **low-level controller command** on a different robot (base below 0.55 m). T2 uses the B2W physical crouch limit −0.33 m (A1 run 2, Section 12) for training and reachability (thesis decision, v0.4); −0.2 m stays as a reporting reference band and a line in the reachability figure. |
| Base pitch range | \|ϕ\| ≤ 0.28 rad | KAIST, Sec. III-C | Same provenance as Δh. KAIST gives no sign convention; T2 uses positive = nose down (Section 3). |
| Payload during training | 0 – 2 kg at the end-effector | KAIST, Sec. III-B (low-level controller) | The KAIST high-level object mass is U(0.2, 2.3) kg (Table I). Z1 effective payload ≈ 2.3 kg. |
| Orientation accuracy reference | ≈ 0.09 rad (hardware) | LimX W1 wheeled-quadruped paper; thesis WP1 target 0.1 rad | Secondary metric in T2. |
| Evaluation scale | 5 seeds, large randomised episode set | KAIST, Sec. IV-B (5 seeds × 10 000 episodes sampled from the training distribution) | KAIST does not state whether the seeds are training or evaluation seeds; T2 interprets them as **policy training seeds** (D12). Scaled to Isaac Lab, Section 8. |
| Base-penalty form | `Vz² + 0.02·|ωx| + 0.02·|ωy|` | KAIST, Table II | Training reward hook only (Section 9). |
| Pitch-RMSE reference | 0.005 rad | KAIST, Fig. 3 | Measured while walking at 2 m/s with joint 3 oscillating at 1 rad / 0.8 Hz; a different test, order-of-magnitude reference only. |
| Height-band reporting | thesis decision | — | KAIST reports a single "large height gap" case (7/10 sim, 6/10 real); bands give more information. |
| Hold-time success criterion | thesis decision | — | KAIST has no hold; a reach task needs one to exclude transient hits. |

---

## 3. Robot and scene

- Robot: the T1 robot (`robots_usd/b2w_z1_gripper.usd`, `T1_ROBOT_CFG`): same articulation, actuator models, PD gains, PhysX self-collision and collider geometry as T1 (D7, D8).
- Terrain: flat rigid plane, friction 1.0 / 1.0, restitution 0 (T1 nominal).
- No obstacles, no table geometry. The target is a virtual pose. Reason: T2 measures posture-dependent reaching; contact with a table is a separate problem (T4/T5 or later). Ground contact of any non-wheel body terminates the episode (Section 7), so targets close to the floor are limited by gripper geometry (4.3).
- Base velocity command: fixed to zero `[vx, vy, ωz] = 0`. The robot stands; stepping to re-balance is allowed, walking away is terminated (Section 7).
- Reference frames:
  - World frame `W`: Isaac Lab world (per-environment origin on the ground).
  - Base frame `B`: robot `base_link`, x forward, z up.
  - Base-horizontal frame `B_h`: base frame with roll and pitch removed (yaw only).
  - **Ground frame `G` (new in v0.3):** fixed per episode; origin = base origin at reset projected onto the ground, axes = `B_h` at reset. Targets are defined and stored in `G` (z = height above ground) and frozen in `W`.
  - Pitch sign: positive pitch = nose down (rotation about +y with x forward, z up; Isaac Lab `euler_xyz_from_quat`).
- Nominal standing height **`h_nom` = 0.6039 m** (D1; verified in code: `standing_ee.protocol.NOMINAL_BASE_POSITION_XYZ`, the settled height of the T1 nominal state).
- Consequence: B2W's base is at a height similar to KAIST's robot (KAIST: below 0.55 m, since its 1.1 m table is "more than twice its base height"). At the T1 nominal pose the TCP is at **≈ 1.21 m** (verified in code), above the High band. High targets need little or no base motion; Mid targets below the base (z < 0.6 m) and Low targets need the arm to reach down past the body, and the lowest targets need base lowering and nose-down pitch.
- End-effector (D2): **TCP = `link06` frame translated by (0.171, 0, 0) m in `link06` coordinates**, i.e. 0.171 m along the tool axis (+x of `link06`). Provenance: `link06` → `gripperStator` is +0.051 m along that axis (verified in the asset), and the fetch pipeline's grasp point is a further +0.12 m (`fetch_env_cfg.GRASP_POINT`, x component). The TCP orientation equals the `link06` orientation. The gripper fingertips extend about 0.03 m beyond the TCP.
- Gripper: held closed by its own drive, not commanded (as T1; D8).

---

## 4. Task definition

### 4.1 Target sampling

Each episode samples one TCP target pose, in `G`.

| Parameter | Range | Note |
|---|---|---|
| Height band | one of {Low, Mid, High} | Bands sampled with equal probability for training (weights configurable, Section 9); evaluated separately |
| z (TCP height above ground) | per band | |
| Position | uniform within the band's reachable set (4.3) | Horizontal distance range `r_min`, `r_max` per band is read from the reachable set, not hand-set |
| Azimuth θ in `G` | within [−60°, +60°] around +x | Front sector; sides and rear excluded |
| Orientation | function of azimuth and height, see 4.2 | |

Height bands (thesis decision; full range 0.0 – 1.1 m adapted from the KAIST real-world protocol):

| Band | z range [m] | Expected posture (h_nom = 0.604 m, nominal TCP at 1.21 m) |
|---|---|---|
| Low | [0.00, 0.30) | base lowered (Δh < 0), nose-down pitch, steep approach (90° → 45° below horizontal) |
| Mid | [0.30, 0.80) | arm reaching down past the body front; base lowering for the lower part |
| High | [0.80, 1.10] | near nominal posture; mostly arm motion |

The v0.2 rule "Mid should contain the T1 workspace" is withdrawn (D5): T1 targets have the TCP above the base (z ≳ 0.6 m) and up to ≈ 1.3 m, which overlaps the upper Mid band, the High band and the region above it. Band boundaries are confirmed or adjusted once the reachability map (A1) is available; record any change in Section 12.

### 4.2 Target orientation

Two modes, selected in config. The benchmark uses Mode A (decided). Mode B stays available as an optional extra.

- Mode A — position + canonical approach direction (default, D6; profile v0.4): the canonical orientation depends on the target azimuth θ and the target height z, not on the band. Tool axis = +x of `link06`.
  - Base frame: tool axis horizontal and **radial** (pointing away from the `G` origin along θ), `link06` y-axis tangential, z-axis up. At θ = 0 this equals the T1 nominal ready-pose orientation (verified in code, < 0.1 rad).
  - This frame is pitched down about the tangential axis by the **approach angle α(z)** below horizontal: α = 90° (vertical, tool axis down) at z = 0, α = 45° at z = 0.30 m, α = 0 (horizontal) for z ≥ 0.60 m, linear in between (`APPROACH_PROFILE`, `approach_angle(z)`).
  - Reason (v0.4): the v0.3 per-band switch (vertical in Low, horizontal in Mid) left no reachable voxel at 0.30 ≤ z < 0.36–0.40 m; the continuous profile removes it (coarse check, Section 12).
  - Orientation error is logged but is not part of the success criterion.
- Mode B — full 6D: orientation sampled as in T1 (random rotation within a cone around the canonical orientation, cone half-angle configurable, default 30°). Success requires position and orientation thresholds.

Reason for Mode A default: KAIST's success criterion is position-based; T2's purpose is the vertical range, not orientation accuracy, which T1 already measures. Note: the reachable set (4.3) is computed with the canonical orientation (6D), while Mode A success is position-only, so the reachable set is conservative for Mode A.

### 4.3 Reachability filter (required before freezing the env)

Targets must be reachable with a feasible whole-body posture. Otherwise success rate measures the sampler, not the controller.

Procedure (offline, CPU, deterministic, run once per asset version; `reach_height/reachability.py`, `evaluation/reachability_t2.py`):

1. Grid: voxel centres at 2 cm resolution over x ∈ [0, 1.2] m, y ∈ [−0.8, 0.8] m, z ∈ [0, 1.2] m in `G`, restricted to centres with |θ| ≤ 60°. Band edges coincide with voxel faces.
2. Base postures (Δh, ϕ): Δh from 0 down to Δh_min in 0.02 m steps, ϕ ∈ [−0.28, 0.28] rad in 0.04 rad steps; base roll = 0, base yaw = yaw of `G`, **base x/y fixed at the nominal position** (no base translation; option (c) rejected, D4).
3. Legs: each wheel centre (foot-joint anchor) stays at its nominal world position; leg joints by IK within the 90 % soft limits. The posture must pass the clearance check below with the arm at its nominal pose.
4. Arm: 6D IK of `link06` for the TCP target with the Mode A canonical orientation α(z) (4.2), within the 90 % soft limits (damped least squares; 4 starts: nominal arm pose + 3 seeded random; tolerance 1 mm / 0.01 rad).
5. Clearance (D10; margins v0.4, 2026-11-13): T1 collider geometry, judged per pair class (`wholebody.clearance_violation`, `CLEARANCE_MARGINS_M`, recorded in the map metadata): every non-wheel collider ≥ 5 mm above the ground; arm–arm pairs (Z1 incl. its mount `link00`) ≥ 5 mm; **arm–body pairs ≥ 2 cm**; **body–body pairs ≥ 2 cm** (legs, wheels, base, head, lidar: leg self-collision, new in v0.4; pairs that are joint-adjacent, USD-filtered or both welded to `base_link` are skipped). Unresolved geometry fails. Applied to the posture with the nominal arm (step 3) and to each arm solution (step 4). Reachability version `t2-reachability-3`.
6. Postures are searched in increasing cost `(Δh/0.20)² + (ϕ/0.28)²`; a voxel is reachable at the first feasible posture, and that **least-deviation posture** (Δh, ϕ) and the arm configuration are recorded.
7. Height range (v0.4): **Δh_min = −0.33 m** (`DH_MIN_M`, run mode `protocol`) = the physical crouch limit found in v0.3 run 2: deepest Δh (0.01 m search steps) at ϕ = 0 for which every shallower posture is feasible under steps 3 and 5. Modes `kaist` (−0.20 m) and `crouch` (recomputes the limit) remain available for comparison. Since `t2-reachability-3` (2 cm leg/body self-clearance, step 5) the **effective crouch limit is ≈ −0.28 m**: no posture is feasible at Δh ≤ −0.30 m. The search range stays [−0.33, 0] m. The figure marks the KAIST reference Δh = −0.20 m (dash-dot contour and colour-bar line).
8. Per band report: absolute reachable voxel count; **annulus fraction** (normative) = reachable / voxels of the band in the sector with r ∈ [r_min, r_max] of that band's reachable voxels; **sector fraction** (information) = reachable / all voxels of the band in the sector; number of connected reachable regions (6-neighbour voxel faces), the size of the largest and its share of the band's reachable voxels; the share of reachable voxels whose least-cost posture needs Δh < −0.20 m (beyond the KAIST range); empty z layers over 0 – 1.1 m. `evaluation/reachability_t2.py --rescore <npz>` recomputes this report and A1 from a saved map without IK.
10. Loading (v0.4): `sampler.ReachableSet.load` refuses a file whose reachability version differs from `REACHABILITY["version"]` (`t2-reachability-3`) and, by default (always in the environment), any grid other than the full protocol grid.
9. The episode sampler draws targets only from the **sampling set** = per band, the largest connected (6-neighbour) region of its reachable voxels; isolated voxels outside it are dropped (v0.4, 2026-11-13). Band → voxel of that band's sampling set (uniform) → uniform point in the voxel, re-drawn until |θ| ≤ 60°. `sampler.ReachableSet` applies the restriction on load and reports the dropped voxels per band; the torch sampler of the environment uses the same set.

Deliverables per run: `reachability_T2_<run>.npz` (boolean voxel grid, per-voxel Δh, ϕ, arm configuration, posture table with leg joints), `reachability_T2_<run>.json` (voxel counts, band fractions, regions, r and z ranges, empty z layers, A1 verdict, rejection counts, asset hashes), `reachability_T2_<run>.png` (side view, colour = required Δh and ϕ). If a band fails A1, shrink or merge the band and record the decision.

**Note (2026-11-23, from the T3 measurement run `diag-measure-20261123T121813Z-2df42c5d`):** the frozen map `t2-reachability-3` evaluates postures down to Δh = −0.32 m (multiples of 0.02 m ≥ −0.33 m), but every posture with Δh ≤ −0.30 m is infeasible at all pitches (leg IK or clearance with the v0.4 per-class margins), and Δh = −0.28 m is feasible only at pitch 0 and +0.04 rad. The map's **effective lowering limit is −0.28 m** (base height 0.324 m), not the −0.33 m crouch limit of the pitch-0 search with T1 margins. The map and its digest are unchanged; T3 uses 0.324 m as the lowest T2 base height (T3 protocol D26).

Known limits of the analysis: kinematic only (no static stability, torque limits or wheel rolling); finite IK starts can miss solutions (under-estimates reachability); voxel membership is decided at the voxel centre. Acceptance test A3 checks dynamic feasibility.

### 4.4 Initial state (D9)

- Evaluation: the deterministic T1 nominal state (nominal base pose, legs, arm ready pose, closed gripper; zero velocities; PD targets = initial positions). No joint or yaw randomisation; base yaw is irrelevant because targets are defined in `G`.
- Training: joint positions ± 0.05 rad and base yaw ± π, to break symmetry; no randomisation of base height or tilt at reset.
- The target is sampled in `G` after the base is placed and converted to `W`.
- After every reset the harness advances one control period with zero action before evaluating terminations (stale contact data after `gym.make`, as in T1 draft.7).

### 4.5 Episode structure

- Control frequency: as T1 (policy 50 Hz, physics 200 Hz, decimation 4).
- Episode length: 10 s (500 control steps). One target per episode; no target switching (T1 handles sequences).
- Hold window `T_hold`: the last 1.0 s of the episode = the last 50 recorded control steps (decided; see 5.1).

---

## 5. Success criterion and metrics

All metrics are computed per episode (`reach_height/metrics.py`) and logged; aggregates are computed by the evaluation script, not inside the env.

### 5.1 Success (binary)

An episode is successful if all of the following hold:

1. Position error `e_p = ‖p_TCP − p_target‖₂ ≤ 0.05 m` for every control step inside the hold window `T_hold` (last 1.0 s).
2. (Mode B only) orientation error `e_R = ‖log(R_target^T R_TCP)‖ ≤ 0.10 rad` over the same window.
3. No termination before the horizon (no fall, no walk-away, no clearance violation; Section 7).

The 5 cm threshold is adapted from KAIST (Section 2); the hold window is a thesis addition that removes successes from transient passes through the target. Report also the "KAIST-strict" variant (threshold met at the final step only, no hold) for comparability. For an episode that terminated early, success is false and all hold-window metrics are NaN.

### 5.2 Final error

- `e_p_final`: position error at the last control step [m].
- `e_R_final`: orientation error at the last control step [rad] (logged in both modes).
- `e_p_hold_mean`: mean position error over `T_hold`.
- `t_reach`: first time at which `e_p ≤ 0.05 m` [s]; NaN if never.

### 5.3 Base height

Base link origin height above ground `h(t)`.

- `h_final`: at the last step.
- `h_hold_mean`: mean over `T_hold`.
- `Δh_hold = h_hold_mean − h_nom`: deviation from nominal [m]. Reference band: [−0.20, 0] m (KAIST command range). Values outside the band are not failures; they are reported as "posture outside reference band".
- `h_min`: minimum over the episode (safety: close to ground = risk of body contact).

### 5.4 Base tilt

Roll `φ(t)` and pitch `ϕ(t)` of the base (Isaac Lab `euler_xyz` convention; independent of yaw).

- `tilt(t) = arccos(−g_B,z / ‖g_B‖)`: angle between base z-axis and world up [rad], with `g_B` the projected gravity vector in `B` (pointing down; upright → `g_B,z = −‖g_B‖` → tilt 0). *v0.2 had the sign wrong (`arccos(g_B,z/‖g‖)` gives π when upright).*
- `tilt_hold_mean`, `tilt_max`.
- `pitch_hold_mean`, `roll_hold_mean` (signed): whether the policy pitches on purpose (expected for Low targets) or tilts by accident.
- `tilt_rms_hold`: RMS of `tilt(t) − tilt_hold_mean` over the hold window: oscillation, not posture. Order-of-magnitude reference: KAIST pitch RMSE 0.005 rad (different test, Section 2).
- Reference band for intentional pitch: |ϕ| ≤ 0.28 rad (KAIST).

### 5.5 Safety and auxiliary

- `fell`: termination by fall (bool).
- `walked_away`: horizontal base displacement from reset position > 0.30 m (bool).
- `max_base_displacement`: maximum horizontal base displacement [m].
- `n_foot_lifts`: number of wheel-contact loss events (true → false transitions, summed over the four wheels). Stepping to balance is allowed and logged only, with no effect on success.
- `joint_limit_violation_frac` (D11): fraction of steps with any arm joint within 5 % of its **soft-limit range** of a soft limit (the 90 % limits that clamp the action targets). KAIST's term uses 5 % of the limit's own value (Table II, `|q − q_min| ≤ 0.05·q_min`), which degenerates for limits near zero; a hard-range definition would read almost always zero because targets are clamped to the soft limits.
- `max_joint_torque_arm`, `max_joint_torque_leg`.

### 5.6 Reporting

Aggregate per height band and overall:

- success rate (mean ± std over policy seeds, sample std), both hold-window and KAIST-strict variants
- median and 90th percentile of `e_p_final`
- mean `Δh_hold`, mean signed `pitch_hold_mean`, mean `tilt_rms_hold`
- fall rate, walk-away rate

Mandatory figure: success rate and median final error versus target height (bins of 0.1 m), with the reachable-set boundary marked.

---

## 6. Observations and actions

Reuse T1 definitions where they exist, so that the same policy architecture can be trained on T1 and T2 and later on a T1+T2 mixture.

- Actions (D7): the T1 22-dim interface: 12 leg and 6 arm joint position offsets from the nominal pose (clamped to the soft limits) and 4 wheel-velocity channels forced to zero. The gripper is not commanded (held closed).
- Observations (policy): joint positions and velocities (22, T1 order), base angular velocity, projected gravity, previous action (22), base velocity command (always zero in T2, kept for interface compatibility), TCP target pose, TCP pose error.
  - **Decided in step 3 (engineering, 2026-11-13):** frame `B`, positions + quaternions wxyz, as T1 (no change to T1). Policy terms in order: joint positions (relative to nominal) and velocities (22 each, T1 order), base angular velocity, projected gravity, previous action (22), base velocity command (3, zero), TCP target pose in `B` (7), measured TCP pose in `B` (7), TCP pose error in `B` (position 3 + rotation 3 as axis-angle of q_target·q_TCP⁻¹). Concatenated, 95 dims (22+22+3+3+22+3+7+7+6). The `B_h` / 6D-rotation proposal of v0.2 is not used.
- Additional observation for T2 (thesis decision, flag `obs_target_height_band`): one-hot of the height band. Default off; a diagnostic to test whether the policy benefits from an explicit posture hint. Keep off for the benchmark.
- Privileged observations (critic): base height, base linear velocity, foot contact forces, payload mass. These do not exist in T1 (T1 has no critic group) and are new in T2.

---

## 7. Terminations (D10)

| Condition | Threshold | Source |
|---|---|---|
| Fall | tilt > 1.0 rad for ≥ 5 control steps (1.1.0; 1.0.0: no debounce), base height < 0.274 m for ≥ 5 steps (1.1.0), or any non-wheel body in contact with the ground | standard legged-RL practice; T3 shared termination set |
| Ground clearance | any non-wheel collider < 5 mm above the ground (T1 collider geometry) | T1 clearance policy |
| Arm self-contact | PhysX arm contact force > 1 N (`collision_arm_contact`) | T1 |
| Invalid state | non-finite state or non-unit quaternion | T1 |
| Walk-away (standing drift D) | horizontal base displacement > 0.30 m from the position at the first control step after reset (1.0.0: from the reset position) | thesis decision (standing task); T3 shared termination set |
| Time-out | 10 s | KAIST |

Not inherited from T1: tilt > 0.5 rad and height < 0.55·h₀ (they would terminate intentional low postures). A tilt threshold of 1.0 rad is far above the 0.28 rad reference band on purpose: intentional pitching must not be terminated; only falls are. The T1 5 mm arm self-clearance stays a logged monitor (as in T1). Implementation (step 3): every row is a separate termination term (`time_out`, `fall_tilt`, `fall_height` (1.1.0), `body_contact`, `arm_contact`, `invalid_state`, `walk_away`, `ground_clearance`). `body_contact` uses the T1 `collision_contact` semantics (net contact force > 1 N on any non-wheel body), which also counts self-contact, not only ground contact. `ground_clearance` uses a vectorised broad phase on all environments (lowest corner of each collider AABB, the same gate as T1) and the exact T1 collider check (CPU) only for environments the broad phase flags.

---

## 8. Evaluation configuration (benchmark mode)

The benchmark is run with a frozen evaluation config, separate from the training config.

- Domain randomisation: off. Nominal masses, friction, zero payload. (A second run with payload 1 kg is optional and reported separately; full payload sweeps belong to T4.)
- Observation noise: off. Initial state deterministic (4.4).
- Target set: fixed, seeded. Generate `N_eval = 2000` targets per band once (`T2_eval_targets_v1.npz`, with the hash of the reachability file it was drawn from) and reuse them for every policy. This is the Isaac Lab counterpart of KAIST's 10 000 randomised episodes per seed, reduced because the per-band split already stratifies the set.
- Seeds: 5 **policy training seeds** (D12). Env reset seed fixed.
- Output: one CSV row per episode with all fields from Section 5 plus `band`, `target_xyz_W`, `target_xyz_G`, `seed`, `policy_id`.

---

## 9. Training configuration (for later; must exist so the env is complete)

Not used in WP1 beyond a smoke test, but the env must expose it. T1 has no rewards, critic group or domain randomisation; these are new in T2.

- Domain randomisation: payload 0–2 kg at the end-effector, friction, link masses ±10 %, motor strength, action latency.
- Rewards (weights to be tuned in WP1 training, not here): TCP position tracking, and the following terms with weights set to zero by default:
  - `base_height_tracking`: not used (height is free; the policy chooses it).
  - `tilt_rate_penalty`: penalise `ω_x`, `ω_y` (KAIST base penalty uses 0.02·|ω_x| + 0.02·|ω_y|).
  - `vertical_velocity_penalty`: penalise `v_z²` (KAIST base penalty).
  - `walk_away_penalty`: penalise horizontal base displacement.
  - `joint_limit_penalty`: arm joints within 5 % of the soft-limit range (5.5).
- Curriculum hook: band weights `w_Low, w_Mid, w_High` exposed as config (`sampler.sample(..., band_weights)`), so that a success-rate-driven curriculum (KAIST style: start with Mid, open Low/High as success exceeds a threshold) can be added later without changing the env.

---

## 10. Implementation checklist (Isaac Lab manager-based env)

1. Done (CPU): `reach_height/protocol.py` (constants), `wholebody.py` (leg/arm IK, postures), `reachability.py` (A1), `sampler.py` (reachable-set sampler, evaluation set, A2 statistics), `metrics.py` (Section 5), `tests/test_t2.py`.
2. Implemented in code, not run in the simulator yet (step 3, 2026-11-13): `reach_height/env_cfg.py`. `T2ReachEnvCfg` (training reset, 4096 envs) and `T2ReachEvalEnvCfg` (deterministic reset, 16 envs) derived from the T1 env cfg. Gym IDs `RobotLab-Isaac-T2-Reach-Unitree-B2W-Z1-v0` and `RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0`. Rewards, critic group, randomisation and PPO runner (Section 9) not yet.
3. Implemented in code, not run in the simulator yet: `reach_height/runtime.py` `ReachTargetCommand` (Mode A only) with the torch sampler `torch_targets.py`. It loads only a full-grid map of reachability version `t2-reachability-3` (`reachability_path` or env var `T2_REACHABILITY_NPZ`, optional `reachability_sha256`), samples band → voxel → uniform position inside voxel → canonical orientation, stores the target and `G` per env, exposes `band_id`, and has an `assign()` hook for the fixed evaluation set.
4. Metric logging via `extras["log"]` at episode end; a separate `T2Recorder` writes per-episode CSV in evaluation mode.
5. Implemented in code, not run in the simulator yet: termination terms, Section 7.
6. Implemented and run (2026-11-13; controllers `hold`, `oracle`; policy not yet). Evaluation script `eval_t2.py`: loads a policy checkpoint (or the scripted oracle, A3), runs the fixed target set over 5 seeds, writes CSV, produces the Section 5.6 table and figure.
7. Config constants: `h_nom` = 0.6039 m, band boundaries (4.1), `r_min/r_max` per band (from reachability), TCP offset (Section 3).

---

## 11. Environment acceptance tests (before freezing, no trained policy needed)

| # | Test | Pass criterion |
|---|---|---|
| A1 | Reachability map generated (4.3, Δh ∈ [−0.33, 0] m); figures inspected | every band: annulus fraction ≥ 20 % and largest connected reachable region (6-neighbour) ≥ 90 % of the band's reachable voxels; absolute reachable voxel counts and share needing Δh < −0.20 m reported (sector fraction for information) |
| A2 | Sampler statistics over 10⁵ targets | band frequencies within 1 % of config; all targets inside the reachable set and their band; \|θ\| ≤ 60° |
| A3a | Gate, **static feasibility** of the map (2026-11-13): for each of 200 targets per band, reset the robot directly into the map's stored whole-body solution (base pose mapped from the map's `G` to the env's `G`, leg and arm joints, zero velocities) and hold 2 s with gravity-compensated PD targets, q_target = q_posture + τ_static / K (`eval_t2.py --controller posture`). τ_static comes from the static equilibrium at the written state: PhysX generalized gravity forces, minimum-norm wheel–ground contact forces carrying the robot, zero static torque at the velocity-controlled wheels. | per band ≥ 95 % with no fall, no illegal contact (no termination of any kind), TCP error < 5 cm at the end, and \|τ_static\| ≤ effort limit for every joint. Logged per episode: torque ratio \|τ_static\| / limit per joint (and its maximum), the maximum applied-torque ratio during the hold, minimum same-leg hip–wheel clearance, maximum leg-joint tracking error, and the contact-force solution (total F_z / weight, minimum F_z, tangential/normal ratio, residual) |
| A3b | **Dropped, out of scope of WP1** (2026-11-13). Dynamic solvability from the nominal pose is controller design and will be shown by the PPO baseline. The earlier scripted-oracle results stay in Section 12 as a naive scripted baseline, informational only. | — |
| A4 | Zero-action rollout (hold nominal pose) | 0 falls over 1000 episodes, `tilt_rms_hold` < 0.01 rad; per band, success rate ≤ the fraction of that band's targets within 5 cm of the nominal TCP (D5; v0.2 required 0 %) |
| A5 | Metric unit test | hand-constructed TCP trajectory gives expected `t_reach`, hold success, final error; tilt computed from a known rotation matches analytic value within 1e-4 rad |
| A6 | Termination test | teleporting the base 0.4 m sideways triggers `walk_away`; pitching 1.2 rad triggers `fall` |
| A7 | Determinism: two runs of `eval_t2.py --controller hold` on the full fixed evaluation set (seed 2026, 2 000 targets per band, 200 envs), same image and GPU, separate containers | tolerance fixed before the run (2026-11-13): identical eval-set digest; per target, identical `outcome`, `success`, `success_kaist_strict`, `completed`, `fell`, `terminations` and `contact_bodies`; \|Δ`e_p_final`\| ≤ 1e-4 m and \|Δ`e_p_hold_mean`\| ≤ 1e-4 m. Maximum differences of all other numeric columns are reported, not judged |
| A8 | Throughput | ≥ 4096 envs step without error; steps/s recorded for WP3 sample-efficiency bookkeeping |

### 11.1 Validation summary (protocol v1.0, map `t2-reachability-3`; all verified at runtime, image `sha256:b52a5210…`, GPU `GPU-fee1ef35-…` where applicable)

| # | Result | Evidence (run under `repos/robot_lab/logs/t2/`) |
|---|---|---|
| A1 | **Pass.** Annulus fraction 20.4 / 61.8 / 90.1 %; largest region 99.8 / 100 / 100 % (Low / Mid / High); 4 427 / 37 694 / 32 932 reachable voxels. Low passes with a small margin, z < 0.10 m unreachable, effective crouch limit ≈ −0.28 m | `diag-reach-A1-v04r3-20261113T084127Z-1d029cc2`, rescore `diag-rescore-A1-v04r3-20261113T122823Z-0be44ce0` |
| A2 | **Pass.** 10⁵ targets: max band-frequency deviation 0.09 %, all in sampling set and band, max \|θ\| = 1.0472 rad. Torch sampler matches the numpy reference (CPU tests). In-env consistency on the frozen map: 2 000 env targets all in the sampling set and their band, band frequencies 0.349 / 0.326 / 0.326, orientation ≤ 2.7e-7 rad and G→W→B ≤ 2.9e-7 m against numpy; observation (16, 95) | `diag-A2-sampler-20261113T140030Z-7898acbc`; `tests/test_t2.py`; smoke `diag-smoke-eval-v1-20261113T142017Z-6cc336b8` |
| A3a | **Pass.** 200/200 static feasible per band; max torque ratio 0.33 (Z1 joint2); min hip–wheel clearance 1.6 cm, no contact | `diag-A3a-static-20261113T134430Z-e415332a`; torque check `diag-static-check-20261113T134420Z-ba3349c2` |
| A3b | Dropped (out of scope of WP1). Naive scripted baseline on `t2-reachability-2`, informational: 26.5 / 29.5 / 42.5 % | `diag-A3-oracle-20261113T081903Z-62fff5a4` |
| A4 | **Pass** on the frozen set (6 000 episodes): 0 falls, max `tilt_rms_hold` 2.2e-5 rad, success 0 % = fraction within 5 cm of the nominal TCP (0 %) in every band | A7 runs below (also `diag-A4-hold-20261113T082427Z-ed7fbfe9` on `t2-reachability-2`) |
| A5 | **Pass.** Metric unit tests (hold window, `t_reach`, final error, early termination, tilt from a known rotation, posture metrics); 119 CPU tests pass | `tests/test_t2.py` |
| A6 | **Pass** on the frozen map (protocol 1.0.0). 0.4 m sideways teleport triggers only `walk_away`; 1.2 rad pitch only `fall_tilt`; no other env terminated | `diag-smoke-eval-v1-20261113T142017Z-6cc336b8` |
| A7 | **Pass, bitwise identical.** Two runs on the full frozen set (6 000 episodes): same eval-set digest, 0 mismatches in every categorical column, max \|Δ\| = 0 in every numeric column (tolerance was 1e-4 m) | `diag-A7-hold-run1-20261113T140000Z-22530d07`, `diag-A7-hold-run2-20261113T140343Z-6cd99d27`, `t2-a7-20261113T140000Z-compare/a7_compare.json` |
| A8 | **Pass.** 4 096 envs, 126 000 env-steps/s (zero action, no policy), 0.23 GB peak torch allocation. Run on `t2-reachability-2` (throughput does not depend on the map) | `diag-smoke-train-20261113T080543Z-59b8a3ff` |

A3a is the gate: it shows that the stored whole-body solutions are statically feasible in the full simulation. The contact forces are statically indeterminate, so A3a shows that a feasible torque distribution exists (the minimum-ratio one), not the one the simulator settles into (holdable with the static torques and within actuator limits, no contacts). It does not test whether a controller can reach them from the nominal pose. That is shown by the PPO baseline, not by a scripted controller in WP1.

---

## 12. Decisions

### 2026-11-12 (v0.2)

1. Bands: Low [0.00, 0.30], Mid [0.30, 0.80], High [0.80, 1.10] m, to be confirmed with the reachability map (A1).
2. Target range follows KAIST: 0.0 – 1.1 m in the main benchmark. No separate Stretch band; 1.0 – 1.1 m is part of High.
3. Δh > 0 is not allowed. Reachability uses the KAIST range Δh ∈ [−0.20, 0] m, |ϕ| ≤ 0.28 rad.
4. Orientation Mode A (position-based success) for the benchmark. Mode B optional.
5. Hold window 1.0 s.
6. Stepping is allowed and only logged (`n_foot_lifts`).

### 2026-11-12 (v0.3, researcher approval of D1–D12)

- D1 `h_nom` = 0.6039 m from the T1 nominal state (replaces ≈ 0.75 m).
- D2 End-effector = TCP at (0.171, 0, 0) m from `link06` (tool axis +x).
- D3 Lower limit of the Low band set by gripper geometry and the 5 mm ground clearance (via the reachability map), with the ground-contact termination kept.
- D4 A1 run twice: Δh_min = −0.20 m and Δh_min = physical crouch limit. Base translation in the reachability search (option (c)) rejected. Fallback if a band fails: shrink the band, or scale the height range to B2W.
- D5 "Mid contains the T1 workspace" withdrawn; A4 criterion relaxed.
- D6 Radial approach direction; Low approach angle decided after A1.
- D7 22-dim action space as T1.
- D8 Gripper closed as T1.
- D9 Deterministic evaluation reset; randomised training reset.
- D10 Terminations as Section 7; the 5 mm ground-clearance check terminates.
- D11 Tilt formula sign fixed; joint-limit metric on the soft-limit range.
- D12 "5 seeds" = policy training seeds.

### 2026-11-13 A1 results under protocol v0.3 (verified at runtime, CPU, 2 cm grid, 4 IK starts, seed 0; per-band orientation switch, superseded by v0.4)

Run 1, Δh_min = −0.20 m: `repos/robot_lab/logs/t2/diag-reach-A1-kaist-20261112T215426Z-72baac11` (image `sha256:b52a5210…`, 5838 s). All 165 postures feasible. 68 279 of 232 680 sector voxels reachable; 4 965 voxels have an IK solution but are rejected by collision/clearance (mostly the arm against `base_link`, `lidar_link` and the front hips).

| Band | sector fraction | annulus fraction | reachable z [m] | reachable r [m] |
|---|---|---|---|---|
| Low | 0.069 | 0.287 | 0.10–0.30 | 0.32–0.68 |
| Mid | 0.235 | 0.371 | 0.40–0.80 | 0.03–0.97 |
| High | 0.566 | 0.894 | 0.80–1.10 | 0.01–0.97 |

A1 verdict: **fails on the sector fraction** (Low 6.9 %), **passes on the annulus fraction**; Low is non-empty with Δh < 0. Findings:
- The TCP cannot get below z ≈ 0.10 m with the top-down Low orientation at Δh ≥ −0.20 m, so the ground (z ≈ 0) is not reachable.
- No reachable voxel lies in 0.30 ≤ z < 0.40 m. Inferred (not yet tested): this comes from the switch between the top-down Low orientation and the horizontal Mid orientation, rather than a positional hole.
- The Low patch needs Δh ≤ −0.10 m and a nose-down pitch of 0.1–0.28 rad, and sits at r ≈ 0.32–0.68 m.

Run 2, Δh_min = crouch limit: `repos/robot_lab/logs/t2/diag-reach-A1-crouch-20261112T215426Z-bceb6d7a` (same image, 10 794 s). Crouch limit Δh = −0.33 m (base 0.274 m; −0.34 m fails leg IK for RL). 224 of 255 postures feasible: below Δh = −0.24 m the feasible pitch range shrinks (at −0.32 m only ϕ = 0 and +0.04 rad), because the legs cannot keep the wheel centres fixed. 79 662 sector voxels reachable; 6 225 have an IK solution but are rejected by collision/clearance.

| Band | sector fraction | annulus fraction | reachable z [m] | reachable r [m] |
|---|---|---|---|---|
| Low | 0.089 | 0.368 | 0.04–0.30 | 0.32–0.68 |
| Mid | 0.340 | 0.536 | 0.36–0.80 | 0.01–0.97 |
| High | 0.566 | 0.894 | 0.80–1.10 | 0.01–0.97 |

A1 verdict: again **fails on the sector fraction** (Low 8.9 %) and **passes on the annulus fraction**. Crouching deeper lowers the lowest reachable TCP from 0.10 to 0.04 m and narrows the empty slab to 0.30–0.36 m; High is unchanged.

### 2026-11-13 (v0.4, researcher decisions after A1)

1. A1 = annulus fraction ≥ 20 % **and** ≥ 1 connected reachable region per band; absolute reachable voxel counts reported (connectivity tightened to ≥ 90 % in the next entry).
2. Δh ∈ [−0.33, 0] m for training and reachability; KAIST −0.20 m kept as a reference line in the figure and as the reporting reference band.
3. Per-band orientation switch replaced by a continuous approach-angle profile: vertical at z = 0, 45° at z = 0.30 m, horizontal at z ≥ 0.60 m, linear in between. First a coarse check; the full map only if the 0.30–0.36 m empty slab disappears.
4. The v0.3 run folders carry a `PROTOCOL_NOTE.md` stating that they were evaluated under v0.3.

Coarse check (verified at runtime; 6 cm grid, Δh ∈ [−0.33, 0] m, 324 s): `repos/robot_lab/logs/t2/diag-reach-coarse-v04-protocol-20261113T063245Z-52b77905`. 224/255 postures feasible.

| Band | reachable voxels | annulus fraction | sector fraction | regions | reachable z [m] |
|---|---|---|---|---|---|
| Low | 293 | 0.362 | 0.135 | 1 | 0.06–0.30 |
| Mid | 1 562 | 0.720 | 0.449 | 1 | 0.30–0.78 |
| High | 1 236 | 0.912 | 0.568 | 1 | 0.78–1.08 |

A1 passes. The only empty z layer is 0.00–0.06 m; the 0.30–0.36 m slab is gone. Caveat: at 6 cm the 0.80 and 1.10 m band edges do not fall on voxel faces, so Mid/High edges are approximate. Full 2 cm map started; results below.

### 2026-11-13 A1 full map, protocol v0.4 (verified at runtime; 2 cm grid, Δh ∈ [−0.33, 0] m, 4 IK starts, seed 0)

Run `repos/robot_lab/logs/t2/diag-reach-A1-v04-protocol-20261113T063821Z-835c4918` (image `sha256:b52a5210…`, CPU, 4 397 s), `reachability_T2_protocol.npz` sha256 `47d0a63f…084d29b`, reachability version `t2-reachability-2`, full grid. Re-scored with the final A1 rule (`--rescore`, `diag-rescore-A1-v04-20261113T075153Z-a673345e`). 224/255 postures feasible; 93 951 of 232 680 sector voxels reachable; 7 302 rejected by collision/clearance after IK.

| Band | reachable voxels | annulus fraction | sector fraction (info) | regions | largest region | needs Δh < −0.20 m | reachable z [m] | reachable r [m] |
|---|---|---|---|---|---|---|---|---|
| Low | 8 037 | 0.327 | 0.138 | 3 | 8 035 (≈ 100 %) | 57.9 % | 0.04–0.30 | 0.36–0.87 |
| Mid | 44 422 | 0.722 | 0.458 | 3 | 44 420 (≈ 100 %) | 27.4 % | 0.30–0.80 | 0.01–0.97 |
| High | 32 934 | 0.894 | 0.566 | 1 | 32 934 (100 %) | 0.1 % | 0.80–1.10 | 0.01–0.97 |

**A1 passes** (annulus fraction ≥ 20 % and largest region ≥ 90 % in every band). The only empty z layers are 0.00–0.04 m; no gap between bands. Low and Mid each have two isolated single-voxel islands outside the main region; the sampler currently includes them. Env loader check (verified at runtime): this file is accepted; the v0.4 coarse map is refused (not full grid) and the v0.3 maps are refused (version `t2-reachability-1`).

### 2026-11-13 (v0.4, after the coarse check)

- Coarse result accepted.
- A1 connectivity tightened: the largest connected region must hold ≥ 90 % of the band's reachable voxels (`min_largest_region_fraction`). The reachability version stays `t2-reachability-2`, because map contents do not change; maps are re-scored with `--rescore`.
- The full-map report also gives, per band, the share of reachable voxels that need Δh < −0.20 m.
- The npz loader checks the reachability version and refuses mismatches; the environment loads only a full-grid `t2-reachability-2` map.
- Step 3 approved. Observation frame decided (Section 6). GPU tests and commits need separate approval.

### 2026-11-13 (v0.4, after the full map)

- Full map accepted (`reachability_T2_protocol.npz`, sha256 `47d0a63f3509ba304fdd809f3b8a250e3901dc5f62ac96d539440c8ee084d29b`), pinned in `env_cfg.REACHABILITY_SHA256`; the environment refuses any other file.
- Isolated voxels are dropped: targets come only from each band's largest connected region (4.3 step 9). On this map that drops 2 Low and 2 Mid voxels.
- GPU smoke test approved (one allocated GPU UUID, unique container, tmux session, log file).

### 2026-11-13 GPU smoke test (verified at runtime; zero actions, no policy)

tmux session `t2-smoke-20261113T080534Z`, log `repos/robot_lab/logs/t2/t2-smoke-20261113T080534Z.log`; GPU `GPU-fee1ef35-…` (RTX 5080), image `sha256:b52a5210…`, torch 2.7.0+cu128. Runs `diag-smoke-eval-20261113T080534Z-b56e94a2` (16 envs) and `diag-smoke-train-20261113T080543Z-59b8a3ff` (4096 envs). Both completed without errors.

- Observation: policy shape (16, 95), term sizes as Section 6.
- Target-map consistency (2 000 targets from the env command): all in the sampling set and in their band, |θ| ≤ 1.0458 rad, band frequencies 0.330 / 0.328 / 0.343; orientation vs. numpy canonical ≤ 2.4e-7 rad; G→W and W→B (command) consistent to ≤ 3e-7 m / 4e-7 rad; G origin and yaw equal the reset root pose. Dropped voxels [2, 2, 0]. Measured TCP in G after one step (0.349, 0.009, 1.205) m vs. map nominal (0.349, 0.009, 1.208) m.
- A6 passed: the 0.4 m sideways teleport triggers only `walk_away`; the 1.2 rad pitch triggers only `fall_tilt`; no other env terminated.
- A8 passed: 4096 envs, 200 control steps, no error, no termination; 126 000 env-steps/s (30.8 control steps/s, zero action, no policy inference); 0.23 GB max torch allocation.
- Ground-clearance term at 4096 envs: broad phase 0.40 ms per call in isolation (456 collider corner points), 0.43 ms inside a step; 0 envs flagged per step while standing, so the exact check never ran; the whole term costs 4.5 ms of a 32.4 ms step (13.8 %, measured by replacing it with a no-op). Inferred: most of that is per-call Python overhead outside the broad phase, not the broad phase itself.

### 2026-11-13 A3 and A4 (verified at runtime; GPU `GPU-fee1ef35-…`, tmux `t2-a3a4-20261113T081903Z`)

Harness `evaluation/eval_t2.py`: fixed evaluation set (seed 2026, `sampler.eval_target_set`, digest-checked), deterministic eval reset, one zero-action harness step outside the episode, 500 recorded steps; the env time-out is set 2 control steps after the harness horizon so no env resets before its last recorded state. Oracle (A3, scripted, not a policy): legs to the map posture in 2 s with the arm nominal, then the arm along the first collider-free candidate path (3 cm, then 5 mm margin) in 4 s, then hold; integral action on the joint targets throughout (arm PD stiffness 100 N m/rad sags 0.13–0.2 rad without it).

**A4 passed** (`diag-A4-hold-20261113T082427Z-ed7fbfe9`, 1 002 episodes, 334 per band): 0 falls, 0 terminations, max `tilt_rms_hold` 2.9e-5 rad, success 0 % in every band = fraction of targets within 5 cm of the nominal TCP (0 in every band; no target is within 5 cm of the nominal TCP at z = 1.208 m).

**A3 failed** (`diag-A3-oracle-20261113T081903Z-62fff5a4`, 200 targets per band, 315 s):

| Band | success (hold) | success (KAIST-strict) | e_p_final median / p90 [m] | fall | clearance | did not reach |
|---|---|---|---|---|---|---|
| Low | 26.5 % | 34.0 % | 0.050 / 0.824 | 93 | 0 | 54 |
| Mid | 29.5 % | 37.0 % | 0.041 / 0.391 | 69 | 0 | 72 |
| High | 42.5 % | 54.5 % | 0.040 / 0.235 | 9 | 0 | 106 |

Failure analysis (from `episodes.csv`):
- Falls during the leg phase (t < 2 s, arm still nominal) with contact only on `FL_hip`/`FR_hip`/`RL_hip`: 58 Low, 26 Mid. Inferred cause: leg self-contact in deep crouched postures. **The reachability map does not check leg self-collision** (the T1 collider check only covers pairs with an arm link), so some map postures are not physically feasible.
- Falls with the arm against `lidar_link` (and a few against the front hips/base) during the arm motion or hold: 29 + 6 Low, 37 + 6 Mid, 9 High. The planned paths have only 5 mm clearance (no candidate cleared 3 cm), and the real arm lags/sags off the path.
- Did not reach: median final error 6 / 13 / 8 cm (Low / Mid / High); commanded joints at a soft limit in 0 / 18 / 13 episodes (joint3, joint4); the largest residual is usually on joint1 or joint2 (median 0.06–0.10 rad). Cause not yet diagnosed (joint1 carries no gravity load, so a limit cycle of the integral action is a candidate); successful episodes have ≤ 0.015 rad error.
- Ground-clearance term: 0 environments flagged per step over 1 503 steps in A3 and A4 (crouched postures included), so the exact check never ran; broad phase 0.19 ms per step at 200 envs. No clearance terminations.

A3 does not pass on this oracle. The failures are dominated by an oracle limitation (path clearance and tracking) and by a map gap (leg self-collision unchecked), not by the environment terms.

### 2026-11-13 (v0.4, decisions after A3)

Contact partner of the A3 hip contacts (verified at runtime, CPU kinematic replay, `diag-contact-partners-20261113T083434Z-37653d59`): all 84 leg-phase falls with only hip contacts have the **same leg's wheel** as the closest collider at the target posture (`FL_hip`–`FL_foot` 70, `RL_hip`–`RL_foot` 14), never the ground. 56 penetrate kinematically (up to 2.3 cm, first at 95 % of the crouch); 28 have 4–9 mm clearance and touched in simulation. Leg self-clearance over the posture grid (`diag-leg-clearance-…`): ≥ 2.9 cm down to Δh = −0.18 m at every pitch; at −0.28 m ≥ 2 cm only at ϕ ∈ {0, +0.04}; below −0.28 m < 2 cm everywhere.

Decisions:
1. The reachability check adds body–body (leg self-collision, including wheels) pairs and keeps the leg/body-to-ground rows (already present in the T1 check, verified in code). Arm–body margin is a parameter, set to 2 cm. Body–body margin set to 2 cm as well (thesis decision of this step, on the A3 evidence above; parameter `CLEARANCE_MARGINS_M["body_body"]`). Map recomputed as `t2-reachability-3`.
2. A3 split into A3a (gate) and A3b (informational), Section 11.

### 2026-11-13 (v0.4, map `t2-reachability-3`)

Run `repos/robot_lab/logs/t2/diag-reach-A1-v04r3-20261113T084127Z-1d029cc2` (image `sha256:b52a5210…`, CPU, 3 937 s, full grid, `--dh-min protocol`), `reachability_T2_protocol.npz` sha256 `22b9e665baa11bced9832107ca17c154ba57751fb55f54c3dbf67d9e1b865404`. Re-scored (`diag-rescore-A1-v04r3-20261113T122823Z-0be44ce0`): A1 pass, `env_loadable` true. 188/255 postures feasible (previously 224); the deepest posture rows lose pitches to `body_body` hip–wheel clearance from Δh = −0.20 m, and Δh ≤ −0.30 m has no feasible posture. 83 611 sector voxels reachable (previously 93 951); 9 148 rejected after IK.

| Band | Reachable voxels (t2-r2 → t2-r3) | Annulus fraction | Largest region | Regions | Δh < −0.20 m | r range (m) |
|---|---|---|---|---|---|---|
| Low (0.0–0.3) | 8 037 → 4 427 | 32.7 % → 20.4 % | 99.8 % | 2 | 33.4 % | 0.39–0.84 |
| Mid (0.3–0.8) | 44 422 → 37 694 | 72.2 % → 61.8 % | 100 % | 1 | 17.4 % | 0.01–0.97 |
| High (0.8–1.1) | 32 934 → 32 932 | 89.4 % → 90.1 % | 100 % | 1 | 0.05 % | 0.01–0.97 |

z < 0.10 m remains empty. The Low band passes the 20 % annulus rule by 0.4 percentage points. The re-pinned hash is in `env_cfg.REACHABILITY_SHA256`; the `t2-reachability-2` map is no longer loadable.

Decisions (researcher, 2026-11-13): map `t2-reachability-3` accepted with the 2 cm leg/body margin.
- The **Low band passes A1 with a small margin**: annulus fraction 20.4 % against the 20 % threshold.
- The 2 cm leg/body self-clearance margin is a **conservative choice, to be checked by A3a**. A3a logs per episode the minimum same-leg hip–wheel clearance and the maximum leg-joint tracking error during the hold. The margin is not changed before these numbers are reviewed.
- **Finding: nothing below z = 0.10 m is reachable while standing** (empty z layers 0.00–0.10 m, in this map and in all earlier maps). The Low band definition stays 0.0–0.30 m. Low-band results are therefore effectively for 0.10–0.30 m, and this is reported, not hidden by redefining the band.
- **Effective crouch limit ≈ −0.28 m** (deepest Δh with a feasible posture; at −0.28 m only ϕ ∈ {0, +0.04}, and none at ≤ −0.30 m), down from the leg-IK limit of −0.33 m. The cause is hip–wheel self-clearance, not leg IK. The search range stays [−0.33, 0] m.

### 2026-11-13 A3a and A3b on map `t2-reachability-3` (verified at runtime; GPU `GPU-fee1ef35-…`, tmux `t2-a3ab-20261113T124612Z`)

A3a (`diag-A3a-posture-20261113T124612Z-348cc403`, 200 targets per band, 2 s hold, 271 s): **fail, 0 % success in every band**.

| Band | Success | Fall | Did not reach | Other | Median e_p final | Hip–wheel min clearance (median / p5 / min) | < 0 / < 2 cm | Leg-joint track error max (median / p95) |
|---|---|---|---|---|---|---|---|---|
| Low | 0 % | 158 | 36 | 6 | 8.9 cm | 0.9 / −0.43 / −0.55 cm | 72 / 116 | 0.117 / 0.126 rad |
| Mid | 0 % | 111 | 86 | 3 | 12.8 cm | 9.2 / −0.07 / −0.55 cm | 16 / 28 | 0.111 / 0.124 rad |
| High | 0 % | 34 | 166 | 0 | 13.0 cm | 28.1 / 15.6 / 7.7 cm | 0 / 0 | 0.102 / 0.112 rad |

- Falls occur 4–13 control steps (0.08–0.26 s) after the state write (contacts: `FR_hip`/`FL_hip` in Low; `lidar_link`, `link02`, `link03` in all bands). Episodes that did not reach run the full 2 s and end 13–17 cm off the target.
- The hip–wheel clearance at the first recorded step is ≥ 1.7 cm in every episode (map value ≥ 2 cm). It drops below 0 only during the hold.
- The leg-joint tracking error is ≥ 0.07 rad in every episode, including High (median Δh ≈ 0). The median final arm error is on joint2: 0.08 / 0.11 / 0.12 rad (Low / Mid / High).
- Inferred cause: **gravity sag under PD-only holding**. Arm stiffness 100 N m/rad and leg stiffness 800 N m/rad give about 0.1 rad steady-state error at the map postures, which have zero static deflection. This drops the TCP by more than 5 cm, pushes the arm onto the lidar/body, and pushes hips onto wheels in deep crouches. A3a as specified therefore measures PD sag, not map feasibility. The hip–wheel numbers during the hold come from a sagged posture, not the map posture, so they do not yet test the 2 cm margin. Margin unchanged.
- Clearance term: 0 flagged environments per step over 303 steps; broad phase 0.26 ms per step.

A3b oracle (`diag-A3b-oracle-pd-20261113T125043Z-007add81`): **no result**. Arm-path planning with the new body–body pairs made batch 0 take 1 347 s, against about 100 s for the earlier A3. The 1 800 s time-box SIGTERM was ignored by Isaac Sim, so the container was stopped manually at 32 min, and no `episodes.csv` was written. The `--final-approach integral` variant was not run.

### 2026-11-13 (v0.4, decisions after the A3a sag result)

Decisions (researcher): controller design is out of scope of WP1.
1. A3a is redefined as a static-feasibility test with gravity-compensated PD targets and a torque-limit criterion (Section 11).
2. A3b is dropped (out of scope). The scripted-oracle results of `diag-A3-oracle-20261113T081903Z-62fff5a4` (map `t2-reachability-2`, Section 12 "A3 and A4") are kept as a **naive scripted baseline, informational only**: hold-window success 26.5 / 29.5 / 42.5 %, KAIST-strict 34 / 37 / 54.5 % (Low / Mid / High). They were measured on the superseded map and are not a WP1 result. The A3b rerun on `t2-reachability-3` (above) produced no result and is not repeated. The leg-pair skip in the oracle's path check is not implemented.
3. Infrastructure: `launch.sh` sends KILL 30 s after the time-limit TERM (`timeout -k 30`; verified at runtime with a TERM-ignoring script: outcome `hang` after 40 s, container removed). `eval_t2.py` appends each episode to `episodes.csv` as it ends.

### 2026-11-13 A3a static-feasibility run 1: invalid (harness bug)

Run `diag-A3a-static-20261113T133500Z-e0760d83` (GPU, tmux `t2-a3a-20261113T133500Z`, 42 s) completed, but its result is **invalid**. The static-torque solve (`eval_t2.static_torques`) satisfies the base force rows (total contact F_z = weight, residual 0) but returns contact forces of −1.3 to −21 kN at single wheels. The resulting leg torque ratios are 3–69× the effort limit, even at High-band postures with Δh ≈ 0, where A4 holds the nominal pose within limits. Arm torque ratios are plausible (≤ 0.33, joint2). The leg target offsets were clamped at the soft limits in 163–200 episodes per band, and the robot fell within a few steps. Not a result about the map or the 2 cm margin. To fix: validate the leg static torques against the settled applied torques at the nominal pose before rerunning.

### 2026-11-13 A3a static feasibility: passed (verified at runtime; GPU `GPU-fee1ef35-…`, tmux `t2-a3a-20261113T134420Z`)

Cause of the invalid run 1 (found with `diagnostics/t2_static_check.py`, `diag-static-check-20261113T133950Z-57df12d6`, offline analysis of its PhysX arrays):
- PhysX Jacobians give link-**COM** velocities, and the base moment rows of the gravity vector are about the base COM. The wheel COM is 5 cm lateral of the wheel link origin.
- The 4 wheel-torque equalities are linearly dependent on the base force rows for parallel wheels (rank 9 of 10). The tiny wheel gravity term (0.003 N m) made that dependent combination slightly inconsistent, so the pseudo-inverse returned kN-scale internal forces.

Fix: COM reference and contacts below the wheel COMs; hard constraints = the 6 base rows only; F chosen to minimise Σ (τ_j / limit_j)² over legs and wheels (wheel pseudo-limit 1 N m, since velocity-controlled wheels hold no static torque). The contact forces are statically indeterminate, so the torques are those of this minimum-ratio distribution: A3a shows that a feasible distribution exists, not which one the simulation settles into.

Validation (`diag-static-check-20261113T134420Z-ba3349c2`, 16 envs, nominal pose settled 2 s, zero action): |τ_static − τ_applied| ≤ 3.4 % of the effort limit for every one of the 18 position joints (pass ≤ 5 %). Solved wheel F_z ≥ 202 N (measured normal forces 192–236 N), tangential/normal ≤ 0.24. The contact sensor reports normal forces only, so friction is not compared directly.

A3a (`diag-A3a-static-20261113T134430Z-e415332a`, 200 targets per band, 2 s gravity-compensated hold, 508 s): **pass, 200/200 in every band** (no fall, no illegal contact, no termination; TCP error < 5 cm; all static torques within limits).

| Band | Static feasible | Median / p90 final TCP error | Max torque ratio (median / max; joint) | Max applied-torque ratio in hold (max) | Hip–wheel min clearance in hold (min / p5 / median) | < 2 cm | Max leg-joint tracking error (median / max) |
|---|---|---|---|---|---|---|---|
| Low | 100 % | 1.6 / 2.7 cm | 0.26 / 0.31 (Z1 joint2; legs ≤ 0.22, calf) | 0.32 | 1.6 / 1.8 / 4.5 cm | 37 | 0.019 / 0.028 rad |
| Mid | 100 % | 1.8 / 2.7 cm | 0.27 / 0.33 (joint2) | 0.37 | 1.7 / 1.9 / 12.0 cm | 13 | 0.021 / 0.028 rad |
| High | 100 % | 2.1 / 3.2 cm | 0.27 / 0.33 (joint2) | 0.39 | 10.7 / 18.2 / 30.2 cm | 0 | 0.023 / 0.039 rad |

- No joint exceeds 33 % of its effort limit. Hips and thighs stay ≤ 4 %, calves ≤ 22 %, Z1 joint2 ≤ 33 %, joint3 ≤ 29 %.
- Solved contact forces: F_z ≥ 162 N at every wheel, tangential/normal ≤ 0.47 (μ = 1.0), residual 0.
- Hold targets beyond the soft limit (clamped by the action term): Z1 joint4 in 50 Mid and 13 High episodes, joint3 in 8 High. All of them succeeded.
- Hip–wheel clearance: no episode goes below 1.6 cm during the hold (first recorded step ≥ 1.9 cm, map ≥ 2 cm). The 2 cm margin leaves ≥ 1.6 cm in simulation with gravity compensation. Below 2 cm: 37 Low, 13 Mid. The margin was not changed (researcher decision pending).
- Ground-clearance term: 0 flagged environments per step over 303 steps; broad phase 0.25 ms per step.

Decisions (researcher, 2026-11-13):
- The 2 cm leg/body self-clearance margin is kept. **A3a validated it**: in simulation with gravity compensation the robot settles up to 0.4 cm closer than the map's kinematic value (minimum hip–wheel clearance 1.6 cm over all 600 holds), and no hip–wheel contact occurred.
- Caveat: the static torques are those of the minimum Σ (τ/limit)² contact-force distribution. A3a shows that a feasible torque distribution exists, not the one the simulator settles into (applied torques in the hold reach at most 39 % of a limit, static ones at most 33 %).

### 2026-11-13 A7 determinism and v1.0 freeze (verified at runtime; GPU `GPU-fee1ef35-…`, tmux `t2-a7-20261113T140000Z`)

A7: `eval_t2.py --controller hold` on the full fixed evaluation set (seed 2026, 2 000 per band, 200 envs), twice in separate containers (213 s each). The tolerance was fixed before the run (Section 11). Result: **bitwise identical**. Same eval-set digest `6260df58…937e`, 0 mismatches in `outcome`, `success`, `success_kaist_strict`, `completed`, `fell`, `terminations`, `contact_bodies`, and max |Δ| = 0 in every numeric column (`t2_compare_runs.py`). These runs also re-verify A4 on the frozen map. A2 was re-run on the frozen map (CPU).

Freeze: protocol 1.0.0. The evaluation-set digest is pinned in `protocol.EVAL_SET_SHA256`, and a CPU test checks that the map file matches `REACHABILITY_SHA256` and that the set regenerates to the pinned digest. A8 was last run on `t2-reachability-2` (throughput does not depend on the map).

Smoke rerun on the frozen map (2026-11-13, protocol 1.0.0, GPU, tmux `t2-smoke-20261113T142017Z`, run `diag-smoke-eval-v1-20261113T142017Z-6cc336b8`): observation (16, 95); target-map consistency passed (2 000 env targets in the sampling set and band, band frequencies 0.349 / 0.326 / 0.326, |θ| ≤ 1.0470 rad, orientation ≤ 2.7e-7 rad, positions ≤ 2.9e-7 m, dropped voxels [10, 0, 0]); A6 passed.

### 2026-11-23 (v1.1.0, interface change for T3; results unchanged)

Changes (T3 protocol v0.4, D6, D20, D22): (1) wheel action channels are velocity targets `a_w · s_w`, s_w = 10 rad/s (1.0.0: ignored, 0). The T2 controllers send zero wheel actions. (2) Shared T1/T2/T3 termination set: `fall_tilt` is debounced (tilt > 1.0 rad for ≥ 5 steps); new `fall_height` (base height < 0.274 m for ≥ 5 steps; h_fall = min(0.5 h_nom, h_T2_min − 0.05) with h_T2_min = 0.324 m); `walk_away` becomes the standing-drift rule (0.30 m from the position latched at the first step after reset). Term names are kept. (3) Smoke test A6 updated: the 1.2 rad pitched env is raised 0.6 m (free fall, no ground contact) and must terminate with `fall_tilt` on exactly the 5th step, not the 4th. Re-run on 1.1.0 (verified at runtime, `logs/t3/diag-smoke-t2-a6-20261123T195054Z-d5097d2f`): env 0 `walk_away` on step 1; env 1 no termination on steps 1–4 and only `fall_tilt` on step 5; no other env terminated; observation and target-map checks passed.

Regression (verified at runtime, GPU `GPU-fee1ef35-…`, image `sha256:b52a5210…`): with both changes, A7 hold (6 000 episodes, `diag-a15-t2-hold-…`, `diag-a14-t2-hold-20261123T170155Z-f955e4e0`) and A3a static postures (600 episodes, `diag-a15-t2-a3a-…`, `diag-a14-t2-a3a-20261123T170155Z-2ccd0aab`) are byte-identical to the 1.0.0 references `diag-A7-hold-run1-20261113T140000Z-22530d07` and `diag-A3a-static-20261113T134430Z-e415332a` (`t3_compare_exact.py`). No reference episode came near the new limits (minimum base height 0.322 m, maximum tilt 0.29 rad). Evaluation-set digest unchanged (`6260df58…937e`; the digest covers the arrays, not the version tag).

### Open

- Training configuration (Section 9): rewards, critic group, randomisation, PPO runner.

---

## 13. Differences from the KAIST protocol (for the thesis text)

- KAIST: mobile pick-and-place with grasping, moving base, pick distance U(0.9, 3.0) m and place distance U(0.5, 3.0) m (Table I), binary success + completion time. T2: standing reach to a virtual pose, no grasp, no locomotion (base displacement ≤ 0.3 m), success + final error + posture metrics.
- Height range: KAIST's 0.0 – 1.1 m (real) and 0.0 – 1.3 m (training) are table heights with a 6 – 10 cm object on the table; T2 uses 0.0 – 1.1 m as TCP height. Finding (map `t2-reachability-3`): while standing (base displacement ≤ 0.3 m, wheels fixed), the B2W+Z1 TCP cannot reach below z ≈ 0.10 m with the T2 orientation profile. The Low band is kept as 0.0 – 0.30 m and is effectively 0.10 – 0.30 m, so T2 does not cover KAIST's ground-level case. KAIST reports a single ground-to-1.1 m case (7/10 sim, 6/10 real). T2 reports success and error as a function of target height in bands.
- Success: KAIST's 5 cm is the horizontal distance of the placed, upright object from the table centre, plus an arm retreat ≥ 10 cm; T2 uses a 3D TCP position error ≤ 5 cm with a 1 s hold.
- Base height: KAIST's robot base is below 0.55 m; B2W's nominal base height is 0.604 m — similar, not "much higher" as stated in v0.2. The B2W TCP rests at ≈ 1.21 m at the nominal pose, so in T2 the difficult part of the range is near the ground (KAIST: both ends, with near-ground grasping reported as hardest, Sec. IV-C failure modes).
- Base height and pitch are commands in KAIST (Δh ∈ [−0.2, 0] m, |ϕ| ≤ 0.28 rad, limits of its low-level controller) and are not reported as metrics. T2 uses these ranges as reference bands, extends the height search range to the B2W leg-IK limit (Δh ≥ −0.33 m, v0.4) for training and reachability, and reports the realised posture. With 2 cm leg self-clearance the effective B2W crouch limit is ≈ −0.28 m (hip–wheel clearance), so T2 can use about 8 cm more crouch than KAIST's −0.20 m; 33 % of Low-band and 17 % of Mid-band reachable voxels need Δh < −0.20 m.
- Hold-window success is a T2 addition; the KAIST-strict variant is reported alongside for comparability.
