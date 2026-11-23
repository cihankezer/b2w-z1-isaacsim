# T3 Protocol: Move with a Base-Velocity Command While Tracking an End-Effector Target

| Field | Value |
|---|---|
| Task ID | T3 (Benchmark Task Suite, WP1) |
| Document type | Implementation specification for the simulation environment |
| Version | v0.2 (decisions frozen, see Section 0) |
| Date | 2026-11-23 |
| Platform | Unitree B2W + Unitree Z1, Isaac Lab v2.3.2 on Isaac Sim 5.1 |
| Depends on | T1 (6D EE tracking while standing), T2 (low/high reach) |
| Reference protocols | Jiang et al., RA-L 2025 (LimX W1, wheeled quadruped manipulator); Deep Whole-Body Control (Fu et al., CoRL 2022), paper and code; Visual Whole-Body Control (Liu et al., CoRL 2024) |

---

## 0. Decision record (v0.2)

These decisions are frozen. They answer the 11 questions from the coding agent's T3 plan.

| ID | Topic | Decision | Section |
|---|---|---|---|
| D1 | Locomotion mode | B2W. The 4 wheel channels of the shared 22-dim action are **wheel velocity targets in T1, T2 and T3**. The forced 0 rad/s in `StandingAction.process_actions` is removed. Scripted oracles in T1/T2 send 0 to the wheels. T1/T2 oracle results must be re-checked once after this change. | 4.1 |
| D2 | EE frame | T2's TCP point (`link06` + 0.171 m along the tool axis) for all tasks. T1 targets are converted to the TCP frame at load time; T1 errors in A10 are recomputed in the TCP frame. | 3, 5.5 |
| D3 | Orientation of random targets (M1) | Joint-space sampling with forward kinematics (FK), extended with a base pose offset (option c). Trajectories interpolate joints and base offset, then apply FK. | 6.2 |
| D4 | Reachability | No 6D grid and no IK bank. M1 targets are reachable by construction (D3). T2-structured targets use the frozen T2 map, shifted into F_T. Collision checks as in the references. | 6.3 |
| D5 | T2 limits | Use the T2 map design limits (not observed values): lowest base height 0.274 m, largest tilt 0.28 rad. Fall thresholds: h_fall = 0.224 m, θ_fall = 1.0 rad. | 3, 8.3 |
| D6 | Terminations | One shared termination set for T1, T2 and T3. Arm ground contact and self-collision are **safety terminations**. T2's 0.30 m base-displacement rule becomes a **standing-drift** rule, active only while the velocity command is zero. | 4.3 |
| D7 | Minimum target height | 0.05 m above terrain (all tasks in T3). 0.10 m while the velocity command is non-zero (**supervisor approval pending**). Acceptance test A5 covers the T2 band above 0.05 m only. | 6.3, 7.3 |
| D8 | Circle scenario | No circle in T3 core. A moving circle is a later extension T3-C that must use T1's circle parameters unchanged. | 7.3, 12 |
| D9 | Evaluation reset | Deterministic reset, as in T1/T2 (decision D9 there). Episodes in a cell differ only by the chosen targets. | 7.4 |
| D10 | Domain randomization | E-DR is deferred until WP1 defines a DR config. v0.2 evaluates E-nom only. | 7.1 |
| D11 | Primary reference | Jiang et al. for platform, action interface, metrics and failure definition. DeepWBC and VBC for the T3 task structure (velocity command + body-relative EE target) and command ranges. No paper reports numbers for the T3 task on a wheeled platform. | 2 |

### 0.1 Change log

| Version | Change |
|---|---|
| v0.1 | First draft (B2/B2W open, spherical M1 sampling, IK reachability map, circle scenario, perturbed eval reset). |
| v0.2 | Decisions D1–D11. Removed S2 (circle), E-DR, spherical M1 sampling and the 6D reachability map. Added shared terminations, standing drift, wheel interface, nominal deviation metric. |

---

## How to read this document

- **MUST** = required for a valid T3 implementation.
- **SHOULD** = default choice; change it only with a written reason.
- **[MEASURE]** = a value that must be measured from the B2W + Z1 asset. Section 10 says how.
- **[OPEN]** = still needs a decision or approval. Section 13 lists all of them.

---

## 1. Scope

### 1.1 What T3 measures

T3 measures how well a whole-body controller tracks a 6D end-effector (EE) target **while the base follows a velocity command**. It is the moving version of T1 and T2. The benchmark does not prescribe a gait: the policy may drive, step, or combine both.

Primary metrics:

1. EE position error and EE orientation error
2. Base velocity tracking error
3. Falls

### 1.2 What this document defines

- The T3 simulation environment: frames, command generators, resets, terminations.
- The training-time command distribution (used later for PPO in WP1).
- The fixed evaluation set and the metrics.
- Acceptance tests that the environment MUST pass before any policy is trained.
- Changes to T1 and T2 needed so that all three tasks share one interface (D1, D2, D6).

### 1.3 Out of scope

| Out of scope | Where it belongs |
|---|---|
| Policy observations, network, rewards | WP1 baseline design |
| Domain randomization, latency | WP1 (D10) |
| Unknown payloads | T4 |
| Pushes and perturbations | T5 |
| Rough terrain, lateral commands, moving circle | Extensions (Section 12) |

---

## 2. References and how T3 uses them

| Item | Jiang et al. (LimX W1) | DeepWBC (paper / code) | VBC | T3 |
|---|---|---|---|---|
| Robot class | Wheeled quadruped + 6-DoF arm | Go1 + WidowX (stepping) | B1 + Z1 (stepping) | B2W + Z1 (wheeled) |
| Action | 18 position targets + 4 wheel velocity targets, τ_w = K_d(a − q̇) | 18 joint position targets | 12 leg targets + IK for the arm | Shared 22-dim layout (D1) |
| Task command | 6D EE target fixed in the world; no base command (Eq. 6 sketches a velocity + EE variant) | Base velocity + EE target in a height-invariant frame | Same as DeepWBC | Base velocity + 6D EE target in F_T |
| EE target sampling | "Randomly resampled around the robot" | Spherical ranges, body box, ground limit | Spherical ranges, collision check | Joint-space FK sampling (D3) |
| Reachability check | None described | None (ranges and collision only) | None (ranges and collision only) | By construction (D4) |
| Position error | Euclidean distance (m) | Unitless L1 | m | cm, Euclidean |
| Orientation error | ‖log(R* Rᵀ)‖_F | not tracked in code | Euler-based (rad) | Same function as T1 (Section 5.5) |
| Failure | Projected gravity z > −0.1, base height < 0.2 m, EE or base collision | Tilt > 0.2 rad toward a goal-dependent side, base z < 0.325 m | not reported | Shared termination set (D6) |
| Errors on failed trials | Excluded | Divided by a fixed 10 s (biased) | — | Averaged over alive steps only |
| Evaluation size | 1000 trials | 1000 episodes × 3 seeds | 100 episodes × 3 seeds | 64 episodes per cell × 3 policy seeds |

**Numerical reference.** No paper reports numbers for "velocity command + body-relative 6D EE target" on a wheeled robot. The closest numbers are VBC App. E Table 4 (walking: 2.9 cm, 0.33 rad; B1 + Z1, stepping, IK arm). The WP1 target (≤ 5 cm, ≤ 0.1 rad) comes from Jiang et al. (world-fixed targets). The thesis MUST state this.

---

## 3. Platform parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| — | Robot | B2W + Z1 with gripper | D1 |
| h_nom | Base height above terrain at the T1 nominal state | 0.6039 m | T1 constants |
| h_T | Height of the Z1 base link `link00` above terrain at the nominal state | **[MEASURE]** | Sec. 10 |
| TCP | EE frame: `link06` + 0.171 m along the tool axis | as in T2 | D2 |
| p_home | TCP pose (in F_T) at the default joint configuration | **[MEASURE]** | Sec. 10 |
| B_body | Box around trunk and legs at nominal state, in F_T, plus 0.05 m margin | **[MEASURE]** | Sec. 10 |
| P_wheels, h_knee | Wheel contact positions and knee height at nominal state, in F_T | **[MEASURE]** | Sec. 10 |
| Δh range | Base height offset used for M1 sampling and the T2 map | [−0.33, 0] m | T2 map limits (D4, D5) |
| θ range | Base pitch offset used for M1 sampling and the T2 map | [−0.28, 0.28] rad, roll 0 | T2 map limits (D4, D5) |
| h_T2_min | Lowest base height | 0.274 m (0.604 − 0.33) | T2 map limit (D5) |
| θ_T2_max | Largest base tilt | 0.28 rad | T2 map limit (D5) |

**Rule:** T3 MUST use the same asset version, the same EE frame (TCP), the same 22-dim action interface and the same termination set as T1 and T2.

---

## 4. Shared environment settings (T1, T2, T3)

### 4.1 Action interface (D1)

- 22 dimensions: 18 non-wheel position targets (12 legs + 6 arm), 4 wheel velocity targets.
- Wheel torque: τ_w = K_d^w · (a_w · s_w − q̇_w), with action scale s_w **[OPEN: value]**.
- The wheel channels are active in all three tasks. They are no longer forced to 0 rad/s.
- Scripted oracles (T1/T2) send a_w = 0.

**Why:** Jiang et al. use exactly this layout on a wheeled quadruped manipulator. One policy must see the same action meaning in every task (Section 4.3 explains the same rule for terminations).

**Consequence:** with a_w = 0 the wheels are damped, not locked. T1/T2 oracle results MUST be re-run once after this change (test A15).

### 4.2 Simulation

| Setting | Value | Why |
|---|---|---|
| Physics dt | 0.005 s | T1 config; DeepWBC; Isaac Lab locomotion default |
| Decimation | 4 → 50 Hz | Same in T1/T2, DeepWBC, VBC |
| Terrain | Flat, friction 1.0 | T1 config; falls must come from the task |
| Training episode length | 20 s | Isaac Lab locomotion default |
| Evaluation episode length | 12 s | Section 7.4 |
| Pushes | Off | T5 |
| Payload | 0 kg | T4 |
| DR, latency | None in v0.2 | D10 |

### 4.3 Shared termination set (D6)

All three tasks use this set, in training and in evaluation. Each termination records its reason.

| ID | Condition | Class | Already in T1/T2? |
|---|---|---|---|
| F1 | Contact force > 1 N between the terrain and any non-wheel body of trunk or legs | Fall | Yes ("any non-wheel contact") |
| F2 | Tilt angle (base z-axis vs. world z) > 1.0 rad for ≥ 5 steps (0.1 s) | Fall | No, add |
| F3 | Base height above terrain < 0.224 m for ≥ 5 steps | Fall | No, add |
| S1 | Contact force > 1 N between the terrain and any arm link or the gripper | Safety | Yes |
| S2 | Arm or gripper contact with the robot's own trunk or legs (self-collision) | Safety | Yes |
| S3 | Ground clearance of the arm/gripper < 5 mm (T2 fast GPU check + exact CPU check) | Safety | Yes |
| D | Standing drift: while the velocity command is zero, horizontal base displacement > 0.30 m from the position latched when the command became zero | Drift | Generalizes T2's 0.30 m rule; add to T1 |
| T | Time-out | — | Yes |

**Why each part:**

- **F1–F3** follow the failure definition of Jiang et al. (tilt, height, collision), with thresholds derived from B2W geometry. θ_fall = max(1.0, θ_T2_max + 0.2) = 1.0 rad and h_fall = min(0.5 · h_nom, h_T2_min − 0.05) = min(0.302, 0.224) = 0.224 m, so they cannot trigger on T2 postures (test A9). Jiang et al. use about 84° and 0.2 m for the LimX W1.
- **S1–S3** keep the existing T1/T2 rules. Jiang et al. also count EE collision as failure.
- **D** is identical to T2's rule when the command is always zero (T1, T2). In T3 it only acts during standing segments, so it does not conflict with commanded motion.
- **F2 and F3 in T1/T2:** with the thresholds above they never trigger on valid T1/T2 behavior, so T1/T2 results stay the same (test A14).

Isaac Lab building blocks: `illegal_contact` (F1, S1), `bad_orientation` (F2), `root_height_below_minimum` (F3). `root_height_below_minimum` uses world z, which equals terrain height on flat ground. The 5-step debounce and the drift rule are custom terms.

---

## 5. Frames and conventions

### 5.1 Target frame F_T

The EE target moves with the base in the horizontal plane and turns with the base yaw. It does not follow base height, roll or pitch.

- Origin: o_T = (x_b, y_b, z_terrain(x_b, y_b) + h_T)
- Axes: x_T = heading (base yaw ψ), y_T = left, z_T = world up

p_world = o_T + R_z(ψ) · p_T,  q_world = q_z(ψ) ⊗ q_T

**Why:** DeepWBC (paper Sec. 2; code `widowGo1.py` lines 597 and 1353) and VBC (App. C.1) use this frame. T3 uses terrain height under the base instead of a fixed world height (identical on flat ground).

**Difference from T2:** T2's frame is frozen at reset. F_T is updated every step.

### 5.2 Heading frame F_H (velocity)

- v_H = R_z(ψ)ᵀ · v_world (x and y components), ω_z = world yaw rate.
- Commands and measurements both use F_H.

**Why:** the body pitches and rolls on purpose when reaching; body-frame velocity would mix in vertical motion. Isaac Lab's yaw-frame velocity tracking uses the same idea.

### 5.3 Base pose offset

For target sampling (Sec. 6.2), a base pose offset b = (Δh, θ) means: base lowered by Δh relative to h_nom and pitched by θ, with roll 0 and the wheels fixed in place. The Z1 base link pose in F_T for a given b follows from the asset kinematics.

### 5.4 Spherical coordinates (reporting only)

l = ‖p_T‖, pitch = atan2(z, √(x² + y²)) (positive up), yaw = atan2(y, x) (positive left). Used only for the coverage report (Sec. 6.4) and legacy metrics.

### 5.5 Pose errors

- Position: e_pos = ‖p_tcp − p_target‖₂, reported in cm.
- Orientation: T3 MUST use **the same orientation-error function as T1** **[OPEN: confirm which]**.
  - If T1 uses Jiang et al.'s d_θ = ‖log(R* Rᵀ)‖_F, note that this equals √2 times the geodesic angle.
  - T3 reports this function as the primary orientation error and also logs the geodesic angle 2·arccos(|⟨q, q*⟩|) for comparison with other work.
- Thresholds (5 cm, 0.1 rad) are applied in the same unit as T1.

---

## 6. Command generators (training distribution)

Two command terms, resampled independently: `base_velocity` and `ee_target`.

### 6.1 Base velocity command `base_velocity`

Use Isaac Lab's stock uniform velocity command (it already resamples on every reset).

| Parameter | Value | Why |
|---|---|---|
| vx* | U[−0.6, 0.9] m/s | DeepWBC [0, 0.9] ∪ VBC backward [−0.6, 0]. Realistic for B2W on wheels (shipped policies reach 0.76–0.97 m/s). |
| vy* | 0 | Neither reference uses lateral commands |
| ω* | U[−1.0, 1.0] rad/s | DeepWBC and VBC |
| Standing fraction | 0.20 | DeepWBC code (implicit ~20 %); explicit here |
| Heading mode | Off | Direct yaw-rate command |
| Resampling period | 3.0 s | DeepWBC code |
| Resample on reset | All resets | Removes the DeepWBC artifact (only time-outs resampled) |

### 6.2 EE target command `ee_target`

At each new goal one mode is drawn:

| Mode | Probability | Content |
|---|---|---|
| M1 random reach | 0.50 | Joint-space FK sampling (below) |
| M2 T1-structured | 0.25 | T1 fixed points and circle segments, converted to the TCP frame and F_T (training only) |
| M3 T2-structured | 0.25 | T2 targets, shifted into F_T |

**M1 random reach (D3).** A goal is the pair g = (q_arm, b) of Z1 joint angles and base pose offset.

1. Sample q_arm uniformly inside the Z1 joint limits, with a 5 % margin at each limit.
2. Sample b: Δh ~ U[−0.33, 0] m, θ ~ U[−0.28, 0.28] rad, roll 0.
3. Target pose = FK of the TCP for (q_arm, b), expressed in F_T.
4. Stratification: draw candidates in batches and accept them so that the target positions fill (l, pitch, yaw) bins evenly (bin size 0.1 m × 0.2 rad × 0.2 rad, within the reachable set). Without this, the target positions cluster in some regions.

**Trajectory between goals.** Interpolate g linearly: g(s) = (1 − s) · g_start + s · g_goal, s = t / T_traj, then apply FK at every step. Every point of the trajectory is therefore reachable.

| Parameter | Value | Why |
|---|---|---|
| Start of a segment | Previous goal | DeepWBC code (continuous targets) |
| T_traj | U[1, 3] s, per goal | DeepWBC range; the code samples it once per env, T3 per goal |
| Hold time | U[0.5, 2.0] s, per goal | DeepWBC code range, per goal |

**Why joint-space interpolation:** DeepWBC interpolates in spherical coordinates; with 6D targets, intermediate orientations may then be unreachable. Interpolating in joint space keeps all intermediate targets reachable. This is a documented deviation from DeepWBC.

**M2/M3 trajectories:** Cartesian linear position interpolation in F_T and slerp, using each task's own transition rule where one exists.

### 6.3 Validity checks (all modes)

A segment is valid only if all 20 samples along it satisfy:

1. Outside the body box B_body.
2. Target height ≥ z_terrain + 0.05 m (D7).
3. If the velocity command is non-zero: target height ≥ z_terrain + 0.10 m **[OPEN: supervisor approval]**, and horizontal distance ≥ 0.15 m from each wheel position in P_wheels for points below h_knee.
4. Reachability:
   - M1: by construction (D4).
   - M2: T1 targets are FK-generated, so reachable by construction.
   - M3: the target lies inside the frozen T2 map. The map is shifted into F_T with the nominal-pose transform, which is valid because the map already includes the base height and pitch offsets of Section 3.

If no valid goal is found after 20 tries, keep the previous goal and try again at the next step. When the velocity command changes from zero to non-zero, the active segment is checked again; if invalid, a new goal is drawn immediately.

**Why:** DeepWBC and VBC check only ranges and collisions; Jiang et al. describe no check. T3 follows them and adds check 3, because a moving robot can drive over low targets near the wheels.

### 6.4 Coverage report

For the full training generator, report the fraction of the VBC sampling region (l ∈ [0.40, 0.95] m, pitch ∈ [−π/2.5, π/3], yaw ∈ [−1.2, 1.2]) covered by valid targets. This replaces the VBC ranges as a sampling rule and shows how the B2W + Z1 workspace compares with B1 + Z1.

---

## 7. Evaluation protocol

### 7.1 Conditions

| Condition | Dynamics | Status |
|---|---|---|
| E-nom | Nominal parameters | v0.2 |
| E-DR | WP1 DR ranges with fixed seeds | Deferred (D10) |

### 7.2 Velocity profiles

Constant during the test phase.

| ID | vx* (m/s) | ω* (rad/s) | Why |
|---|---|---|---|
| V0 | 0.0 | 0.0 | Standing; T1 consistency (A10) |
| V1 | 0.3 | 0.0 | Slow forward |
| V2 | 0.6 | 0.0 | VBC maximum speed |
| V3 | 0.9 | 0.0 | DeepWBC maximum training speed |
| V4 | −0.3 | 0.0 | Backward (VBC) |
| V5 | 0.0 | +0.6 | Turn in place, left |
| V6 | 0.0 | −0.6 | Turn in place, right |
| V7 | 0.5 | +0.5 | Curve |

Stress set (secondary): X1 = (1.0, 0), X2 = (0, +1.0), X3 = (0, −1.0). From the DeepWBC test bands.

### 7.3 Target scenarios

| ID | Content | Schedule in the 10 s test phase |
|---|---|---|
| S1 | T1 fixed points (TCP frame) | 4 points, each: 1.0 s transition + 1.5 s hold |
| S3 | T2 targets, filtered by Sec. 6.3 | low → high → low → high, each: 1.0 s transition + 1.5 s hold |
| S4 | M1 generator with evaluation seeds disjoint from training seeds | Continuous for 10 s |

Transitions in S1 and S3: Cartesian linear interpolation in F_T and slerp.

There is no S2 (circle) in v0.2 (D8). The scenario IDs keep their v0.1 numbers for traceability.

### 7.4 Episode structure

| Phase | Time | Velocity command | EE target | In metrics |
|---|---|---|---|---|
| Settle | 0–2 s | V0 | p_home | No |
| Test | 2–12 s | Profile | Scenario | Yes |

- Reset: deterministic nominal state (D9), same as T1/T2.
- The 64 episodes of a cell differ only by the targets: different 4-point subsets of the T1 set (S1), different T2 target sequences (S3), different seeds (S4).
- **Cell (V0, S1) MUST be built exactly like the T1 evaluation**, so that A10 compares like with like.
- An episode ends at any termination of Sec. 4.3. No reset inside an episode.

**Evaluation set size:**

| Set | Cells | Episodes |
|---|---|---|
| Core | 8 profiles × 3 scenarios = 24 | 24 × 64 = 1536 |
| Stress | 3 profiles × 3 scenarios = 9 | 9 × 64 = 576 |
| Per method | 3 policy seeds | (1536 + 576) × 3 = 6336 |

The full list is generated once and stored as a versioned file (e.g. `t3_eval_set_v2.npz`) with its SHA-256 hash. Results MUST cite the version and hash.

---

## 8. Metrics

All per-step metrics are computed at 50 Hz on the **alive steps of the test phase**, against the target sent to the policy at that step. Jiang et al. likewise exclude failed trials from the error metrics.

### 8.1 EE tracking (primary)

- e_pos (cm) and e_ori (T1's function, Sec. 5.5), plus the geodesic angle (rad).
- Windows:
  - **All:** all test-phase steps.
  - **Hold:** hold phases of S1 and S3, without the first 0.5 s of each hold.
- Hold success: a hold phase succeeds if mean e_pos ≤ 5 cm and mean e_ori ≤ 0.1 (T1 unit). Report the hold success rate.

### 8.2 Velocity tracking (primary)

In F_H:

- e_vx = |v_H,x − vx*| (m/s)
- e_vy = |v_H,y| (m/s)
- e_ω = |ω_z − ω*| (rad/s)

The first 1.0 s of the test phase is excluded from the steady-state values. Velocity settling time = first time after the command step with e_vx < 0.1 m/s and e_ω < 0.1 rad/s for 0.5 s continuously.

### 8.3 Falls and other terminations (primary)

Report separately, each with a Wilson 95 % interval:

- **Fall rate:** episodes ended by F1, F2 or F3.
- **Safety termination rate:** episodes ended by S1, S2 or S3.
- **Drift rate:** episodes ended by D (can only happen in standing segments).
- Mean time to termination, per class.

### 8.4 Secondary metrics

| Metric | Definition | Why |
|---|---|---|
| Base height | Mean and min above terrain (m) | Same as T2 |
| Base tilt | Mean and max (rad) | Same as T2 |
| Nominal deviation | (1/T) Σ_t Σ_wheels ‖p_i,xy − p_n,i,xy‖ in the body frame (m) | Jiang et al. Eq. 16; quality of wheel placement |
| Joint-limit violations | Fraction of steps with any non-wheel joint within 1 % of a limit | Safety; WP3 (G1) |
| Torque saturation | Fraction of steps with any torque at its limit | Sim-to-real feasibility |
| Mechanical power | Σ \|τ · q̇\| over all joints (W), mean | Jiang et al. "Aver. Power"; DeepWBC "Tot. Energy" |

### 8.5 Legacy metrics (cross-reference only)

- `legacy_ee_sphere_l1`: DeepWBC's normalized spherical L1 with the VBC ranges.
- `legacy_vx_l1`: body-frame |vx error|.

Both as per-step means over alive steps. Never primary.

---

## 9. Aggregation and reporting

1. Per episode: mean of each per-step metric over alive test-phase steps, per window.
2. Per cell: mean over 64 episodes, per policy seed.
3. Per method: mean ± std over 3 policy seeds; 95 % bootstrap CI over episodes (10 000 resamples).
4. Termination rates: pooled over episodes, Wilson 95 % interval.

**Required tables:** Main (all core cells pooled), per speed (V0–V7), per scenario (S1, S3, S4), stress (X1–X3).

**Required figure:** e_pos and e_ori against commanded speed |vx*| for V0–V3.

### 9.1 Log format

One row per episode (CSV, appended as episodes finish, as in T2):

`method, policy_seed, eval_set_version, eval_set_hash, cell_V, cell_S, episode_index, termination_class (none/fall/safety/drift), termination_id (F1/F2/F3/S1/S2/S3/D), t_termination, n_alive_steps, e_pos_mean_all, e_pos_mean_hold, e_pos_p95, e_ori_mean_all, e_ori_mean_hold, e_geo_mean_all, hold_success_rate, e_vx_mean, e_vy_mean, e_w_mean, vel_settle_time, base_h_mean, base_h_min, tilt_mean, tilt_max, nominal_deviation, joint_limit_frac, torque_sat_frac, power_mean`

Plus a manifest file: asset hash, protocol version, evaluation-set hash, code commit.

---

## 10. Measurement procedure

Measured from the training asset, stored in the T3 constants with the asset hash. Run in the Isaac image (CPU mode is enough).

| Value | How |
|---|---|
| h_T, p_home | Spawn at the T1 nominal state on flat ground, fix the base, settle the joints for 1 s, read `link00` height and the TCP pose in F_T. |
| B_body | AABB of trunk and leg links at nominal state, in F_T, + 0.05 m per side (collider model). |
| P_wheels, h_knee | Wheel contact points and knee height at nominal state, in F_T. |
| Base-offset kinematics | `link00` pose in F_T as a function of (Δh, θ) over the ranges of Section 3, for M1 sampling. Use the same base-pose model as the T2 map. |

---

## 11. Acceptance tests

Scripted inputs only; no trained policy except where stated.

| ID | Test | Pass criterion | Where |
|---|---|---|---|
| A1 | Change base height, roll, pitch with a fixed target in F_T | Target world position changes < 1 mm | CPU |
| A2 | Rotate base yaw by 90° | Target rotates by 90° around o_T (< 1 mm) | CPU |
| A3 | Joint-space interpolation: endpoints, monotonic s, FK consistency; M2/M3 slerp across ±π | Exact endpoints; shortest arc | CPU |
| A4 | Sample 10⁶ goals from M1–M3 | 0 segments violate Sec. 6.3 | CPU |
| A5 | Height histogram of M3 and S3 targets | Covers the T2 band above 0.05 m (0–0.05 m excluded by D7) | CPU |
| A6 | Robot pitches ±0.28 rad in place (scripted) | \|v_H,x\|, \|v_H,y\| < 0.01 m/s | Sim |
| A7 | Synthetic trajectories with known errors | Metrics match within 1e-4 | CPU |
| A8 | Tip the robot over | Fall detected with correct ID | Sim |
| A9 | Scripted T2-limit postures: crouch to Δh = −0.33 m, pitch ±0.28 rad | No F2/F3 termination | Sim |
| A10 | Same policy on T1 and on cell (V0, S1) | Hold e_pos, e_ori within 5 % relative or 0.5 cm / 0.01 (T1 unit) | **Blocked** until a trained policy exists |
| A11 | Zero action on V0 and V1 | V0: no termination, e_vx ≈ 0; V1: e_vx ≈ 0.3 m/s | Sim |
| A12 | Evaluation set run twice with a deterministic controller | Same metrics within GPU tolerance (report max difference) | Sim |
| A13 | Regenerate the evaluation set from its seed | Same SHA-256 | CPU |
| A14 | Run T1 and T2 oracle evaluations with the shared termination set | F2, F3 and D never trigger; other outcomes unchanged | Sim |
| A15 | T1 and T2 oracle evaluations with active wheel channels (a_w = 0) | Results recorded and compared with the braked-wheel results; differences documented | Sim |
| A16 | Coverage report (Sec. 6.4) | Report produced; no pass threshold | CPU |

Simulator tests run through the existing launcher on an allocated GPU, logged under `logs/t3`.

---

## 12. Extensions (not in v0.2)

| ID | Extension |
|---|---|
| T3-C | Moving circle. MUST use T1's circle parameters unchanged (0.05 Hz, 2 s quintic lead-in, full period). |
| T3-R | Rough terrain (F_T already uses terrain height). |
| T3-L | Lateral velocity commands. |
| E-DR | DR evaluation condition, after WP1 defines DR. |

---

## 13. Open items

| # | Item | Default |
|---|---|---|
| 1 | Minimum target height 0.10 m while moving | Supervisor approval pending |
| 2 | Which orientation-error function T1 uses (Frobenius log or geodesic) | T3 follows T1 |
| 3 | Wheel action scale s_w and wheel K_d | Shared with T1/T2; set once in the action config |
| 4 | Adding F2, F3 and the drift rule to T1 | Required by D6; verified by A14 |
| 5 | [MEASURE] values of Section 10 | Measurement script |

---

## 14. Sources

1. K. Jiang, Z. Fu, J. Guo, W. Zhang, H. Chen. *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator.* IEEE RA-L, 2025. Sec. III-C (action space, Eq. 2), Sec. IV-A (Eq. 4, 6), Sec. V-B (failure conditions, Eq. 15, 16, Table III), Table IV.
2. Z. Fu, X. Cheng, D. Pathak. *Deep Whole-Body Control.* CoRL 2022. Sec. 2, Tables 1–4, Fig. 5, 7.
3. Deep Whole-Body Control code: `widowGo1.py` (frame: lines 597, 1353; goal generator: 1316–1350; termination: 937–963; command zeroing: 842; metric normalization: 745–750), `widowGo1_config.py`.
4. M. Liu, Z. Chen, X. Cheng et al. *Visual Whole-Body Control for Legged Loco-Manipulation.* CoRL 2024. Sec. 3.1, App. C.1, App. E Table 4.
5. Coding-agent repository report (2026-11-23): T1/T2 action layout, EE frames, T2 map limits, terminations, Isaac Lab v2.3.2 components.
6. Project plan "Semester Thesis": WP1 targets, Benchmark Task Suite T1–T5.
