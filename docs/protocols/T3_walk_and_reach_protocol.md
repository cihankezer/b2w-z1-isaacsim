# T3 Protocol: Move with a Base-Velocity Command While Tracking an End-Effector Target

| Field | Value |
|---|---|
| Task ID | T3 (Benchmark Task Suite, WP1) |
| Document type | Implementation specification for the simulation environment |
| Version | v0.5 (decisions D1–D37 frozen, see Section 0) |
| Date | 2026-11-23 |
| Platform | Unitree B2W + Unitree Z1, Isaac Lab v2.3.2 on Isaac Sim 5.1 |
| Depends on | T1 (6D EE tracking while standing), T2 (low/high reach) |
| Reference protocols | Jiang et al., RA-L 2025 (LimX W1, wheeled quadruped manipulator); Deep Whole-Body Control (Fu et al., CoRL 2022), paper and code; Visual Whole-Body Control (Liu et al., CoRL 2024) |
| Code | `training/robot_lab_overlay/.../manipulation/walk_reach/` (constants in `protocol.py`, `VERSION = "0.4.0"`) |

---

## 0. Decision record

These decisions are frozen. D1–D11 answer the questions of the first T3 plan; D12–D25 answer the review of v0.2; D26–D32 answer the review of v0.3. Where a later decision changes an earlier one, the later one applies (marked "superseded").

| ID | Topic | Decision | Section |
|---|---|---|---|
| D1 | Locomotion mode | B2W. The 4 wheel channels of the shared 22-dim action are **wheel velocity targets in T1, T2 and T3**. The forced 0 rad/s in `StandingAction.process_actions` is removed. Scripted oracles in T1/T2 send 0 to the wheels. | 4.1 |
| D2 | EE frame | T2's TCP point (`link06` + 0.171 m along the tool axis) for all tasks. | 3, 5.5 |
| D3 | Orientation of random targets (M1) | Joint-space sampling with forward kinematics (FK), extended with a base pose offset. | 6.2 |
| D4 | Reachability | No 6D grid and no IK bank. Goals are FK samples, reachable by construction. | 6.3 |
| D5 | T2 limits | Superseded by D26 (v0.3: lowest base height 0.274 m, h_fall = 0.224 m). Largest tilt 0.28 rad, θ_fall = 1.0 rad unchanged. | 3, 8.3 |
| D6 | Terminations | One shared termination set for T1, T2 and T3. Arm ground contact and self-collision are safety terminations. T2's 0.30 m rule becomes a standing-drift rule, active only while the velocity command is zero. | 4.3 |
| D7 | Minimum target height | 0.05 m above terrain. 0.10 m while the velocity command is non-zero (**supervisor approval pending**). | 6.3, 7.3 |
| D8 | Circle scenario | No circle in T3 core. A moving circle is extension T3-C with T1's circle parameters unchanged. | 7.3, 12 |
| D9 | Evaluation reset | Deterministic reset, as in T1/T2. | 7.4 |
| D10 | Domain randomization | E-DR deferred until WP1 defines a DR config. v0.3 evaluates E-nom only. | 7.1 |
| D11 | Primary reference | Jiang et al. for platform, action interface, metrics and failure definition. DeepWBC and VBC for the task structure and command ranges. | 2 |
| D12 | Base posture feasibility | M1 samples (Δh, θ) only from a precomputed **feasibility table**: the posture table of the frozen T2 map (T2 leg IK with the wheel centres fixed + T2 clearance check, every grid posture evaluated). A continuous (Δh, θ) is allowed if the four corner postures of its grid cell are all feasible. Roll is 0. | 3, 6.2 |
| D13 | Collision-checked sample pool | An offline pool of FK samples g = (q_arm, Δh, θ, roll). Every goal is checked for arm self-clearance, arm–body clearance, body–body clearance and ground clearance (margins per D28) and the TCP height check of Sec. 6.3. Every stored segment is checked at all 20 interpolation points. This is a **collision-checked FK pool, not the IK reachability bank excluded by D4**: no IK is solved; each target pose is the FK pose of its sample. | 6.2, 6.3 |
| D14 | One interpolation rule | Every training goal carries g = (q_arm, b). M1: sampled. M2 (T1 rule): the stored joint angles, b = nominal base pose. M3 (T2): the stored posture and arm pose of the voxel; the target pose is the FK pose of that stored posture (not the voxel centre). All training segments interpolate g linearly in joint space. | 6.2 |
| D15 | Evaluation schedule rule | Evaluation scenarios inherit the schedule of their source task; T3 only adds the velocity command. S1 = T1 schedule, S3 = T2 schedule. A10 compares only the first S1 point with T1. | 7.3 |
| D16 | Training transitions | All modes: T_traj ~ U[1, 3] s, hold ~ U[0.5, 2] s, except that 20 % of transitions are steps (T_traj = 0). | 6.2 |
| D17 | M2 content | T1 fixed points only (no circle segments). | 6.2 |
| D18 | Orientation error | Geodesic angle, threshold 0.1 rad, as in T1/T2 (both use the geodesic angle as primary). Jiang et al.'s Frobenius form (√2 × geodesic) is logged as secondary only. | 5.5, 8.1 |
| D19 | D2 scope | T1 observations and commands switch to the TCP frame (new T1 version). | 3, 4 |
| D20 | D6 scope | T1 moves to the shared termination set (new T1 version). T1's current height threshold 0.55 · h_nom ≈ 0.332 m already conflicts with T2's lowest base height 0.274 m. | 4.3 |
| D21 | Termination details | S2 includes arm-to-arm contact. S3 keeps the current T1/T2 scope (all non-wheel colliders). The F1/S2 split only affects the reported reason, not training (both terminate). Test A17 checks a ground-filtered contact sensor; if it does not work, the reason is classified by geometry. | 4.3, 11 |
| D22 | Wheel action scale | s_w = 10 rad/s. A15 is a regression test: with zero wheel action T1/T2 results must be reproduced exactly. | 4.1, 11 |
| D23 | Tests | A6 becomes a CPU kinematic test with synthetic states. A9 uses the extreme postures stored in the T2 map. | 11 |
| D24 | Measurements | Section 10 adds the nominal wheel positions in the body frame. | 10 |
| D25 | Open item "T1 orientation function" | Removed (answered by D18). | 13 |
| D26 | Lowest base height | The frozen map's deepest feasible posture is Δh = −0.28 m, so h_T2_min = 0.604 − 0.28 = **0.324 m**. With the existing rule h_fall = min(0.5 · h_nom, h_T2_min − 0.05) = min(0.302, 0.274) = **0.274 m**. θ_fall stays 1.0 rad. The T2 protocol notes that the map's effective lowering limit is −0.28 m. | 3, 4.3 |
| D27 | M3 height stratification | T2-derived goals are stratified by 0.10 m TCP-height bins within each band (equal count per bin, capped by availability), so the lowest T2 heights are represented (A5). | 6.2 |
| D28 | Inheritance rule (extends D15) | Evaluation scenarios inherit the **schedule, the success rule and the clearance margins** of their source task. S1 primary success = T1's rule; S3 primary success = T2's rule (every scored step within 5 cm, position only); T3's mean position + orientation hold success is secondary. M2 training goals (T1-derived) use T1's clearance margins (5 mm); M1 and M3 keep T2's margins. S1 uses T1's evaluation targets unfiltered except by the moving-cell rules; S3 uses T2's targets with the 0.05 m minimum and the moving-cell rules. | 6.2, 6.3, 7.3, 8.1 |
| D29 | Body box removed | The single AABB B_body is no longer a validity check. Pool goals and segments are checked with the real colliders at their own posture; S1/S3 targets are valid in their source task. No box pre-filter is needed anywhere in v0.4 (a 5-box trunk + legs model at the nominal state is the rule if one is added later). | 3, 6.3 |
| D30 | Pool size | Training pool ≈ 30 000 goals (15 000 M1, 7 500 M2, 7 500 M3), same stratification. | 6.2 |
| D31 | Moving minimum height | 0.10 m while the velocity command is non-zero stays. How often it binds for M1 goals is reported (T2 goals start at 0.129 m, so it never binds for M3 / S3). | 6.3 |
| D32 | Target-stranding rules | The three online rules of Sec. 6.3 are approved. The training environment logs how often each fires per 1000 transitions. | 6.3 |
| D33 | T1 version for D19/D20 | T1 is bumped once, at the TCP switch (run-order step 4); the shared termination set (D20, step 3, A14) ships in the same new version. The T1 target file states its EE frame explicitly (`ee_frame: tcp`); target poses change by the 0.171 m TCP offset, the generating joint angles do not. The v1c target file stays archived. Step 4 cannot be byte-identical to v1c; regression check per step of every T1 trial: orientation error identical to v1c (the TCP is a rigid translation of link06) and \|e_pos_tcp − e_pos_flange\| ≤ 0.171 · θ_geo + 1e-6. The T1 v2 run becomes the new T1 reference. | 3, 4.3, 11 |
| D34 | Dead-end goals | Goals without a stored successor are excluded at sampling time by a mask (iterated until no goal's successors are all masked); the pool file is not rebuilt. Training pool v4: 51 goals without a stored successor (0.17 %); the iterated mask excludes 52 (40 M1, 12 M2; one more goal whose successors were all dead ends), verified by `t3_sampler_rollout.py`. | 6.2, 6.3 |
| D35 | Moving minimum is a guard | The 0.10 m moving minimum (D31) stays as a guard. It is non-binding for pool v4: 0 of 15 000 M1 goals lie below 0.10 m. | 6.3 |
| D36 | Per-height-band reporting | Evaluation outputs include a table per T2 height band (low 0–0.30, mid 0.30–0.80, high 0.80–1.10 m, target height above the terrain) plus an `outside` row (> 1.10 m): e_pos, e_ori and hold success. | 9 |
| D37 | T2 version | T2 is bumped to 1.1.0 for the wheel channel (D22) and the shared termination set (D6). Its A7/A3a results are byte-identical to 1.0.0 (A15, A14). | 4.3 |

### 0.1 Change log

| Version | Change |
|---|---|
| v0.1 | First draft (B2/B2W open, spherical M1 sampling, IK reachability map, circle scenario, perturbed eval reset). |
| v0.2 | Decisions D1–D11. Removed S2 (circle), E-DR, spherical M1 sampling and the 6D reachability map. Added shared terminations, standing drift, wheel interface, nominal deviation metric. |
| v0.3 | Decisions D12–D25. Feasibility table and collision-checked FK goal pool; joint-space interpolation for all training goals; 20 % step transitions; M2 = T1 fixed points; S1/S3 inherit the T1/T2 schedules (S1: 2 points × 5 s, S3: 1 target × 10 s); geodesic orientation error; s_w = 10 rad/s; A6 CPU, A9 map postures, A15 regression, A17 contact filter; measured values of Section 10 filled in; online step rules of Sec. 6.3; new open items 6–10 (Section 13). |
| v0.5 | Decisions D33–D37: T1 version and step-4 regression rule (D33), dead-end mask (D34), moving minimum as a non-binding guard (D35), per-height-band table (D36), T2 1.1.0 (D37). Artifacts pinned: training pool `6e4a18f7…0c70`, evaluation pool `86b45c40…d911`, evaluation set v4 `779f76ff…7658`. A14 and A15 passed (Section 11). |
| v0.4 | Decisions D26–D32. Body box removed (D29); h_T2_min = 0.324 m, h_fall = 0.274 m (D26); M3 height stratification (D27); inheritance of success rule and clearance margins, M2 with T1 margins, S1 unfiltered (D28); training pool ≈ 30 000 goals (D30); stranding-rule logging (D32); evaluation set v4. Open items 6–11 closed or updated. |

---

## How to read this document

- **MUST** = required for a valid T3 implementation.
- **SHOULD** = default choice; change it only with a written reason.
- **[OPEN]** = still needs a decision or approval. Section 13 lists all of them.
- Values marked "measured" come from `evaluation/measure_t3.py`, run `diag-measure-20261123T121813Z-2df42c5d`, stored in `walk_reach/data/measurements.json` and pinned by hash in `walk_reach/protocol.py`.

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
- Changes to T1 and T2 needed so that all three tasks share one interface (D1, D2, D6, D19, D20).

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
| Task command | 6D EE target fixed in the world; no base command | Base velocity + EE target in a height-invariant frame | Same as DeepWBC | Base velocity + 6D EE target in F_T |
| EE target sampling | "Randomly resampled around the robot" | Spherical ranges, body box, ground limit | Spherical ranges, collision check | Collision-checked FK pool (D3, D13) |
| Target interpolation | — | Spherical coordinates | Spherical coordinates | Joint space (D14), 20 % steps (D16) |
| Reachability check | None described | None (ranges and collision only) | None (ranges and collision only) | By construction (D4) |
| Position error | Euclidean distance (m) | Unitless L1 | m | cm, Euclidean |
| Orientation error | ‖log(R* Rᵀ)‖_F | not tracked in code | Euler-based (rad) | Geodesic angle (D18); Frobenius form secondary |
| Failure | Projected gravity z > −0.1, base height < 0.2 m, EE or base collision | Tilt > 0.2 rad toward a goal-dependent side, base z < 0.325 m | not reported | Shared termination set (D6) |
| Errors on failed trials | Excluded | Divided by a fixed 10 s (biased) | — | Averaged over alive steps only |
| Evaluation size | 1000 trials | 1000 episodes × 3 seeds | 100 episodes × 3 seeds | 64 episodes per cell × 3 policy seeds |

**Numerical reference.** No paper reports numbers for "velocity command + body-relative 6D EE target" on a wheeled robot. The closest numbers are VBC App. E Table 4 (walking: 2.9 cm, 0.33 rad; B1 + Z1, stepping, IK arm). The WP1 target (≤ 5 cm, ≤ 0.1 rad) comes from Jiang et al. (world-fixed targets). The thesis MUST state this.

**Documented deviations from DeepWBC/VBC:** joint-space instead of spherical interpolation (D14); FK pool instead of spherical sampling ranges (D3, D13); 20 % step transitions (D16); 6D orientation tracked; heading-frame velocity; physical fall definition; errors over alive steps; fixed evaluation file.

---

## 3. Platform parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| — | Robot | B2W + Z1 with gripper (`robots_usd/b2w_z1_gripper.usd`, asset tree SHA-256 `4e55713b…f1b4`) | D1 |
| h_nom | Base height above terrain at the T1 nominal state | 0.6039 m | T1 constants |
| h_T | Height of the Z1 base link `link00` origin above terrain at the nominal state | **0.6910 m** (measured) | Sec. 10 |
| TCP | EE frame: `link06` + 0.171 m along the tool axis | as in T2 | D2 |
| p_home | TCP pose in F_T at the nominal state | position (0.3489, 0.0088, 0.5168) m, quaternion (0.99997, −0.00730, 0.00005, 0.0000) (measured); TCP height 1.208 m | Sec. 10 |
| b_nom | Base pose offset of the T1 nominal state (Δh, θ, roll) | (0, 0.0001, −0.0146) rad (measured) | Sec. 10 |
| B_body | AABB of all non-arm colliders at the nominal state, in F_T, + 0.05 m per side. **Measured but not used as a check since v0.4 (D29).** | x ∈ [−0.720, 0.538], y ∈ [−0.339, 0.339], z ∈ [−0.792, 0.209] m | Sec. 10 |
| P_wheels | Wheel contact points in F_T (FR, FL, RR, RL) | (0.306, −0.193), (0.336, 0.191), (−0.351, −0.207), (−0.321, 0.177) m; z = −h_T (measured) | Sec. 10 |
| h_knee | Highest knee (calf joint) above terrain | 0.368 m (measured; per leg 0.368 / 0.352 / 0.368 / 0.350) | Sec. 10 |
| Feasibility table | (Δh, θ) grid postures of the frozen T2 map (Δh ∈ {−0.32, …, 0} m step 0.02; θ ∈ [−0.28, 0.28] rad step 0.04; roll 0), feasible or not | 188 of 255 feasible; deepest feasible Δh = −0.28 m (θ = 0 or +0.04 rad only) | D12, Sec. 10 |
| h_T2_min | Lowest base height = deepest feasible map posture | **0.324 m** (0.604 − 0.28) | D26 |
| θ_T2_max | Largest base tilt (map limit) | 0.28 rad | D5 |

**Rule:** T3 MUST use the same asset version, the same EE frame (TCP), the same 22-dim action interface and the same termination set as T1 and T2. T1 switches its observations and commands to the TCP frame in a new T1 version (D19).

---

## 4. Shared environment settings (T1, T2, T3)

### 4.1 Action interface (D1, D22)

- 22 dimensions: 18 non-wheel position targets (12 legs + 6 arm), 4 wheel velocity targets.
- Wheel torque: τ_w = K_d^w · (a_w · s_w − q̇_w), with **s_w = 10 rad/s** (D22). With the 0.108 m wheel radius (measured), 0.9 m/s needs about 8.3 rad/s, i.e. a_w ≈ 0.83.
- Wheel actuator (verified in code): stiffness 0, damping K_d^w = 20, effort limit 20 N·m.
- The wheel channels are active in all three tasks. They are no longer forced to 0 rad/s.
- Scripted oracles (T1/T2) send a_w = 0.

**Why:** Jiang et al. use this layout on a wheeled quadruped manipulator. One policy must see the same action meaning in every task.

**Consequence:** with a_w = 0 the applied wheel target is 0 rad/s with damping 20, which is exactly what T1/T2 apply today. Test A15 is therefore a regression test: T1/T2 oracle results MUST be reproduced exactly.

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
| DR, latency | None in v0.3 | D10 |

### 4.3 Shared termination set (D6, D20, D21)

All three tasks use this set, in training and in evaluation. Each termination records its reason. T1 moves to this set in a new T1 version (D20).

| ID | Condition | Class | Already in T1/T2? |
|---|---|---|---|
| F1 | Contact force > 1 N between the terrain and any non-wheel body of trunk or legs | Fall | Partly: T1/T2 terminate on net force > 1 N on any non-wheel body, without separating terrain from self-contact |
| F2 | Tilt angle (base z-axis vs. world z) > 1.0 rad for ≥ 5 steps (0.1 s) | Fall | No, add (T1 today: 0.5 rad, no debounce) |
| F3 | Base height above terrain < 0.274 m for ≥ 5 steps (D26) | Fall | No, add (T1 today: < 0.55 · h ≈ 0.332 m, no debounce) |
| S1 | Contact force > 1 N between the terrain and any arm link or the gripper | Safety | Yes (arm contact sensor) |
| S2 | Arm or gripper contact with the robot's own trunk or legs, **or arm-to-arm contact** (D21) | Safety | Yes (arm contact sensor) |
| S3 | Ground clearance of **any non-wheel collider** < 5 mm (T2 fast GPU check + exact CPU check), as in T1/T2 (D21) | Safety | Yes |
| D | Standing drift: while the velocity command is zero, horizontal base displacement > 0.30 m from the position latched when the command became zero | Drift | Generalizes T2's 0.30 m rule; add to T1 |
| T | Time-out | — | Yes |

**Why each part:**

- **F1–F3** follow the failure definition of Jiang et al. (tilt, height, collision), with thresholds derived from B2W geometry. θ_fall = max(1.0, θ_T2_max + 0.2) = 1.0 rad and h_fall = min(0.5 · h_nom, h_T2_min − 0.05) = min(0.302, 0.274) = 0.274 m (D26), so they cannot trigger on T2 postures (test A9).
- **S1–S3** keep the existing T1/T2 rules. Jiang et al. also count EE collision as failure.
- **F1 versus S2 (D21):** both terminate, so the split only changes the reported reason. The reason uses a ground-filtered contact sensor (`ContactSensorCfg.filter_prim_paths_expr` = ground) if test A17 shows that it works with the multi-body sensor; otherwise the reason is classified by geometry (lowest collider point of the body within 5 mm of the terrain = terrain contact).
- **D** is identical to T2's rule when the command is always zero (T1, T2). In T3 it only acts during standing segments. The latch is kept when a zero command is resampled to zero.
- **T1 height conflict (D20):** T1's current height termination 0.55 · h_nom ≈ 0.332 m lies above T2's lowest base height 0.324 m (D26), so a shared policy could be terminated in T1 for a posture that is valid in T2. F3 (0.274 m) removes this conflict.

Isaac Lab building blocks: `illegal_contact` (F1, S1, S2), `bad_orientation` (F2), `root_height_below_minimum` (F3). The 5-step debounce and the drift rule are custom terms.

---

## 5. Frames and conventions

### 5.1 Target frame F_T

The EE target moves with the base in the horizontal plane and turns with the base yaw. It does not follow base height, roll or pitch.

- Origin: o_T = (x_b, y_b, z_terrain(x_b, y_b) + h_T), h_T = 0.6910 m
- Axes: x_T = heading (base yaw ψ), y_T = left, z_T = world up

p_world = o_T + R_z(ψ) · p_T,  q_world = q_z(ψ) ⊗ q_T

**Why:** DeepWBC (paper Sec. 2; code `widowGo1.py` lines 597 and 1353) and VBC (App. C.1) use this frame. T3 uses terrain height under the base instead of a fixed world height (identical on flat ground).

**Relation to T2:** T2's ground frame G is frozen at reset. At reset, F_T = G shifted up by h_T (same origin x/y, same yaw). F_T is updated every step.

### 5.2 Heading frame F_H (velocity)

- v_H = R_z(ψ)ᵀ · v_world (x and y components), ω_z = world yaw rate.
- Commands and measurements both use F_H.

### 5.3 Base pose offset

A base pose offset b = (Δh, θ, roll) means: base height h_nom + Δh, base orientation R_z(ψ) R_y(θ) R_x(roll), base x/y at the F_T origin, wheel centres fixed in place (T2 base model). In F_T the base is at (0, 0, h_nom + Δh − h_T). θ positive = nose down. Roll is 0 for M1 and M3; M2 uses b_nom.

### 5.4 Spherical coordinates (reporting and stratification only)

l = ‖p_T‖, pitch = atan2(z, √(x² + y²)) (positive up), yaw = atan2(y, x) (positive left).

### 5.5 Pose errors (D18)

- Position: e_pos = ‖p_tcp − p_target‖₂, reported in cm.
- Orientation (primary): geodesic angle e_ori = 2·arccos(|⟨q, q*⟩|) in rad, as in T1 and T2.
- Orientation (secondary): Jiang et al.'s d_θ = ‖log(R* Rᵀ)‖_F = √2 · e_ori, logged only.
- Thresholds: 5 cm and 0.1 rad (geodesic).

---

## 6. Command generators (training distribution)

Two command terms, resampled independently: `base_velocity` and `ee_target`.

### 6.1 Base velocity command `base_velocity`

Use Isaac Lab's stock uniform velocity command (it already resamples on every reset).

| Parameter | Value | Why |
|---|---|---|
| vx* | U[−0.6, 0.9] m/s | DeepWBC [0, 0.9] ∪ VBC backward [−0.6, 0]. Realistic for B2W on wheels. |
| vy* | 0 | Neither reference uses lateral commands |
| ω* | U[−1.0, 1.0] rad/s | DeepWBC and VBC |
| Standing fraction | 0.20 | DeepWBC code (implicit ~20 %); explicit here |
| Heading mode | Off | Direct yaw-rate command |
| Resampling period | 3.0 s | DeepWBC code |
| Resample on reset | All resets | Removes the DeepWBC artifact |

### 6.2 EE target command `ee_target`

The target is a sequence of goals from the **collision-checked FK goal pool** (D13). A goal is g = (q_arm, b) with q_arm the 6 Z1 joint angles and b the base pose offset (Sec. 5.3). Its target pose is the TCP pose FK(g) in F_T. At each new goal one mode is drawn:

| Mode | Probability | Goals in the pool |
|---|---|---|
| M1 random reach | 0.50 | q_arm uniform inside the Z1 joint limits with a 5 % margin at each limit (= Isaac Lab soft limits, factor 0.9); b uniform over the feasible cells of the feasibility table (D12); stratified (below) |
| M2 T1-structured | 0.25 | T1 fixed-point rule (TASK_T1 1.0.0 §5.1: q_arm uniform in the soft limits, T1 clearance filters, `link06` not below the base) with training seeds disjoint from T1's evaluation seed; b = b_nom (D14, D17). No circle segments. **T1 clearance margins (5 mm)** (D28). |
| M3 T2-structured | 0.25 | Voxels of the frozen T2 map's sampling set (largest connected region per band, centre azimuth ≤ 60°), equal count per band and, within a band, equal count per 0.10 m TCP-height bin (D27); g = stored arm pose and stored posture (Δh, θ, roll 0) of the voxel; target pose = FK(g), not the voxel centre (D14) |

**Stratification (M1).** Valid candidates are drawn in excess (4×) and selected round-robin over occupied (l, pitch, yaw) bins of the TCP position (bin size 0.1 m × 0.2 rad × 0.2 rad), so that target positions fill the reachable set evenly.

**Dead ends (D34).** A goal without a stored successor is never drawn: `GoalPool` masks every stored successor that points to such a goal, repeated until no goal is left whose successors are all masked. Training pool v4 has 51 goals without a stored successor (0.17 %; 29 689 goals have all 8).

**Pool sizes (v0.4, D30).** Training pool: home goal (nominal state) + 15 000 M1 + 7 500 M2 + 7 500 M3 goals, seed 3101, 8 stored successors per goal (up to 32 candidates drawn by the mode mixture). Evaluation pool (S4 only): home + 512 M1 goals, seed 3901. The pool files live outside git; their digests are pinned in `walk_reach/protocol.py` and listed with path, size and SHA-256 in `walk_reach/data/artifacts_manifest.json`.

**Trajectory between goals (D14, D16).** Interpolate g linearly in joint space: g(s) = (1 − s) · g_start + s · g_goal, s = t / T_traj, then apply FK. Every point of the trajectory is therefore reachable. Z1 joint ranges are below ±π, so the interpolation needs no angle wrapping.

| Parameter | Value | Why |
|---|---|---|
| Start of a segment | Current target (the previous goal) | DeepWBC code (continuous targets) |
| Transition type | 80 % interpolated, 20 % step (T_traj = 0) | D16; covers the step targets of the evaluation scenarios S1, S3 |
| T_traj | U[1, 3] s, per goal (interpolated transitions) | DeepWBC range, per goal |
| Hold time | U[0.5, 2.0] s, per goal | DeepWBC code range, per goal |
| Settle target | Home goal (p_home) at every reset | Same as the evaluation settle phase |

**Why joint-space interpolation:** with 6D targets, intermediate orientations of a spherical or Cartesian interpolation may be unreachable. Interpolating in joint space keeps every intermediate target reachable. This is a documented deviation from DeepWBC.

### 6.3 Validity checks

Checks per point (TCP position p_T and goal g):

1. (Removed in v0.4, D29: the body box. Points 2–4 keep their numbers.)
2. Target height ≥ z_terrain + 0.05 m (D7).
3. If the velocity command is non-zero: target height ≥ z_terrain + 0.10 m (D31; a guard, non-binding for pool v4, D35; **supervisor approval pending**), and horizontal distance ≥ 0.15 m from each wheel contact point in P_wheels for points below h_knee.
4. Feasible whole-body sample: legs by T2 leg IK with the wheel centres fixed (within soft limits), and clearance with the margins of the goal's source (D28): M1, M3 T2 per-class margins (ground 5 mm, arm–arm 5 mm, arm–body 2 cm, body–body 2 cm); M2 T1 margins (5 mm; body–body 2 cm because T1 has no moving legs); a segment uses T1 margins if either end is an M2 goal. Unresolved geometry fails.

**Offline (pool build).** Checks 2 and 4 for every goal and at all 20 points s = k/19 of every stored segment (D13).

**Online (command term).** Check 3, because it depends on the velocity command:

- An interpolated segment is valid if all 20 points pass. A step sends only its goal, so only the goal must pass.
- A new goal is a random stored successor of the current goal with the drawn mode. If no valid goal is found after 20 tries, the previous goal is kept and the draw is repeated at the next step.
- If the current target itself fails check 3 (the command became non-zero while at a low target), the next transition is a forced step.
- When the velocity command changes from zero to non-zero, the rest of the active segment is checked again. If it fails, a forced step to a stored successor of the segment's start goal is drawn at once (that stored segment is clearance-checked; only its goal is sent).

**Logging (D32).** The command term counts, per 1000 accepted transitions: steps; steps accepted only because a step is checked at its goal; forced steps because the current target failed check 3; mid-segment re-checks that forced a step; kept goals after 20 failed tries; and M1 candidates rejected by the 0.10 m minimum or the wheel zone while moving.

**Why:** DeepWBC and VBC check only ranges and collisions; Jiang et al. describe no check. T3 never sends a target that fails checks 2–3, and every target is a collision-checked FK pose.

### 6.4 Coverage report

For the training pool, report the fraction of the VBC sampling region (l ∈ [0.40, 0.95] m, pitch ∈ [−π/2.5, π/3], yaw ∈ [−1.2, 1.2]) covered by valid targets, as the fraction of its (l, pitch, yaw) bins (0.1 m × 0.2 rad × 0.2 rad) that contain at least one target. This shows how the B2W + Z1 workspace compares with B1 + Z1. Spherical coordinates are relative to the F_T origin (Z1 base height), so the VBC ranges are a reference, not a sampling rule.

---

## 7. Evaluation protocol

### 7.1 Conditions

| Condition | Dynamics | Status |
|---|---|---|
| E-nom | Nominal parameters | v0.3 |
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

### 7.3 Target scenarios (D15)

Evaluation scenarios inherit the schedule of their source task; T3 only adds the velocity command.

| ID | Source | Schedule in the 10 s test phase | Scored ("hold") window |
|---|---|---|---|
| S1 | T1 fixed targets, T1 evaluation target set (seed 1101, file SHA-256 `505128a1…f190`), converted to the TCP frame and F_T; unfiltered except by the moving-cell rules (D28) | **2 points**, each a step target held 5 s (T1 schedule) | last 2 s of each point (T1 steady window) |
| S3 | T2 evaluation target set (regenerated from the frozen map, seed 2026, T2 `EVAL_SET_SHA256`), shifted into F_T | **1 target**, step, held 10 s (T2 schedule: `EPISODE_S` = 10 s) | last 1.0 s (T2 `HOLD_S`) |
| S4 | M1 generator on the evaluation pool (seed disjoint from training) | Continuous goal chain from p_home, training transition rule (D16) and online checks (Sec. 6.3) | none (all-window metrics only) |

- **S1:** the first points of the 64 episodes cycle through a seeded permutation of the valid T1 targets (all 50 appear as first point); the second point is a random other valid T1 target.
- **S3:** episodes alternate low and high band (32 each), drawn without replacement from the band's valid targets.
- S1 targets are excluded only by check 3 in moving cells; S3 targets by checks 2–3. The numbers are recorded in the evaluation-set metadata.
- Primary success follows the source task (D28): S1 = T1's rule (a started point succeeds if it completes without a termination); S3 = T2's rule (the episode completes and every scored step is within 5 cm, position only). S4 has no success rule.
- There is no S2 (circle) in v0.3 (D8). The scenario IDs keep their v0.1 numbers.

**A10 (D15):** only the **first S1 point** of cell (V0, S1) is compared with T1, because it is the only point that starts from the settled nominal state as in T1.

### 7.4 Episode structure

| Phase | Time | Velocity command | EE target | In metrics |
|---|---|---|---|---|
| Settle | 0–2 s | V0 | p_home | No |
| Test | 2–12 s | Profile | Scenario | Yes |

- Reset: deterministic nominal state (D9), same as T1/T2.
- The 64 episodes of a cell differ only by the targets.
- An episode ends at any termination of Sec. 4.3. No reset inside an episode.

**Evaluation set size:**

| Set | Cells | Episodes |
|---|---|---|
| Core | 8 profiles × 3 scenarios = 24 | 24 × 64 = 1536 |
| Stress | 3 profiles × 3 scenarios = 9 | 9 × 64 = 576 |
| Per method | 3 policy seeds | (1536 + 576) × 3 = 6336 |

The full list is generated once (`evaluation/build_t3_eval_set.py`, seed 3301) and stored as `t3_eval_set_v4.npz` with its SHA-256 digest (pinned in `walk_reach/protocol.py`). Results MUST cite the version and digest.

---

## 8. Metrics

All per-step metrics are computed at 50 Hz on the **alive steps of the test phase**, against the target sent to the policy at that step. Jiang et al. likewise exclude failed trials from the error metrics.

### 8.1 EE tracking (primary)

- e_pos (cm) and e_ori (geodesic, rad); the Frobenius form is logged as secondary.
- Windows:
  - **All:** all test-phase steps.
  - **Hold:** the scored windows of the source schedule (Sec. 7.3): S1 last 2 s of each point, S3 last 1.0 s. S4 has no hold window.
- **Primary success (D28):** the source task's rule (Sec. 7.3), reported as `primary_success_rate`.
- Hold success (secondary): a hold window succeeds if mean e_pos ≤ 5 cm **and** mean e_ori ≤ 0.1 rad, over hold windows that are fully alive.
- First-point values of S1 (e_pos, e_ori over the first scored window) are logged separately for A10.

### 8.2 Velocity tracking (primary)

In F_H:

- e_vx = |v_H,x − vx*| (m/s)
- e_vy = |v_H,y| (m/s)
- e_ω = |ω_z − ω*| (rad/s)

The first 1.0 s of the test phase is excluded from the steady-state values. Velocity settling time = first time after the command step with e_vx < 0.1 m/s and e_ω < 0.1 rad/s for 0.5 s continuously (NaN if never reached).

### 8.3 Falls and other terminations (primary)

Report separately, each with a Wilson 95 % interval:

- **Fall rate:** episodes ended by F1, F2 or F3.
- **Safety termination rate:** episodes ended by S1, S2 or S3.
- **Drift rate:** episodes ended by D.
- Mean time to termination, per class.

### 8.4 Secondary metrics

| Metric | Definition | Why |
|---|---|---|
| Base height | Mean and min above terrain (m) | Same as T2 |
| Base tilt | Mean and max (rad) | Same as T2 |
| Nominal deviation | (1/T) Σ_t Σ_wheels ‖p_i,xy − p_n,i,xy‖ in the body frame (m), p_n = nominal wheel centres in the body frame (Sec. 10) | Jiang et al. Eq. 16 |
| Joint-limit violations | Fraction of steps with any non-wheel joint within 1 % of the range from a limit | Safety; WP3 (G1) |
| Torque saturation | Fraction of steps with any torque at its limit | Sim-to-real feasibility |
| Mechanical power | Σ \|τ · q̇\| over all joints (W), mean | Jiang et al. "Aver. Power"; DeepWBC "Tot. Energy" |

### 8.5 Legacy metrics (cross-reference only)

- `legacy_ee_sphere_l1`: DeepWBC's normalized spherical L1 in F_T with the VBC range widths.
- `legacy_vx_l1`: body-frame |vx error|.

Both as per-step means over alive steps. Never primary.

---

## 9. Aggregation and reporting

1. Per episode: mean of each per-step metric over alive test-phase steps, per window.
2. Per cell: mean over 64 episodes, per policy seed.
3. Per method: mean ± std (ddof 1) over 3 policy seeds; 95 % percentile bootstrap CI over episodes (10 000 resamples).
4. Termination rates: pooled over episodes, Wilson 95 % interval.

**Required tables:** Main (all core cells pooled), per speed (V0–V7), per scenario (S1, S3, S4), stress (X1–X3), per height band (D36: T2 bands low / mid / high by target height above the terrain, plus `outside` above 1.10 m; e_pos and e_ori as per-seed means of the episode band means, mean ± std over seeds; hold success pooled over holds with a Wilson 95 % interval; a hold's band is that of its target).

**Required figure:** e_pos and e_ori against commanded speed |vx*| for V0–V3.

### 9.1 Log format

One row per episode (CSV, appended as episodes finish, as in T2):

`method, policy_seed, eval_set_version, eval_set_hash, cell_V, cell_S, episode_index, termination_class (none/fall/safety/drift), termination_id (F1/F2/F3/S1/S2/S3/D), t_termination, n_alive_steps, e_pos_mean_all, e_pos_mean_hold, e_pos_p95, e_ori_mean_all, e_ori_mean_hold, e_frob_mean_all, primary_success_rate, hold_success_rate, n_holds_completed, e_pos_mean_first_hold, e_ori_mean_first_hold, e_vx_mean, e_vy_mean, e_w_mean, vel_settle_time, base_h_mean, base_h_min, tilt_mean, tilt_max, nominal_deviation, joint_limit_frac, torque_sat_frac, power_mean, legacy_ee_sphere_l1, legacy_vx_l1`

Plus a manifest file: asset hash, protocol version, evaluation-set digest, goal-pool digests, code commit, image ID.

---

## 10. Measurement procedure and results

Measured by forward kinematics on the gripper USD at the T1 nominal state (`evaluation/measure_t3.py`, CPU, run `diag-measure-20261123T121813Z-2df42c5d`). The T1 nominal state is itself the final sample of a 30 s settled standing simulation (T1 draft.5), so no extra physics settling is done. Results are stored in `walk_reach/data/measurements.json` with the asset tree hash and the T2 map hash, and pinned in `walk_reach/protocol.py`.

| Value | How | Result |
|---|---|---|
| h_T | `link00` origin height at the nominal state | 0.6910 m |
| p_home | TCP pose in F_T at the nominal state | (0.3489, 0.0088, 0.5168) m; TCP height 1.208 m |
| b_nom | Base pose offset of the nominal state (ZYX decomposition) | (0, 0.0001, −0.0146) rad; rebuild error 1e-16 |
| B_body | AABB of all non-arm collider bounds in F_T, + 0.05 m per side | Sec. 3 |
| P_wheels | Wheel centres (foot-joint anchors) in F_T; contact point = centre x/y on the terrain | Sec. 3; wheel radius 0.107–0.109 m |
| Nominal wheel positions, body frame (D24) | Wheel centres in `base_link` | FR (0.306, −0.186, −0.499), FL (0.336, 0.198, −0.493), RR (−0.351, −0.200, −0.500), RL (−0.321, 0.184, −0.492) m |
| h_knee | Calf joint origin heights | max 0.368 m |
| Feasibility table (D12) | Posture table of the frozen T2 map (`posture_feasible`, T2 leg IK + clearance) | 188 / 255 feasible; Δh ∈ {−0.32, −0.30} infeasible at every θ; Δh = −0.28 feasible at θ ∈ {0, +0.04} only |
| Base-offset kinematics | `link00` pose in F_T over the (Δh, θ) grid; Z1 chain exported to `walk_reach/data/kinematics.json` for the torch FK | chain FK error vs. USD tree 2e-16 m; torch FK vs. USD < 1e-9 (test) |
| Clearance check cost | one pool point (leg IK + whole-body clearance) | mean 45 ms, p95 0.12 s (single core) |

---

## 11. Acceptance tests

Scripted inputs only; no trained policy except where stated. "CPU" tests are in `tests/test_t3.py` and run in the Isaac image without a simulator (`evaluation/diagnostics/run_pytest_t3.py`).

| ID | Test | Pass criterion | Where |
|---|---|---|---|
| A1 | Change base height, roll, pitch with a fixed target in F_T | Target world position changes < 1 mm | CPU |
| A2 | Rotate base yaw by 90° | Target rotates by 90° around o_T (< 1 mm) | CPU |
| A3 | Joint-space interpolation: exact endpoints, monotonic s, FK consistency; goal sampler: every new segment starts at the previous target, step fraction 0.20 ± 0.05 | All hold | CPU |
| A4 | Draw 10⁶ segments from the training pool with random velocity commands | 0 segments violate checks 2–3 of Sec. 6.3 at the sent points; check 4 holds for stored segments (re-checked with the USD model on a random subset) | CPU |
| A5 | Height histogram of M3 pool goals and S3 targets | Every 0.10 m height bin above 0.05 m that contains T2 evaluation targets contains M3 goals; S3 spans the low and high bands above 0.05 m (0–0.05 m excluded by D7) | CPU |
| A6 | Synthetic states: base pitched ±0.28 rad, pure pitching motion and forward motion with vertical velocity (D23) | \|v_H,x\|, \|v_H,y\| < 0.01 m/s for pure pitching; v_H equals the horizontal heading velocity | CPU |
| A7 | Synthetic trajectories with known errors | Metrics match within 1e-4 | CPU |
| A8 | Tip the robot over | Fall detected with correct ID | Sim |
| A9 | Extreme postures stored in the T2 map (deepest Δh, largest \|θ\|), written as robot state and held (D23) | No F2/F3 termination | Sim |
| A10 | Same policy on T1 and on the first point of cell (V0, S1) | Scored-window e_pos, e_ori within 5 % relative or 0.5 cm / 0.01 rad | **Blocked** until a trained policy exists |
| A11 | Zero action on V0 and V1 | V0: no termination, e_vx ≈ 0; V1: e_vx ≈ 0.3 m/s | Sim |
| A12 | Evaluation set run twice with a deterministic controller | Same metrics within GPU tolerance (report max difference) | Sim |
| A13 | Regenerate the evaluation set from its seed and inputs | Same SHA-256 digest | CPU |
| A14 | Run T1 and T2 oracle evaluations with the shared termination set | F2, F3 and D never trigger; other outcomes unchanged | Sim |
| A15 | T1 and T2 oracle evaluations with active wheel channels and zero wheel action (D22) | Results reproduced exactly (regression) | Sim |
| A16 | Coverage report (Sec. 6.4) | Report produced; no pass threshold | CPU |
| A17 | Ground-filtered contact sensor (`filter_prim_paths_expr`) on the multi-body robot sensor, small scripted scene: one body on the ground, one arm–body contact (D21) | Terrain contact reported only for the ground contact; otherwise fall back to geometric classification | Sim |

Simulator tests run through the existing launcher on an allocated GPU, in tmux, logged under `logs/t3`.

### 11.1 Results (verified at runtime unless stated; image `sha256:b52a5210…`, GPU `GPU-fee1ef35-…`, runs under `repos/robot_lab/logs/t3/`)

| Test | Result | Run(s) |
|---|---|---|
| A17 | Fallback. A ground-filtered sensor on the multi-body robot reports no force for `/World/ground`, `/World/ground/.*` and `/World/ground/.*/.*` (Isaac Lab filters one-to-many only). Terrain vs. self contact is classified by geometry (D21); the self-contact case was not exercised (the arm zero pose gave no self contact) and is added to the step-5 env tests. | `diag-a17-20261123T140727Z-ac2ba2f4` |
| A9 | **Pass.** 430 extreme map postures held 2 s with T2's posture controller: no F2/F3. Deepest Δh (200): min base height 0.322 m; nose-down +0.28 (200): max tilt 0.29 rad; nose-up −0.24 (30): max tilt 0.25 rad. | `diag-a9-20261123T140727Z-dee3769c` |
| A15 | **Pass.** s_w = 10, zero wheel action: T2 hold (6 000 ep.) and A3a (600 ep.) byte-identical to the 1.0.0 references; T1 full validation 8/8, trial summaries, trajectories and metrics byte-identical to `diag-validate-v1c-…` (120 trials, 73 384 rows). | `diag-a15-t2-hold-…`, `diag-a15-t2-a3a-…`, `diag-a15-t1-r2-20261123T154553Z-ed2d756c` |
| CPU suite (A1–A7, A13, A16, pool, sampler) | **18/18 pass** on the v0.4 artifacts: 17 in the full run; `test_pool_segments_recheck` failed there because it re-checked M2 segments with T2 margins (stale test, fixed to the D28 margins) and passed in the re-run with the other changed tests. A5 passes (height stratification, D27). | `diag-cpu-suite-v4-20261123T195054Z-1bf982e5`, `diag-cpu-recheck-20261123T195851Z-d61d35f9` |
| Stranding counters (CPU emulation, 4 096 envs × 200 s, 311 880 transitions) | Per 1000 transitions: steps 199.7; step accepted only at its goal 0.003; forced step because the current target failed 0.003; re-check forced step 0.003; kept goal 0. M1 candidates while moving: 0.10 m minimum rejected 0 of 130 907 (non-binding, D35), wheel zone rejected 20 (0.015 %). Dead-end mask: 52 goals masked, none drawn. To be repeated in the Isaac training env (step 5). | `diag-sampler-v4-20261123T195054Z-f3d5641a` |
| A14 | **Pass.** Shared termination set in T1 and T2: T2 hold and A3a byte-identical to the references; T1 validation 8/8 incl. the new fault tests (tilt stops on the 5th step only, injected 0.2 m height on the 5th evaluation only, 0.31 m drift); all non-fault T1 rows byte-identical to v1c (73 374 rows, 120 trials). F2, F3 and D never fired. | `diag-a14-t1-20261123T170135Z-cfb5c47b`, `diag-a14-t2-hold-20261123T170155Z-f955e4e0`, `diag-a14-t2-a3a-20261123T170155Z-2ccd0aab` |

---

## 12. Extensions (not in v0.3)

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
| 1 | Minimum target height 0.10 m while moving | Supervisor approval pending; binding rate reported (D31) |
| 3 | Wheel K_d (s_w fixed by D22) | K_d = 20 (current asset config) |

Closed in v0.5: 4 (F2, F3 and the drift rule in T1: implemented, A14 passed 2026-11-23). Closed in v0.4: 6 (lowest base height, D26), 7 (pool size, D30), 8 (S3 success rule, D28), 9 (body box, D29), 10 (M2 margins, D28), 11 (A5 height coverage, D27). Closed earlier: 2 (D18/D25), 5 (Section 10).

---

## 14. Sources

1. K. Jiang, Z. Fu, J. Guo, W. Zhang, H. Chen. *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator.* IEEE RA-L, 2025. Sec. III-C (action space, Eq. 2), Sec. IV-A (Eq. 4, 6), Sec. V-B (failure conditions, Eq. 15, 16, Table III), Table IV.
2. Z. Fu, X. Cheng, D. Pathak. *Deep Whole-Body Control.* CoRL 2022. Sec. 2, Tables 1–4, Fig. 5, 7.
3. Deep Whole-Body Control code: `widowGo1.py` (frame: lines 597, 1353; goal generator: 1316–1350; termination: 937–963; command zeroing: 842; metric normalization: 745–750), `widowGo1_config.py`.
4. M. Liu, Z. Chen, X. Cheng et al. *Visual Whole-Body Control for Legged Loco-Manipulation.* CoRL 2024. Sec. 3.1, App. C.1, App. E Table 4.
5. Coding-agent repository reports (2026-11-23): T1/T2 action layout, EE frames, T2 map limits, terminations, Isaac Lab v2.3.2 components; T3 measurement run and pool builds.
6. Project plan "Semester Thesis": WP1 targets, Benchmark Task Suite T1–T5.
