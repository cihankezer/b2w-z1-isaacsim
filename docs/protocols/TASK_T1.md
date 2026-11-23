# T1 — Standing 6D End-Effector Pose Tracking

**Protocol ID:** `T1-STANDING-6D-EE`  
**Version:** `2.0.0` (2026-11-23: TCP end-effector frame, shared T1/T2/T3 termination set, active wheel channel; see "Protocol 2.0.0" below. 1.0.0: draft.7 + oracle scripted controller, fast shutdown, metrics_export completeness contract; `validate_t1` PASS 8/8 on 2026-11-12; advisor review of the [OPEN] points pending)  
**Date:** 2026-11-12  
**Platform:** Unitree B2W + Unitree Z1 arm, Isaac Lab / Isaac Sim (researcher confirmed)  
**Work package:** WP1 — implement and validate T1–T3 task environments **before** model-free PPO training  
**Primary reference:** Jiang et al., *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator*, IEEE RA-L (2025), [arXiv:2412.03012](https://arxiv.org/abs/2412.03012).  
**Project reference:** `Thesis-Definition.txt`, “Benchmark Task Suite” (T1) and WP1.

> **Scientific status.** This is a literature-grounded, reproducible **proposed protocol**, not a verbatim reproduction or an externally approved standard. The LimX paper does not publish sufficient information to recover its exact fixed-point or circle trajectories. Decisions not supported by the paper are explicitly marked **[PROPOSED]** or **[OPEN]**. Obtain supervisor approval before treating these defaults as the frozen benchmark.

**Researcher decisions (first implementation):** use B2W; compute tracking statistics on the windows defined in §4.3 (draft.7); exclude entire failed trials from tracking-error averages while retaining them in the paper's survival-success denominator (§V-B, Eq. 15). Preserve all raw traces. The major-version prerelease records the changed statistics relative to draft `0.1.0`; it does not declare runtime completion.

**Researcher collision decision (2026-11-09, draft.2):** the existing USD robot
and collision geometry are authoritative. Preserve the collision exclusions and
PhysX filtering; do not replace capsules or modify geometry. The 5 mm requirement
is measured between **physical surfaces**. OBB overlap only selects narrow-phase
candidates. Distinguish true penetration, nonnegative clearance below 5 mm,
broad-phase false positives, and unresolved geometry. Physical contact force is
a separate observation: filtered geometry can overlap without generating force.
Visual-only links are not substituted as physical colliders.

**Asset switch (2026-11-11, draft.4; researcher decision, developer guidance).** The T1
robot is now `robots_usd/b2w_z1_gripper.usd` (fetch robot: arm 11 cm further back,
D405/D455 holder, gripper), replacing `b2w_z1_rl.usd`. PhysX articulation
self-collision is used as authored (enabled; FilteredPairs head_Link/lidar_link and
link00/base_link). `jointGripper` is held closed (0 rad) by its own implicit drive
(fetch gains) and is **not** in the 22-dim action space; the TCP stays the `link06`
flange. Per the asset developer, self-collision on this asset is judged on the physical
colliders, not visual meshes. Asset hashes change; draft.3 and earlier results are
not comparable. First finding: nominal settling produces a real PhysX self-contact
(link02 cylinder vs link06 wrist-camera ring collider, 408 N at 0.155 s), which stops
the run via `collision_arm_contact`. **[OPEN]**

**Nominal initial state (2026-11-11, draft.5; researcher decision).** Episodes start at a
defined nominal state instead of dropping from the 0.65 m spawn height. The former `ARM_STOW`
(joint2 0.15, joint3 -0.15) sits exactly at Isaac Lab's 90 % soft limits, i.e. it is a clamp
artifact, not a designed pose; T1 therefore defines a **nominal arm ready pose** [PROPOSED]:
joint1..6 = (0, 1.0, -1.3, 0.3, 0, 0) rad. It was verified on CPU (collider surfaces) with
every joint inside the middle 80 % of its range, self-clearance >= 15 mm at the pose and along
quintic transitions to +/-0.1 rad on each joint (minimum 19.98 mm, link00-link02), EE (link06) in
front of the base (x_B 0.178 m, z_B 0.604 m) and inside the draft.5 generator's workspace bounds (removed in draft.7), and
the wrist camera pointing 92.9 deg away from the trunk. Base height 0.603874 m and orientation
(wxyz) (0.999889, -0.007303, 0.000146, -0.013016), and leg/wheel joints, come from the final
sample (t = 30.0 s) of `diag-standing30-20261110T000140Z-c935ca00` (`protocol.NOMINAL_*`, draft.7);
base x/y at the spawn origin; all velocities zero; initial positions = PD targets. Joint
clamping, arm gains and the failure rule are unchanged (no settling exemption). `jointGripper`
stays closed (clamped to about -0.0785 rad). The nominal arm pose is **approved** (draft.7). The initial state is
deterministic (zero-range reset events); the seed governs target sampling only.

**Clearance policy (2026-11-11, draft.5; supersedes draft.3; pending advisor confirmation).**
Machine-readable copy: `protocol.CLEARANCE_POLICY`, recorded in sampled target sets and run metadata.

| Check | Geometry | Threshold | Mode |
| --- | --- | --- | --- |
| Arm self-clearance (non-adjacent pairs with a Z1 link or gripper body; adjacency = joints + asset filters) | PhysX collider surfaces: capsule analytic, capsule/convex certified QP, convex/convex hpp-fcl GJK/EPA (signed); cylinders as circumscribed 96-gon prisms (conservative, error <= r(1/cos(pi/96)-1) <= 0.04 mm here) | 5 mm | **Logged monitor** (per pair clearance, classification, per-episode counts) |
| Physical self-contact | PhysX self-collision (enabled by the asset) | 1 N arm contact | **Termination** (`collision_arm_contact`) |
| Ground/environment | PhysX colliders (mesh colliders: approx_hull, see below) | 5 mm | Termination; a selected candidate with unresolved geometry is a failure |

Target feasibility rejects any ground or self collider row below 5 mm or unresolved.
The draft.3 visual/CAD-mesh metric is dropped. Results under draft.3/4/5/7 must not be mixed.

**Draft.7 decisions (2026-11-11; researcher/advisor).**

- *Mesh colliders (approx_hull).* PhysX's cooked hull of this asset's mesh colliders (gripperStator,
  gripperMover, 4 calf, 4 wheel; CollisionAPI on an Xform over the mesh) cannot be read:
  `request_convex_collision_representation` returns `RESULT_ERROR_COOKING_FAILED` at runtime (and
  omni.physx is unavailable on CPU). The T1 monitor geometry for these colliders is the **exact convex
  hull of the collider's USD mesh points** (each mesh in its own frame). PhysX itself collides with its
  own convex approximation of the same points, so this is an approximation of the simulated surface.
  Rows touching such a collider carry `geometry = "approx_hull"`; the list of approx_hull colliders and
  the cooking failure are recorded in `protocol.CLEARANCE_POLICY`, run metadata (`approx_hull_colliders`)
  and every sampled target set. No pair remains `unresolved` for this reason.
- *Guardrail test change accepted:* a failed capsule/convex QP certificate falls back to the exact
  hpp-fcl GJK/EPA distance, never to a bounding-box value.
- *Target sampler, windows and statistics:* §4.3 and §5 (draft.7). The prepared-target-set path
  (`--mode prepare`, frozen 24 + 9 manifest, world workspace bounds, `validate_manifest`) is removed.
- *Runtime checks order:* `t1_runtime_checks.py` runs the fault checks first and the FK teleport check
  last, each from its own reset (`--checks` selects a subset). The failure term is unchanged.

**Protocol 2.0.0 (2026-11-23; researcher; T3 protocol D19, D20, D22, D33).** Shared interface with T2/T3.

- *End-effector frame = TCP:* `link06` pose × a fixed translation of 0.171 m along +x of `link06`
  (`protocol.TCP_OFFSET_M`, shared with T2/T3). Targets, the reference, the `ee_pose_body` observation,
  the command, the logged `ee_pos_world_xyz` and all errors use the TCP. The target file states its frame
  (`ee_frame: tcp`, `tcp_parent: link06`, `tcp_offset_m`).
- *Joint samples unchanged:* the sampler draws the same joint-space samples as 1.0.0. The filters (incl.
  `ee_min_height` on the `link06` origin), the circle IK audit and the oracle joint paths stay on `link06`;
  only the stored target poses move by R·(0.171, 0, 0). Quaternions are unchanged bit-for-bit.
- *Reference = rigid map:* the TCP reference is the `link06` reference (quintic + slerp transition, circle)
  right-multiplied by the offset at every instant (`geometry.tcp_reference`). During the (unscored) circle
  transition the TCP path is therefore not a quintic in TCP position.
- *Shared termination set* (`standing_ee/terminations.py`): tilt > 1.0 rad and base height < 0.274 m, each
  for ≥ 5 consecutive control steps (1.0.0: tilt > 0.5 rad, height < 0.55·h₀, no debounce), and the
  standing-drift rule (base x/y > 0.30 m from the position at the first control step after reset; reason
  `standing_drift`). Contact, clearance and invalid-state rules unchanged. Fault checks test the debounce.
- *Wheel channel:* wheel actions are velocity targets a_w · 10 rad/s (1.0.0: ignored). Zero wheel action
  reproduces 1.0.0 exactly (T3 A15).
- *Regression against 1.0.0 (D33):* per step of every trial, orientation error identical to 1.0.0 and
  |e_pos_tcp − e_pos_flange| ≤ 0.171·θ_geo + 1e-6 (`evaluation/t1_tcp_regression.py`). The 1.0.0 target
  file and run `diag-validate-v1c-20261112T142731Z-72ac7c0a` stay archived; the 2.0.0 validation run
  becomes the new T1 reference.

**Protocol 1.0.0 decisions (2026-11-12; researcher).**

- *metrics_export* checks **completeness and consistency of every trial**, successful or terminated
  (`validate_t1.check_metrics`): no failure flag before the last row; a successful trial has the full
  row count, ends truncated and its summary flags match; a terminated trial ends terminated with failure
  and a failure reason, its summary is consistent, its aggregate errors are null and its sample counts
  match the window masks; successful trials' mean/RMS/max are recomputed independently. Success is not required.
- *Teardown closed:* all T1 scripts use Isaac Sim's default `fast_shutdown=True` (`AppLauncher(args)`;
  evidence in the 2026-11-12 teardown bisection below). Outcomes are still judged by `completion.json`.
- *Second ArticulationRootAPI* on `/Unitree_B2W/z1/root_joint` of `b2w_z1_gripper.usd` is an **asset
  finding, no change**: Isaac Lab binds `base_link` (36 bodies, floating base); the extra root did not
  cause the teardown crash.
- *Oracle scripted controller* (`eval_t1.oracle_arm_q`, `--controller oracle|ik`, default `oracle`), used
  by `validate_t1` and for demos: fixed targets follow a 2 s quintic joint-space transition from the
  nominal arm pose to the target's `joint_space_sample_rad`, then hold it; circles replay the stored
  sampler-audit IK joint path (`sampler_audit.ik_path`, linear interpolation). Joint targets pass
  through the unchanged action term (same gains, same soft-limit clamping). Terminations remain genuine
  failures. The DLS-IK controller (`--controller ik`) is kept as a diagnostic.
- `validate_t1` child timeout 3600 s (was 1800 s; a concurrent RTX render slowed one rollout past 1800 s).
- *Reset contact = stale spawn data (2026-11-12, researcher decision; developer input).* The robot is
  spawned at the USD default pose, i.e. the **CAD zero pose** (folded arm/legs; not an intended pose).
  `T1_ROBOT_CFG.init_state` is already the nominal state (`env_cfg.py`), but Isaac Lab only stores
  `init_state` as defaults at spawn (`Articulation._process_cfg`, IsaacLab `articulation.py:1666`) and
  writes it on reset events; the physics step inside `sim.reset()` during `gym.make` runs at the CAD
  zero pose (verified at runtime: |q - init_state| max 1.461 rad right after `gym.make`). The contact
  sensor history (4 x 5 ms) keeps that step's rows across `env.reset()` until physics advances. The
  harness therefore advances **one control period** (4 physics steps, zero action = hold the nominal PD
  targets, no termination/command managers, not a T1 sample) after every reset before evaluating the
  failure term, and logs the contact rows after `gym.make`, after reset and after that period
  (`contact_log.json`). The failure term is unchanged. Episode rollouts were never affected: `env.step`
  advances physics before termination.
- *D405 ring collider:* real size per the asset developer; link02-camera contacts are genuine
  collisions (not an asset artifact) and remain terminations.
- *Statistics:* std over targets and over seeds is the sample std (`ddof=1`; null for n < 2).
  `trial_summary.csv` keeps per-trial max position/orientation error (`tracking_*_max`); pooled p95 stays dropped.
- *validate_t1 child processes* are judged by `completion.json` (written before teardown), not by the
  exit code; the exit code is recorded in `process.json`. Timeouts, a missing/unfinished
  `completion.json` or an `error.json` still fail. Runtime checks run as two processes
  (`--checks standing_termination`, then `--checks coordinate_frames`).
- *Known limitation (teardown):* processes with `num_envs >= 2` abort in native teardown after
  `app.close()` returned and after Python's atexit hook ("Fatal Python error: PyThreadState_Get ...
  finalizing"), even without stepping. Outcomes are judged by `completion.json` / `lifecycle.json`
  written before teardown; the exit code is not overridden.

`diagnose_t1_standing.py` may measure nominal PD dynamics by stepping physics
without automatic resets on geometric failures. It logs those failures, retains
physical-contact/fall/invalid-state stopping checks, and cannot declare T1 passed.
This diagnostic separates dynamics behavior from an unresolved clearance gate;
it does not alter the production task's termination configuration.

## 1. Purpose, scope, and non-goals

### Objective

Implement an Isaac Lab task in which a quadruped manipulator **remains standing** while its end-effector (EE) tracks a desired full 6D pose (3D position + 3D orientation). The task must support both **discrete fixed-pose goals** and **continuous spatial circles**. The main outputs are EE position error in metres and orientation error in radians.

### Deliverable for the current milestone

A validated, reproducible **simulation environment and benchmark evaluator**, consisting of:

1. A robot and actuator configuration capable of stepping safely in Isaac Lab.
2. A world-frame SE(3) target generator and body-frame command interface.
3. Fixed-pose and circular trajectory evaluation suites with frozen test seeds.
4. Error metrics, standing/fall diagnostics, time-series logging, and summary reports.
5. Scripted (non-learned) test controllers for smoke tests and kinematic checks.

**Out of scope at this stage:** PPO training, world-model training, teacher–student training, reward-weight tuning, real-robot transfer, and claims of learned tracking performance. T1 is only the **first** task: WP1 will also implement T2 (height-range reaching) and T3 (walking while tracking the EE), then train one or more model-free baselines on the completed task suite. **Standing-only is a T1 task constraint, not a limitation on the eventual controller.**

## 2. Evidence provenance

| Item | Source status | Evidence / decision |
| --- | --- | --- |
| Direct 6D EE pose target with world-frame semantics; expressed in body frame to the policy | **[PAPER]** | Jiang et al., §III-A, Eq. (1) |
| EE pose command represented as position (3) + rotation matrix (9) | **[PAPER]** | §III-A, Table I; command dimension = 12 |
| Joint-level control via PD targets | **[PAPER]** | §III-C, Eq. (2); W1 specifically has 18 non-wheel position targets + 4 wheel velocity targets |
| Position error as Euclidean distance and orientation error via SO(3) logarithm | **[PAPER]** | §IV-A, Eq. (4) |
| Fixed points and spatial circles in hardware tracking | **[PAPER]** | §V-C, Table IV (p. 8) |
| T1 = standing-only 6D tracking on B2/B2W + Z1 | **[THESIS]** | Thesis definition, T1; standing-only restriction clarified for this implementation |
| Number of targets, circle radius/plane/frequency, hold time, orientation programme | **[PROPOSED]** | Not specified by the reference paper; see §5 |
| Nominal stance, foot-slip bounds and controller frequency | **[PROPOSED]** | Engineering choices; must be validated on the actual robot asset |
| Platform | **[RESEARCHER DECISION]** | B2W + Z1; confirm runtime asset mapping |
| Definition of “standing” to freeze | **[OPEN]** | Requires supervisor/lab confirmation |

### Reference results — contextual, not our benchmark targets

The LimX paper reports these **real-hardware** tracking results (Table IV):

| Target type | Mean EE position error | Mean EE orientation error |
| --- | --- | --- |
| Fixed points | `0.028 ± 0.019 m` | `0.089 ± 0.073 rad` |
| Spatial circle | `0.048 ± 0.022 m` | `0.085 ± 0.025 rad` |

These are from a **different, wheeled robot and a controller allowed to move its base**. They must **not** be presented as standing-only B2(W)+Z1 targets, and the publication does not unambiguously identify what the `±` dispersion represents. It separately reports simulation ablation numbers; do not mix the two experiments.

## 3. Task definition and MDP/POMDP interface

### 3.1 Formalisation

**[DESIGN]** Represent the benchmark as a goal-conditioned, partially observed control environment:

\[
\mathcal M = (\mathcal S,\mathcal A,\mathcal O,P,r,\gamma,\rho_0),\quad
c_t=T^{W}_{E,\mathrm{des}}(t)\in SE(3).
\]

The paper does **not** specify this entire tuple; this is our implementation formalisation. For the **task-construction stage**, `P` is Isaac Lab physics, and the target generator and evaluator must work **independently of any learned policy**. `r` and `γ` are reserved for subsequent PPO development; do not interpret placeholder rewards as the paper's reward.

| Interface | T1 specification | Status |
| --- | --- | --- |
| Simulation state `s_t` | Floating-base pose/twist, joint positions/velocities, contacts, EE pose, actuator state, command, time | **[DESIGN]**; physics truth available to task manager |
| Observation `o_t` | Joint positions and velocities, base angular velocity, projected gravity, measured EE pose, previous action; target supplied separately | **[DESIGN]**; full policy layout frozen later with T2/T3 |
| Goal `c_t` | Desired SE(3) EE pose in world; transformed to body frame for future actor interface | **[PAPER-ALIGNED]** |
| Action `a_t` | Joint target command exposed by the simulated articulation; source-independent task/action interface | **[DESIGN]** |
| Transition `P` | Isaac Lab dynamics and actuator model | **[DESIGN]** |
| Reward `r_t` | **Not finalised or optimised in this milestone**; diagnostics exposed for future reward definition | **[DEFERRED]** |
| Termination | Time limit, fall, collision, invalid simulation state; distinguish failure from time-out | **[PROPOSED]** |
| Initial distribution `ρ₀` | Stable nominal stance, nominal arm configuration, deterministic evaluation targets | **[PROPOSED]** |

### 3.2 Robot DOFs and action conventions

- **B2 + Z1:** nominally **12 leg position targets + 6 arm position targets = 18 actions** (subject to verifying the actual asset's actuation map).
- **B2W + Z1:** if the loaded asset has four independently actuated wheel joints, use the above 18 position targets **plus 4 wheel-velocity targets = 22 actions**. During T1, wheel commands are constrained to zero velocity. Do **not** silently apply the LimX W1 actuator map to B2W; verify available joints and gains first.
- Keep gripper actuation (if present) fixed in a known neutral configuration and **out of T1's active action space**, unless the loaded asset requires otherwise.
- Apply commands through an explicit position-PD (and, if needed, wheel-velocity) actuator layer, with joint limits, torque limits and motor gains taken from the actual robot asset/configuration. A **[PROPOSED]** action convention is `q_des = q_nominal + Δq_cmd`; record the exact scaling/clipping. Do not assume an untrained policy exists.
- For task smoke tests, allow deterministic scripted actions (hold nominal pose, joint-space interpolation, arm IK + PD). These are **test instruments**, not RL baselines.

### 3.3 Future actor observation compatibility

The source W1 policy inputs are: 18 non-wheel joint positions, 22 joint velocities, 3D EE position, 9D EE rotation matrix, 22D previous action, 3D base angular velocity, 3D projected gravity, 1D SE(3) reference distance, 12D command, and 32D estimated privileged latent (§III-A, Table I).

**Do not copy the W1 observation dimensions verbatim**: the B2/B2W articulation and T1–T3 shared policy interface may differ. Export each feature in a structured dictionary (`joint_pos`, `joint_vel`, `base_ang_vel`, `projected_gravity`, `ee_pose_body`, `previous_action`, `ee_goal_body`, `base_velocity_goal`); flattening, history length, and privileged latent estimation are deferred until all task suites are defined. The T1 base-velocity goal is identically zero; T3 will use nonzero commands.

## 4. Coordinates and error metrics (normative)

### 4.1 Frame conventions

- `W`: fixed simulation world frame (z upward).
- `B`: robot floating-base frame as supplied by the articulation.
- `E`: **one explicitly named EE/tool frame**, such as the Z1 tool flange or TCP; select, document and lock it before creating target sets. Do not mix wrist link and TCP.
- `T^A_C`: transform that maps coordinates in frame `C` into frame `A`.
- Store all canonical target poses and all evaluator poses in **world coordinates**.

Use:

\[
T^W_E(t)=T^W_B(t)T^B_E(t),\qquad
T^B_{E,\mathrm{des}}(t)=[T^W_B(t)]^{-1}T^W_{E,\mathrm{des}}(t).
\]

This preserves the reference paper's distinction: **the desired pose remains world-fixed**, while the command expressed in the body frame changes as the base moves. For circles, only the **specified time-dependent reference** moves; never move the target in response to actual tracking error or base drift.

**[DESIGN]** Internal rotations may use unit quaternions; document `wxyz` versus `xyzw` conversion at every Isaac Lab API boundary. Export the original quaternion and optionally its flattened rotation matrix. Never subtract Euler angles to compute orientation error.

### 4.2 Primary errors

At every controller tick, compute:

\[
e_p(t)=\|p^W_E(t)-p^W_{E,\mathrm{des}}(t)\|_2\quad[\mathrm m],
\]

\[
e_R(t)=\|\operatorname{Log}(R^{W\top}_{E,\mathrm{des}}(t)R^W_E(t))^\vee\|_2
=\cos^{-1}\!\left(\operatorname{clip}\left(\frac{\operatorname{tr}(R_{\mathrm{des}}^\top R)-1}{2},-1,1\right)\right)
\quad[\mathrm{rad}].
\]

`Log(·)^vee` is the 3-vector rotation logarithm. Use a numerically stable quaternion/rotation implementation near zero and π rotations.

**Paper comparability:** Jiang et al. define `d_theta = ||log(R^T R_des)||_F` (matrix-log Frobenius norm, §IV-A, Eq. (4)). Mathematically, for a regular rotation with angle `e_R`, **`d_theta = sqrt(2) * e_R`**. Record **both** `orientation_error_geodesic_rad` (human-readable diagnostic) and `orientation_error_paper_frobenius` (first-version primary, source-aligned summary). The paper's printed radian results and exact log implementation are not fully reconstructable; **do not claim exact numerical comparability** without clarification.

### 4.3 Aggregation

Record raw errors at every **controller step**, with simulation time stamps (not wall-clock timing). **[RESEARCHER DECISION, draft.7]** Windows per target (endpoint sampling at 50 Hz, logged per step as `scored` = tracking window and `steady`):

- Fixed: **tracking error** = mean and RMS of position / orientation error over the **whole 5 s hold, including the transient** (250 ticks); **steady-state error** = mean over the **last 2 s** (100 ticks).
- Circle: **tracking error** = mean and RMS over the scored window, circle-phase times `(20, 40]` s (1000 ticks); no separate steady window.

Per trial (`trial_summary.csv`): `tracking_{position_m, orientation_geodesic_rad, orientation_paper_frobenius}_{mean, rms}`, `steady_{...}_mean` (fixed), `tracking_sample_count`, `steady_sample_count`, `evaluated_duration_s`, `failure`, `failure_reason`, `success`, and:
- Standing diagnostics: max base XY displacement from settled stance, base roll/pitch, minimum base height, foot contact fractions, foot drift, and (for B2W) net wheel rotation and cumulative absolute wheel travel. Foot drift is not itself a true contact-point slip estimate.

**[RESEARCHER DECISION / PAPER-ALIGNED]** Report each trajectory family separately. Exclude the **entire failed trial** from tracking-error summaries, even if steady-state samples existed before failure; preserve its raw samples, reason and duration. Retain every started trial in the denominator of `success_rate = completed_without_failure / n_trials`, following the paper's survival-success convention. An interrupted/incomplete trial is not a success and contributes no tracking-error samples. Report incomplete counts separately; a complete paper-style comparison requires no incomplete trials.

**[RESEARCHER DECISION, draft.7]** The statistical unit is the **target**: per-target means/RMS first, then **mean ± std over successful targets** per family (`aggregate_summary.json`, sample std `ddof=1`; null for fewer than 2 units). With several seeds, `eval_t1 --seeds` also reports **mean ± std over seeds** of each per-seed family statistic, including success rate (`seeds_summary.json`). Primary orientation summary: geodesic; the matrix-log Frobenius norm (`sqrt(2)` × geodesic) is retained for paper comparison. Empty or all-failed groups have null errors, never zero/NaN. The draft.5 pooled-tick statistics (p95/max) are dropped; `trial_summary.csv` keeps the per-trial maximum position/orientation error over the tracking window (`tracking_*_max`).

## 5. Frozen evaluation tasks and target generators

All numeric values here are **[PROPOSED DEFAULTS]**, **not recovered from the LimX paper**. They form an executable first version, subject to supervisor approval and feasibility testing on the confirmed robot asset. Keep them in a versioned evaluation configuration; do not retune them to make a particular policy appear successful.

### 5.1 Target sampler (draft.7; applies to both suites)

**[RESEARCHER/ADVISOR DECISION, draft.7]** Target **sets are not frozen**. The protocol freezes the **sampler** (`protocol.SAMPLER`: version `t1-joint-space-sampler-1`, parameters, counts); every evaluation run draws its targets from its seed (`targets.sample(model, seed, asset_hashes)`), records them in `seed-<seed>/targets.json` with all rejection counts, and validates them with `protocol.validate_targets`. Default counts: **50 fixed + 10 circles** per seed.

1. Arm joints are drawn **uniformly within the 90 % soft joint limits** (Isaac Lab `soft_joint_pos_limit_factor`), with legs, base and gripper at the nominal state (§ draft.5 block; base x/y at the env origin). FK gives the 6D `link06` target.
2. Filters only: self-clearance ≥ 5 mm and ground clearance ≥ 5 mm on the monitor geometry (collider surfaces, approx_hull for mesh colliders), and EE (`link06` origin) **not below the base** (`base_link` origin z). No workspace box; **no IK for fixed targets**.
3. Uniform in joint space is **not** uniform in task space; this is an accepted choice. The sampler report (`evaluation/diagnostics/sampler_report.py`) gives position ranges and 3-bin histograms of EE height and orientation deviation from the nominal EE orientation.
4. Attempts are capped at `max_attempts_per_target × count` per family; exceeding it fails sampling explicitly (no relaxed counts). Every rejection is logged by category (`self_clearance`, `ground_clearance`, `*_unresolved`, `ee_below_base`, `ik_residual`, `ik_branch_or_velocity`) and detail.

### 5.2 T1-A — Fixed 6D EE pose tracking

**[PAPER]** The hardware experiment tests several random individual target points; the number, workspace and hold time are not specified (§V-C).

**[DECISION, draft.7]** 50 joint-space-sampled 6D targets per seed (§5.1).

- `1 trial = 1 target`, with a fresh nominal-state reset per trial.
- After the 2 s settle, issue the constant target `T^W_E,des` for **`5.0 s`** (was 10 s).
- Tracking window: the whole hold `(0, 5]` s (250 ticks); steady-state window: the last 2 s `(3, 5]` s (100 ticks).
- Never resample the goal within a trial.

### 5.3 T1-B — Spatial circle 6D pose tracking

**[PAPER]** The paper tracks several spatial circular trajectories (§V-C), but does **not** report radius, plane orientation, speed, centre distribution, or orientation profile.

**[DECISION, draft.7]** 10 circles per seed; plane by index (`XY`, `XZ`, `YZ`, `XY`, ...). Each centre is the `link06` position of a joint-space sample (§5.1); the **orientation is constant = that sample's `link06` orientation, i.e. the orientation at circle start**. The whole transition + circle is IK/clearance-checked at 10 Hz (6D IK 1 mm / 0.01 rad, joint margin 0.05 rad, the §5.1 filters at every sample, arm branch/speed guard 0.25 rad per 0.1 s); the accepted IK path is stored. Parameters (unchanged):

- Radius `r = 0.10 m`.
- Frequency `f = 0.05 Hz` (**20 s per revolution**, angular speed `ω = 2πf ≈ 0.314 rad/s`).
- Tracking interval `40.0 s` (**two complete revolutions**).
- Score only the second revolution: circle-phase times `(20.0, 40.0]`, giving 1000 controller ticks at 50 Hz.
- A quintic reference transition of `2.0 s` moves the command from the settled EE pose to the circle's starting pose. For **C² continuity**, set the final translation boundary conditions to match the circle at `t = 0`: position `c + r u`, velocity `r ω v`, and acceleration `−r ω² u`; start with the settled EE position and zero velocity/acceleration. Interpolate orientation smoothly to the fixed `R₀` (zero final angular velocity). **Do not score the transition.** Start the circle phase clock at the end of that transition.

Given the circle centre `c ∈ R³` and orthonormal plane axes `u,v ∈ R³`:

\[
p^W_{E,\mathrm{des}}(t)=c+r\,[u\cos(2\pi ft)+v\sin(2\pi ft)],\quad
R^W_{E,\mathrm{des}}(t)=R_0.
\]

Use `t = 0` at the **end of the transition**. The command is evaluated at the physics/control timestamp; do not derive its phase from rendering frames or elapsed wall-clock time.

A constant orientation still constitutes a **6D pose command**, but only the position component follows a periodic trajectory. **[OPEN]** If the scientific objective requires *dynamic* 6D orientation tracking, add a clearly labelled **T1-B-rot** sub-suite with a bounded, specified SO(3) trajectory and report it separately. Do not claim that this was the original LimX orientation profile.

### 5.4 Sampler manifest (stable across algorithms)

The frozen object is the sampler, not a target list. Machine-readable copy: `protocol.SAMPLER`, recorded in run metadata and in every `targets.json`:

```yaml
benchmark: T1-STANDING-6D-EE
version: 2.0.0
robot_variant: B2W
end_effector_frame: tcp
ee_frame: tcp
tcp_parent: link06
tcp_offset_m: [0.171, 0.0, 0.0]
sampler:
  version: t1-joint-space-sampler-1
  fixed_count: 50
  circle_count: 10
  arm_joint_distribution: uniform within the 90 % soft joint limits; legs/base/gripper nominal
  filters: {self_clearance_min_m: 0.005, ground_clearance_min_m: 0.005, ee_min_height: base_link origin z}
  fixed: {hold_s: 5.0, steady_state_last_s: 2.0, ik: false}
  circle: {radius_m: 0.10, frequency_hz: 0.05, transition_s: 2.0, duration_s: 40.0,
           steady_state_start_s: 20.0, planes: [XY, XZ, YZ], plane_assignment: index mod 3,
           centre: TCP position of a joint-space sample, orientation: constant, sample orientation (circle start),
           reference: rigid map of the link06 reference by the TCP offset}
  max_attempts_per_target: 400
seed: per evaluation run (governs target sampling only)
# Each run's seed-<seed>/targets.json holds the resolved targets, their joint-space samples,
# circle IK paths, digests, attempts and all rejection counts.
```

## 6. Standing constraint, failures, and task boundaries

**[DESIGN]** T1 has **zero commanded base velocity**; leg and arm joints may coordinate to adjust base height/attitude while maintaining ground contact. The robot **must not walk, intentionally roll, or replant its feet**. The entire controller is **not** required to lock base pose rigidly; doing so would eliminate relevant whole-body coordination.

Proposed diagnostics and termination:

| Condition | Handling | Status |
| --- | --- | --- |
| Time budget reached | `timeout`; not a failure | **[PROPOSED]** |
| Base height below safe threshold | `failure: fall` (`fall_height`) | 2.0.0: < 0.274 m for ≥ 5 control steps (shared T1/T2/T3 set) |
| Excessive base tilt | `failure: fall/instability` (`fall_tilt`) | 2.0.0: > 1.0 rad for ≥ 5 control steps (shared set) |
| Standing drift | `failure: standing_drift` | 2.0.0: base x/y > 0.30 m from the position at the first control step after reset |
| Base/EE contact with ground or forbidden object | `failure: collision` | **[PAPER-INSPIRED]**; distinguish allowed foot-ground contacts |
| NaN/Inf state, invalid quaternion or broken simulation | `failure: invalid_state` | **[PROPOSED]** |
| Foot lift/stance shift above tolerance | `standing_violation` flag; optionally failure after threshold approved | **[OPEN]** |
| B2W wheel displacement or rotation | `standing_violation` flag; optionally failure after threshold approved | **[OPEN]** |

The W1 paper's ablation failure rules are `projected_gravity_z > -0.1`, base height `< 0.2 m`, and collision involving EE or base (§V-B). These thresholds are **specific to the W1 setup and must not be transferred unchanged** to the B2/B2W + Z1 configuration. Log the raw values so alternative thresholds can be audited.

Recommended first-pass stance checks are `foot contact fraction` and horizontal foot movement from the initial settled contact positions. Suggested diagnostic threshold: `max foot drift = 0.05 m` **[PROPOSED]**, but make hard failure enforcement configurable and pending adviser approval. Foot contact is not equivalent to true no-slip; record both.

**Important separation:** a task's **structural correctness** is not determined by how well a policy tracks a trajectory. A failed tracking rollout may indicate an inadequate controller while the environment, command generator and metric logger are working correctly.

## 7. Simulation and command-loop configuration

- **Simulator:** Isaac Lab on the lab's configured Isaac Sim installation; note that the thesis plan mentions Isaac Sim 5.1. Pin and record the actual versions, asset hash, actuator configuration and physics parameters.
- **Level ground** and no task objects for the core T1 suite. Camera/depth perception is not required.
- **[PROPOSED]** Physics integration `dt = 0.005 s` (200 Hz); command/action period `dt_ctrl = 0.02 s` (50 Hz), with 4 physics substeps per command. **Validate** these settings against the loaded actuator model; they are not specified by the LimX paper as reproducible T1 values.
- Set gravity, solver configuration, foot contact settings, base pose, arm mounting transform, PD gains and torque limits explicitly; store them in the run metadata.
- Start with **nominal dynamics only** and `payload_mass = 0 kg`. Keep domain-randomization hooks but disable them during initial task validation. Later training and robustness evaluation can activate them without changing the frozen nominal T1 protocol.
- Use deterministic target generation and stable simulation seeds where supported. Isaac Lab/GPU physics may not be bitwise deterministic; report tested reproducibility tolerances instead of assuming exact replay.
- For vectorised environments, each instance must maintain independent episode time, target ID, target phase, initial stance pose and failure status.

## 8. Metric and log schema

Required per-step fields (Parquet/CSV; SI units):

```text
run_id, protocol_version, robot_variant, target_id, trial_id, seed,
sim_time_s, episode_time_s, phase, phase_time_s,
base_pos_world_xyz, base_quat_world_wxyz, base_lin_vel_world_xyz,
base_ang_vel_body_xyz, projected_gravity_body_xyz,
ee_pos_world_xyz, ee_quat_world_wxyz,
ee_goal_pos_world_xyz, ee_goal_quat_world_wxyz,
ee_goal_pos_body_xyz, ee_goal_quat_body_wxyz,
position_error_m, orientation_error_geodesic_rad,
orientation_error_paper_frobenius,
joint_pos, joint_vel, action_command, applied_joint_targets,
foot_contact_flags, foot_contact_forces, foot_positions_world_xyz,
base_xy_displacement_m, foot_drift_m, wheel_rotation_rad,
terminated, truncated, failure, failure_reason
```

If a field is inapplicable (e.g., wheels on B2), mark it explicitly as `null`, not a misleading zero. In addition, write:

- `run_config.yaml`: simulator/asset versions, numerical config, robot and joint-name maps, TCP definition, actuator configuration, seed and git commit.
- `targets.json` or `targets.parquet`: immutable **resolved** benchmark target poses and complete trajectory parameters.
- `trial_summary.csv`: per-trial metrics, failures, evaluated duration, and reachability status.
- `aggregate_summary.json`: per-suite statistics with the correct denominator for success rate, plus `n_trials`, `n_failures`, `n_valid_trajectories`.

**[RESEARCHER DECISION]** The first implementation's `success` and `success_rate` mean survival to the normal timeout, following the paper. They do not certify standing compliance or tracking accuracy. Standing violations are reported independently until hard enforcement is approved. The thesis's rough `≤ 0.05 m` / `≤ 0.10 rad` objectives remain future accuracy targets. **[OPEN]** Define a separately named accuracy-success criterion after deciding its steady-state aggregation rule; do not overload paper-style survival success.

### First implementation defaults and limitations

- Task ID: `RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0`. The task has no PPO runner, learned controller, reward shaping or curriculum.
- Asset: existing no-gripper `b2w_z1_rl.usd`; provisional TCP is flange `link06`, zero additional offset. Use the existing nominal leg pose and `ARM_STOW`, and the standalone reach task's implicit standing actuators. Their suitability remains **runtime-unverified**.
- Actions are 22 values: FR/FL/RR/RL leg positions (hip/thigh/calf per leg), FR/FL/RR/RL wheel velocities, then joint1–joint6. Position inputs are nominal offsets in radians, clamped to soft position limits; wheel targets are always zero rad/s. Observation joint fields use this same explicit order. Applied targets are recorded separately.
- **[PROPOSED engineering defaults]** Height failure below 55% of settled nominal base height (spawn height during settling); tilt above 0.5 rad; forbidden-contact threshold 1 N. Foot drift above 5 cm, wheel joint displacement above 0.05 rad, or absent support flags a standing violation. Hard stance enforcement defaults off. These choices are not copied W1 failure thresholds and require stance validation.
- *(Superseded by draft.7: there is no target-preparation stage.)*
- *(Superseded by draft.7 §5.1, kept for history:)* Deterministic target generation uses 512 collision-filtered random arm configurations to estimate workspace. Candidate bounds are the coordinate-wise 10th–90th percentiles; actual sampled bounds are recorded. Fixed targets allocate eight per height band, lateral sectors 3/2/3, with four nominal and four offset orientations per band. Orientation offsets compose in the settled EE frame, with ±10° roll/pitch and ±15° yaw bounds. Circle axes are world XY/XZ/YZ; the middle circle in each plane has an offset orientation. Each target has at most 50 candidates before explicit failure, never a relaxed count/radius.
- Feasibility uses independent USD FK and 6D bounded IK (1 mm / 0.01 rad residual limits, 0.05 rad joint margin). OBBs are only a broad phase; capsule pairs use analytic segment/radius clearance, capsule/convex-box or cooked-hull pairs use a certified convex-distance calculation, and shape/ground clearance uses physical surfaces. Mesh queries require PhysX's cooked convex representation; unavailable or uncertified geometry is reported as **unresolved**, not labelled a collision from its bound. The physical clearance threshold is 5 mm. Circles and transitions are sampled every 0.1 s with a 2.5 rad/s arm branch/speed guard; this does not prove continuous collision avoidance or dynamic trackability. Preserve all accepted paths and rejected/unresolved candidates. No runtime-validated targets are frozen yet.
- Known intentional mount overlap `base_link/link00` and sensor overlap `head_Link/lidar_link`, plus directly joint-connected bodies, are excluded from self-collision checks. Moving-arm interactions with other robot bodies remain checked. Physical self-collision remains as configured in the existing asset; physical-surface monitoring independently reports clearance violations and unresolved queries. Do not alter the assets to clear this gate.
- Evaluation logs physical terminal samples through the recorder's post-step hook **before reset**, computes references at the matching simulation timestamp, and stores all raw phases. Each vectorized instance has its own target, clock and stance reference. Sampled target positions (env frame) are translated only by fixed scene environment origins. `eval_t1 --mode evaluate --seeds S1,S2,... --num-envs N` runs each seed's targets in batches of N (fixed first, then circles), resetting all envs between batches, and writes `seed-<seed>/{targets.json, steps.csv, trial_summary.csv, aggregate_summary.json, completion.json}` plus top-level `seeds_summary.json` and `completion.json`, all before teardown.
- The evaluator copies source and the protocol into each new run directory. It rejects changed asset hashes and target definitions. See `evaluation/T1.md` for entrypoints, validation status and remaining acceptance checks.

**Runtime findings (2026-11-11, draft.7).**
- *Runtime checks* (`diag-rtchecks-ordered-20261111T103935Z-f3388b72`): with `standing_termination` first,
  the nominal state still fails directly after the first `env.reset()` + `sim.forward()` (both envs:
  `collision_contact` + `collision_arm_contact`; contact rows link02 and link06 317 N each, FL_calf 126-143 N,
  FR_calf 63 N, FL_foot 102-108 N). Fault checks therefore did not run. `coordinate_frames` passes
  (18 states, max FK error 0.53 µm / 1.3 µrad). *(Resolved 2026-11-12, below: stale spawn data.)*
- *Sampler* (seeds 1101/1102/1103, 50 + 10 each): fixed acceptance 50/61, 50/58, 50/62; circle 10/207,
  10/201, 10/155 (circle rejections mostly `ik_branch_or_velocity`, then `ik_residual`, self-clearance).
  Seed 1101 drawn on CPU and in the runtime are identical except the approx_hull provenance field.
  Target orientations deviate 46-175 deg (median ~130 deg) from the nominal EE orientation.
- *Evaluation* (`diag-eval7b-seeds-20261111T105609Z-79c354db`, scripted DLS-IK controller, 4 envs):
  about 23 min per seed (about 5 min sampling + 18 min rollout); success fixed 0.787 ± 0.025, circle
  0.733 ± 0.047 over seeds. All failures are contact terminations after 2.2 s (controller motion), none
  at reset. A repeat of seed 1101 (`diag-eval7b-repeat-20261111T120420Z-3b1ee484`) gives byte-identical
  `targets.json`, `steps.csv`, `trial_summary.csv` and `aggregate_summary.json`. Both processes abort
  in teardown after writing `completion.json` (known limitation).
- *Teardown*: `num_envs=2` without stepping aborts 3/3 after `python_atexit`
  (`diag-shutdown-u2-{1,2,3}-*`); 1-env 50-step runs were clean 3/3 (draft.5).

**Runtime findings (2026-11-12, draft.7).**
- *Reset contact resolved* (`diag-rt8b-standing-20261112T113948Z-81632db2`,
  `diag-rt8b-coordinate-20261112T114000Z-854f2198`): right after `gym.make` the joints are 1.461 rad
  from `init_state` (CAD zero pose) and the contact rows are link02/link06 317 N, FL_calf 126-143 N,
  FR_calf 63 N, FL_foot 102-108 N; identical rows remain after `env.reset()`; after one control period only
  the four feet touch the ground (101-121 N) and no arm contact remains. Later resets inside the check show the same
  pattern for the previous case's contacts (fall: thighs/calves up to 743 N; obstacle: link03/04/06).
  With the one-control-period hold, `standing_termination` PASSes all seven cases (fall tilt, fall height,
  physical arm collision with a kinematic cube, invalid cached state, normal timeout, pre-reset terminal
  capture, asynchronous subset reset) and `coordinate_frames` PASSes (18 states, 0.53 µm / 1.3 µrad),
  each in its own process.
- *Steady-state error decomposition* (seed 1101, 38 successful fixed targets, last 2 s;
  `diag-errdecomp-1101-20261112T114140Z-fc68267f`; USD FK; mean over targets): measured 0.121 m /
  0.493 rad (equals the logged error); with the nominal instead of the actual base pose 0.124 m (base
  deviation 3.0 cm / 0.012 rad; base motion does not contribute); with perfect PD tracking (FK of the
  applied targets) 0.100 m; with the unclipped IK command 0.090 m. Mean |applied - actual| per arm joint
  0.081/0.143/0.082/0.030/0.025/0.017 rad (joint1..6). 20/38 targets clip at least one joint at the soft
  limit in the last 2 s (joint4 15, joint6 13, joint5 7, joint3 6; max 0.35 rad). **The dominant term is
  the scripted DLS-IK controller's own residual** (about 0.09 m of 0.12 m): the IK command does not reach the
  target; the actual joints differ from the target's joint-space sample by 0.26-1.29 rad per joint (mean),
  i.e. the controller settles away from the sampled configuration. PD tracking (about 0.02 m) and clipping (about 0.01 m)
  are secondary. Inferred, not proven: large orientation targets (median about 130 deg) with the
  0.04 rad/step, +/-0.2 rad-around-measured DLS update.
- *Articulation audit of `b2w_z1_gripper.usd`* (composed, LoadAll; `diag-artaudit-20261112T114251Z-f2a3aacc`):
  two prims carry `ArticulationRootAPI`: `/Unitree_B2W/base_link` (rigid body; all 36 rigid bodies
  reachable through joints) and `/Unitree_B2W/z1/root_joint` (a `PhysicsFixedJoint`, enabled, body0
  **empty = world**, body1 `z1/world`; not an instance proxy). No rigid body lies outside the
  articulation. `z1/world`: Xform with `PhysicsRigidBodyAPI` + `PhysicsMassAPI` (no CollisionAPI); joints
  `z1/world/FixedJoint` (base_link -> z1/world), `z1/joints/base_static_joint` (z1/world -> link00),
  `z1/root_joint` (world -> z1/world). `b2w_z1_rl.usd` has a single root (base_link) and no root_joint;
  `b2w_z1.usd` has roots on base_link and z1/world. At runtime Isaac Lab binds `base_link`
  (36 bodies, floating base, `is_fixed_base` False) in both T1 and the stock config.
- *Teardown bisection* (2 envs, unstepped open/close, lifecycle files): T1 (draft.7 `u2`, 3/3), stock
  `RobotLab-Isaac-Velocity-Flat-Unitree-B2W-Z1-v0` with the gripper USD (`diag-oc-stockgrip-*`, 3/3)
  and `Isaac-Cartpole-v0` (`diag-oc-cartpole-*`, 3/3) **all** abort with `Fatal Python error:
  PyThreadState_Get` after `app_close_returned` and `python_atexit`, all with `fast_shutdown=False`
  (what every T1 script uses). The developer's entrypoint (robot_lab `rsl_rl/train.py`, task
  `RobotLab-Isaac-RFM-Sphere-Unitree-B2W-Z1-v0`, `b2w_z1_rl.usd`, 2 envs, 1 iteration;
  `diag-train-rfm-1it-20261112T114534Z-468681b7`) exits 0 after `learn_returned`; `app.close()` with Isaac Sim's default
  `fast_shutdown=True` ends the process without Python finalization. T1 with `fast_shutdown=True`
  (`diag-oc-t1-fast-{1,2,3}-*`) exits 0 3/3; the control with `False` (`diag-oc-t1-slow-1-*`) aborts.
  Conclusion: the crash follows `fast_shutdown=False`, not the asset or its second root. **Adopted in
  1.0.0:** all T1 scripts use the default fast shutdown (all artifacts are written before teardown);
  the second root on the gripper USD is real but did not cause this crash (asset finding, no change).
- *validate_t1 end to end* (`diag-validate-full2-20261112T114913Z-5bf2b0e1`, 47.4 min wall): **FAIL 7/8**.
  PASS: robot_physics (30 s hold), coordinate_frames, error_metrics, fixed_pose_generator,
  circle_generator, standing_termination, reproducibility (target set byte-identical; traces max
  difference 0.0 over 32,493 samples). FAIL: metrics_export, "Incomplete target rollout":
  `check_metrics` requires every sampled target to complete without failure, but the scripted IK
  controller fails 12/50 fixed and 3/10 circle targets of seed 1101 by contact termination (identical in
  both repetitions and to `diag-eval7b-seeds-*`). **[OPEN, researcher decision]** whether the check should
  verify failed trials' terminal flags/summary instead of requiring success. Every child aborted in
  teardown (exit -6) after writing `completion.json`; children were judged by that file (decision C).
  The first attempt (`diag-validate-full-20261112T114732Z-6808879a`) stopped at robot_physics on a stale
  validator rule (exactly 22 articulation joints; the gripper asset has jointGripper), fixed to accept the
  22 action joints plus at most jointGripper.

**Runtime findings (2026-11-12, protocol 1.0.0).**
- *validate_t1 end to end* (`diag-validate-v1c-20261112T142731Z-72ac7c0a`; image `sha256:b52a5210…`;
  GPU `GPU-fee1ef35-…`; host wall 3656 s): **PASS 8/8**, all children exit 0 with `fast_shutdown=True`
  (evaluate 2018 s / 1526 s, hold 53 s, runtime checks 12 s each). robot_physics (30 s hold, 1500
  samples); coordinate_frames (18 states, 0.53 µm / 1.3 µrad); error_metrics; fixed_pose_generator (50
  targets, 61 attempts); circle_generator (10 targets, 207 attempts); standing_termination (7/7 cases);
  reproducibility (target set byte-identical, traces max difference 0.0 over 35,937 samples);
  metrics_export (60 trials: 56 completed, 4 terminated, both repetitions).
- *Oracle results, seed 1101* (`runs/evaluate_1/.../aggregate_summary.json`; mean ± std over successful
  targets, ddof 1): fixed 48/50 success (fixed-14, fixed-46: arm contact), steady position 7.79 ± 3.67 cm,
  steady orientation 7.85 ± 4.46 deg; whole-hold position 13.3 ± 6.6 cm, orientation 32.5 ± 9.8 deg.
  Circle 8/10 success (circle-02, circle-08: arm contact), tracking position 9.26 ± 2.56 cm,
  orientation 11.2 ± 2.2 deg. The oracle does not reach the 5 mm threshold; the residual comes from the
  unchanged PD tracking under load and clamping (inferred: the oracle commands the sampled joints exactly). It is a reference controller
  for validation, not a performance baseline.

**Runtime finding (2026-11-08):** the first B2W smoke test initialized but failed at 0.02 s on link00/link02 overlap; independent capsule geometry confirms approximately 12.5 mm penetration. The six-second standing check has not passed, no validated target manifest was generated, and no collision exemption was added. Guide/invisible collider discovery and visual-bound coverage for links without physical colliders were corrected. See `evaluation/T1_SMOKE_2026-11-08.md` for source snapshots, logs, the single-pose FK check and the required asset review. The protocol remains an unaccepted draft.

**Historical review (2026-11-09, superseded geometry direction):** body-relative bounds corrected rotation/gap inflation. A GPU probe recorded assertions for 18 FK states, mapping, selected faults and pre-reset recording, but crashed during shutdown. A diagnostic cylinder layer was investigated but never selected. The researcher's subsequent decision fixes the existing USD as authoritative and prohibits collider replacement. See `evaluation/T1_REVIEW_2026-11-09.md` for historical evidence.

**Physical-surface and dynamics finding (2026-11-09, draft.2):** 64 host tests and seven CPU-only USD SDK tests pass. Analytic capsule surfaces confirm link00/link02 penetration of approximately 12.5 mm, while other overlapping boxes have safe surface clearance. Existing articulation self-collision is disabled; two directed contact views recorded no link00/link02 contacts or normal/friction forces over 6,000 physics steps. A separate diagnostic observed 30 s nominal PD standing without resets, nonfinite states or physical-stop conditions. Geometric violations remained active, and process cleanup hung (exit 137 after stopping only that container), so full runtime acceptance still fails. Eight mesh cooking queries remain unresolved; none was a nominal narrow-phase candidate. **[OPEN]** Resolve the geometric safety gate under the unchanged USD and existing exclusions, obtain actual cooked geometry when needed, and fix native shutdown before a clean acceptance run. No targets or training were started. See `evaluation/T1_COLLISIONS_2026-11-09.md`.

## 9. Implementation modules and non-learning verification

Suggested components (module names are illustrative, not claims about existing repository paths):

```text
T1 standing task
├── RobotAssetAdapter          # names, actuators, TCP and joint ordering
├── StandingResetManager       # nominal stance + settled initial state
├── EEGoalCommandManager       # world pose -> body-frame command
│   ├── FixedPoseGenerator
│   └── SpatialCircleGenerator
├── ActionAdapter              # scripted / future policy actions -> actuators
├── TrackingMetrics            # world-frame pose errors and statistics
├── StandingMonitor            # contacts, foot drift, base displacement
├── EpisodeManager             # timeout vs failure
└── BenchmarkRecorder          # immutable targets + step and trial logs
```

### Acceptance tests: complete these **before** policy training

- [ ] **Asset/actuator audit:** document B2 or B2W, 12 leg / 6 arm joint mapping, optional 4 wheel joints, TCP frame and gripper handling. Reject mismatched action lengths or joint ordering.
- [ ] **Basic physics:** nominal PD hold keeps the robot standing for **30 simulated seconds**, with finite states and no numerical explosion (researcher completion checklist). Investigate any failure; do not weaken the benchmark to mask a broken asset.
- [ ] **FK identity:** `T^W_E = T^W_B T^B_E` agrees with the simulator's named EE frame to numerical tolerance. Unit-test multiple base poses and arm configurations.
- [ ] **World/body invariance:** moving/rotating the base while holding the **same world-frame target** changes `T^B_E,des` as mathematically required. The world-frame target must not drift.
- [ ] **Rotation metric:** identity orientation → 0 rad; 90° relative rotation → π/2 rad geodesic and π/√2 Frobenius matrix-log norm (approximately 2.221 rad in the latter convention).
- [ ] **Position metric:** known `(0.03, 0.04, 0)` metre offset → exactly `0.05 m` within floating-point tolerance.
- [ ] **Fixed targets:** each sampled set has the frozen sampler and counts; target remains constant during each scored episode; reset does not reorder trials.
- [ ] **Circle geometry:** every reference point lies in its declared plane at distance `r` from centre; path closes after `1/f`, with continuous velocity across the phase wrap.
- [ ] **Smooth transition:** reference translation/rotation enters the circle without a command jump; transition samples never enter scored circle metrics.
- [ ] **Feasibility audit:** IK/collision checks pass over all recorded target poses and densely sampled circle paths; failed targets are never silently replaced at runtime.
- [ ] **Standing diagnostics:** only feet may contact the ground; foot drift, contact status, base pose and optional wheel rotation are logged and distinguishable from EE tracking error.
- [ ] **Termination semantics:** manual triggered fall, forbidden collision, NaN and normal time limit produce distinct labels; metrics preserve partial trials.
- [ ] **Reproducibility:** a repeated scripted test with the same seed produces a byte-identical target set and matching metric traces within documented solver tolerance; different seeds produce different target sets.
- [ ] **Metrics export:** validate per-step error calculations against an independent unit-test implementation; summaries contain trial denominators and no unannounced filtering.
- [ ] **No-learning isolation:** all tests can run with simple scripted controllers; no PPO model, RWM, replay buffer or privileged teacher is required.

### T1 completion criterion

`evaluation/validate_t1.py` implements the researcher's eight-group completion
checklist. It produces `validation_report.json`, the seed-1101 `targets.json` sampled
by the first evaluation repetition, `trajectories.csv`, `metrics.csv`, `config.yaml` and
`trial_summary.csv`, retaining the underlying raw artifacts. Overall `PASS`
requires eight executed, passing groups and finished child processes (`completion.json`; draft.7 decision C); the
evaluation children use the oracle controller. Status: **PASS 8/8** (`diag-validate-v1c-20261112T142731Z-72ac7c0a`); blocked
or missing checks cannot pass. `--cpu-only` deliberately reports incomplete
validation. It introduces recorded **[PROPOSED]** repeatability tolerances of
`1e-4 m` / `1e-4 rad`; target/reference comparisons use `1e-9` in stored units.
These do not change the benchmark's error definitions or accuracy objectives.
See `evaluation/T1.md` for commands, evidence semantics and pending runtime status.

**T1 environment is complete** when all structural/unit/integration checks above pass, the target sampler is frozen, and a reproducible scripted smoke-test run produces valid logs and metrics. **Reaching the 5 cm / 0.1 rad tracking goal is *not* required at this stage**; that is evaluated only after T1–T3 exist and model-free policy training begins.

## 10. Relationship to the remaining WP1 task suite

| Task | Command family | Extra constraints | Development dependency |
| --- | --- | --- | --- |
| **T1** Standing 6D EE tracking | World-frame desired EE SE(3) | No commanded locomotion; feet remain supporting | Base task and metric infrastructure |
| **T2** Height-range reaching | Low/high EE goals, later task success | Broader postural adjustment and reachable height range | Reuse T1 SE(3) generator and reachability tools |
| **T3** Walk + EE tracking | Desired base velocity **plus** desired EE pose | Walking and simultaneous EE tracking | Reuse T1 pose tracking metrics, lift standing-only restriction |

Do not hard-code standing-only restrictions deep inside the shared robot, observation or action adapter. Implement standing as an **independent T1 task constraint**, so T2 and T3 can reuse the same infrastructure while activating different commands, terminations and target distributions.

## 11. Freeze checklist and change control

Before marking this document `v1.0`:

1. Record advisor decisions on the **[OPEN]** points and finalise variant, TCP, stance semantics, trajectories, action/control settings and metrics.
2. Freeze the target sampler (`protocol.SAMPLER`) with a source/provenance note for each setting; target sets are drawn per run from its seed.
3. Run the §9 tests and attach reproducibility logs with simulator/asset hashes.
4. Declare protocol changes via semantic versioning: changes to target distributions, error definitions, failure rules or action interfaces require a **major** benchmark-version update and must not be mixed in a single result table without labelling.
5. Keep reward shaping and learning hyperparameters **out** of the frozen benchmark specification; add them to the later **PPO baseline training protocol** after T1–T3 task implementations are complete.

## References

1. K. Jiang, Z. Fu, J. Guo, W. Zhang, H. Chen, *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator*. IEEE Robotics and Automation Letters; arXiv:2412.03012, revised 2025. Especially **§III-A/C, §IV-A, §V-A/B/C, Tables I, III and IV**. https://arxiv.org/abs/2412.03012
2. Project thesis proposal, `Thesis-Definition.txt`, **WP1** and **T1–T3 Benchmark Task Suite**. Internal research planning document; thresholds and deadlines are goals, not experimentally validated findings.

---

**Implementation note for Claude:** Read this protocol before implementing T1. Preserve all source-status labels. Do not invent original LimX trajectory parameters, do not start PPO training, and do not change benchmark targets or metric definitions without updating the protocol and manifest.
