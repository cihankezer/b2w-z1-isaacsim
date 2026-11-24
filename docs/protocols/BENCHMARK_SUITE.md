# Benchmark Suite v1 (T1, T2, T3)

Release tag: `benchmark-suite-v1.1` (see the version history at the end). Robot: Unitree B2W + Z1 with gripper (`robots_usd/b2w_z1_gripper.usd`).
Stack: Isaac Sim 5.1, Isaac Lab 2.3.2, robot_lab 2.3.2, image `sha256:b52a5210…`.
Detailed rules are in the task protocols: [TASK_T1.md](TASK_T1.md), [T2_reach_low_high_protocol.md](T2_reach_low_high_protocol.md),
[T3_walk_and_reach_protocol.md](T3_walk_and_reach_protocol.md). This page is the summary.

## 1. What all tasks share

**Action (22 dims).** 12 leg and 6 arm channels are joint position offsets (rad) from the nominal pose,
clamped to the soft limits. 4 wheel channels are wheel velocity targets a_w · 10 rad/s. Zero action means
"hold the nominal pose, wheels at 0 rad/s". Control at 50 Hz (physics 200 Hz, decimation 4).

**End-effector frame.** The TCP: `link06` + 0.171 m along the tool axis. All targets and errors use the TCP.
Position error in m (reported in cm), orientation error = geodesic angle (rad). Success thresholds 5 cm, 0.1 rad.

**Policy observation (95 dims, the same in the T1, T2 and T3 environments; one module, `standing_ee/observations.py`).** In this order:
joint positions (22), joint velocities (22), base angular velocity (3), projected gravity (3), previous action (22),
base-velocity goal (3: vx*, vy*, ωz*; zero in T1/T2), TCP goal in the base frame (7: position + quaternion wxyz),
measured TCP pose in the base frame (7), TCP pose error (6). No task-only dimensions.

**Termination set (all tasks, training and evaluation).**

| ID | Condition | Class |
|---|---|---|
| F1 | Trunk or leg body touches the terrain (> 1 N) | fall |
| F2 | Base tilt > 1.0 rad for 5 control steps | fall |
| F3 | Base height < 0.274 m for 5 control steps | fall |
| S1 | Arm or gripper touches the terrain | safety |
| S2 | Arm or gripper touches the robot's trunk, legs or another arm link | safety |
| S3 | Any non-wheel collider < 5 mm above the ground | safety |
| S4 | Trunk or leg bodies touch each other (body self-contact) | safety |
| D | Base drifts > 0.30 m while the velocity command is zero | drift |
| T | Time-out | — |

## 2. Training and evaluation environments

The **T3 environment is the only training environment**. The T1 and T2 environments are for evaluation only.
Three training presets select the command distribution; everything else (robot, action, observation,
terminations, physics) is identical.

| Preset | Gym ID | Velocity command | EE goal modes |
|---|---|---|---|
| `train-t1` | `RobotLab-Isaac-Suite-Train-T1-Unitree-B2W-Z1-v0` | always zero (standing fraction 1.0) | M2 only (T1-style fixed points); from a goal without an M2 successor it steps to any M2 goal |
| `train-t1t2` | `RobotLab-Isaac-Suite-Train-T1T2-Unitree-B2W-Z1-v0` | always zero (standing fraction 1.0) | M1 + M2 + M3, weights 0.50 / 0.25 / 0.25 |
| `train-t3` | `RobotLab-Isaac-Suite-Train-T3-Unitree-B2W-Z1-v0` | full T3 range, 20 % standing | M1 + M2 + M3, weights 0.50 / 0.25 / 0.25 |

M1 = random reach, M2 = T1 rule, M3 = T2 map postures; all goals come from the collision-checked goal pool.
Accepted mode mix measured at 4 096 envs (M1 / M2 / M3): `train-t1` 0 / 100 / 0 (7 488 of 7 500 M2 goals usable),
`train-t1t2` 52.5 / 24.4 / 23.1, `train-t3` 52.6 / 24.3 / 23.0. The mix leans to M1 because a draw whose mode has
no stored successor at the current goal is redrawn.
The standing fraction and the mode weights are config parameters (`base_velocity.rel_standing_envs`,
`tcp_target.mode_weights`).

| Task | Evaluated in | Script |
|---|---|---|
| T1 (6D target while standing) | T1 environment | `evaluation/validate_t1.py`, `evaluation/eval_t1.py` |
| T2 (low and high reach while standing) | T2 evaluation environment | `evaluation/eval_t2.py` |
| T3 (6D target while following a base velocity) | T3 evaluation environment | `evaluation/eval_t3.py` |

## 3. Versions and pinned artifacts

| Task | Version | Status | Pinned artifacts (SHA-256) |
|---|---|---|---|
| T1 | 2.1.0 | final | Targets `4ffb525c99a62131f5c778f28ecc9d378c706dcdf277ce244a86c7288037439a` (ee_frame tcp; identical to the 2.0.0 file `908f0ad6…3c3d` except `protocol_version`, which T3's S1 is built from); reference run `diag-reg-t1-v21-20261124T130420Z-949b131f` (trials byte-identical to 2.0.0) |
| T2 | 1.1.0 | final | Reachability map `22b9e665baa11bced9832107ca17c154ba57751fb55f54c3dbf67d9e1b865404`; evaluation set `6260df589d63db4d8633959e2244e9ec0320ab4c7a716bab3279b8a7a4f5937e` |
| T3 | v0.9 (code 0.9.0) | final except open item 12 | Training pool `6e4a18f7fc531e1cb9799efa99c0dee9f63ff9e222a766f0a3cc5118fc270c70`; evaluation pool `86b45c40638009b40250d92595b6f3697bfe55543c7bebf365d59f923001d911`; evaluation set v5 `0a7dd22d8d7bf3fcee335b70e658d12b990d52399c0c8f07c0fd0bf6cd0d4151` |

Large files are not in git. They live under `repos/robot_lab/logs/` and are listed with path, size and SHA-256 in
`walk_reach/data/artifacts_manifest.json`. The T2 and T3 loaders check the pinned digest and refuse a different file;
T1 regenerates its targets deterministically from seed 1101 (reproducibility is checked by `validate_t1.py`).

## 4. Rules for running evaluations

- **One fresh process per evaluation.** Two evaluations in separate processes are byte-identical. A second pass in
  the same process after `env.reset` is not (PhysX keeps solver state; differences ~1e-4).
- Deterministic nominal reset; fixed evaluation sets; report the evaluation-set version and digest with every result.
- One allocated GPU per job; run through `evaluation/diagnostics/launch.sh`, which stores the command, source
  snapshot, git state and image ID with the outputs.

## 5. Open items

| Item | Status |
|---|---|
| Wheel yaw authority (T3 open item 12) | In the scripted test, open-loop wheel speeds produce no yaw. Cause unknown; waiting for the original developer. Does not block this release. |
| A10 (T1 vs T3 consistency) | Blocked until a trained policy exists. |
| E-DR (domain-randomized evaluation) | Deferred until WP1 defines the randomization config. |
| Payload ξ (T3 open item 13) | Added with T4 as a separate privileged observation group; the policy observation layout does not change. |

## 6. Version history

| Release | Changes |
|---|---|
| `benchmark-suite-v1` | Initial release: T1 2.0.0, T2 1.1.0, T3 v0.8; training presets. |
| `benchmark-suite-v1.1` | T1 2.1.0: the T1 environment uses the shared 95-dim observation (v1 had the 89-dim T1 layout). `train-t1`: a goal without an M2 successor steps to any M2 goal instead of being masked (31.5 % of the M2 goals were lost in v1). T3 v0.9. |

**Evaluate policies with v1.1 or later.** In v1 the T1 environment cannot evaluate suite-trained policies, because its
observation does not match the training observation.
