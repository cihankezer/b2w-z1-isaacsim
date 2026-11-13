Research Overview: World-Model-Guided Whole-Body Control
Status: Research plan — objectives, benchmarks, and success criteria below are proposed, not achieved results.
Source: Thesis-Definition.txt (Master's thesis topic definition).
Scope: Joint-level control of a Unitree B2 or B2W quadruped with a Unitree Z1 arm in Isaac Lab.
Planning horizon: Approximately 26 weeks, starting mid-November 2026.

1. Research Question
Can a learned world model of coupled base–arm dynamics improve the sample efficiency, robustness, and payload adaptation of reinforcement-learned whole-body controllers for quadruped manipulators?

The working hypothesis is that a dynamics model trained on robot transitions can predict how arm motion, payload changes, contacts, and shifts in the center of mass affect the quadruped. This model can then provide cheap, informative rollouts for policy optimization and be updated with limited data when dynamics change.

In this project, world model means a learned state-space dynamics simulator, not a video-generation model. It predicts future robot states/observations under actions; the goal is control, not visual scene generation.
Intended contributions
1. A reproducible Isaac Lab whole-body-control (WBC) environment and model-free PPO baseline for B2(W) + Z1.
2. A learned coupled base–arm world model, with multi-step predictions and epistemic uncertainty, suitable for joint-level actions.
3. A comparison of three increasingly model-dependent RL strategies (G1–G3) against model-free PPO.
4. A few-shot dynamics-adaptation pipeline for previously unseen payloads and sim-to-real discrepancies.
5. A shared benchmark and releaseable datasets, with hardware validation if the platform is available.
These are research goals. Novelty relative to the complete literature, quantitative improvements, and hardware feasibility must be established experimentally.
2. Motivation and Research Gap
A quadruped manipulator must coordinate a mobile, contact-rich base with an attached arm. The arm is not dynamically independent of locomotion: arm acceleration and payload mass affect base attitude, balance, ground contacts, and reachable workspace.

Established whole-body controllers often use model-free PPO trained with many physics-simulator transitions. Learned dynamics models such as Robotic World Model (RWM) and RWM-U have demonstrated promising model-based training for robot control, especially legged locomotion; existing model-based loco-manipulation on Spot does not provide the same joint-level leg-and-arm control setting.

The gap targeted here is therefore model-based RL using a jointly learned base–arm dynamics model for low-level quadruped loco-manipulation, with a controlled comparison of training efficiency and adaptation to unseen physical conditions. This is a research positioning statement, not a claim that no related work exists.
3. System and Control Problem
Component	Initial specification
Robot	Unitree B2 or B2W with Z1 arm; exact variant to be confirmed
Simulation	Isaac Lab / Isaac Sim 5.1; existing B2(W) + Z1 USD assets are the starting point
Control	Joint-position targets through PD controllers: 12 leg joints + 6 arm joints = 18 actions
Commands	6D end-effector pose target and base-velocity command
Policy inputs	Joint positions/velocities, base motion, projected gravity, previous actions, and task commands; precise observation schema to be fixed in implementation
Model outputs	Future joint states, base twist/attitude, contact-related signals, and end-effector pose; exact targets to be fixed before dataset collection
Training	rsl_rl for PPO; rsl_rl_rwm and PyTorch as starting points for learned dynamics
Transfer	ROS 2 Jazzy and physical B2(W) + Z1 if validated and available


Baseline controller
Train a unified PPO controller to track an end-effector target while preserving locomotion and stability. Candidate reward components include:
- End-effector position and orientation tracking.
- Base-velocity command tracking.
- Base stability, safe body posture, and fall avoidance.
- Smooth actions, energy/effort regularization, and joint-limit avoidance.
Use domain randomization over payload mass, friction, link mass/inertia, motor strength, and latency. The source proposes a privileged teacher/deployable student setup; implementation details must be verified against the actual environment and deployment constraints.

Initial target for T1: end-effector position error ≤ 5 cm, orientation error ≤ 0.1 rad on fixed targets, and no falls on flat ground. These are goals, not existing baseline measurements.
4. Research Architecture
B2(W) + Z1 Isaac Lab task
          |
          v
Model-free PPO baseline (real simulator rollouts)
          |
          +----> D1: transitions across commands, actions, payloads, dynamics
                          |
                          v
             Coupled base-arm world model
             (RWM-style + uncertainty ensemble)
                          |
        +-----------------+---------------------+
        |                 |                     |
        v                 v                     v
   G1: look-ahead    G2: Dyna mixing     G3: model-based PPO
   guidance          short imagined       in imagination with
   for PPO           + real rollouts      uncertainty penalty
        |                 |                     |
        +-----------------+---------------------+
                          |
                          v
                Simulation evaluation
                          |
           D2 / D3: new-dynamics data
                          |
                          v
             World-model fine-tuning
             + policy adaptation
                          |
                          v
               Sim-to-real evaluation
World-model learning
Learn an approximate transition model from observations/states and actions:
\[
\hat{s}_{t+1} = f_{\theta}(s_t, a_t, h_t),
\]
where h_t represents relevant historical context. The main candidate is RWM, using autoregressive multi-step prediction; an ensemble (RWM-U style) estimates epistemic uncertainty. Train on mixed-policy and randomized-dynamics rollouts to cover both ordinary and disturbed conditions.

Core model evaluation:
- Open-loop autoregressive predictions for 1, 10, 50, and 100 steps.
- Base orientation and velocity error; joint-state error; end-effector position/orientation error; contact-related prediction quality.
- Seen vs. held-out payloads and dynamics conditions.
- Correlation/calibration between ensemble uncertainty and actual prediction error.
- Model rollout speed and compute cost compared with Isaac Lab.
For sequence models, preserve ordered trajectory windows, reset/termination boundaries, action timing, and the distinction between simulator-only privileged state and deployable observations.
5. World-Model-Guided RL Strategies
ID	Strategy	Use of learned dynamics	Primary hypothesis
G1	Look-ahead guidance	Predict falls, instability, or tracking errors over a short horizon and add a training-time penalty to PPO running in Isaac Lab	Safer training / fewer falls and joint-limit violations
G2	Dyna-style mixing	Combine actual Isaac Lab transitions with short imagined rollouts, varying rollout horizon and mixing ratio	Reach PPO-level accuracy with fewer real simulator interactions
G3	Imagination-based optimization	Optimize a policy inside the learned model using MBPO-PPO, with a penalty for uncertain predictions (RWM-U/MOPO-PPO style)	Train and adapt policies with much less high-fidelity interaction


Important implementation distinction: G1, G2, and G3 are separate experimental conditions, not interchangeable names for the same algorithm. For G2, PPO's on-policy assumptions must be handled explicitly: imagined trajectories should be generated by the current policy (or a clearly documented compatible update scheme), and stale replay should not be naively treated as on-policy PPO data.

Model exploitation risk: Policies may discover behavior that scores well only because the learned dynamics are wrong. Start with short, grounded imagination horizons; monitor uncertainty, model-vs-simulator return gaps, and performance after transferring each learned policy back to Isaac Lab. If long rollouts drift, shorten the horizon or fall back to G2; a residual learned correction to a physics model is a secondary fallback.
6. Datasets and Data Splits
ID	Description	Proposed size / setting	Purpose
D1	Isaac Lab transitions from PPO and exploratory/noisy policies with randomized dynamics and 0–2 kg payloads	10^7–10^8 transitions	World-model training and model-guided RL
D2	Held-out, perturbed simulation ('pseudo-real'): unfamiliar payloads, arm friction changes, latency shifts	1, 5, 15 minutes of new data per condition	Out-of-distribution testing and fast adaptation
D3	Real-robot logs: joint states, IMU, commands, estimated end-effector pose	A few minutes per payload, if hardware available	Model fine-tuning and sim-to-real evaluation
D4	Optional retargeted motion data from a related project	As available	Broader motion-state coverage


Data collection and evaluation must preserve a distinction between training dynamics, model validation dynamics, and held-out adaptation/test dynamics. The exact payload split and random seeds must be recorded before model comparison to prevent leakage. D2 should be assessed both before adaptation and after adaptation.

Minimum transition metadata: episode and step IDs; state/observation; action; next state/observation; command; reward and termination; timing/control frequency; payload and domain-randomization parameters (if available); policy identifier; simulator/version; random seed. This metadata schema is an implementation recommendation, not a finalized specification in the topic definition.
7. Benchmark Suite
Task	Description	Main metrics	Related reference
T1	6D end-effector pose tracking while standing: fixed targets and circular trajectories	Position error (cm), orientation error (rad), fall rate	LimX W1 whole-body pose tracking
T2	Reach targets from ground level to approximately 1 m	Success rate, final pose error, base height/tilt	KAIST dynamic pick-and-place
T3	Walk under base-velocity commands while tracking an end-effector target	End-effector tracking, velocity error, falls	Deep Whole-Body Control; VBC
T4	Carry and adapt to unfamiliar payloads in the proposed 0–2 kg range	Tracking error vs. amount of new data; recovery/adaptation time	Thesis-specific adaptation task
T5	Reach while recovering from pushes and other perturbations	Falls, recovery time, tracking degradation	Standard robustness testing


Reproduction / transfer checks:
- Sanity check: reproduce a published RWM result on Isaac Lab's ANYmal D velocity-tracking task before extending to B2(W) + Z1.
- Public cross-check: evaluate a compatible version of the methodology on Isaac Lab's Isaac-Tracking-LocoManip-Digit-v0 task, if feasible.
The names of the reference papers do not by themselves define identical commands, tolerances, coordinate conventions, or test distributions. Exact evaluation protocols must be specified and versioned before reporting cross-paper comparisons.
8. Baselines, Ablations, and Reporting
Required comparison groups
1. PPO: strong model-free WBC with domain randomization (main reference).
2. RWM-based training: plain learned-model optimization without additional uncertainty-aware guidance, when feasible.
3. G1, G2, G3: isolated proposed variants on the same environment and benchmark.
4. One-step probabilistic ensemble / MLP: simple dynamics prediction reference.
5. Adaptation baselines: robust PPO without adaptation, PPO retraining under the new dynamics, and an RMA-style online estimator if implementable.
Secondary / conditional references: RWM-U, DreamerV3, TD-MPC2; optional NeRD or a residual dynamics model. Their implementation is not a prerequisite for the minimum viable thesis comparison, unless explicitly prioritized later.
Controlled ablations
- World model with vs. without arm state, arm-action history, and payload variation.
- Ensemble uncertainty on/off; uncertainty penalty coefficient.
- Imagination horizon, G2 mixing fraction, and model retraining frequency.
- Seen vs. unseen payloads; frozen vs. fine-tuned world model.
- Teacher/privileged inputs vs. deployable inputs, where applicable.
Metrics and fair comparison
Report tracking quality, stability and safety, environment sample efficiency, wall-clock/GPU cost, world-model error, and adaptation speed. Use common evaluation commands, simulator settings, train/test splits, and termination rules across policies. For primary claims, use at least three seeds, show dispersion, and retain learning curves rather than only final checkpoints.

Distinguish carefully between:
- Isaac Lab interactions vs. imagined model transitions.
- Total compute cost vs. simulator sample count.
- Offline dataset size vs. new dynamics-adaptation data.
- Prediction accuracy vs. downstream control performance.
An accurate model is not by itself evidence of improved policy learning; policy comparisons must run in the same evaluation simulator/hardware.
9. Proposed Success Criteria
Milestone	Target from the thesis definition	Status
WBC baseline	T1 errors ≤ 5 cm position and ≤ 0.1 rad orientation; no flat-ground falls	Target, not verified
World model	Better 50-step predictions than a one-step baseline; meaningful uncertainty on unseen payloads	Target, not verified
Sample efficiency (G2)	At least 2× fewer Isaac Lab environment steps to comparable PPO tracking accuracy	Target, not verified
Training safety (G1)	Fewer falls and joint-limit violations	Target, not verified
Adaptation (G3)	Recover at least 80% of nominal tracking accuracy with ≤ 15 min of data	Target, not verified
Hardware transfer	Real T1/T2 tracking error within about 2× simulation error	Conditional target


Before experiments, operationalize ambiguous terms such as accuracy recovered, fall, target error, and adaptation time in the benchmark configuration and analysis code.
10. Work Plan and Decision Gates
Work package	Weeks	Deliverable / decision
WP1 — Environment + PPO	1–4	Working Isaac Lab task, PPO baseline, T1–T3; commit to B2 or B2W
WP2 — Dataset + world-model comparison	4–9	RWM sanity reproduction; D1; RWM / ensemble baseline; 1/10/50/100-step evaluation
WP3 — G1/G2	9–15	Sample-efficiency and fall comparisons on T1–T3; mid-February paper go/no-go
WP4 — G3 + adaptation	15–20	D2 adaptation curves at 1/5/15 minutes; T4/T5 results
WP5 — Transfer and final evaluation	20–24	Hardware tests if available, otherwise second-simulator robustness cross-check
WP6 — Writing + release	23–26	Thesis, reproducible code/configs, benchmark protocol, releasable datasets/models


The original plan proposes an optional IROS 2027 simulation paper and later CoRL 2027 or ICRA 2028 submission. Those are aspirations, not committed deliverables or verified submission dates.
Initial execution priorities
1. Confirm access to the robot USD assets, existing simulation code/Docker environment, and training hardware; verify whether any RL task already exists.
2. Choose B2 vs. B2W and document actual robot/arm payload limits and actuation behavior.
3. Fix observation/action conventions, control rates, coordinate frames, rewards, train/test splits, and benchmark success definitions.
4. Reproduce an RWM sanity benchmark and establish a tracked PPO baseline before model-guided modifications.
5. Implement a deterministic dataset recorder and multi-step evaluation harness before training large world-model candidates.
11. Risks and Fallbacks
Risk	Response
B2W wheel control destabilizes learning	Start with legged B2; add wheels after the base task is stable
Insufficient model accuracy over long rollouts	Use shorter imagination horizons and more frequent resets from true transitions
Policy exploits incorrect model dynamics	Use uncertainty penalties, model-vs-simulator checks, and conservative G2 mixing
Unknown dynamics cannot be adapted with little data	Report the true data/performance curve; compare against robust PPO and the non-adapted model
Real robot not available or unsafe to deploy	Validate across a second simulator (e.g. MuJoCo); do not label this as hardware sim-to-real success
Full baseline zoo exceeds thesis budget	Prioritize PPO, one-step ensemble, RWM, and G1/G2/G3; clearly mark optional external baselines


12. References and Code Starting Points
Core methodology
- Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics — RWM, long-horizon learned state-space simulation and imagined policy training.
- Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots — RWM-U and MOPO-PPO.
- When to Trust Your Model: Model-Based Policy Optimization — short model rollouts and model-error control (MBPO).
- Mastering Diverse Control Tasks through World Models — DreamerV3, imagination-based actor–critic.
- Learning More With Less: Sample-Efficient Model-Based RL for Loco-Manipulation — learned dynamics on Spot with an arm; important related work, but a different control interface.
Whole-body-control and benchmark context
- Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion.
- Visual Whole-Body Control for Legged Loco-Manipulation (VBC).
- Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator (LimX W1).
- Learning Dynamic Pick-and-Place for a Legged Manipulator (KAIST).
Software starting points (as specified in the topic definition)
- Isaac Lab
- rsl_rl
- rsl_rl_rwm
See the original Thesis-Definition.txt for additional candidate papers (QWM, TD-MPC2, NeRD, MLM) and full project context. Paper names here are references to review, not a claim that the repo already reproduces them.
Notes for Researchers and Coding Agents
- Treat this document as the research specification, not evidence of completed experiments or an exact software API contract.
- Preserve T1–T5, D1–D4, G1–G3, and WP1–WP6 identifiers in code, configs, figures, and experiment logs where useful.
- Do not assume paths, modules, checkpoints, trained models, hardware availability, or installed dependencies exist merely because they are mentioned here; inspect the repo first.
- Keep the simulator, baseline, data recorder, world model, guided RL, and evaluation interfaces modular so each research claim can be tested independently.
- Version configs and random seeds. Save both raw measurements and scripts needed to regenerate plots/tables.
- Do not silently change benchmark definitions, payload splits, or training budgets between baselines. Record deviations from this plan explicitly.