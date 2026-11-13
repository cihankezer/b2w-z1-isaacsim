# AGENTS.md — World-Model-Guided WBC Thesis

> Persistent instructions and project context for Codex in `b2w-z1-isaacsim`.
> Read this file before making a plan or editing. **Current source code and validated runtime behavior override this document.**

## 1. Repository baseline: inspected source, not runtime-verified

A previous Codex source inspection examined `thesis/world-model-rl` at commit **`ec3272e`**. It **did not** run Docker, setup, simulator or GPU training. Reinspect current code before acting. The local dependency checkouts and USD assets were absent in that inspection.

### Existing architecture

- **WBC RL pipeline:** `run.py train` → `isaac/run_train.sh` → Dockerized Isaac Sim/Isaac Lab/`robot_lab` → registered `ManagerBasedRLEnv` → **RSL-RL PPO**.
- **Separate fetch pipeline:** `run.py fetch` → `project/fetch_box.py` → navigation (scripted/Nav2/DualVLN), ETH locomotion controller, MoveIt/IK manipulation. **Not the WBC RL training loop.**
- **Existing:** Isaac Lab task overlay, PPO training configs, shipped WBC checkpoints and evaluation tools.
- **Not yet present as a thesis pipeline:** persistent reset-correct WBC transition dataset, learned coupled world model, imagined-rollout algorithm, model-based PPO integration or a fully automated experiment manifest.

Known WBC task ID:

```text
RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0
```

| Key file / directory                | What to inspect                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| `run.py`                            | Main CLI; training dispatch                                                      |
| `isaac/run_train.sh`                | Docker train launcher, headless defaults, container naming                       |
| `isaac/isaac_env.sh`                | Mounts, GPU selection, caches, container lifecycle/permissions                   |
| `isaac/docker/Dockerfile.isaaclab`  | Isaac Sim/Isaac Lab training image                                               |
| `setup.sh`                          | Dependency cloning, overlay installation, assets, image builds; has side effects |
| `training/robot_lab_overlay/`       | B2W+Z1 task, reward/observation/action/reset and PPO configs                     |
| `training/inspect_task.py`          | Runtime inspection of resolved task interface                                    |
| `project/play_task.py`              | Checkpoint playback                                                              |
| `evaluation/`                       | Reaching, approach, locomotion, stability evaluation                             |
| `training/README.md`                | Warm-start history of shipped policy                                             |
| `repos/IsaacLab`, `repos/robot_lab` | Expected external checkouts; confirm installed versions                          |
| `robots_usd/`                       | Asset definitions/docs; verify actual robot files                                |

WBC task sources are nested under:

```text
training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/
```


## 2. Docker execution policy on the shared workstation

The earlier source inspection found:

- `isaac/docker/Dockerfile.isaaclab` bases WBC training on `nvcr.io/nvidia/isaac-sim:5.1.0`, with **Isaac Lab v2.3.2** and **robot_lab v2.3.2**; example image tag `isaac-lab:2.3.2`.
- `setup.sh` clones dependencies, overlays the local WBC task and can build additional Nav2/MoveIt images. Review all side effects before running it.
- The project uses `isaac/run_train.sh` and `isaac/isaac_env.sh`; **Docker Compose was not found**. Do not create a parallel Compose stack by default.
- WBC training is headless and detaches by default; logs/checkpoints normally persist under `repos/robot_lab/logs/rsl_rl/<experiment>/<timestamp>/`. Confirm other output locations in code.
- Docker bind mounts point to paths on the **remote Docker host**, not the local laptop. Existing source mounts can obscure files baked into the image.
- The inspected container used UID **1234**. Check bind-mount writability and ownership without broad permission changes.

**Mandatory shared-workstation precautions:**

- Verify Docker + NVIDIA GPU availability, image ID, dependency checkouts, `config.yaml` and required WBC USD before running anything.
- Use an **allocated GPU UUID**, not implicit `--gpus all`, unless the machine administrator explicitly permits all GPUs. GPU exposure is **not** a reservation.
- Use **unique container names for every run**. Some existing launchers delete stopped same-name containers, and Nav2/MoveIt launchers can forcibly remove named containers.
- Use project-specific cache, dependency and results paths; verify mount lists before launching. Existing helpers may modify permissions broadly on paths they configure.
- Check host-network ports / ROS domain collisions if launching other services. Container naming does not isolate ports, files, processes or GPUs.
- Do not stop/delete other people's containers, use global Docker cleanup, change shared permissions, overwrite shared caches or alter global Docker settings.
- Do not kill other people's processes on the GPU without asking.
- Persist stdout, checkpoints, resolved configs and artifacts outside ephemeral containers. Prefer detached long-running jobs.

## 3. Reproducibility and reporting contract

For each experiment, preserve an immutable or otherwise fixed source snapshot and a manifest containing:

- Git commit/dirty status; Isaac Lab, robot_lab, RSL-RL and world-model library versions.
- Docker image tag **and digest/ID**, Isaac Sim/PyTorch versions, asset hashes, allocated GPU and relevant machine/runtime info.
- Task ID, full resolved config, reward terms, observation/action ordering, policy/checkpoint provenance, seed, env count, step counts and experiment command.
- Payload and dynamics randomization, dataset/scenario split, evaluation protocol, stdout/stderr, metrics, checkpoints and run ID.
- Separate totals for physics-simulator samples, imagined samples, wall-clock time and world-model compute.

**Never claim a test passed if it was not executed.** Use labels: **verified in code**, **verified at runtime**, **inferred**, or **proposed/unverified**. Benchmarks, numerical targets and publication expectations in this file are *proposals*, not results.

## 4. Codex operating rules

1. At the beginning of each task, check Git status and the current branch. Inspect only files directly relevant to the requested change. Reuse previously verified repository context when still valid. Do not perform repository-wide exploration unless explicitly requested.
2. Before meaningful changes, provide a compact plan: objective, affected modules, validation and remote side effects. Favor minimal diffs and avoid unrelated refactors.
3. **Ask permission before** SSH operations that modify remote files/state, `git push`/commit/merge/rebase, Docker builds/start/stop/removal, setup/install, GPU jobs, destructive operations or long experiments. Read-only local inspection is fine.
4. **Do not assume remote connectivity or credentials**; never use an SSH password/private key in prompts, docs, code or logs. Use the configured local SSH alias when permitted.
5. Preserve baseline PPO and shipped checkpoints; introduce distinct task/runner IDs and new output paths for thesis experiments to avoid accidental overrides.
6. Explicitly distinguish the **existing repository implementation** from the **thesis proposal** wherever they differ (especially 18 vs. 22 actions, B2 vs. B2W, proposed vs. existing environment).
7. Avoid inventing or hard-coding remote hostname/IP/port, GPU UUID, file paths or image tags. If placeholders are unset, report the missing configuration rather than guessing.
8. For every response, concisely summarize what changed, exact files/functions touched, tests run/not run, risks and the next concrete action. Assume the researcher knows motion prediction/RL; explain unfamiliar world-model or VLM concepts when needed.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

**Before implementing:**

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

**When editing existing code:**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

**When your changes create orphans:**

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define clear success criteria before implementation. Run only focused, relevant verification steps. If verification fails, investigate the specific failure without repeating unchanged tests. Stop and report blockers when the required environment or dependencies are unavailable. Do not expand the task scope without approval.

---

## 5. Token-Efficient Codex Operation

- Inspect only task-relevant files; avoid repository-wide scans unless necessary.
- Keep plans and responses concise, proportional to task complexity.
- Reuse valid context rather than repeatedly reading unchanged files.
- Run focused verification; avoid redundant tests and unbounded debugging loops.
- Stop and report blockers if the required runtime or dependencies are unavailable.
- Ask before expanding scope or launching expensive operations.
- Never sacrifice correctness, security, or reproducibility to reduce token usage.

