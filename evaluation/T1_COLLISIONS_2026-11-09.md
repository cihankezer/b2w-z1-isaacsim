# T1 physical surfaces and nominal standing — 2026-11-09

**Outcome:** actual USD capsule penetrations are confirmed. The link00–link02
pair produced no measured physical contact forces with the existing filtering.
Nominal PD standing was observed for 30 simulated seconds, but geometric safety
failed and simulator cleanup hung. **This is not a clean runtime-validation pass
or an 8/8 T1 completion.** No benchmark targets or training results are claimed.

## Authoritative model and implementation

The existing `robots_usd/b2w_z1_rl.usd` and its referenced assets are unchanged.
All 54 files in the recorded asset manifest retain their hashes. Root USD SHA256:
`89a3adaf7a54387e60227b2ed34cdab5a41536af4dabce4c9fc0e2ea1e7b35f0`.
No capsule/cylinder replacement layer was used. No exclusions were added and no
physical filtering, mass, joint or actuator settings were repaired to pass tests.
The earlier cylinder diagnostic is superseded by the researcher's decision.

Changes verified in code:

- `standing_ee/clearance.py`: exact capsule centreline/radius queries; certified
  convex-distance queries for capsule/box or cooked-hull candidates; physical
  shape/ground clearance; explicit unresolved results for unsupported geometry.
- `standing_ee/kinematics.py`: body AABBs and per-collider OBBs select candidates
  only. Surface results decide penetration, sub-5 mm clearance, or a broad-phase
  false positive. Visual-only links acquire no invented physical collision shape.
  Existing directly connected, base_link/link00 and head_Link/lidar_link
  exclusions remain unchanged. Filtering configuration is recorded separately.
- `standing_ee/runtime.py`, `eval_t1.py`: export classified geometric evidence
  and geometry/filtering provenance. Physical-stop rules remain unchanged.
- `standing_ee/protocol.py`, `targets.py`, `validate_t1.py`: draft.2 identifies
  the physical-surface rule; target feasibility and optional metric export carry
  the new geometric evidence. The eight-check validator still fails closed.
- `diagnose_t1_standing.py`: separate nominal PD diagnostic with two directed,
  one-body filtered contact views, raw contact/friction buffers at 200 Hz and
  stability/clearance records at 50 Hz. It steps below the environment's automatic
  reset path to observe standing while preserving geometric failure reports.
  Physical faults still stop collection. It is never full T1 acceptance.

The physical threshold remains **0.005 m**; numerical classification tolerance is
`1e-8 m`. Negative clearance denotes intersecting solids. Nonnegative clearance
below the threshold denotes a safety-clearance violation without necessarily
penetration. Candidates at or above the threshold are not rejected. For
capsule/convex queries whose centreline enters the solid, a negative value is a
penetration lower bound, explicitly marked as not an exact penetration depth.
The measured pairs below are capsule/capsule queries.

## Geometry findings verified against the USD

Here `mesh_0`/`mesh_1` identifies the USD capsule's collision group. Distances are
minimum surface clearances among recorded broad-phase candidates over the 30 s
trace; this is a discrete audit, not continuous collision detection.

| Collider pair | Minimum clearance | Classification and consequence |
| --- | ---: | --- |
| link00 mesh_0 – link02 mesh_0 | −12.5001 mm | Actual penetration; present at all 1,500 control samples |
| link04 mesh_0 – link06 mesh_0 | −17.8001 mm | Actual penetration; present at all 1,500 control samples |
| link02 mesh_1 – link04 mesh_0 | −20.7302 mm | Actual penetration during the hold; 1,491 penetrating samples |
| link02 mesh_0 – link06 mesh_0 | +4.1119 mm | Genuine sub-5 mm safety violation, without penetration, at two samples |
| link00 mesh_0 – link02 mesh_1 | +16.9261 mm | Conservative-box false positive; all 1,500 candidates accepted geometrically |
| link02 mesh_0 – link04 mesh_0 | +5.7470 mm | Conservative-box false positive; 1,496 candidates accepted geometrically |

The two link02–link04 mesh_1/mesh_0 capsules also had four positive but sub-5 mm
samples during the motion. The raw trace distinguishes these from penetration.
At the first 0.02 s sample the link00/link02 penetrating capsule clearance was
−12.50005 mm, whereas link00's clearance to link02's other capsule was
+19.07656 mm. A pair can therefore contain both a real intersection and a
separate harmless box candidate; the latter is never counted as a collision.

Independent USD FK/surface recomputation at physics ticks 4, 3,000 and 6,000
matched 13 recorded candidate distances within **1.40e-7 m**. The separate actual
asset regression confirms the approximately 12.5 mm mount penetration and that
neither link00/link02 nor link04/link06 has been added to geometric exclusions.

## Physical filtering and contact-force findings

Both the default USD articulation and the composed runtime articulation have
`physxArticulation:enabledSelfCollisions = false`. No explicit USD FilteredPairs
relationships were found. Existing joint-connected-body exclusions also remain.
The global self-collision setting physically filters link00/link02; it does not
make their authored surfaces disjoint and is not silently converted into a new
geometric safety exemption.

Two one-body contact views resolved exactly one target each: link00 → link02 and
link02 → link00. Across **all 6,000 physics ticks**, both reported:

- zero raw normal contact count;
- zero sum of normal-force magnitudes and zero normal-force matrix;
- zero sum of friction-force magnitudes.

Thus no unintended physical contact force from this pair was observed in the
nominal configuration. Raw counts and summed magnitudes avoid confusing opposing
forces with a zero net force. Foot sensors recorded a peak of **711.11 N**,
providing a positive control for contact reporting, and arm net-contact forces
were zero. The other penetrating arm pairs did not receive separate raw pair
views, so the direct pair-force finding applies specifically to link00/link02.
Self-collision was never enabled as a counterfactual experiment.

## Nominal standing dynamics

One B2W+Z1 environment, seed 1101; zero nominal-offset actions, 22-joint mapping,
existing implicit PD drives, zero wheel-velocity targets. Physics dt 0.005 s;
control dt 0.02 s. The recorded diagnostics show:

| Check | Observation |
| --- | --- |
| Duration and clocks | 30 simulated seconds; 6,000 physics / 1,500 control samples, complete monotonic clocks |
| Finite states | All physics samples finite; recorded joint/base arrays independently checked |
| Falls / physical-stop reasons / automatic resets | None / none / none |
| Standing compliance after 2 s settling | Clear at every control sample |
| Maximum control-sampled base tilt | 0.020977 rad |
| Minimum control-sampled base height | 0.597423 m |
| Maximum logged foot drift | 0.00009997 m (approximately 0.10 mm) |
| Maximum logged wheel displacement | 0.001414 rad |
| Geometric safety | Failed; the surface violations above remained logged |

This separates observed standing behavior from the geometric gate. It does not
validate target tracking, prove continuous clearance, or change normal task
termination. There was no learning, checkpoint replacement or imagined data.

## Process outcome and unresolved issues

Two preliminary 1 s contact diagnostics completed their data assertions but
segfaulted during native cleanup. In the 30 s diagnostic an explicit timeline
stop was tried before environment cleanup; all data and the diagnosis were saved,
but cleanup remained unresponsive for over a minute. Only that uniquely named
container was stopped; its exit code was **137**. The current diagnostic removes
that explicit-stop attempt and retains normal environment/application cleanup.
It does not use a fast exit to manufacture success. No further clean 30 s run
has been demonstrated with the current cleanup path.

The historical `diagnosis.json` field `standing_dynamics_passed: true` describes
the saved pre-cleanup dynamics assertions only. The host outcome and independent
analysis explicitly reject a formal pass. Current source names this field
`standing_dynamics_checks_satisfied` to avoid implying process acceptance.
`full_t1_passed` remains false. The host's `native_shutdown_failure: false` only
means its segfault-text detector did not fire; exit 137 and the recorded cleanup
hang are still process failures.

Eight leg/wheel mesh colliders did not return a cooked PhysX representation via
the installed query API. They are recorded as **unresolved**, never approximated
by visual/raw mesh hulls or labelled penetrating from their boxes. None became a
narrow-phase candidate in this nominal trace. Future arm/mesh candidates cannot
be certified until the actual cooked geometry is available. USD also emitted
pre-existing unresolved payload/reference warnings, retained in the audit log;
no paths or assets were rewritten to suppress them.

**Open decisions/work before full acceptance:** resolve how the retained
geometric safety gate can be satisfied under the authoritative USD and existing
exclusions; obtain cooked geometry for any relevant mesh candidates; diagnose
native cleanup and obtain a clean process. The current implementation keeps
these failures visible. It does not lower 5 mm, add pair exemptions or repair
assets. Target freezing and the complete eight-check suite remain pending.

## Validation and reproducibility

- Host: **64 passed, seven skipped**, using
  `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_t1.py tests/test_t1_clearance.py tests/test_t1_usd.py tests/test_validate_t1.py`.
  The skips require the USD SDK. Fixtures that stub simulator workers are only
  validator-plumbing tests.
- Existing Isaac image's CPU-only USD SDK: **seven passed**; independent saved
  trace FK/surface readback also passed. No SimulationApp or GPU was used for
  these SDK regressions.
- The 30 s dynamics assertions were independently recomputed from saved CSVs,
  including clock lengths, finiteness, contact values, geometric classifications,
  post-settling stance flags and unchanged asset hashes. Process acceptance failed.

Persistent artifacts under `repos/robot_lab/logs/t1/`:

- `surfaces-20261109T192100Z-9cd5a8d4/`: original USD audit, analytic surface audit,
  CPU launch/SDK logs; `final_readback.py`, `final_trace_fk_check.json`,
  `final_cpu_checks.log`, `final_cpu_result.json` preserve the final regressions.
- `standing-surfaces-20261109T193600Z-e01a9c30/` and
  `standing-surfaces-20261109T193750Z-80190cdc/`: preliminary 1 s diagnostics,
  native-shutdown failures and the cooking-query investigation.
- `standing-surfaces-20261109T194032Z-acd56018/`: full 30 s trace;
  `artifacts/physics.csv`, `control.csv`, `diagnosis.json`, `interface.json`,
  `config.yaml` and source snapshot; host `launch.json`, `preflight.json`,
  `stdout.log`, `host_result.json`, `shutdown_observation.json` and
  `independent_analysis.json`.

Runtime identity: project commit `1e3069a2a2345841761dc9649d1d811a6d2e0b64`
plus retained uncommitted source; Isaac Lab v2.3.2 / robot_lab v2.3.2 with overlay;
existing image `isaac-lab:2.3.2`, ID
`sha256:b52a521085e561319e84e79e157b8cd8846e1a5089fb946d8e3909f4a6c0f8f4`;
configured GPU `GPU-fee1ef35-de0d-198e-6c95-df5a9940cc06` (RTX 5080).
Each launch used a unique name, read-only repository and separate writable
run/cache mounts. After cleanup only the pre-existing `nav2`/`moveit` containers
remained running, and GPU memory returned to 802 MiB. They were not modified.
