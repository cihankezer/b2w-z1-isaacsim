"""Validator reporting/analysis tests. Synthetic traces do not validate the robot."""

import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_validate_t1_cpu",ROOT/"evaluation/validate_t1.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
g, p = v.pure_modules()


def fixed():
    return dict(target_id="fixed-test",family="fixed",position_world_xyz=[.3,0,.8],
                quaternion_world_wxyz=[1,0,0,0],hold_s=5.,steady_state_last_s=2.)


def circle():
    return dict(target_id="circle-test",family="circle",plane="XY",u=[1,0,0],v=[0,1,0],
                centre_world_xyz=[.3,0,.8],quaternion_world_wxyz=[1,0,0,0],
                radius_m=.1,frequency_hz=.05,transition_s=2.,duration_s=40.,steady_state_start_s=20.)


def trace(target):
    settled = g.transform([.4,0,.8],[1,0,0,0])
    duration = 7 if target["family"] == "fixed" else 44
    rows = []
    for tick in range(1,round(duration/.02)+1):
        t = tick*.02
        reference,phase,phase_time,scored = g.tcp_reference(target,t,settled,p.TCP,p.TCP_INV)   # T1 2.0.0
        desired = g.pose(reference)
        actual = settled.copy() if t <= 2 else reference.copy()
        if t > 2:
            actual[:3,3] += [.03,.04,0]
            actual[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        actual = g.pose(actual)
        errors = v.independent_errors(actual[:3],actual[3:],desired[:3],desired[3:])
        row = dict(env_id=0,target_id=target["target_id"],family=target["family"],episode_time_s=t,
                   phase=phase,phase_time_s=phase_time,scored=scored,steady=g.steady_state(target,t),
                   ee_pos_world_xyz=actual[:3].tolist(),ee_quat_world_wxyz=actual[3:].tolist(),
                   ee_goal_pos_world_xyz=desired[:3].tolist(),ee_goal_quat_world_wxyz=desired[3:].tolist(),
                   position_error_m=float(errors[0]),orientation_error_geodesic_rad=float(errors[1]),
                   orientation_error_paper_frobenius=float(errors[2]),
                   base_pos_world_xyz=[0,0,.65],base_quat_world_wxyz=[1,0,0,0],
                   base_lin_vel_world_xyz=[0,0,0],base_ang_vel_body_xyz=[0,0,0],base_height_m=.65,
                   tilt_rad=0.,base_roll_pitch_rad=[0,0],base_xy_displacement_m=0.,
                   foot_contact_flags=[True]*4,foot_contact_forces=[[0,0,100]]*4,
                   foot_positions_world_xyz=[[0,0,0]]*4,foot_drift_m=[0]*4,
                   wheel_rotation_rad=[0]*4,wheel_travel_rad=[0]*4,standing_violation=False,
                   terminated=False,truncated=tick==round(duration/.02),failure=False,failure_reason=[])
        rows.append(row)
    is_fixed = target["family"] == "fixed"
    summary = dict(target_id=target["target_id"],family=target["family"],success=True,completed=True,failure=False,failure_reason=[],
                   tracking_sample_count=250 if is_fixed else 1000,steady_sample_count=100 if is_fixed else 0)
    for name,value in (("position_m",.05),("orientation_geodesic_rad",math.pi/2),("orientation_paper_frobenius",math.pi/math.sqrt(2))):
        summary.update({f"tracking_{name}_mean":value,f"tracking_{name}_rms":value,f"tracking_{name}_max":value,
                        f"steady_{name}_mean":value if is_fixed else None})
    return rows,summary


def test_synthetic_metric_check():
    assert v.check_error_metrics(g)["position_error_m"] == pytest.approx(.05)
    assert v.independent_errors([0,0,0],[-1,0,0,0],[0,0,0],[1,0,0,0])[1] == 0


@pytest.mark.parametrize("angle",[0.,1e-10,math.pi/2,math.pi])
def test_independent_angle_handles_quaternion_sign_and_half_turn(angle):
    q = [math.cos(angle/2),0,math.sin(angle/2),0]
    for sign in (-1,1):
        assert v.independent_errors([0,0,0],np.asarray(q)*sign,[0,0,0],[1,0,0,0])[1] == pytest.approx(angle,abs=1e-12)


def test_cpu_only_cli_cannot_claim_runtime_pass(tmp_path):
    out = tmp_path/"cpu"
    result = subprocess.run([sys.executable,str(ROOT/"evaluation/validate_t1.py"),"--cpu-only","--out",str(out)],capture_output=True,text=True)
    assert result.returncode == 1, result.stderr
    report = json.loads((out/"validation_report.json").read_text())
    assert report["status"] == "INCOMPLETE" and not report["all_tests_passed"]
    assert report["progress"] == {"passed":1,"total":8}
    assert report["checks"]["robot_physics"]["status"] == "NOT_RUN"
    assert not (out/"targets.json").exists()
    for name in ("trajectories.csv","metrics.csv","trial_summary.csv","config.yaml"):
        assert (out/name).stat().st_size > 0
    repeated = subprocess.run([sys.executable,str(ROOT/"evaluation/validate_t1.py"),"--cpu-only","--out",str(out)],capture_output=True,text=True)
    assert repeated.returncode != 0
    assert json.loads((out/"validation_report.json").read_text()) == report


def test_child_is_judged_by_completion_artifact_not_exit_code(tmp_path):
    """draft.7 decision C: a teardown abort after completion.json is recorded, not a failure."""
    with pytest.raises(ValueError,match="did not finish"):
        v.completed(tmp_path,{"exit_code":1,"timed_out":False})
    v.write_json(tmp_path/"completion.json",{"status":"started"})
    with pytest.raises(ValueError,match="did not finish"):
        v.completed(tmp_path,{"exit_code":0,"timed_out":False})
    v.write_json(tmp_path/"completion.json",{"status":"finished"})
    v.completed(tmp_path,{"exit_code":1,"timed_out":False})
    v.completed(tmp_path,{"exit_code":0,"timed_out":False})
    with pytest.raises(ValueError,match="timed out"):
        v.completed(tmp_path,{"exit_code":None,"timed_out":True})
    v.write_json(tmp_path/"error.json",{"message":"failure masked by shutdown"})
    with pytest.raises(ValueError,match="recorded an error"):
        v.completed(tmp_path,{"exit_code":0,"timed_out":False})


def test_report_requires_eight_passes_and_artifacts(tmp_path):
    report = v.new_report(False)
    for name in v.CHECKS:
        report["checks"][name] = {"status":"PASS"}
    v.save_report(tmp_path,report)
    assert report["status"] == "FAIL" and not report["all_tests_passed"]
    report = v.new_report(False)
    for name in v.CHECKS:
        report["checks"][name] = {"status":"PASS"}
    for name in ("targets.json","trajectories.csv","metrics.csv","config.yaml","trial_summary.csv"):
        (tmp_path/name).write_text("fixture\n")
    v.save_report(tmp_path,report)
    assert report["status"] == "PASS" and report["all_tests_passed"]
    report["checks"]["robot_physics"] = {"status":"BLOCKED"}
    v.save_report(tmp_path,report)
    assert report["status"] == "INCOMPLETE" and not report["all_tests_passed"]


@pytest.mark.parametrize("target",[fixed(),circle()])
def test_independent_metrics_and_steady_state_summary(target):
    rows,summary = trace(target)
    evidence = v.check_metrics(rows,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    assert evidence["trials"] == 1
    assert v.compare_repetitions(rows,copy.deepcopy(rows))["samples_per_repetition"] == len(rows)
    rows[-1]["position_error_m"] += .001
    with pytest.raises(AssertionError,match="pose error mismatch"):
        v.check_metrics(rows,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])


def test_reference_tampering_cannot_hide_behind_recomputed_errors():
    target = fixed();rows,summary = trace(target)
    rows[-1]["ee_goal_pos_world_xyz"][0] += .1
    errors = v.independent_errors(rows[-1][v.POSE_FIELDS[0]],rows[-1][v.POSE_FIELDS[1]],rows[-1][v.POSE_FIELDS[2]],rows[-1][v.POSE_FIELDS[3]])
    for key,error in zip(v.METRIC_FIELDS[:3],errors):rows[-1][key] = float(error)
    with pytest.raises(ValueError,match="sampled target"):
        v.check_metrics(rows,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])


def terminated(target, ticks=200, reason=("collision_arm_contact",)):
    """A trial ended by a genuine termination at tick `ticks`, with its partial summary."""
    rows,summary = trace(target)
    rows = rows[:ticks]
    rows[-1].update(failure=True,terminated=True,truncated=False,failure_reason=list(reason))
    scored = sum(r["scored"] for r in rows); steady = sum(r["steady"] for r in rows)
    summary = {k:(None if k.startswith(("tracking_","steady_")) else val) for k,val in summary.items()}
    summary.update(success=False,completed=False,failure=True,failure_reason=list(reason),
                   tracking_sample_count=scored,steady_sample_count=steady)
    return rows,summary


@pytest.mark.parametrize("target",[fixed(),circle()])
def test_terminated_trials_are_checked_for_completeness_not_success(target):
    rows,summary = terminated(target)
    assert v.check_metrics(rows,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])["outcomes"] == {"completed":0,"terminated":1}
    bad = copy.deepcopy(summary);bad["failure_reason"] = ["fall_tilt"]
    with pytest.raises(ValueError,match="summary disagrees"):
        v.check_metrics(rows,[bad],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    bad = copy.deepcopy(summary);bad["tracking_position_m_mean"] = .01
    with pytest.raises(ValueError,match="reports aggregate errors"):
        v.check_metrics(rows,[bad],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    bad = copy.deepcopy(summary);bad["tracking_sample_count"] += 1
    with pytest.raises(ValueError,match="sample count"):
        v.check_metrics(rows,[bad],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    other = copy.deepcopy(rows);other[-1].update(terminated=False,failure_reason=[])
    with pytest.raises(ValueError,match="terminal flags"):
        v.check_metrics(other,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    other = copy.deepcopy(rows);other[-1]["tilt_rad"] = float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        v.check_metrics(other,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])


def test_incomplete_inconsistent_and_duplicate_trials_are_rejected():
    target = fixed();rows,summary = trace(target)
    with pytest.raises(ValueError,match="Incomplete completed rollout"):
        v.check_metrics(rows[:-1],[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    failed = copy.deepcopy(rows);failed[-1].update(failure=True,terminated=True,truncated=False,failure_reason=["fall_tilt"])
    with pytest.raises(ValueError,match="Completed trial flags"):
        v.check_metrics(failed,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    early = copy.deepcopy(rows);early[100].update(terminated=True)
    with pytest.raises(ValueError,match="continued after its end"):
        v.check_metrics(early,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    with pytest.raises(ValueError,match="Duplicate"):
        v.check_metrics(rows,[summary,summary],{**p.TARGET_FRAME,"targets":[target,target]},g,[[0,0,0]])


def test_summary_and_stability_repeatability_failures():
    target = fixed();rows,summary = trace(target)
    summary["tracking_position_m_mean"] += .01
    with pytest.raises(ValueError,match="Incorrect summary"):
        v.check_metrics(rows,[summary],{**p.TARGET_FRAME,"targets":[target]},g,[[0,0,0]])
    other = copy.deepcopy(rows);other[-1]["tilt_rad"] += .001
    with pytest.raises(ValueError,match="Repeatability failed"):
        v.compare_repetitions(rows,other)
    other = copy.deepcopy(rows);other[-1]["ee_goal_pos_world_xyz"][1] += 1e-4
    with pytest.raises(AssertionError,match="Reference changed"):
        v.compare_repetitions(rows,other)


def test_partial_export_retains_failure_and_injection_label(tmp_path):
    source = tmp_path/"partial";source.mkdir()
    target = fixed();rows,_ = trace(target)
    row = dict(rows[-1],failure=True,terminated=True,truncated=False,failure_reason=["fall_tilt"],validation_case="fall_tilt")
    with (source/"steps.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
        writer.writerow({k:json.dumps(val) if isinstance(val,(list,dict)) else val for k,val in row.items()})
    v.export_rows(tmp_path,[("runtime_checks",0,source)])
    metric = v.read_steps(tmp_path/"metrics.csv")[0]
    pose = v.read_steps(tmp_path/"trajectories.csv")[0]
    assert metric["failure"] and metric["failure_reason"] == ["fall_tilt"]
    assert metric["validation_case"] == pose["validation_case"] == "fall_tilt"
    assert metric["target_id"] == pose["target_id"]


def hold_fixture(directory):
    names = ([f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf')] +
             [f'{leg}_foot_joint' for leg in ('FR','FL','RR','RL')] + [f'joint{i}' for i in range(1,7)])
    meta = dict(robot_variant='B2W',end_effector_frame='tcp',num_envs=1,seed=1101,action_order=names,joint_order=names,
                actuator_types={'legs':'ImplicitActuator','wheel':'ImplicitActuator','arm':'ImplicitActuator'},
                loaded_joint_parameters={'joint_stiffness':[800]*12+[0]*4+[100]*6,'joint_damping':[20]*22,
                                         'joint_vel_limits':[10]*22,'joint_effort_limits':[200]*22},
                physics_dt_s=.005,control_dt_s=.02,hold_duration_s=30)
    v.write_json(directory/'interface.json',meta)
    row = trace(fixed())[0][0]
    row.update(joint_pos=[0]*22,joint_vel=[0]*22,action_command=[0]*22,applied_joint_targets=[0]*22)
    rows = []
    for tick in range(1,1501):
        rows.append(dict(row,episode_time_s=tick*.02))
    def save():
        with (directory/'steps.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            for r in rows:
                writer.writerow({k:json.dumps(val) if isinstance(val,(list,dict)) else val for k,val in r.items()})
    save()
    return meta,rows,save


def test_hold_gate_requires_full_30_seconds_and_stance(tmp_path):
    _,rows,save = hold_fixture(tmp_path)
    assert v.check_hold(tmp_path)['samples'] == 1500
    rows[-1]['standing_violation'] = True;save()
    with pytest.raises(ValueError,match='Standing compliance'):
        v.check_hold(tmp_path)
    rows[-1]['standing_violation'] = False
    rows.pop();save()
    with pytest.raises(ValueError,match='1500'):
        v.check_hold(tmp_path)


def test_hold_gate_rejects_bad_mapping_and_nonzero_wheels(tmp_path):
    meta,rows,save = hold_fixture(tmp_path)
    rows[-1]['applied_joint_targets'] = [0]*12+[.1]*4+[0]*6;save()
    with pytest.raises(ValueError,match='wheel target'):
        v.check_hold(tmp_path)
    meta['action_order'] = list(reversed(meta['action_order']))
    v.write_json(tmp_path/'interface.json',meta)
    with pytest.raises(ValueError,match='mapping'):
        v.check_hold(tmp_path)


def sampled_fixture():
    """Sampler-shaped fixture only; no physical reachability is claimed."""
    targets = [dict(fixed(),target_id=f'fixed-{i:02d}') for i in range(2)]
    for i,(plane,u,w) in enumerate((('XY',[1,0,0],[0,1,0]),('XZ',[1,0,0],[0,0,1]),('YZ',[0,1,0],[0,0,1]))):
        targets.append(dict(circle(),target_id=f'circle-{i:02d}-{plane}',plane=plane,u=u,v=w))
    for t in targets:
        t['joint_space_sample_rad'] = [0.]*6
        t['sampler_audit'] = dict(attempt_index=1,target_sha256=p.target_digest(t))
        if t['family'] == 'circle':
            t['sampler_audit']['ik_path'] = [dict(episode_time_s=2+.1*j,arm_joint_pos_rad=[0]*6) for j in range(421)]
    return dict(protocol_version=p.VERSION,robot_variant='B2W',**p.TARGET_FRAME,seed=1101,settle_s=2.,
                sampler=p.SAMPLER,counts={'fixed':2,'circle':3},asset_hashes={'fixture.usd':'fixture'},
                nominal_ee_pose_env=[.4,0,.8,1,0,0,0],rejections={'fixed':{},'circle':{}},
                attempts={'fixed':2,'circle':3},targets=targets)


@pytest.mark.parametrize('crash_worker',[False,True])
def test_orchestration_contract_with_stubbed_simulator(tmp_path,monkeypatch,crash_worker):
    """Tests report wiring and crash rejection, never simulator acceptance."""
    monkeypatch.setenv('T1_IMAGE_ID','sha256:fixture')
    monkeypatch.setenv('ISAAC_GPU_UUID','GPU-fixture')
    calls = []
    sampled = sampled_fixture()
    rows,summary = trace(fixed())
    def child(out,name,script,arguments):
        calls.append((name,arguments))
        directory = out/'runs'/name/'artifacts';directory.mkdir(parents=True)
        crashed = name=='runtime_coordinate_frames' and crash_worker
        process = dict(exit_code=1,timed_out=False)   # every child aborts in native teardown (decision C)
        v.write_json(directory.parent/'process.json',process)
        if not crashed:
            v.write_json(directory/'completion.json',{'status':'finished'})
        metadata = {k:'fixture' for k in ('actuator_types','loaded_joint_parameters','asset_hashes','image_id','gpu_uuid','controller','git','versions')}
        metadata.update(robot_variant='B2W',end_effector_frame='tcp',seed=1101,num_envs=4 if name.startswith('evaluate') else 1,
                        physics_dt_s=.005,control_dt_s=.02,environment_origins_world_xyz=[[0,0,0]])
        v.write_json(directory/'interface.json',metadata)
        if name.startswith('runtime_'):
            v.write_json(directory/'checks.json',{name[len('runtime_'):]:{'status':'PASS'}})
        else:
            if name.startswith('evaluate'):
                directory = directory/'seed-1101';directory.mkdir()
                v.write_json(directory/'targets.json',sampled)
            with (directory/'steps.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerow({k:json.dumps(val) if isinstance(val,(list,dict)) else val for k,val in rows[0].items()})
            with (directory/'trial_summary.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(summary));writer.writeheader();writer.writerow(summary)
        return out/'runs'/name/'artifacts',process
    monkeypatch.setattr(v,'run_child',child)
    monkeypatch.setattr(v,'check_hold',lambda directory:{'fixture':'hold'})
    monkeypatch.setattr(v,'check_metrics',lambda *args:{'fixture':'metrics'})
    out = tmp_path/'validation'
    code = v.main(['--out',str(out)])
    report = v.read_json(out/'validation_report.json')
    assert calls[0][0] == 'hold'
    hold_args = calls[0][1]
    assert hold_args[hold_args.index('--hold-seconds')+1] == '30'
    if crash_worker:
        assert code == 1 and report['status'] == 'FAIL'
        assert report['checks']['coordinate_frames']['status'] == 'FAIL'
        assert report['checks']['standing_termination']['status'] == 'PASS'
        assert report['checks']['reproducibility']['status'] == 'BLOCKED'
        assert report['checks']['fixed_pose_generator']['status'] == 'BLOCKED'
    else:
        assert code == 0 and report['status'] == 'PASS'
        assert report['progress'] == {'passed':8,'total':8}
        assert [name for name,_ in calls] == ['hold','runtime_standing_termination','runtime_coordinate_frames',
                                              'evaluate_1','evaluate_2']
        assert [args[args.index('--checks')+1] for name,args in calls if name.startswith('runtime_')] == \
            ['standing_termination','coordinate_frames']
        assert all(args[args.index('--seeds')+1] == '1101' for name,args in calls if name.startswith('evaluate'))
        assert all(value['exists'] for value in report['artifacts'].values())


def test_runtime_config_must_match_across_hold_and_repetitions():
    meta = {key:'fixture' for key in v.RUNTIME_IDENTITY}
    v.match_configuration(meta,copy.deepcopy(meta))
    changed = dict(meta,asset_hashes={'robot.usd':'changed'})
    with pytest.raises(ValueError,match='asset_hashes'):
        v.match_configuration(meta,changed)


def test_circle_plane_label_must_match_geometry():
    sampled = sampled_fixture()
    assert v.check_references(sampled,'circle',g,p)['target_count'] == 3
    assert v.check_references(sampled,'fixed',g,p)['target_count'] == 2
    target = sampled['targets'][2]
    target['v'] = [0,0,1]  # XZ geometry still incorrectly labelled XY.
    target['sampler_audit']['target_sha256'] = p.target_digest(target)
    with pytest.raises(ValueError,match='named world plane'):
        v.check_references(sampled,'circle',g,p)


def test_hold_mapping_accepts_only_the_gripper_as_extra_joint(tmp_path):
    """draft.4 gripper asset: 22 action joints + jointGripper in the articulation (as StandingAction)."""
    meta,_,_ = hold_fixture(tmp_path)
    params = meta['loaded_joint_parameters']
    for extra,ok in (('jointGripper',True),('jointOther',False)):
        changed = dict(meta,joint_order=meta['action_order']+[extra],
                       loaded_joint_parameters={k:list(val)+[1] for k,val in params.items()})
        v.write_json(tmp_path/'interface.json',changed)
        if ok:
            assert v.check_hold(tmp_path)['samples'] == 1500
        else:
            with pytest.raises(ValueError,match='mapping'):
                v.check_hold(tmp_path)
