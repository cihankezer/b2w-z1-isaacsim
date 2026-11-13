"""Measure nominal PD dynamics/contact forces without hiding geometric issues.

Lower-level physics stepping deliberately avoids automatic resets on known
geometric failures. Production terminations/exclusions are unchanged. This is a
separate diagnostic, never a full T1 acceptance result.
"""

import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
import time

from eval_t1 import ROOT, OVERLAY, CsvLog, git_info, write_json


def run(args, app, out):
    import numpy as np
    import torch
    import gymnasium as gym
    import omni.usd
    import robot_lab.tasks  # noqa: F401
    from pxr import Usd, UsdPhysics
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.io import dump_yaml

    parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
    parent.__path__.insert(0,str(OVERLAY))
    package = "robot_lab.tasks.manager_based.manipulation.standing_ee"
    importlib.import_module(package)
    cfg = importlib.import_module(package+".env_cfg").StandingEnvCfg()
    protocol = importlib.import_module(package+".protocol")
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    # One body per filtered view; record both directions and all contact data.
    for name,source,target in (("pair_00_02","link00","link02"),("pair_02_00","link02","link00")):
        setattr(cfg.scene,name,ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/"+source,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/z1/"+target],update_period=.005,
            history_length=4,track_contact_points=True,track_friction_forces=True,max_contact_data_count_per_prim=128))
    env = None
    physics_log = CsvLog(out/"physics.csv")
    control_log = CsvLog(out/"control.csv")
    summary = dict(status="in_progress",standing_dynamics_checks_satisfied=False,full_t1_passed=False,
                   geometric_clearance_passed=False,automatic_resets=False,physics_samples=0,
                   diagnostic_only=True,requested_duration_s=args.seconds,physical_stop_reasons=[])
    try:
        env = gym.make(protocol.TASK_ID,cfg=cfg).unwrapped
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joints")
        command = env.command_manager.get_term("ee_pose")
        monitor = env.termination_manager.get_term_cfg("failure").func
        env.reset(seed=cfg.seed)
        stage = omni.usd.get_context().get_stage()
        model = importlib.import_module(package+".kinematics").UsdArmKinematics(stage,"/World/envs/env_0/Robot")
        snapshot = dict(protocol_version=protocol.VERSION,seed=cfg.seed,device=args.device,robot="B2W+Z1",tcp="link06",
            physics_dt_s=env.physics_dt,control_dt_s=env.step_dt,requested_duration_s=args.seconds,
            image_id=os.environ["T1_IMAGE_ID"],gpu_uuid=os.environ["ISAAC_GPU_UUID"],git=git_info(ROOT),
            asset_path=robot.cfg.spawn.usd_path,asset_hashes=protocol.asset_hashes(robot.cfg.spawn.usd_path),
            filtering=model.physics_filtering,geometric_exclusions=sorted(sorted(p) for p in model.adjacent),
            clearance_policy=protocol.CLEARANCE_POLICY,
            bound_sources=model.bounds_source,joint_order=robot.joint_names,
            action_order=[robot.joint_names[j] for j in action.joint_ids],
            loaded_parameters={n:getattr(robot.data,n)[0].tolist() for n in ("joint_stiffness","joint_damping","joint_effort_limits")},
            sensors={n:dict(bodies=env.scene[n].body_names,filter_count=env.scene[n].contact_physx_view.filter_count)
                     for n in ("pair_00_02","pair_02_00")},
            physical_colliders=[],unresolved_geometry=[dict(path=s["path"],issue=s["issue"])
                for shapes in model.shapes.values() for s in shapes if not s.get("resolved",True)])
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                snapshot["physical_colliders"].append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),
                    enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                    contact_offset=str(prim.GetAttribute("physxCollision:contactOffset").Get()),
                    rest_offset=str(prim.GetAttribute("physxCollision:restOffset").Get())))
        write_json(out/"interface.json",snapshot)
        dump_yaml(str(out/"config.yaml"),cfg.to_dict())
        for name in ("pair_00_02","pair_02_00"):
            if env.scene[name].contact_physx_view.filter_count != 1:
                raise ValueError("Pair contact filter did not resolve exactly one target")
        zeros = torch.zeros(1,22,device=env.device)
        steps = round(args.seconds/env.step_dt)
        maximum_normal = maximum_friction = maximum_net = 0.
        maximum_contact_count = 0
        maximum_foot_force = 0.
        geometry_seen, self_seen, leg_mesh_ground_candidates = {}, {}, 0
        settled_stance = True
        max_tilt = max_drift = max_wheel = 0.
        min_height = float("inf")
        start = time.monotonic()
        for step in range(steps):
            if not app.is_running():
                raise RuntimeError("Simulator interrupted")
            with torch.inference_mode():
                env.action_manager.process_action(zeros)
                for substep in range(cfg.decimation):
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    tick = step*cfg.decimation+substep+1
                    values = {}
                    for name in ("pair_00_02","pair_02_00"):
                        sensor = env.scene[name]
                        view = sensor.contact_physx_view
                        matrix = sensor.data.force_matrix_w[0,0,0]
                        normal,_,_,separations,counts,starts = view.get_contact_data(dt=env.physics_dt)
                        count,offset = int(counts[0,0]),int(starts[0,0])
                        normal_sum = float(normal[offset:offset+count].abs().sum()) if count else 0.
                        friction,_,fcounts,fstarts = view.get_friction_data(dt=env.physics_dt)
                        fcount,foffset = int(fcounts[0,0]),int(fstarts[0,0])
                        friction_sum = float(friction[foffset:foffset+fcount].norm(dim=-1).sum()) if fcount else 0.
                        maximum_normal = max(maximum_normal,normal_sum)
                        maximum_friction = max(maximum_friction,friction_sum)
                        maximum_contact_count = max(maximum_contact_count,count)
                        values[name] = dict(normal_force_world_n=matrix.tolist(),normal_force_sum_n=normal_sum,
                            friction_force_sum_n=friction_sum,contact_count=count,
                            min_separation_m=float(separations[offset:offset+count].min()) if count else None)
                    d = robot.data
                    finite = all(bool(torch.isfinite(v).all()) for v in (d.root_state_w,d.joint_pos,d.joint_vel,d.body_state_w))
                    forces = env.scene["arm_contacts"].data.net_forces_w[0].norm(dim=-1)
                    maximum_net = max(maximum_net,float(forces.max()))
                    feet = env.scene["contacts"].data.net_forces_w[0]
                    foot_ids = [env.scene["contacts"].body_names.index(robot.body_names[j]) for j in command.foot_ids]
                    maximum_foot_force = max(maximum_foot_force,float(feet[foot_ids].norm(dim=-1).max()))
                    physics_log.append(dict(physics_tick=tick,time_s=tick*env.physics_dt,finite_state=finite,
                        pair_contacts=values,arm_contact_forces_n=forces.tolist(),foot_contact_forces_n=feet[foot_ids].tolist(),
                        base_position_world_xyz=d.root_pos_w[0].tolist(),base_quaternion_world_wxyz=d.root_quat_w[0].tolist(),
                        joint_pos=d.joint_pos[0].tolist(),joint_vel=d.joint_vel[0].tolist()))
                    summary["physics_samples"] = tick
                    if not finite:
                        raise ValueError("Nonfinite physical state")
            env.episode_length_buf += 1
            env.common_step_counter += 1
            monitor(env)
            diag = monitor.diagnostics[0]
            reasons = monitor.reasons[0]
            for row in diag["collision_geometry"]:
                key = (tuple(row["pair"]),row["shape_a"],row["shape_b"],row["classification"])
                previous = geometry_seen.setdefault(key,dict(row,min_clearance_m=row.get("clearance_m"),samples=0))
                previous["samples"] += 1
                if row.get("clearance_m") is not None:
                    previous["min_clearance_m"] = min(previous["min_clearance_m"],row["clearance_m"])
                if row["gating"] and "/collisions/" in row["shape_a"] and any(
                        f"/{leg}_{part}/" in row["shape_a"] for leg in ("FL","FR","RL","RR") for part in ("calf","foot")):
                    leg_mesh_ground_candidates += 1
            for row in diag["self_clearance"]:
                s = self_seen.setdefault(":".join(row["pair"]),dict(samples=0,below_threshold_samples=0,
                    penetration_samples=0,unresolved_samples=0,min_clearance_m=None,worst_time_s=None,
                    min_clearance_first_2s_m=None,min_clearance_after_2s_m=None,worst_shapes=None))
                s["samples"] += 1
                s["below_threshold_samples"] += row["classification"] in ("clearance_violation","penetration")
                s["penetration_samples"] += row["classification"] == "penetration"
                s["unresolved_samples"] += row["classification"] == "unresolved"
                d, t = row["clearance_m"], (step+1)*env.step_dt
                if d is not None and (s["min_clearance_m"] is None or d < s["min_clearance_m"]):
                    s.update(min_clearance_m=d,worst_time_s=t,worst_shapes=[row["shape_a"],row["shape_b"]])
                window = "min_clearance_first_2s_m" if t <= 2.0 else "min_clearance_after_2s_m"
                if d is not None and (s[window] is None or d < s[window]):
                    s[window] = d
            max_tilt = max(max_tilt,diag["tilt_rad"])
            min_height = min(min_height,diag["base_height_m"])
            max_drift = max(max_drift,max(diag["foot_drift_m"]))
            max_wheel = max(max_wheel,max(abs(v) for v in diag["wheel_rotation_rad"]))
            if (step+1)*env.step_dt > 2:
                settled_stance &= not diag["standing_violation"]
            control_log.append(dict(time_s=(step+1)*env.step_dt,joint_targets=action.processed_actions[0].tolist(),
                failure_reasons=reasons,**diag))
            physical = [r for r in reasons if not r.startswith("collision_geometry:")]
            if physical:
                summary["physical_stop_reasons"] = physical
                break
            if (step+1) % 250 == 0:
                print(f"Nominal dynamics: {(step+1)*env.step_dt:.1f}/{args.seconds:.1f} s",flush=True)
        complete = summary["physics_samples"] == round(args.seconds/env.physics_dt)
        summary.update(status="finished" if complete else "stopped",elapsed_simulated_s=summary["physics_samples"]*env.physics_dt,
            nominal_pd_survival=complete and not summary["physical_stop_reasons"],
            standing_compliance_after_settling=bool(settled_stance) if args.seconds>2 else None,
            standing_dynamics_checks_satisfied=bool(args.seconds==30 and complete and settled_stance and not summary["physical_stop_reasons"]),
            max_pair_normal_force_n=maximum_normal,max_pair_friction_force_n=maximum_friction,
            max_pair_contact_count=maximum_contact_count,max_arm_net_contact_force_n=maximum_net,
            max_foot_contact_force_n=maximum_foot_force,contact_reporting_positive_control=maximum_foot_force>1,
            max_tilt_rad=max_tilt,min_base_height_m=min_height,max_foot_drift_m=max_drift,max_wheel_rotation_rad=max_wheel,
            geometry=list(geometry_seen.values()),
            # policy draft.3: only PhysX-collider ground rows gate; self-clearance is a monitored metric
            geometric_clearance_passed=all(r["classification"]=="broad_phase_false_positive"
                                           for r in geometry_seen.values() if r["gating"]),
            self_clearance_summary=self_seen,leg_mesh_ground_candidates=leg_mesh_ground_candidates,
            wall_time_s=time.monotonic()-start,imagined_samples=0)
        if protocol.asset_hashes(robot.cfg.spawn.usd_path) != snapshot["asset_hashes"]:
            raise ValueError("Asset files changed during diagnosis")
        write_json(out/"diagnosis.json",summary)
        print(f"Dynamics survival={summary['nominal_pd_survival']}; clearance={summary['geometric_clearance_passed']}; full T1 remains unaccepted",flush=True)
        return summary
    except BaseException as exc:
        summary.update(status="error",error=dict(type=type(exc).__name__,message=str(exc)))
        write_json(out/"diagnosis.json",summary)
        raise
    finally:
        physics_log.close();control_log.close()
        if env is not None:
            env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=30.)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args();args.headless = True
    if not 0 < args.seconds <= 30 or abs(args.seconds/.02-round(args.seconds/.02)) > 1e-6:
        parser.error("Use a positive control-aligned duration up to 30 s")
    if not os.environ.get("T1_IMAGE_ID") or not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("Set actual T1_IMAGE_ID and one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot = args.out/"source";snapshot.mkdir()
    for path in [Path(__file__),ROOT/"evaluation/eval_t1.py",ROOT/"docs/protocols/TASK_T1.md",*sorted((OVERLAY/"standing_ee").glob("*.py"))]:
        shutil.copy2(path,snapshot/path.name)
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        summary = run(args,app,args.out)
    finally:
        app.close()
        write_json(args.out/"lifecycle.json",dict(application_close_returned=True))
    return 0 if summary["standing_dynamics_checks_satisfied"] and summary["geometric_clearance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
