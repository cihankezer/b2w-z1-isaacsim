"""Isolated Isaac worker for validate_t1.py: faults, then FK and fixed-world commands.

Exit zero means the worker completed; checks.json contains each check's verdict.
The parent owns the overall verdict and also rejects native shutdown failures.
"""

import argparse
import importlib
from pathlib import Path

from validate_t1 import ROOT, TOLERANCES, record_check, require, write_json


def checks(env, app, out, result, selected):
    import numpy as np
    import torch
    import omni.usd
    from scipy.spatial.transform import Rotation
    from eval_t1 import CsvLog

    package = "robot_lab.tasks.manager_based.manipulation.standing_ee"
    g = importlib.import_module(package + ".geometry")
    k = importlib.import_module(package + ".kinematics")
    TR = importlib.import_module(package + ".terminations")
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joints")
    cmd = env.command_manager.get_term("ee_pose")
    monitor = env.termination_manager.get_term_cfg("failure").func
    model = k.UsdArmKinematics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
    zero = torch.zeros(env.num_envs, 22, device=env.device)
    samples = CsvLog(out/"steps.csv")

    def step(label):
        require(app.is_running(), "Simulator stopped during fault checks")
        with torch.inference_mode():
            env.step(zero)
        result["physics_samples"] = result.get("physics_samples",0) + env.num_envs
        for row in env.t1_samples:
            samples.append(dict(row, validation_case=label))
        return env.t1_samples

    contact_log = []

    def contact_rows(label):
        """Contact rows > 1 N in both sensors (max over the sensor history) and the spawn/reset pose."""
        rows = {}
        for sensor in ("contacts","arm_contacts"):
            forces = env.scene[sensor].data.net_forces_w_history.norm(dim=-1).amax(dim=1).cpu().numpy()
            rows[sensor] = [[dict(body=n,force_n=float(f)) for n,f in zip(env.scene[sensor].body_names,forces[i]) if f > 1.0]
                            for i in range(env.num_envs)]
        deviation = (robot.data.joint_pos-robot.data.default_joint_pos).abs().amax(dim=1)
        contact_log.append(dict(label=label,rows=rows,max_abs_joint_pos_minus_init_state_rad=deviation.tolist(),
                                root_pos_env_xyz=(robot.data.root_pos_w-env.scene.env_origins).tolist()))
        write_json(out/"contact_log.json",contact_log)
        return rows

    def hold_one_control_period():
        """draft.7: one control period (decimation physics steps, zero action = hold the nominal PD
        targets) after reset, without the termination/command managers, so the contact history
        holds post-reset data only. Not a T1 sample."""
        with torch.inference_mode():
            env.action_manager.process_action(zero)
            for _ in range(env.cfg.decimation):
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
        result["settle_physics_integration_steps"] = result.get("settle_physics_integration_steps",0)+env.cfg.decimation*env.num_envs

    def reset(label="reset"):
        env.reset(seed=1101)
        env.sim.forward()
        env.scene.update(env.physics_dt)
        contact_rows(f"{label}:after_reset")
        hold_one_control_period()
        contact_rows(f"{label}:after_one_control_period")

    def frames_check():
        reset("coordinate_frames")
        goal = g.transform(robot.data.body_pos_w[0,cmd.hand].cpu().numpy(),
                           robot.data.body_quat_w[0,cmd.hand].cpu().numpy())
        old_targets = list(cmd.targets)
        cmd.targets[0] = dict(family="fixed",target_id="frame-probe",position_world_xyz=goal[:3,3].tolist(),
                             quaternion_world_wxyz=g.pose(goal)[3:].tolist(),hold_s=5.,steady_state_last_s=2.)
        env.episode_length_buf[:] = 150
        rng = np.random.default_rng(1101)
        cases = []
        try:
            for case in range(9):
                q = robot.data.default_joint_pos.clone()
                if case:
                    q[:,action.joint_ids[16:]] = torch.tensor(
                        rng.uniform(model.limits[:,0]+.1,model.limits[:,1]-.1,(2,6)),device=env.device,dtype=q.dtype)
                root = robot.data.default_root_state[:,:7].clone()
                root[:,:3] += env.scene.env_origins
                if case:
                    root[:,:3] += torch.tensor(rng.uniform(-.2,.2,(2,3)),device=env.device,dtype=root.dtype)
                    root[:,2] += 1
                    quat = Rotation.from_euler("xyz",rng.uniform(-.6,.6,(2,3))).as_quat()[:,[3,0,1,2]]
                    root[:,3:] = torch.tensor(quat,device=env.device,dtype=root.dtype)
                robot.write_root_pose_to_sim(root)
                robot.write_root_velocity_to_sim(torch.zeros(env.num_envs,6,device=env.device))
                robot.write_joint_state_to_sim(q,torch.zeros_like(q))
                env.sim.forward(); env.scene.update(env.physics_dt)
                for i in range(2):
                    d = robot.data
                    base = g.transform(d.root_pos_w[i].cpu().numpy(),d.root_quat_w[i].cpu().numpy())
                    frames = model.forward(base,dict(zip(robot.joint_names,d.joint_pos[i].tolist())))
                    errors = [g.pose_error(frames[name],g.transform(d.body_pos_w[i,j].cpu().numpy(),
                               d.body_quat_w[i,j].cpu().numpy()))[:2] for j,name in enumerate(robot.body_names)]
                    maximum = np.max(errors,axis=0)
                    require(maximum[0] <= TOLERANCES["fk_position_m"] and maximum[1] <= TOLERANCES["fk_orientation_rad"], "FK differs from simulator")
                    cases.append(dict(case=case,env_id=i,root_pose=root[i].tolist(),joint_pos=q[i].tolist(),
                                      position_error_m=float(maximum[0]),orientation_error_rad=float(maximum[1])))
                cmd._update_command()
                body_goal = cmd.command[0].cpu().numpy()
                base = g.transform(robot.data.root_pos_w[0].cpu().numpy(),robot.data.root_quat_w[0].cpu().numpy())
                reconstructed = base @ g.transform(body_goal[:3],body_goal[3:])
                dp,dr,_ = g.pose_error(reconstructed,goal)
                require(dp < 1e-5 and dr < 1e-5,"World target changed with base motion")
        finally:
            cmd.targets[:] = old_targets
            reset()
        write_json(out/"fk_cases.json",cases)
        return dict(states=len(cases),max_position_error_m=max(c["position_error_m"] for c in cases),
                    max_orientation_error_rad=max(c["orientation_error_rad"] for c in cases),world_target_invariance="PASS")

    def faults_check():
        reset("standing_termination")
        monitor(env)
        if any(monitor.reasons):
            rows = contact_rows("standing_termination:nominal_failure")
            raise ValueError(f"Nominal state already fails: {monitor.reasons}; contact rows: {rows}")
        # Cached-state injection avoids sending NaN into the physics engine.
        state = robot.data.joint_pos
        before = state[0,0].clone()
        try:
            state[0,0] = float("nan")
            monitor(env)
            require(monitor.reasons[0] == ["invalid_state"],"Invalid state did not get a distinct label")
            env.reset_terminated = torch.tensor([True,False],device=env.device)
            env.reset_time_outs = torch.zeros(2,dtype=torch.bool,device=env.device)
            env.recorder_manager.record_post_step()
            for row in env.t1_samples:
                samples.append(dict(row,validation_case="invalid_cached_state"))
        finally:
            state[0,0] = before
        reset()
        root = robot.data.root_pose_w.clone()
        root[0,2] = 1.2
        # Shared termination set (T3 D6/D20): roll 1.2 rad > 1.0 rad must terminate on the 5th step only.
        root[0,3:] = torch.tensor([np.cos(.6),np.sin(.6),0,0],device=env.device)
        robot.write_root_pose_to_sim(root)
        env.sim.forward(); env.scene.update(env.physics_dt)
        for n in range(TR.DEBOUNCE_STEPS - 1):
            require(not step("fall_tilt")[0]["failure"], f"Tilt terminated before the debounce ({n+1} steps)")
        rows = step("fall_tilt")
        require("fall_tilt" in rows[0]["failure_reason"] and rows[0]["terminated"] and not rows[0]["truncated"], "Fall label/flags incorrect")
        require(rows[0]["base_pos_world_xyz"][2] > .9 and robot.data.root_pos_w[0,2] < .7,
                "Terminal state was contaminated by automatic reset")
        reset()
        # Cached-state injection (no physics): base 0.2 m < 0.274 m for 5 evaluations, then 0.31 m x/y drift.
        with torch.inference_mode():       # the cached buffers are inference tensors
            pos = robot.data.root_pos_w
            before = pos[0].clone()
            try:
                for n in range(TR.DEBOUNCE_STEPS):
                    pos[0,2] = env.scene.env_origins[0,2] + .2
                    require(abs(float(robot.data.root_pos_w[0,2]-env.scene.env_origins[0,2]) - .2) < 1e-6,
                            "Cached height injection failed")
                    monitor(env)
                    require(("fall_height" in monitor.reasons[0]) == (n == TR.DEBOUNCE_STEPS - 1),
                            f"Height debounce incorrect at evaluation {n+1}: {monitor.reasons[0]}")
                pos[0] = before
                pos[0,0] += TR.DRIFT_M + .01
                monitor(env)
                require("standing_drift" in monitor.reasons[0] and not monitor.reasons[1],
                        f"Standing drift missing: {monitor.reasons}")
            finally:
                pos[0] = before
        reset()
        # A separate kinematic cube physically contacts the arm; no fake force history.
        obstacle = env.scene["validation_obstacle"]
        obstacle_pose = obstacle.data.default_root_state[:,:7].clone()
        obstacle_pose[:,:3] = 100
        obstacle_pose[0,:3] = robot.data.body_pos_w[0,cmd.hand]
        obstacle.write_root_pose_to_sim(obstacle_pose)
        try:
            rows = step("physical_arm_collision")
            require("collision_arm_contact" in rows[0]["failure_reason"], "Physical arm collision not detected by contact sensor")
            require(rows[0]["terminated"] and not rows[0]["truncated"],"Collision flags incorrect")
        finally:
            obstacle_pose[:,:3] = 100
            obstacle.write_root_pose_to_sim(obstacle_pose)
        reset()
        env.episode_length_buf[:] = torch.tensor([100,2999],device=env.device)
        rows = step("timeout_and_async_reset")
        require(rows[1]["truncated"] and not rows[1]["terminated"] and not rows[1]["failure"],"Normal timeout was not isolated")
        require(rows[1]["episode_time_s"] == 60 and not rows[0]["truncated"] and not rows[0]["terminated"],"Asynchronous timeout flags incorrect")
        require(env.episode_length_buf.tolist() == [101,0],"Asynchronous reset changed surviving episode clock")
        for field in ("foot_contact_forces","foot_drift_m","base_height_m","tilt_rad"):
            require(rows[0][field] is not None and np.isfinite(rows[0][field]).all(), f"Missing standing diagnostic: {field}")
        return dict(fall_tilt="PASS",fall_height="PASS",standing_drift="PASS",physical_arm_collision="PASS",invalid_cached_state="PASS",
                    normal_timeout="PASS",pre_reset_terminal_capture="PASS",asynchronous_reset="PASS")

    # draft.7: fault checks first; the FK teleport check (arbitrary root/arm states) runs last.
    # Each check starts from its own reset + one control period. validate_t1 runs each in its own process.
    contact_rows("after_gym_make_before_first_reset")
    try:
        for name,function in (("standing_termination",faults_check),("coordinate_frames",frames_check)):
            if name in selected:
                # Whole check in inference mode: the post-reset hold creates inference tensors.
                with torch.inference_mode():
                    record_check(result,name,function)
                write_json(out/"checks.json",result["checks"])
    finally:
        samples.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--checks",default="standing_termination,coordinate_frames",
                        help="comma-separated subset, run in this order: standing_termination, coordinate_frames")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(); args.headless = True
    args.out.mkdir(parents=True,exist_ok=False)
    result = dict(checks={})
    app = env = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import gymnasium as gym
        import robot_lab.tasks  # noqa: F401
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.utils.io import dump_yaml
        parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
        parent.__path__.insert(0,str(ROOT/"training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"))
        package = "robot_lab.tasks.manager_based.manipulation.standing_ee"
        importlib.import_module(package)
        cfg = importlib.import_module(package+".env_cfg").StandingEnvCfg()
        cfg.scene.num_envs = 2; cfg.sim.device = args.device
        cfg.scene.validation_obstacle = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/validation_obstacle",collision_group=-1,
            spawn=sim.CuboidCfg(size=(.12,.12,.12),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                collision_props=sim.CollisionPropertiesCfg(),mass_props=sim.MassPropertiesCfg(mass=1.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(100.,100.,100.)))
        env = gym.make("RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0",cfg=cfg).unwrapped
        dump_yaml(str(args.out/"config.yaml"),cfg.to_dict())
        checks(env,app,args.out,result,args.checks.split(","))
        write_json(args.out/"completion.json",dict(status="finished",physics_samples=result.get("physics_samples",0),
                   physics_integration_steps=result.get("physics_samples",0)*cfg.decimation,imagined_samples=0))
    except BaseException as exc:
        write_json(args.out/"error.json",dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
