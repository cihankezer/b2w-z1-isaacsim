"""Render the 30 s T2 presentation video (GPU, offscreen RTX camera, 1 env, 1920x1080 @ 30 fps, MP4).

    render_t2.py --reachability <frozen npz> --figure <reachability png> --ids HIGH,MID,LOW --out <dir> [--preview]

No learned policy: a kinematic visualization of the map postures (stated on every frame). Structure:
  0-3 s    nominal standing pose (settled under physics, zero action), title card
  3-24 s   three 7 s segments (ids in the order given, High, Mid, Low), each:
           1 s nominal + target sphere; 3 s joint-space interpolation (base pose and joints smoothstepped from
           the settled nominal state to the map's stored posture of the target's voxel, written kinematically,
           no physics step); 2 s physics hold with gravity-compensated PD targets as in A3a
           (eval_t2.static_torques); 1 s freeze on the last hold frame with the measured TCP error
  24-30 s  reachability side-view figure (KAIST -0.20 m line) and the validation line
Targets come from the frozen evaluation set (regenerated, digest checked against protocol.EVAL_SET_SHA256).
Fixed camera (front-left side view), identical for all segments. Overlay drawn per frame with PIL.
PNG stills (nominal, three final postures, figure) are written as preview_*.png while rendering; --preview
stops after them (no video).
Outputs in --out: t2_presentation.mp4 (+ .json sidecar) or preview_*.png.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
OVERLAY = ROOT / "training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.reach_height"
FONT = "/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
FPS = 30
TITLE_S, NOMINAL_S, INTERP_S, HOLD_S, RESULT_S, FIGURE_S = 3.0, 1.0, 3.0, 2.0, 1.0, 6.0
TITLE = "T2 — Reach Low and High Targets While Standing"
DISCLAIMER = "kinematic visualization of map postures, no learned policy"
VALIDATION = "100 % static feasibility (A3a) · bitwise-deterministic evaluation (A7)"
BAND_LABEL = {"low": "Low (0.0–0.3 m)", "mid": "Mid (0.3–0.8 m)", "high": "High (0.8–1.1 m)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reachability", required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="three target_ids of the frozen eval set, in display order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    if not os.environ.get("ISAAC_GPU_UUID") or os.environ["ISAAC_GPU_UUID"] == "all":
        parser.error("needs one allocated ISAAC_GPU_UUID")
    args.out.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    import gymnasium as gym
    import numpy as np
    import torch
    import imageio_ffmpeg
    import isaaclab.sim as sim_utils
    import robot_lab.tasks  # noqa: F401
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sensors import CameraCfg
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation, Slerp
    parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
    if str(OVERLAY) not in parent.__path__:
        parent.__path__.insert(0, str(OVERLAY))
    importlib.import_module(PACKAGE)
    P = importlib.import_module(PACKAGE + ".protocol")
    S = importlib.import_module(PACKAGE + ".sampler")
    E = importlib.import_module("eval_t2")
    W, H = args.width, args.height

    cfg = importlib.import_module(PACKAGE + ".env_cfg").T2ReachEvalEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.commands.tcp_target.reachability_path = args.reachability
    cfg.sim.render_interval = 10**6          # render explicitly (markers and written states in sync)
    cfg.scene.render_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/RenderCam", update_period=0.0, width=W, height=H,
                                     data_types=["rgb"],
                                     spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 50.0)))
    env = gym.make("RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
    robot, cmd, camera = env.scene["robot"], env.command_manager.get_term("tcp_target"), env.scene["render_cam"]
    jid = env.action_manager.get_term("joints").joint_ids
    joint_names = [robot.joint_names[j] for j in jid]
    default = robot.data.default_joint_pos[0, jid].cpu().numpy()
    soft = robot.data.soft_joint_pos_limits[0, jid].cpu().numpy()
    leg_cols, arm_cols = list(range(12)), list(range(16, 22))
    pos_cols = leg_cols + arm_cols
    dev = env.device

    # Frozen evaluation set and the map's stored solutions (as eval_t2.py).
    rs = S.ReachableSet.load(args.reachability)
    targets, meta = S.eval_target_set(rs, P.EVAL_SEED, cmd.reachability_sha256, P.N_EVAL_PER_BAND)
    if meta["targets_sha256"] != P.EVAL_SET_SHA256:
        raise RuntimeError("evaluation set digest differs from protocol.EVAL_SET_SHA256")
    with np.load(args.reachability, allow_pickle=False) as d:
        data = {k: d[k] for k in ("posture_index", "arm_q", "posture_leg_q", "leg_joint_names", "posture_dh",
                                  "posture_pitch", "posture_base_world", "G_world")}
    ids = [int(x) for x in args.ids.split(",")]
    rows = [int(np.flatnonzero(targets["target_id"] == i)[0]) for i in ids]

    target_vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/t2_target", markers={"s": sim_utils.SphereCfg(
        radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)))}))
    tcp_vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/t2_tcp", markers={"s": sim_utils.SphereCfg(
        radius=0.018, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)))}))
    hidden = torch.tensor([[0.0, 0.0, -10.0]], device=dev)

    def show_markers(target_w):
        target_vis.visualize(translations=hidden if target_w is None else
                             torch.tensor(target_w[None], dtype=torch.float32, device=dev))
        tcp_vis.visualize(translations=hidden if target_w is None else cmd.tcp_pose_w()[0][:1])

    def capture():
        env.sim.render()
        camera.update(env.step_dt, force_recompute=True)
        return camera.data.output["rgb"][0, ..., :3].contiguous().cpu().numpy().astype(np.uint8)

    def write_state(pose7, q):
        q_full = robot.data.default_joint_pos[:1].clone()
        q_full[0, jid] = torch.as_tensor(q, dtype=q_full.dtype, device=dev)
        robot.write_root_pose_to_sim(torch.as_tensor(pose7[None], dtype=torch.float32, device=dev))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
        env.sim.forward()

    def settle_nominal():
        env.reset()
        zero = torch.zeros(1, 22, device=dev)
        for _ in range(25):
            env.step(zero)
        return (torch.cat([robot.data.root_pos_w[0], robot.data.root_quat_w[0]]).cpu().numpy(),
                robot.data.joint_pos[0, jid].cpu().numpy())

    big, mid, small = (ImageFont.truetype(FONT, s) for s in (58, 34, 28))

    def overlay(frame, lines=None, title=False, result=None):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, 0, W, 64], fill=(0, 0, 0, 160))
        draw.text((24, 14), f"T2 · protocol {P.VERSION} · {DISCLAIMER}", font=small, fill=(255, 255, 255))
        if title:
            box = draw.textbbox((0, 0), TITLE, font=big)
            x, y = (W - (box[2] - box[0])) // 2, H // 2 - 330
            draw.rectangle([x - 30, y - 20, x + box[2] - box[0] + 30, y + box[3] - box[1] + 90], fill=(0, 0, 0, 170))
            draw.text((x, y), TITLE, font=big, fill=(255, 255, 255))
            sub = "Unitree B2W + Z1 · nominal standing pose"
            sbox = draw.textbbox((0, 0), sub, font=mid)
            draw.text(((W - (sbox[2] - sbox[0])) // 2, y + 80), sub, font=mid, fill=(225, 225, 225))
        if lines:
            draw.rectangle([0, H - 40 - 46 * len(lines), W, H], fill=(0, 0, 0, 170))
            for i, line in enumerate(lines):
                draw.text((24, H - 26 - 46 * (len(lines) - i)), line, font=mid, fill=(255, 255, 255))
        if result:
            draw.rectangle([W - 620, 84, W - 20, 150], fill=(20, 110, 40, 200))
            draw.text((W - 600, 98), result, font=mid, fill=(255, 255, 255))
        return np.asarray(image)

    def figure_frame():
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        fig = Image.open(args.figure).convert("RGB")
        scale = min((W - 80) / fig.width, (H - 260) / fig.height)
        fig = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
        canvas.paste(fig, ((W - fig.width) // 2, 90))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, W, 64], fill=(30, 30, 30))
        draw.text((24, 14), f"T2 · protocol {P.VERSION} · {DISCLAIMER}", font=small, fill=(255, 255, 255))
        box = draw.textbbox((0, 0), VALIDATION, font=mid)
        draw.text(((W - (box[2] - box[0])) // 2, H - 120), VALIDATION, font=mid, fill=(20, 20, 20))
        return np.asarray(canvas)

    nominal_pose, nominal_q = settle_nominal()
    # Fixed camera from the settled nominal pose: front-left side view (yaw + 70 deg), same for all segments.
    yaw0 = float(Rotation.from_quat(nominal_pose[[4, 5, 6, 3]]).as_euler("ZYX")[0])
    fwd = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    look = nominal_pose[:3] * np.array([1, 1, 0]) + 0.35 * fwd + np.array([0, 0, 0.55])
    view = np.array([math.cos(yaw0 + math.radians(70)), math.sin(yaw0 + math.radians(70)), 0.0])
    eye = look + 2.7 * view + np.array([0, 0, 0.35])
    camera.set_world_poses_from_view(torch.tensor(eye[None], dtype=torch.float32, device=dev),
                                     torch.tensor(look[None], dtype=torch.float32, device=dev))
    show_markers(None)
    for _ in range(30):                     # RTX warm-up
        env.sim.render()

    writer = None
    if not args.preview:
        writer = imageio_ffmpeg.write_frames(str(args.out / "t2_presentation.mp4"), (W, H), fps=FPS, codec="libx264",
                                             macro_block_size=8,
                                             output_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"])
        writer.send(None)
    frames = 0

    def emit(frame, n=1):
        nonlocal frames
        for _ in range(n):
            if writer is not None:
                writer.send(frame.tobytes())
            frames += 1

    start = time.monotonic()
    nominal_frame = capture()
    Image.fromarray(overlay(nominal_frame, title=True)).save(args.out / "preview_0_nominal.png")
    emit(overlay(nominal_frame, title=True), round(TITLE_S * FPS))

    segments = []
    for s, row in enumerate(rows):
        band = P.BAND_NAMES[int(targets["band"][row])]
        pos_G = targets["position_G"][row]
        k = int(data["posture_index"][targets["voxel"][row]])
        dh, pitch = float(data["posture_dh"][k]), float(data["posture_pitch"][k])
        nominal_pose, nominal_q = settle_nominal()
        cmd.assign(torch.arange(1, device=dev), targets["position_G"][row:row + 1], targets["quat_G_wxyz"][row:row + 1],
                   targets["band"][row:row + 1])
        target_w = cmd.position_w[0].cpu().numpy().copy()
        q_goal, _ = E.oracle_targets(np, data, targets["voxel"][row:row + 1], joint_names, default)
        q_goal = np.clip(q_goal, soft[:, 0], soft[:, 1])[0]
        G_env = np.eye(4)
        G_env[:3, :3] = Rotation.from_euler("z", float(cmd.yaw[0])).as_matrix()
        G_env[:3, 3] = cmd.origin_w[0].cpu().numpy()
        base = G_env @ np.linalg.inv(data["G_world"]) @ data["posture_base_world"][k]
        goal_pose = np.r_[base[:3, 3], Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]]]
        info = (f"Band {BAND_LABEL[band]} · target z = {pos_G[2]:.2f} m · Δh = {dh:+.2f} m · "
                f"base pitch = {pitch:+.2f} rad ({math.degrees(pitch):+.0f}°) · target id {ids[s]}")
        # 1 s nominal with target
        write_state(nominal_pose, nominal_q)
        show_markers(target_w)
        frame = capture()
        emit(overlay(frame, [info, "nominal pose · orange: target · cyan: TCP"]), round(NOMINAL_S * FPS))
        # 3 s kinematic joint-space interpolation (smoothstep), no physics
        slerp = Slerp([0, 1], Rotation.from_quat([nominal_pose[[4, 5, 6, 3]], goal_pose[[4, 5, 6, 3]]]))
        n_interp = round(INTERP_S * FPS)
        for i in range(1, n_interp + 1):
            u = i / n_interp
            w = u * u * (3 - 2 * u)
            pose = np.r_[(1 - w) * nominal_pose[:3] + w * goal_pose[:3], slerp(w).as_quat()[[3, 0, 1, 2]]]
            q = (1 - w) * nominal_q + w * q_goal
            q[12:16] = nominal_q[12:16]
            write_state(pose, q)
            show_markers(target_w)
            emit(overlay(capture(), [info, "joint-space interpolation to the stored map posture (kinematic, 3 s)"]))
        Image.fromarray(overlay(capture(), [info, "stored map posture"])).save(args.out / f"preview_{s + 1}_{band}.png")
        if args.preview:
            segments.append(dict(target_id=ids[s], band=band))
            continue
        # 2 s physics hold, gravity-compensated PD targets as in A3a
        env.episode_length_buf[:] = 0
        env.sim.physics_sim_view.update_articulations_kinematic()
        tau, statics = E.static_torques(env, robot, torch, np, 1, jid, E.foot_body_ids(robot))
        stiffness = robot.data.joint_stiffness[0, jid].cpu().numpy()
        effort = robot.data.joint_effort_limits[0, jid].cpu().numpy()
        q_hold = q_goal.copy()
        q_hold[pos_cols] += tau[0, pos_cols] / stiffness[pos_cols]
        action = torch.as_tensor(q_hold - default, dtype=torch.float32, device=dev)[None]
        action[:, 12:16] = 0.0
        steps_per_frame = env.step_dt * FPS             # 50 Hz control, 30 fps video
        n_hold, t_sim, err, terminated = round(HOLD_S * FPS), 0.0, None, []
        last = None
        for i in range(n_hold):
            while t_sim < (i + 1) / FPS - 1e-9:
                env.step(action)
                t_sim += env.step_dt
                if bool(env.reset_buf[0]):
                    terminated = [t for t in env.termination_manager.active_terms
                                  if bool(env.termination_manager.get_term(t)[0])]
                    break
            if terminated:
                break
            err = float(torch.linalg.norm(cmd.tcp_pose_w()[0][0] - cmd.position_w[0]))
            show_markers(target_w)
            last = overlay(capture(), [info, f"physics hold, gravity-compensated PD as in A3a · t = {t_sim:.1f} s · "
                                             f"TCP error {100 * err:.1f} cm"])
            emit(last)
        if terminated:
            raise RuntimeError(f"segment {ids[s]} terminated during the hold: {terminated}")
        result = f"held 2 s · TCP error {100 * err:.1f} cm"
        emit(overlay(np.asarray(last), None, result=result), round(RESULT_S * FPS))
        segments.append(dict(target_id=ids[s], band=band, target_z_m=float(pos_G[2]), position_G=pos_G.tolist(),
                             dh_m=dh, pitch_rad=pitch, tcp_error_end_of_hold_m=err,
                             torque_ratio_max=float((np.abs(tau[0]) / effort)[pos_cols].max()), steps_per_frame=steps_per_frame))
        print("segment", segments[-1], flush=True)
    Image.fromarray(figure_frame()).save(args.out / "preview_4_figure.png")
    if not args.preview:
        emit(figure_frame(), round(FIGURE_S * FPS))
        writer.close()
        final = args.out / "t2_presentation.mp4"
        sidecar = dict(video=str(final), protocol_version=P.VERSION, resolution=[W, H], fps=FPS, frames=frames,
                       duration_s=frames / FPS, eval_set_sha256=meta["targets_sha256"], eval_seed=P.EVAL_SEED,
                       reachability_sha256=cmd.reachability_sha256, figure=str(args.figure), segments=segments,
                       camera=dict(eye=eye.tolist(), look=look.tolist(), focal_length_mm=18.0,
                                   placement="fixed, front-left side view (yaw + 70 deg), same for all segments"),
                       timing_s=dict(title=TITLE_S, nominal=NOMINAL_S, interpolation=INTERP_S, hold=HOLD_S,
                                     result=RESULT_S, figure=FIGURE_S),
                       note="no learned policy; interpolation is kinematic (state writes, no physics); "
                            "hold is simulated with gravity-compensated PD targets (eval_t2.static_torques, as A3a)",
                       render_wall_s=time.monotonic() - start, command_argv=sys.argv,
                       sha256=hashlib.sha256(final.read_bytes()).hexdigest())
        (args.out / "t2_presentation.json").write_text(json.dumps(sidecar, indent=1) + "\n")
        print("written", final, frames, "frames", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:   # an exception before app.close() otherwise hangs Kit's shutdown
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
