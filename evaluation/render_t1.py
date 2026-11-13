"""Render T1 demo videos with the oracle controller (GPU, offscreen RTX camera, 1 env).

    render_t1.py --targets <targets.json> --ids fixed-00,fixed-01,... --name fixed_targets --out <dir> [--trail]
One frame per control step (50 Hz, played at 50 fps = real time), full protocol episodes
(settle 2 s + fixed hold 5 s incl. the 2 s quintic transition; circle: settle 2 s + transition 2 s +
circle 40 s). Scene markers: target frame (long axes) and EE/link06 frame (short axes), RGB = xyz;
optional EE trail. Pass 1 renders <name>_raw.mp4 and writes this run's steps.csv; pass 2 draws the
overlay (time, target index, position/orientation error) from steps.csv into <name>.mp4.
<name>.json is the sidecar: command, per-target errors (protocol.target_metrics), frame counts.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT/"training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
PACKAGE = "robot_lab.tasks.manager_based.manipulation.standing_ee"
FONT = "/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"


def load_eval():
    spec = importlib.util.spec_from_file_location("_eval_t1", ROOT/"evaluation/eval_t1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose(raw, steps_csv, out, labels, protocol_version, seed, width, height):
    """Pass 2: overlay from steps.csv (one row per frame, same order)."""
    import csv
    import math
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    csv.field_size_limit(1 << 30)
    with open(steps_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    big, small = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 26)
    reader = imageio_ffmpeg.read_frames(str(raw))
    meta = next(reader)
    assert tuple(meta["size"]) == (width, height), meta
    writer = imageio_ffmpeg.write_frames(str(out), (width, height), fps=50, codec="libx264", macro_block_size=8,
                                         output_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"])
    writer.send(None)
    frames = 0
    for row, frame in zip(rows, reader):
        image = Image.fromarray(np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, height-70, width, height], fill=(0, 0, 0, 170))
        index, total = labels[row["target_id"]]
        status = " · TERMINATED: " + ",".join(json.loads(row["failure_reason"])) if row["failure"] == "True" else ""
        text = (f"t = {float(row['episode_time_s']):5.2f} s   ·   target {index}/{total}: {row['target_id']}   ·   "
                f"phase {row['phase']}   ·   position error {100*float(row['position_error_m']):6.2f} cm   ·   "
                f"orientation error {math.degrees(float(row['orientation_error_geodesic_rad'])):6.1f} deg{status}")
        draw.text((24, height-54), text, font=big, fill=(255, 255, 255))
        draw.rectangle([0, 0, 760, 92], fill=(0, 0, 0, 150))
        draw.text((20, 12), f"T1 · protocol {protocol_version} · seed {seed}", font=big, fill=(255, 255, 255))
        draw.text((20, 54), "long axes: target frame · short axes: EE (link06) · RGB = xyz", font=small, fill=(220, 220, 220))
        writer.send(np.asarray(image).tobytes())
        frames += 1
    writer.close()
    if frames != len(rows):
        raise ValueError(f"Frame/row mismatch: {frames} frames, {len(rows)} steps.csv rows")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trail", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    args.out.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    import robot_lab.tasks  # noqa: F401
    import imageio_ffmpeg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sensors import CameraCfg
    parent = importlib.import_module("robot_lab.tasks.manager_based.manipulation")
    parent.__path__.insert(0, str(OVERLAY))
    importlib.import_module(PACKAGE)
    geometry = importlib.import_module(PACKAGE + ".geometry")
    protocol = importlib.import_module(PACKAGE + ".protocol")
    ev = load_eval()

    sampled = protocol.load_targets(args.targets)
    by_id = {t["target_id"]: t for t in sampled["targets"]}
    ids = args.ids.split(",")
    targets = [by_id[i] for i in ids]
    cfg = importlib.import_module(PACKAGE + ".env_cfg").StandingEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.seed = sampled["seed"]
    cfg.sim.render_interval = 10**6   # render explicitly after each control step (markers in sync)
    cfg.scene.render_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/RenderCam", update_period=0.0, width=args.width,
                                     height=args.height, data_types=["rgb"],
                                     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 50.0)))
    env = gym.make(protocol.TASK_ID, cfg=cfg).unwrapped
    robot, command, camera = env.scene["robot"], env.command_manager.get_term("ee_pose"), env.scene["render_cam"]
    action = env.action_manager.get_term("joints")

    def axes(name, length, thick):
        return {f"{name}{a}": sim_utils.CuboidCfg(size=tuple(length if k == i else thick for k in range(3)),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(1.0 if k == i else 0.1 for k in range(3))))
                for i, a in enumerate("xyz")}
    frames_vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/t1_frames",
                                      markers={**axes("t", 0.18, 0.014), **axes("e", 0.11, 0.009)}))
    trail_vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/t1_trail", markers={"dot": sim_utils.SphereCfg(
        radius=0.006, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)))}))
    lengths = (0.18, 0.11)

    def draw_frames(goal, ee):
        positions, quats = [], []
        for pose, length in ((goal, lengths[0]), (ee, lengths[1])):
            for i in range(3):
                positions.append(pose[:3, 3] + pose[:3, i] * length / 2)
                quats.append(geometry.pose(pose)[3:])
        frames_vis.visualize(translations=torch.tensor(np.array(positions), dtype=torch.float32, device=env.device),
                             orientations=torch.tensor(np.array(quats), dtype=torch.float32, device=env.device),
                             marker_indices=torch.arange(6, device=env.device))

    def aim_camera():
        base = robot.data.root_pos_w[0].cpu().numpy()
        yaw = float(torch.atan2(robot.data.root_quat_w[0, 3], robot.data.root_quat_w[0, 0])) * 2
        direction = np.array([np.cos(yaw + np.pi/4), np.sin(yaw + np.pi/4), 0.0])   # front-left of the robot
        look = base + np.array([0.15, 0.0, 0.35])
        eye = look + 2.3 * direction + np.array([0.0, 0.0, 0.95])                  # |eye - look| ~ 2.49 m
        camera.set_world_poses_from_view(torch.tensor(eye[None], dtype=torch.float32, device=env.device),
                                         torch.tensor(look[None], dtype=torch.float32, device=env.device))
        return float(np.linalg.norm(eye - look))

    def capture():
        env.sim.render()
        camera.update(env.step_dt, force_recompute=True)
        return camera.data.output["rgb"][0, ..., :3].contiguous().cpu().numpy().astype(np.uint8)

    raw = args.out/f"{args.name}_raw.mp4"
    writer = imageio_ffmpeg.write_frames(str(raw), (args.width, args.height), fps=50, codec="libx264", macro_block_size=8,
                                         output_params=["-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p"])
    writer.send(None)
    log = ev.CsvLog(args.out/"steps.csv")
    nominal_q = np.array([protocol.NOMINAL_ARM_POSE[f"joint{i}"] for i in range(1, 7)])
    trials, frame_count, start, distance = [], 0, time.monotonic(), None
    with torch.inference_mode():
        for n, target in enumerate(targets):
            command.assign(0, target)
            env.reset(seed=sampled["seed"])
            trail = []
            trial = ev.new_trial(target, n, n, 0)
            if n == 0:   # RTX warm-up on the reset state (no physics step)
                distance = aim_camera()
                for _ in range(30):
                    env.sim.render()
            last = None
            for _ in range(round(command.duration(0) / env.step_dt) + 1):
                t = float(env.episode_length_buf[0]) * env.step_dt
                offsets = torch.zeros(1, 22, device=env.device)
                q = ev.oracle_arm_q(target, t, nominal_q, geometry.quintic)
                offsets[0, 16:] = torch.as_tensor(q, device=env.device, dtype=torch.float32) - \
                    robot.data.default_joint_pos[0, action.joint_ids[16:]]
                env.step(offsets)
                sample = env.t1_samples[0]
                sample.update(protocol_version=protocol.VERSION, robot_variant="B2W", seed=sampled["seed"], batch=n, trial_id=n)
                log.append(sample)
                ended = ev.update_trial(trial, sample)
                if not ended:   # after a terminal step the env has already auto-reset: repeat the last image
                    goal = geometry.transform(sample["ee_goal_pos_world_xyz"], sample["ee_goal_quat_world_wxyz"])
                    ee = geometry.transform(sample["ee_pos_world_xyz"], sample["ee_quat_world_wxyz"])
                    draw_frames(goal, ee)
                    if args.trail:
                        trail.append(sample["ee_pos_world_xyz"])
                        trail_vis.visualize(translations=torch.tensor(trail, dtype=torch.float32, device=env.device))
                    aim_camera()
                    last = capture()
                writer.send(last.tobytes())
                frame_count += 1
                if ended:
                    break
            trials.append(trial)
            print(f"rendered {target['target_id']}: success={trial['completed'] and not trial['failure']}", flush=True)
    writer.close()
    log.close()
    render_s = time.monotonic() - start
    labels = {t["target_id"]: (i + 1, len(targets)) for i, t in enumerate(targets)}
    final = args.out/f"{args.name}.mp4"
    composed = compose(raw, args.out/"steps.csv", final, labels, protocol.VERSION, sampled["seed"], args.width, args.height)
    per_target = []
    for trial in trials:
        m = protocol.target_metrics(trial)
        per_target.append(dict(target_id=trial["target_id"], family=trial["family"],
                               success=trial["completed"] and not trial["failure"], failure_reason=trial["failure_reason"],
                               samples=trial["samples"],
                               tracking_position_mean_cm=100*m["tracking_position_m_mean"],
                               tracking_position_rms_cm=100*m["tracking_position_m_rms"],
                               tracking_position_max_cm=100*m["tracking_position_m_max"],
                               tracking_orientation_mean_deg=np.degrees(m["tracking_orientation_geodesic_rad_mean"]),
                               tracking_orientation_max_deg=np.degrees(m["tracking_orientation_geodesic_rad_max"]),
                               steady_position_mean_cm=None if m["steady_position_m_mean"] is None else 100*m["steady_position_m_mean"],
                               steady_orientation_mean_deg=None if m["steady_orientation_geodesic_rad_mean"] is None
                               else float(np.degrees(m["steady_orientation_geodesic_rad_mean"]))))
    sidecar = dict(video=str(final), raw_video=str(raw), steps_csv=str(args.out/"steps.csv"), protocol_version=protocol.VERSION,
                   seed=sampled["seed"], targets_json=str(args.targets), target_ids=ids, controller="oracle", num_envs=1,
                   resolution=[args.width, args.height], fps=50, control_dt_s=env.step_dt, real_time=True,
                   frames=composed, duration_s=composed/50, camera=dict(placement="front-left of the robot (yaw+45 deg)",
                   distance_to_look_point_m=distance, focal_length_mm=24.0), render_wall_s=render_s,
                   episode_definition="full protocol episodes: settle 2 s; fixed: 5 s hold incl. 2 s quintic transition; "
                                      "circle: 2 s transition + 40 s circle",
                   metrics="protocol.target_metrics on this run's steps.csv (fixed: whole 5 s hold / last 2 s; circle: (20, 40] s)",
                   command_argv=sys.argv, per_target=[{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}
                                                      for r in per_target],
                   sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    (args.out/f"{args.name}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print("written", final, flush=True)
    env.close()
    app.close()



if __name__ == "__main__":
    main()
