"""Visual evidence for the flagged Z1 collider pairs (GPU render, no physics stepping).

The robot is posed kinematically from samples of the saved 30 s nominal trace
(write_root_pose/joint_state + sim.forward). Collider overlays are separate,
non-physical capsule prims under /World/T1Overlay built from the USD collider
parameters (radius/height/axis/local transform) of an independently opened copy
of the asset; the asset file, collision prims and physics are never modified.
Visual-hiding passes only toggle visibility of in-memory visual prims.
Shots: link02-link04 at nominal stow, link02-link06 at the capsule trace minimum;
labels show capsule (logged) and CAD (policy draft.3) distances. Afterwards the
leg/wheel cooked hulls (CollisionAPI Xform query) are checked against USD points.
lifecycle.json records shutdown stages.

    render_pairs.py --trace <artifacts dir> --cad <cad_sweep.json> [--out /t1-run/artifacts]
"""
import argparse
import atexit
import csv
import importlib
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True)
parser.add_argument("--cad", required=True, help="cad_sweep.json (trace minima)")
parser.add_argument("--out", default="/t1-run/artifacts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
stages = []


def mark(stage):
    stages.append(dict(stage=stage, t=time.time()))
    with open(os.path.join(args.out, "lifecycle.json"), "w") as f:
        json.dump(dict(stages=stages), f, indent=1)
    print("LIFECYCLE:", stage, flush=True)


atexit.register(mark, "python_atexit")
app = AppLauncher(args).app
mark("app_started")

COLORS = {"link00": (1.0, 0.55, 0.0), "link02": (0.9, 0.1, 0.1), "link03": (0.6, 0.6, 0.6),
          "link04": (0.1, 0.75, 0.2), "link06": (0.15, 0.35, 1.0)}
PAIR_COLLIDERS = {  # pair -> collider pair as named in the reports
    "link02-link06": ("link02", "mesh_0", "link06", "mesh_0"),
    "link00-link02": ("link00", "mesh_0", "link02", "mesh_0"),
    "link04-link06": ("link04", "mesh_0", "link06", "mesh_0"),
    "link02-link04": ("link02", "mesh_1", "link04", "mesh_0"),
}


def main():
    import numpy as np
    import torch
    import gymnasium as gym
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    import robot_lab.tasks  # noqa: F401

    root = "/workspace/training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation"
    importlib.import_module("robot_lab.tasks.manager_based.manipulation").__path__.insert(0, root)
    pkg = "robot_lab.tasks.manager_based.manipulation.standing_ee"
    importlib.import_module(pkg)
    k, cl, g, cadm = (importlib.import_module(f"{pkg}.{m}") for m in ("kinematics", "clearance", "geometry", "cad"))
    cfg = importlib.import_module(pkg + ".env_cfg").StandingEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.t1_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/T1Cam", update_period=0.0, height=720, width=1280,
                                 data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                                     focal_length=18.0, clipping_range=(0.01, 50.0)))
    env = gym.make("RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0", cfg=cfg).unwrapped
    env.reset(seed=1101)
    mark("env_ready")
    stage = omni.usd.get_context().get_stage()
    robot, cam = env.scene["robot"], env.scene["t1_cam"]
    dev = env.device

    # independent model of the authoritative asset (not the live stage)
    asset = Usd.Stage.Open(robot.cfg.spawn.usd_path)
    model = k.UsdArmKinematics(asset, "/Unitree_B2W")
    interface = json.load(open(os.path.join(args.trace, "interface.json")))
    physics = list(csv.DictReader(open(os.path.join(args.trace, "physics.csv"))))
    cad_audit = json.load(open(args.cad))
    order = interface["joint_order"]

    def state(tick):
        row = physics[tick - 1]
        assert int(row["physics_tick"]) == tick
        return (json.loads(row["base_position_world_xyz"]), json.loads(row["base_quaternion_world_wxyz"]),
                dict(zip(order, json.loads(row["joint_pos"]))), float(row["time_s"]))

    # ---- overlay capsules ------------------------------------------------------
    looks = {}
    for link, col in COLORS.items():
        path = f"/World/T1Overlay/Looks/{link}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[0.35 * x for x in col]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        looks[link] = mat
    overlays = []
    for link in COLORS:
        for shape in model.shapes.get(link, []):
            if shape["kind"] != "capsule":
                continue
            e0, e1 = shape["ends"]
            axis = (e1 - e0) / np.linalg.norm(e1 - e0)
            z = np.array([0, 0, 1.0])
            v, cth = np.cross(z, axis), float(z @ axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx / (1 + cth) if cth > -0.999999 else np.diag([1, -1, -1.0])
            local = np.eye(4)
            local[:3, :3], local[:3, 3] = rot, (e0 + e1) / 2
            name = shape["path"].split("/collisions/")[-1].split("/")[0]
            prim_path = f"/World/T1Overlay/{link}_{name}"
            capg = UsdGeom.Capsule.Define(stage, prim_path)
            capg.CreateAxisAttr("Z")
            capg.CreateRadiusAttr(shape["radius"])
            capg.CreateHeightAttr(float(np.linalg.norm(e1 - e0)))
            UsdShade.MaterialBindingAPI.Apply(capg.GetPrim()).Bind(looks[link])
            op = UsdGeom.Xformable(capg).AddTransformOp()
            overlays.append(dict(link=link, name=name, local=local, op=op))

    names = robot.joint_names
    zeros6 = torch.zeros(1, 6, device=dev)
    sim_vs_fk = []

    def pose(tick):
        pos, quat, joints, t = state(tick)
        robot.write_root_pose_to_sim(torch.tensor([pos + quat], device=dev, dtype=torch.float32))
        robot.write_root_velocity_to_sim(zeros6)
        jp = torch.tensor([[joints[n] for n in names]], device=dev, dtype=torch.float32)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        env.sim.forward()
        frames = model.forward(g.transform(pos, quat), joints)
        for o in overlays:
            o["op"].Set(Gf.Matrix4d((frames[o["link"]] @ o["local"]).T.tolist()))
        # cross-check: live PhysX link poses vs independent FK (xyzw from PhysX)
        tf = robot.root_physx_view.get_link_transforms()[0].cpu().numpy()
        err = max(float(np.linalg.norm(tf[robot.body_names.index(n), :3] - frames[n][:3, 3]))
                  for n in COLORS if n in robot.body_names)
        sim_vs_fk.append(dict(tick=tick, max_link_position_diff_m=err))
        return frames, t

    def clearance_lines(frames, pair):
        a, b = pair.split("-")
        lines = []
        for sa in model.shapes.get(a, []):
            for sb in model.shapes.get(b, []):
                d = cl.surface_clearance(cl.moved_shape(sa, frames[a]), cl.moved_shape(sb, frames[b]))["clearance_m"]
                na = sa["path"].split("/collisions/")[-1].split("/")[0]
                nb = sb["path"].split("/collisions/")[-1].split("/")[0]
                lines.append(f"capsule {a}/{na} - {b}/{nb}: {1000 * d:+.2f} mm (logged, non-gating)")
        d, hit = cadm.mesh_distance(model.cad[a]["bvh"], frames[a], model.cad[b]["bvh"], frames[b])
        lines.append(f"CAD visual mesh {a} - {b}: {'INTERSECTING' if hit else f'{1000 * d:.2f} mm'} (policy draft.3 metric)")
        return lines

    def visuals(visible, links=None):
        for link in links or [f"link0{i}" for i in range(7)]:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/z1/{link}/visuals")
            if prim.IsValid():
                (UsdGeom.Imageable(prim).MakeVisible if visible else UsdGeom.Imageable(prim).MakeInvisible)()

    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=18)
    except TypeError:
        font = small = ImageFont.load_default()

    def shoot(eye, target, path, title, lines, renders=12):
        cam.set_world_poses_from_view(torch.tensor([eye], device=dev, dtype=torch.float32),
                                      torch.tensor([target], device=dev, dtype=torch.float32))
        for _ in range(renders):
            env.sim.render()
        cam.update(0.0, force_recompute=True)
        img = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 1280, 34 + 24 * len(lines)], fill=(0, 0, 0))
        d.text((10, 6), title, fill=(255, 255, 255), font=font)
        for i, line in enumerate(lines):
            d.text((10, 36 + 24 * i), line, fill=(230, 230, 230), font=small)
        x = 10
        for link, col in COLORS.items():
            d.rectangle([x, 690, x + 18, 708], fill=tuple(int(255 * c) for c in col))
            d.text((x + 24, 688), link, fill=(255, 255, 255), font=small)
            x += 110
        img.save(path)
        return path

    def view(center, dist, az, el):
        a, e = math.radians(az), math.radians(el)
        return [center[0] + dist * math.cos(e) * math.cos(a), center[1] + dist * math.cos(e) * math.sin(a),
                center[2] + dist * math.sin(e)]

    def pair_center(frames, pair):
        a, na, b, nb = PAIR_COLLIDERS[pair]
        pts = []
        for link, name in ((a, na), (b, nb)):
            s = next(s for s in model.shapes[link] if f"/collisions/{name}/" in s["path"])
            pts.append(cl.moved_shape(s, frames[link])["ends"].mean(0))
        return ((pts[0] + pts[1]) / 2).tolist()

    out = args.out
    media = dict(images=[], videos=[], frames_dirs=[])
    final_tick = len(physics)
    tick_0206 = cad_audit["trace"]["pairs"]["link02-link06"]["capsule_min_tick"]
    shots = [("link02-link04", final_tick, "nominal stow (final trace sample)"),
             ("link02-link06", tick_0206, "capsule trace minimum")]
    # wide context shot
    frames, t = pose(final_tick)
    base = frames["base_link"][:3, 3].tolist()
    media["images"].append(shoot(view(base, 2.2, 215, 20), base, f"{out}/wide_nominal.png",
                                 f"B2W+Z1 nominal standing, trace tick {final_tick} (t={t:.2f} s); overlays = Z1 USD capsules", []))
    for pair, tick, label in shots:
        frames, t = pose(tick)
        center = pair_center(frames, pair)
        lines = clearance_lines(frames, pair)
        for az, el in ((30, 25), (120, 25), (210, 25), (300, 25), (0, 80)):
            for mode in ("visual+collider", "collider-only"):
                visuals(mode == "visual+collider")
                fn = f"{out}/{pair}_tick{tick}_az{az}_el{el}_{mode.replace('+', '_')}.png"
                media["images"].append(shoot(view(center, 0.32, az, el), center, fn,
                                             f"{pair} | {label} tick {tick} (t={t:.2f} s) | {mode} | az {az} el {el}",
                                             lines))
        visuals(True)
    mark("stills_done")

    # ---- task 6: cooked hulls of the leg/wheel CollisionAPI prims, frame check vs USD points ----
    cooking = {}
    try:
        live = k.UsdArmKinematics(stage, "/World/envs/env_0/Robot")
        for leg in ("FL", "FR", "RL", "RR"):
            for part in ("calf", "foot"):
                name = f"{leg}_{part}"
                shapes = live.shapes.get(name, [])
                entry = dict(sources=[s.get("source", s["kind"]) for s in shapes],
                             issues=[s.get("issue") for s in shapes if not s.get("resolved", True)])
                body = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
                pts = []
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                    if prim.IsA(UsdGeom.Mesh) and "/collisions/" in str(prim.GetPath()):
                        rel = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim) *
                                       UsdGeom.XformCache().GetLocalToWorldTransform(body).GetInverse()).T
                        pts.append(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()) @ rel[:3, :3].T + rel[:3, 3])
                hulls = [s for s in shapes if s["kind"] == "polytope"]
                if hulls and pts:
                    P = np.vstack(pts)
                    outside = max(float((P @ h["planes"][:, :3].T + h["planes"][:, 3]).max()) for h in hulls)
                    from scipy.spatial import ConvexHull
                    exact = ConvexHull(P)
                    overshoot = max(float((h["vertices"] @ exact.equations[:, :3].T + exact.equations[:, 3]).max()) for h in hulls)
                    entry.update(hull_vertex_counts=[len(h["vertices"]) for h in hulls], usd_points=len(P),
                                 max_usd_point_outside_cooked_hull_m=outside,
                                 max_cooked_vertex_outside_exact_hull_m=overshoot)
                cooking[name] = entry
        from pxr import PhysxSchema
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/FL_calf/collisions/FL_calf/Scene")
        api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        cooking["FL_calf_schema"] = dict(has_api=prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI),
            hull_vertex_limit=api.GetHullVertexLimitAttr().Get(), min_thickness=api.GetMinThicknessAttr().Get())
    except Exception as exc:  # recorded, not hidden
        cooking["error"] = f"{type(exc).__name__}: {exc}"
    mark("cooking_done")
    json.dump(dict(media=media, sim_vs_fk=sim_vs_fk, cooking=cooking,
                   max_sim_vs_fk_m=max(x["max_link_position_diff_m"] for x in sim_vs_fk)),
              open(f"{out}/render_result.json", "w"), indent=1, default=str)
    env.close()
    mark("env_close_returned")


main()
mark("main_returned")
app.close()
mark("app_close_returned")
