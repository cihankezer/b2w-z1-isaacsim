"""T2 diagnostic (CPU): contact partners of the hip contacts in A3 leg-phase falls.

For every A3 episode that fell during the leg phase (n_steps < 100, contact bodies only hips), walk the
crouch kinematically from the nominal posture to the episode's map posture (dh, pitch scaled by s in 0..1,
wheel centres fixed, leg IK as in the reachability map) and measure the clearance of each contacting hip's
colliders to the ground and to every link PhysX collides it with (non-adjacent per the USD joint tree and
filtered pairs, as in the T1 collider model). Reports per episode the first s with any hip clearance < 0 and
the partner, and a summary over episodes. Writes contact_partners.json to --out.
"""

import argparse
import collections
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import usd_loader  # noqa: E402
import numpy as np  # noqa: E402
from reachability_t2 import t2_module  # noqa: E402


def profile(model, wb, W, C, P, args):
    """Minimum leg self clearance (body_body pairs) per posture (dh, pitch) of the protocol grid."""
    pairs = W._body_pairs(model)
    out = {}
    for dh in P.dh_values(P.DH_MIN_M):
        for pitch in P.pitch_values():
            base = W.base_pose(wb.G, P.H_NOM_M + dh, pitch)
            legs, reason = wb.solve_legs(base)
            if legs is None:
                out[f"{dh:+.2f}/{pitch:+.2f}"] = reason
                continue
            frames = model.forward(base, dict(wb.joints_nominal, **legs))
            best = (1.0, None)
            for a, b in pairs:
                for sa in model.shapes[a]:
                    for sb in model.shapes[b]:
                        d = C.surface_clearance(C.moved_shape(sa, frames[a]), C.moved_shape(sb, frames[b]))["clearance_m"]
                        if d < best[0]:
                            best = (d, f"{a}:{b}")
            out[f"{dh:+.2f}/{pitch:+.2f}"] = [round(best[0], 4), best[1]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leg_clearance_profile.json").write_text(json.dumps(out, indent=1) + "\n")
    print("pitch:", " ".join(f"{p:+.2f}" for p in P.pitch_values()))
    for dh in P.dh_values(P.DH_MIN_M):
        cells = []
        for pitch in P.pitch_values():
            v = out[f"{dh:+.2f}/{pitch:+.2f}"]
            cells.append(f"{v[0] * 100:5.1f}" if isinstance(v, list) else "  IK ")
        print(f"dh {dh:+.2f}:", " ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/t1-run/artifacts"))
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--profile", action="store_true", help="min hip-wheel clearance over the posture grid")
    parser.add_argument("--check", type=float, nargs=2, default=None, metavar=("DH", "PITCH"))
    args = parser.parse_args()
    from pxr import Usd
    K = t2_module("standing_ee.kinematics")
    C = t2_module("standing_ee.clearance")
    W = t2_module("reach_height.wholebody")
    P = t2_module("reach_height.protocol")
    stage = Usd.Stage.Open(str(usd_loader.ASSET))
    root = next(p for p in stage.Traverse() if p.GetName() == "base_link").GetParent()
    model = K.UsdArmKinematics(stage, str(root.GetPath()))
    wb = W.WholeBody(model)

    if args.check:
        dh, pitch = args.check
        base, joints, reason = wb.posture(dh, pitch)
        print("posture", dh, pitch, "->", reason)
        legs, _ = wb.solve_legs(W.base_pose(wb.G, P.H_NOM_M + dh, pitch))
        frames = model.forward(base, dict(wb.joints_nominal, **legs))
        print("clearance_violation:", W.clearance_violation(model, frames))
        for a, b in W._body_pairs(model):
            d = min(C.surface_clearance(C.moved_shape(x, frames[a]), C.moved_shape(y, frames[b]))["clearance_m"]
                    for x in model.shapes[a] for y in model.shapes[b])
            if d < 0.04:
                print(f"  {a}:{b} {d:.4f}  shapes {len(model.shapes[a])}x{len(model.shapes[b])}")
        return
    if args.profile:
        return profile(model, wb, W, C, P, args)
    rows = [r for r in csv.DictReader(open(args.episodes))
            if r["outcome"] == "fall" and int(r["n_steps"]) < 100 and r["contact_bodies"]
            and all(b.endswith("_hip") for b in r["contact_bodies"].split("|"))]
    names = sorted(model.shapes)
    report, first_partner, final_partner = [], collections.Counter(), collections.Counter()
    for r in rows:
        dh, pitch = float(r["oracle_posture_dh"]), float(r["oracle_posture_pitch"])
        hips = r["contact_bodies"].split("|")
        found = None
        final = {}
        for s in np.linspace(0.0, 1.0, args.steps):
            base = W.base_pose(wb.G, P.H_NOM_M + s * dh, s * pitch)
            legs, reason = wb.solve_legs(base)
            if legs is None:
                continue
            frames = model.forward(base, dict(wb.joints_nominal, **legs))
            dist = {}
            for hip in hips:
                shapes_h = [C.moved_shape(x, frames[hip]) for x in model.shapes[hip]]
                dist[(hip, "ground")] = min(C.ground_clearance(x)["clearance_m"] for x in shapes_h)
                for other in names:
                    if other == hip or frozenset((hip, other)) in model.adjacent:
                        continue
                    shapes_o = [C.moved_shape(x, frames[other]) for x in model.shapes[other]]
                    dist[(hip, other)] = min(C.surface_clearance(a, b)["clearance_m"] for a in shapes_h for b in shapes_o)
            if s == 1.0:
                final = dist
            neg = {k: v for k, v in dist.items() if v < 0}
            if neg and found is None:
                found = dict(s=float(s), pairs={f"{a}:{b}": round(v, 4) for (a, b), v in sorted(neg.items(), key=lambda kv: kv[1])})
                first_partner.update(b for (_, b) in neg)
        closest = sorted(final.items(), key=lambda kv: kv[1])[:3]
        final_partner.update(b for (a, b), v in final.items() if v < 0)
        report.append(dict(target_id=int(r["target_id"]), band=r["band"], dh=dh, pitch=pitch, hips=hips,
                           n_steps=int(r["n_steps"]), first_penetration=found,
                           final_closest={f"{a}:{b}": round(v, 4) for (a, b), v in closest}))
    summary = dict(episodes=len(rows), with_kinematic_penetration=sum(x["first_penetration"] is not None for x in report),
                   partners_at_first_penetration=dict(first_partner), partners_penetrating_at_final_posture=dict(final_partner),
                   first_penetration_s=[x["first_penetration"]["s"] for x in report if x["first_penetration"]])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "contact_partners.json").write_text(json.dumps(dict(summary=summary, episodes=report), indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for x in report[:8]:
        print(x)


if __name__ == "__main__":
    main()
