"""
Compare candidate third-person camera placements for one layout, the way a VLA sees them.

    MUJOCO_GL=egl python camera_compare.py              # layout "both" from layout.json
    python camera_compare.py --layout pull

For each candidate it reports what matters for trainable data:
  part_px      width of a 30 mm part at the plate center in a 224 px wide model input
  grip_frame   share of all task waypoints with the gripper inside the image
  grip_seen    share of waypoints where the gripper is visible (not hidden by printer/arm)
  plate_seen   share of plate points visible through the door with the arm parked
  plate_clear  share of those plate points NOT hidden by the arm while it works inside
  elbow_frame  share of waypoints with the elbow in the image
  base_frame   whether the arm base is in the image
  keep_out     closest approach of the arm to the camera (m)
and writes results/figures/camera_compare_<layout>.png: three task moments per camera
plus the grasp moment squashed to 224 x 224 (what the model gets).

Add your own candidates to CANDIDATES (printer frame, metres: x right, y into printer,
z above the printer's base; the table top is at min(0, rel_height)).
"""
import argparse
import json
import os
import warnings

import numpy as np
import mujoco
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "egl")

from config import Config, plate_top_z
from evaluate import Evaluator
from camera_study import CameraStudy, look_at, robot_points
from render import Shots, set_scene_cam, _font

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_W = 224
IMG_W, IMG_H, VFOV = 640, 360, 42.0     # D435 color 16:9, ~69 x 42 deg


def candidates(L, P, zt):
    """Named (position, look-at target) pairs. Targets are in the printer frame."""
    base = np.array([L["base_x"], -L["base_dist"], L["rel_height"] + 0.25])
    task = np.array([-0.03, -0.05, zt + 0.04])            # centroid of door swing, opening, pull path
    opening = np.array([0.0, P.plate_front_y + 0.06, zt + 0.05])
    mid = 0.5 * (base + opening)
    return {
        "A  current (study pick)": (np.array(L["camera"]), None),
        "B  side-on, right of printer": (np.array([0.62, -0.30, zt + 0.42]), task),
        "C  high front-right, looking down": (np.array([0.42, -0.55, zt + 0.80]), task),
        "D  left-front, beyond the open door": (np.array([-0.45, -1.05, zt + 0.55]), task),
        "E  whole cell, high and far": (np.array([0.95, -1.15, zt + 0.95]), mid),
        # found by search: elbow + gripper always in frame, base left out, plate >= 70 % visible
        "F  middle ground: arm from elbow out": (np.array([0.60, -0.80, L["rel_height"] + 0.95]), "elbow"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="both")
    a = ap.parse_args()
    L = json.load(open(os.path.join(HERE, "layout.json")))["layouts"][a.layout]
    cfg = Config()
    cfg.scenario = L["scenario"]
    ev = Evaluator(cfg)
    r = ev.evaluate(L["base_x"], L["base_dist"], L["rel_height"], L["bed_z"])[L["objective"]]
    cell, m, d = ev.cell, ev.cell.model, ev.cell.data
    cell.set_layout(r.base_x, r.base_dist, r.rel_height, r.bed_z)
    cs = CameraStudy(ev, r, vfov=VFOV)
    zt = plate_top_z(cfg, r.bed_z)
    segs = ev.trajectory(r)
    wps = [(q, door, name) for name, seg in segs for q, door in seg]
    # moments to show
    door = next((s[1] for s in segs if s[0] == "door"), None)
    picks = [s[1] for s in segs if s[0].startswith("pick")]
    pull = next((s[1] for s in segs if s[0] == "pull"), None)
    moments = []
    if door:
        moments.append(("door half open", door[len(door) // 2]))
    if picks:
        mid = picks[len(picks) // 2]
        moments.append(("grasp", mid[2]))
    if pull:
        moments.append(("plate pulled out", pull[-1]))
    f_px = (IMG_H / 2) / np.tan(np.deg2rad(VFOV) / 2)        # focal length in 640x360 pixels
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "g_base")
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forearm_link")
    shots = Shots(cell, IMG_W, IMG_H)
    rows, report = [], []
    for name, (pos, tgt) in candidates(L, cfg.printer, zt).items():
        if isinstance(tgt, str):        # aim 60 % at the opening, 40 % at the elbow's average position
            ee = []
            eid_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forearm_link")
            for q, dr_, _ in wps:
                cell.set_q(q)
                mujoco.mj_kinematics(m, d)
                ee.append(d.xpos[eid_].copy())
            opening = np.array([0.0, cfg.printer.plate_front_y + 0.06, zt + 0.05])
            tgt = 0.6 * opening + 0.4 * np.mean(ee, 0)
        target = cs.target if tgt is None else tgt
        cs.target = target
        s = cs.score(pos)
        x, y, z = look_at(pos, target)
        R = np.stack([x, y, z], 1)
        gf = gs = ef = 0
        keep = 9.0
        for q, dr, _ in wps:
            cell.set_q(q)
            cell.set_door(dr)
            mujoco.mj_kinematics(m, d)
            gp = 0.5 * (d.xpos[gid] + d.site_xpos[cell.pinch_sid])
            gf += cs._in_frame(pos, R, gp)
            gs += cs._ray(pos, gp) in ("clear", "grip")
            ef += cs._in_frame(pos, R, d.xpos[eid])
            keep = min(keep, np.min(np.linalg.norm(robot_points(cell) - pos, axis=1)))
        n = len(wps)
        base_in = cs._in_frame(pos, R, cell.base_pose[:3, 3] + np.array([0, 0, 0.05]))
        pc = np.array([0.0, cfg.printer.plate_front_y + cfg.printer.plate_size / 2, zt])
        depth = -(R.T @ (pc - pos))[2]
        part_px = f_px * 0.03 / depth * (MODEL_W / IMG_W)
        rec = dict(camera=name.split("  ")[1], key=name.split("  ")[0], pos=pos.round(2).tolist(),
                   above_table_m=round(float(pos[2] - min(0.0, r.rel_height)), 2),
                   part_px=round(float(part_px), 1), grip_frame=round(gf / n, 2), grip_seen=round(gs / n, 2),
                   plate_seen=round(float(s["bed_visible"]), 2), plate_clear=round(float(s["arm_clear"]), 2),
                   elbow_frame=round(ef / n, 2), base_frame=bool(base_in), keep_out=round(float(keep), 2))
        report.append(rec)
        # renders
        set_scene_cam(cell, pos, target, VFOV)
        imgs = []
        for label, (q, dr) in moments:
            cell.set_grip("pinch" if label != "grasp" else "open")
            imgs.append(shots.shot(q, dr, "scene_cam", label=label))
        q, dr = moments[[mm[0] for mm in moments].index("grasp")][1] if picks else moments[0][1]
        cell.set_grip("open")
        g = shots.shot(q, dr, "scene_cam").resize((MODEL_W, MODEL_W), Image.BILINEAR)
        rows.append((name, rec, imgs, g))
    # contact sheet
    tw, th = 384, 216
    pad, head = 10, 34
    W = pad + len(moments) * (tw + pad) + th + pad + 300
    H = pad + len(rows) * (th + head + pad)
    sheet = Image.new("RGB", (W, H), (245, 245, 247))
    dr = ImageDraw.Draw(sheet)
    f1, f2 = _font(17), _font(14)
    for i, (name, rec, imgs, g) in enumerate(rows):
        y0 = pad + i * (th + head + pad)
        dr.text((pad, y0 + 6), name, fill=(20, 20, 25), font=f1)
        for j, im in enumerate(imgs):
            sheet.paste(im.resize((tw, th)), (pad + j * (tw + pad), y0 + head))
        x0 = pad + len(moments) * (tw + pad)
        sheet.paste(g.resize((th, th), Image.NEAREST), (x0, y0 + head))
        dr.text((x0, y0 + 10), "grasp at 224 x 224", fill=(90, 90, 100), font=f2)
        tx = x0 + th + 14
        lines = [f"30 mm part: {rec['part_px']} px wide",
                 f"gripper in frame: {rec['grip_frame']:.0%}",
                 f"gripper visible: {rec['grip_seen']:.0%}",
                 f"plate seen thru door: {rec['plate_seen']:.0%}",
                 f"plate not hidden by arm: {rec['plate_clear']:.0%}",
                 f"elbow in frame: {rec['elbow_frame']:.0%}",
                 f"base in frame: {'yes' if rec['base_frame'] else 'no'}",
                 f"{rec['above_table_m']:.2f} m above table"]
        for k, t in enumerate(lines):
            dr.text((tx, y0 + head + 2 + k * 25), t, fill=(30, 30, 35), font=f2)
    out = os.path.join(HERE, "results", "figures", f"camera_compare_{a.layout}.png")
    sheet.save(out)
    json.dump(report, open(os.path.join(HERE, "results", f"camera_compare_{a.layout}.json"), "w"), indent=1)
    for rec in report:
        print(rec)
    print("->", out)


if __name__ == "__main__":
    main()
