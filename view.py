"""
Watch a layout in the interactive MuJoCo viewer, with the task motions playing.

    python view.py                       # active layout from layout.json, loops its motions
    python view.py --layout pull         # one of the other recommended layouts
    python view.py --x 0.10 --dist 0.50 --rel 0.0 --bedz 200      # try your own placement
    python view.py --static              # no playback: drag the arm with the viewer's joint sliders

macOS: the passive viewer needs   mjpython view.py   (mjpython ships with `pip install mujoco`).
Linux/Windows: plain python works.

Tweak loop: change numbers here or in config.py -> `python evaluate.py ...` for the
score -> `python view.py ...` to see it.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import mujoco
import mujoco.viewer

from config import Config
from evaluate import Evaluator, describe
from render import interp

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", help="pick | pull | both | top (from layout.json)")
    ap.add_argument("--x", type=float)
    ap.add_argument("--dist", type=float)
    ap.add_argument("--rel", type=float)
    ap.add_argument("--bedz", type=float)
    ap.add_argument("--scenario", choices=["door", "top"])
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="print the score and motion size, no window")
    a = ap.parse_args()

    with open(os.path.join(HERE, "layout.json")) as f:
        lay = json.load(f)
    key = a.layout or lay["active"]
    L = dict(lay["layouts"][key])
    for k, v in (("base_x", a.x), ("base_dist", a.dist), ("rel_height", a.rel), ("bed_z", a.bedz),
                 ("scenario", a.scenario)):
        if v is not None:
            L[k] = v
    cfg = Config()
    cfg.scenario = L["scenario"]
    ev = Evaluator(cfg)
    res = ev.evaluate(L["base_x"], L["base_dist"], L["rel_height"], L["bed_z"])
    r = res[L.get("objective", key)] if L.get("objective", key) in res else list(res.values())[0]
    print(describe(r))
    cell = ev.cell
    m, d = cell.model, cell.data
    cell.set_layout(r.base_x, r.base_dist, r.rel_height, r.bed_z)
    if "camera" in L:
        from render import set_scene_cam
        from camera_study import CameraStudy
        cs = CameraStudy(ev, r)
        set_scene_cam(cell, L["camera"], cs.target, L.get("camera_fovy", 42.0))
    segs = ev.trajectory(r)
    frames = interp(segs, steps=25) if segs else []
    cell.set_grip("open")
    if frames:
        cell.set_q(frames[0][0])
    mujoco.mj_forward(m, d)

    if a.dry_run:
        print(f"{len(segs)} motion segments, {len(frames)} playback frames")
        return

    if a.static or not frames:
        d.ctrl[:6] = d.qpos[cell.qadr]
        mujoco.viewer.launch(m, d)
        return

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (r.base_x / 2, -r.base_dist / 2, 0.25)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, -60, -25
        i = 0
        while v.is_running():
            q, door, name = frames[i % len(frames)]
            grip = "pinch" if name in ("door", "pull") else "open"
            cell.set_grip(grip)
            cell.set_q(q)
            cell.set_door(door)
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(0.02 / a.speed)
            i += 1


if __name__ == "__main__":
    main()
