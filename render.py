"""
Images for a layout: overview renders, third-person and wrist camera views,
a motion GIF, and a dimensioned top-down drawing of the table.
Headless rendering: MUJOCO_GL=egl python report.py
"""
import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

from config import plate_top_z
from camera_study import look_at


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def set_scene_cam(cell, pos, target, fovy):
    m = cell.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
    x, y, z = look_at(np.asarray(pos, float), target)
    m.cam_mode[cid] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = Rot.from_matrix(np.stack([x, y, z], 1)).as_quat(scalar_first=True)
    m.cam_fovy[cid] = fovy


class Shots:
    def __init__(self, cell, w=960, h=640):
        self.cell = cell
        self.r = mujoco.Renderer(cell.model, h, w)
        self.w, self.h = w, h
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]   # hide collision-only gripper geoms and sites

    def free(self, lookat, dist, az, el):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        return cam

    XRAY = ("pr_wall_R", "pr_lid", "ams", "ams_lid", "pr_front_R", "pr_front_top", "rail_R", "pr_topring_F")

    def shot(self, q, door, camera, label=None, w=None, h=None, xray=False):
        cell = self.cell
        m = cell.model
        saved = {}
        if xray:
            for n in self.XRAY:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                if g >= 0:
                    saved[g] = m.geom_rgba[g].copy()
                    m.geom_rgba[g][3] = 0.12
        cell.set_q(q)
        cell.set_door(door)
        mujoco.mj_forward(cell.model, cell.data)
        r = self.r if (w is None or (w == self.w and h == self.h)) else mujoco.Renderer(cell.model, h, w)
        r.update_scene(cell.data, camera=camera, scene_option=self.opt)
        img = Image.fromarray(r.render())
        for g, c in saved.items():
            m.geom_rgba[g] = c
        if label:
            dr = ImageDraw.Draw(img)
            f = _font(max(14, img.width // 45))
            tw = dr.textlength(label, font=f)
            dr.rectangle([8, 8, 20 + tw, 16 + f.size + 6], fill=(255, 255, 255))
            dr.text((14, 11), label, fill=(20, 20, 20), font=f)
        return img


def interp(segs, steps=8):
    """Joint-space interpolation between waypoints, for playback."""
    frames = []
    for name, seg in segs:
        for (q0, d0), (q1, d1) in zip(seg[:-1], seg[1:]):
            dq = np.angle(np.exp(1j * (q1 - q0)))
            for t in np.linspace(0, 1, steps, endpoint=False):
                frames.append((q0 + t * dq, d0 + t * (d1 - d0), name))
        frames.append((seg[-1][0], seg[-1][1], name))
    return frames


def motion_gif(shots, segs, camera, path, steps=6, max_frames=160):
    frames = interp(segs, steps)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    imgs = [shots.shot(q, door, camera, label=name.split(" p")[0]).convert("P", palette=Image.ADAPTIVE)
            for q, door, name in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0, optimize=True)
    return path


def top_down_plot(cfg, layout, res, cam_pos, path, layout_kind="end", reach=0.85):
    """Dimensioned top view of the table, printer, door swing, arm and camera."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Wedge, Arc, Polygon

    P, T = cfg.printer, cfg.table
    fig, ax = plt.subplots(figsize=(7.2, 11) if layout_kind == "end" else (12, 7))
    ink, soft = "#2b2f36", "#8a94a6"
    if layout_kind == "end":
        tw, td = T.depth, T.length
    else:
        tw, td = T.length, T.depth
    ty0 = P.depth + 0.03 - td
    ax.add_patch(Rectangle((-tw / 2, ty0), tw, td, fc="#efe6d8", ec=ink, lw=1.5))
    # printer
    ax.add_patch(Rectangle((-P.width / 2, 0), P.width, P.depth, fc="#3a3f47", ec=ink, lw=1.2))
    ax.add_patch(Rectangle((-P.plate_size / 2, P.plate_front_y), P.plate_size, P.plate_size, fc="#c9a46a", ec="none", alpha=0.9))
    ax.text(0, P.depth / 2 + 0.06, "X1E\n(AMS on top)" if cfg.scenario == "door" else "X1E\n(lid off)",
            ha="center", va="center", color="white", fontsize=9)
    if cfg.scenario == "top":
        ax.add_patch(Rectangle((P.width / 2 + 0.03, (P.depth - P.ams_depth) / 2), P.ams_width, P.ams_depth,
                               fc="#d9d9db", ec=ink, lw=1))
        ax.text(P.width / 2 + 0.03 + P.ams_width / 2, P.depth / 2, "AMS", ha="center", va="center", fontsize=8)
    # door swing
    hs = -1 if P.door_hinge_side == "left" else 1
    hx = hs * (P.door_open_x_half + 0.02)
    if cfg.scenario == "door":
        # closed panel points +x (hinge left) or -x (hinge right) and swings toward the robot
        th0, th1 = (-(P.door_open_angle_deg + 5), 0) if hs < 0 else (180, 180 + P.door_open_angle_deg + 5)
        ax.add_patch(Wedge((hx, 0), P.door_panel_w, th0, th1, fc="#9ec1e0", ec=soft, alpha=0.35, lw=0.8))
        a = np.deg2rad(P.door_open_angle_deg)
        ex = hx + (-hs) * P.door_panel_w * np.cos(a)
        ey = -P.door_panel_w * np.sin(a)
        ax.plot([hx, ex], [0, ey], color="#4f86c6", lw=3)
        ax.text(ex - hs * 0.03, ey, f"door\nopen {P.door_open_angle_deg:.0f}°", ha="right" if hs < 0 else "left",
                va="center", fontsize=8, color="#4f86c6")
    # arm
    bx, by = layout["base_x"], -layout["base_dist"]
    ax.add_patch(Circle((bx, by), reach, fc="none", ec="#d9822b", ls="--", lw=1))
    ax.add_patch(Circle((bx, by), 0.0745, fc="#d9822b", ec=ink, lw=1.2))
    ax.text(bx, by, "UR5", ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.text(bx + reach * 0.7, by - reach * 0.72, "850 mm reach", color="#d9822b", fontsize=8, rotation=45)
    # dimensions
    def dim(x0, y0, x1, y1, txt, off=(0, 0)):
        ax.annotate("", (x0, y0), (x1, y1), arrowprops=dict(arrowstyle="<->", color=ink, lw=0.9))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], txt, fontsize=8, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1))
    dim(bx, by, bx, 0, f"{layout['base_dist'] * 1000:.0f} mm\nbase center\nto printer front", off=(0.14, 0))
    dim(0, by - 0.12, bx, by - 0.12, f"{bx * 1000:+.0f} mm", off=(0, -0.03)) if abs(bx) > 0.01 else None
    ax.plot([0, 0], [by - 0.16, 0], color=soft, lw=0.6, ls=":")
    # camera
    if cam_pos is not None:
        cx, cy, cz = cam_pos
        tgt = np.array([0.0, P.plate_front_y + 0.08])
        ang = np.degrees(np.arctan2(tgt[1] - cy, tgt[0] - cx))
        ax.add_patch(Wedge((cx, cy), 0.45, ang - 34.5, ang + 34.5, fc="#5aa469", ec="none", alpha=0.25))
        ax.add_patch(Rectangle((cx - 0.045, cy - 0.0125), 0.09, 0.025, angle=ang - 90, rotation_point="center",
                               fc="#5aa469", ec=ink))
        ax.text(cx, cy - 0.06, f"scene camera\n{(cz - min(0.0, res.rel_height)) * 1000:.0f} mm above table", ha="center", va="top",
                fontsize=8, color="#2f6b3a")
    ax.set_aspect("equal")
    pad = 0.12
    ax.set_xlim(min(-tw / 2, bx - reach, (cam_pos[0] if cam_pos is not None else 0)) - pad,
                max(tw / 2, bx + reach, (cam_pos[0] if cam_pos is not None else 0)) + pad)
    ax.set_ylim(min(ty0, by - reach, (cam_pos[1] if cam_pos is not None else 0)) - pad, P.depth + 0.25)
    ax.set_xlabel("x (m)  —  printer frame, origin = bottom-front-center of printer")
    ax.set_ylabel("y (m)  —  into the printer")
    title = {"pick": "Pick in place through the door", "pull": "Pull the plate out",
             "both": "Supports both pick and pull", "top": "Top access (lid off, AMS beside)"}.get(res.objective, "")
    setup = ("arm on a %.0f mm riser" % (res.rel_height * 1000) if res.rel_height > 0 else
             "printer on a %.0f mm stand" % (-res.rel_height * 1000) if res.rel_height < 0 else "both directly on the table")
    ax.set_title(f"{title}\n{setup}, bed park Z = {res.bed_z:.0f} mm",
                 fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
