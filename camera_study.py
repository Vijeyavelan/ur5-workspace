"""
Third-person camera placement for a chosen arm layout.

For each candidate camera position (aimed at the printer opening) it measures:
  bed_visible   share of plate points visible past the printer frame / open door (arm parked)
  arm_clear     share of plate points NOT hidden by the arm while it works in the printer
  grip_visible  share of task waypoints where the gripper is visible
  in_frame      share of key points (plate, door handle, gripper path) inside the image
  keep_out      camera is at least 15 cm from the arm at every waypoint
The RealSense field of view is taken from config (D435 color 69 x 42 deg by default).
"""
import numpy as np
import mujoco

from config import Config, plate_top_z

D435_HFOV, D435_VFOV = 69.0, 42.0


def look_at(cam, target, up=(0, 0, 1.0)):
    f = np.asarray(target, float) - cam
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return x, y, -f          # camera x (right), y (up), z (backward)


def robot_points(cell):
    """Points along the arm centerline (for keep-out distance)."""
    m, d = cell.model, cell.data
    names = ["ur_mount", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
             "wrist_3_link", "g_base"]
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    P = [d.xpos[i].copy() for i in ids] + [d.site_xpos[cell.pinch_sid].copy()]
    pts = []
    for a, b in zip(P[:-1], P[1:]):
        for t in np.linspace(0, 1, 6):
            pts.append(a + t * (b - a))
    return np.array(pts)


class CameraStudy:
    def __init__(self, ev, result, hfov=D435_HFOV, vfov=D435_VFOV):
        self.ev, self.r, self.cell = ev, result, ev.cell
        self.hfov, self.vfov = np.deg2rad(hfov), np.deg2rad(vfov)
        cfg: Config = ev.cfg
        m = self.cell.model
        self.cfg = cfg
        P = cfg.printer
        zt = plate_top_z(cfg, result.bed_z)
        n = 5
        xs = np.linspace(-P.plate_size / 2 + 0.03, P.plate_size / 2 - 0.03, n)
        ys = np.linspace(P.plate_front_y + 0.03, P.plate_front_y + P.plate_size - 0.03, n)
        self.bed_pts = np.array([(x, y, zt + 0.004) for y in ys for x in xs])
        # aim between the printer opening and the arm base so the whole arm stays in view
        door_c = np.array([0.0, P.plate_front_y + 0.05, zt + 0.06])
        base_c = np.array([result.base_x, -result.base_dist, result.rel_height + 0.25])
        self.target = 0.7 * door_c + 0.3 * base_c
        self.robot_body = np.array([m.body(b).id in self.cell.robot_bodies for b in range(m.nbody)])
        self.grip_body = np.array([(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("g_")
                                   for b in range(m.nbody)])
        self.plate_body = self.cell.plate_id
        # task poses to test: every waypoint of the chosen motions
        self.segs = ev.trajectory(result)
        self.wp = [(q, door, name) for name, seg in self.segs for q, door in seg]
        # gripper-in-printer poses for arm occlusion (the inner part of each pick)
        self.work = [(q, door) for name, seg in self.segs if name.startswith(("pick", "pull"))
                     for k, (q, door) in enumerate(seg) if k in (1, 2, 3)]
        if len(self.work) > 40:
            self.work = [self.work[i] for i in np.linspace(0, len(self.work) - 1, 40).astype(int)]
        if not self.work:
            self.work = [(q, door) for q, door, _ in self.wp]
        self.park_q = self.segs[0][1][0][0] if self.segs else np.zeros(6)

    def _ray(self, cam, p):
        """Returns ('clear'|'robot'|'grip'|'env', geom)."""
        m, d = self.cell.model, self.cell.data
        v = p - cam
        dist = np.linalg.norm(v)
        gid = np.array([-1], dtype=np.int32)
        h = mujoco.mj_ray(m, d, cam, v / dist, None, 1, -1, gid)
        if gid[0] < 0 or h < 0 or h >= dist - 0.012:
            return "clear"
        b = m.geom_bodyid[gid[0]]
        if b == self.plate_body:
            return "clear" if h >= dist - 0.03 else "env"
        if self.grip_body[b]:
            return "grip"
        if self.robot_body[b]:
            return "robot"
        return "env"

    def _in_frame(self, cam, R, p):
        v = R.T @ (p - cam)
        if v[2] > -0.05:
            return False
        return abs(v[0] / -v[2]) <= np.tan(self.hfov / 2) and abs(v[1] / -v[2]) <= np.tan(self.vfov / 2)

    def score(self, cam):
        cell = self.cell
        m, d = cell.model, cell.data
        x, y, z = look_at(cam, self.target)
        R = np.stack([x, y, z], axis=1)
        # parked arm: plate visibility through the door
        cell.set_q(self.park_q)
        cell.set_door(self.cfg.printer.door_open_angle_deg if self.cfg.scenario == "door" else 0.0)
        mujoco.mj_kinematics(m, d)
        vis = [self._ray(cam, p) for p in self.bed_pts]
        bed_visible = np.mean([v == "clear" for v in vis])
        # arm occlusion while working
        blocked, total = 0, 0
        for q, door in self.work:
            cell.set_q(q)
            cell.set_door(door)
            mujoco.mj_kinematics(m, d)
            for p, v0 in zip(self.bed_pts, vis):
                if v0 != "clear":
                    continue
                total += 1
                blocked += self._ray(cam, p) in ("robot", "grip")
        arm_clear = 1 - blocked / total if total else 0.0
        # gripper visibility + framing + keep-out along all waypoints
        gv, fr, nf, keep = 0, 0, 0, 1e9
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "g_base")
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forearm_link")
        for q, door, _ in self.wp:
            cell.set_q(q)
            cell.set_door(door)
            mujoco.mj_kinematics(m, d)
            gp = 0.5 * (d.xpos[gid] + d.site_xpos[cell.pinch_sid])
            gv += self._ray(cam, gp) in ("clear", "grip")
            # gripper and elbow both in the image
            fr += self._in_frame(cam, R, gp) + self._in_frame(cam, R, d.xpos[eid])
            nf += 2
            keep = min(keep, np.min(np.linalg.norm(robot_points(cell) - cam, axis=1)))
        n = max(1, len(self.wp))
        frame_pts = list(self.bed_pts[[0, 4, 20, 24]]) + [self.cell.base_pose[:3, 3] + np.array([0, 0, 0.1])]
        in_frame = (fr + sum(self._in_frame(cam, R, p) for p in frame_pts)) / (nf + len(frame_pts))
        dist = float(np.linalg.norm(self.target - cam))
        s = dict(bed_visible=bed_visible, arm_clear=arm_clear, grip_visible=gv / n, in_frame=in_frame,
                 keep_out=keep, dist_to_scene=dist)
        s["score"] = (0.25 * bed_visible + 0.30 * arm_clear + 0.25 * s["grip_visible"] + 0.20 * in_frame) \
            * (1.0 if keep >= 0.15 else 0.3) * min(1.0, dist / 0.7)
        return s

    def sweep(self, table_top_z=0.0):
        P = self.cfg.printer
        out = []
        for cx in np.arange(-0.9, 0.901, 0.1):
            for cy in np.arange(-1.4, -0.099, 0.1):
                for ch in (0.45, 0.65, 0.85, 1.05):
                    cam = np.array([cx, cy, table_top_z + ch])
                    # not inside the printer/AMS footprint
                    if abs(cx) < P.width / 2 + 0.05 and cy > -0.05:
                        continue
                    s = self.score(cam)
                    s["pos"] = cam.round(3).tolist()
                    out.append(s)
        out.sort(key=lambda s: -s["score"])
        return out
