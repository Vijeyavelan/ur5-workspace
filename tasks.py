"""
Task targets, as chains of gripper (TCP = 'pinch' point between the fingertips)
poses in the printer frame. A chain is feasible for an arm branch only if every
waypoint is reachable in that branch, joint motion between waypoints is small,
and no waypoint brings the robot within the clearance margin of anything.

Tool frame convention: z = approach direction (out of the fingers),
x = finger closing direction, wrist camera on the tool -y side.

Tasks
  pick  : 5x5 grasp points on the plate x tool tilt (0/20/40 deg, leaning into
          the printer) x 4 gripper yaws. Chain = pre-grasp -> mid -> grasp ->
          vertical lift -> retreat out through the door.
  door  : grab a printed handle on the door's free edge, swing the door 0 -> 105 deg.
  pull  : grab the plate's front tab with the tool horizontal, slide the plate out.
"""
from dataclasses import dataclass, field
import numpy as np
from config import Config, plate_top_z


@dataclass
class Chain:
    task: str
    poses: list            # list of 4x4 TCP poses (printer frame)
    door_deg: list         # door angle to set at each waypoint
    meta: dict = field(default_factory=dict)


def tool_pose(p, z_axis, x_axis):
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    x = np.asarray(x_axis, float); x -= z * (x @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, p
    return T


def pick_chains(cfg: Config, bed_z_mm: float):
    P, task = cfg.printer, cfg.task
    zt = plate_top_z(cfg, bed_z_mm)
    n, inset, s = task.pick_grid_n, task.pick_inset, P.plate_size
    xs = np.linspace(-s / 2 + inset, s / 2 - inset, n)
    ys = np.linspace(P.plate_front_y + inset, P.plate_front_y + s - inset, n)
    top = cfg.scenario == "top"
    chains = []
    for iy, py in enumerate(ys):
        for ix, px in enumerate(xs):
            for it, tdeg in enumerate(task.pick_tilts_deg):
                t = np.deg2rad(tdeg)
                gh = task.side_grasp_height if tdeg >= 60 else task.grasp_height
                g = np.array([px, py, zt + gh])
                z = np.array([0.0, np.sin(t), -np.cos(t)])      # leans into the printer
                x0 = np.array([1.0, 0, 0])
                y0 = np.cross(z, x0)
                for iw, wdeg in enumerate((0.0, 90.0, 180.0, 270.0)):
                    w = np.deg2rad(wdeg)
                    x = np.cos(w) * x0 + np.sin(w) * y0
                    pre = g - z * task.approach_dist
                    mid = g - z * 0.5 * task.approach_dist
                    lift = g + np.array([0, 0, task.lift])
                    poses = [tool_pose(pre, z, x), tool_pose(mid, z, x), tool_pose(g, z, x), tool_pose(lift, z, x)]
                    if top:
                        # carry the part straight up and out of the open top
                        for hz in (zt + 0.15, P.height + 0.02, P.height + 0.10):
                            if hz > lift[2]:
                                poses.append(tool_pose(np.array([px, py, hz]), z, x))
                    else:
                        # carry the part straight out through the door at lift height
                        for ry in (P.plate_front_y * 0.5, -0.04, -0.12):
                            if ry < py:
                                poses.append(tool_pose(np.array([px * 0.5, ry, lift[2]]), z, x))
                    door = 0.0 if top else P.door_open_angle_deg
                    chains.append(Chain("pick", poses, [door] * len(poses),
                                        dict(ix=ix, iy=iy, tilt=it, yaw=iw, point=(px, py), gripper="open",
                                             touch=("plate_top", "plate_tab"))))
    return chains


def door_chains(cfg: Config):
    """
    Grab a printed vertical bar handle on the door's free edge and swing the door.
    The gripper closes around the bar, so it can rotate about the bar while the door
    swings; each variant is a schedule of that grip angle (relative to the door's
    outward normal) from the start to the end of the swing.
    """
    P = cfg.printer
    hs = -1 if P.door_hinge_side == "left" else 1
    hinge = np.array([hs * (P.door_open_x_half + 0.02), -0.008, 0.0])
    hz = P.door_open_z_min - 0.02 + P.door_panel_h / 2
    r_along = P.door_panel_w - 0.03      # handle distance from hinge along the panel
    r_out = 0.035                        # handle bar stands off the glass toward the robot
    angles = list(cfg.task.door_angles_deg)
    chains = []
    for d0 in (-40, -20, 0, 20, 40):            # grip angle at the start (deg)
        for d1 in (-40, -20, 0, 20, 40):        # grip angle at the end
            poses, degs = [], []
            for k, a in enumerate([angles[0]] + angles):
                th = np.deg2rad(a)
                u = np.array([-hs * np.cos(th), -np.sin(th), 0.0])     # hinge -> free edge
                nrm = np.array([hs * np.sin(th), -np.cos(th), 0.0])    # outer face normal
                frac = (a - angles[0]) / (angles[-1] - angles[0])
                dl = np.deg2rad(d0 + (d1 - d0) * frac)
                handle = hinge + u * r_along + nrm * r_out + np.array([0, 0, hz])
                z0, x0 = -nrm, u                 # gripper pushes toward the glass, fingers along the panel
                z = np.cos(dl) * z0 + np.sin(dl) * x0
                x = -np.sin(dl) * z0 + np.cos(dl) * x0
                p = handle - (z * 0.06 if k == 0 else 0)   # first waypoint = pre-grasp 6 cm back
                poses.append(tool_pose(p, z, x))
                degs.append(a)
            chains.append(Chain("door", poses, degs, dict(angles=[angles[0]] + angles, grip=(d0, d1),
                                                          gripper="pinch", touch=("door_glass",))))
    return chains


def pull_chains(cfg: Config, bed_z_mm: float):
    P = cfg.printer
    zt = plate_top_z(cfg, bed_z_mm)
    tab_y = P.plate_front_y - P.plate_tab_depth + 0.015      # pad tips stop ~1 cm short of the heatbed
    z = np.array([0.0, 1.0, 0.0])          # tool horizontal, pointing into the printer
    chains = []
    for iw, x in enumerate((np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))):   # camera left / right
        poses = [tool_pose(np.array([0.0, tab_y - 0.06, zt - P.plate_thickness / 2]), z, x)]
        for s in np.linspace(0, cfg.task.plate_pull_dist, 5):
            poses.append(tool_pose(np.array([0.0, tab_y - s, zt - P.plate_thickness / 2]), z, x))
        chains.append(Chain("pull", poses, [P.door_open_angle_deg] * len(poses), dict(yaw=iw, gripper="pinch",
                                                                                    touch=("heatbed", "plate_top", "plate_tab"))))
    return chains


def plate_fits_door(cfg: Config, bed_z_mm: float, margin=0.015):
    """The plate can only slide out if it sits inside the door opening height."""
    zt = plate_top_z(cfg, bed_z_mm)
    P = cfg.printer
    return (zt - 0.025 > P.door_open_z_min + margin) and (zt + margin < P.door_open_z_max)


if __name__ == "__main__":
    cfg = Config()
    pc = pick_chains(cfg, 150)
    print("pick chains:", len(pc), "poses:", sum(len(c.poses) for c in pc))
    dc = door_chains(cfg)
    print("door variants:", len(dc), "poses each:", len(dc[0].poses))
    for T, a in zip(dc[12].poses, dc[12].door_deg):
        print(f"  door {a:5.1f} deg  handle at {T[:3, 3].round(3)}  tool z {T[:3, 2].round(2)}")
    print("pull chains:", len(pull_chains(cfg, 150)), "plate fits door at Z=150:", plate_fits_door(cfg, 150))
