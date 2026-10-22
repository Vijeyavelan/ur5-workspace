"""
Builds the MuJoCo scene (UR5 CB3 + Robotiq 2F-85 + wrist RealSense + Bambu X1E
with AMS + 3x6 ft table) from config.py, and provides helpers to move the arm
base, bed, and door without recompiling.

    python build_scene.py            -> writes scene.xml (default layout)
    python -m mujoco.viewer --mjcf scene.xml
"""
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

from config import Config, plate_top_z

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# collision bits: robot = 1, environment = 2
ROBOT_CT, ROBOT_CA = 1, 3   # robot collides with robot (self) and environment
ENV_CT, ENV_CA = 2, 1       # environment collides only with robot

# UR5 CB3 link chain, transcribed from ros-industrial ur_description (ur5 config).
# (body, parent offset xyz, rpy, mesh, mesh origin xyz, mesh rpy)
SHOULDER_OFFSET, ELBOW_OFFSET = 0.13585, 0.0165
UR5_CHAIN = [
    ("shoulder_link", (0, 0, 0.089159), (0, 0, 0), "shoulder", (0, 0, 0), (0, 0, np.pi)),
    ("upper_arm_link", (0, 0, 0), (np.pi / 2, 0, 0), "upperarm", (0, 0, SHOULDER_OFFSET), (np.pi / 2, 0, -np.pi / 2)),
    ("forearm_link", (-0.425, 0, 0), (0, 0, 0), "forearm", (0, 0, ELBOW_OFFSET), (np.pi / 2, 0, -np.pi / 2)),
    ("wrist_1_link", (-0.39225, 0, 0.10915), (0, 0, 0), "wrist1", (0, 0, -0.093), (np.pi / 2, 0, 0)),
    ("wrist_2_link", (0, -0.09465, 0), (np.pi / 2, 0, 0), "wrist2", (0, 0, -0.095), (0, 0, 0)),
    ("wrist_3_link", (0, 0.0823, 0), (np.pi / 2, np.pi, np.pi), "wrist3", (0, 0, -0.0818), (np.pi / 2, 0, 0)),
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
JOINT_RANGE = [(-2 * np.pi, 2 * np.pi)] * 6
JOINT_RANGE[2] = (-np.pi, np.pi)


def quat_rpy(rpy):
    """URDF rpy (fixed-axis XYZ) -> MuJoCo quat (w, x, y, z)."""
    return Rot.from_euler("xyz", rpy).as_quat(scalar_first=True)


def quat_z(deg):
    return Rot.from_euler("z", deg, degrees=True).as_quat(scalar_first=True)


def _tool0_in_wrist3():
    # URDF: wrist_3 -> flange (rpy 0,-pi/2,-pi/2) -> tool0 (rpy pi/2,0,pi/2)
    R = Rot.from_euler("xyz", (0, -np.pi / 2, -np.pi / 2)) * Rot.from_euler("xyz", (np.pi / 2, 0, np.pi / 2))
    return R.as_quat(scalar_first=True)


def _box(body, name, center, half, rgba, env=True, margin=None, group=0, contact=True):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = center
    g.size = half
    g.rgba = rgba
    g.group = group
    if contact:
        g.contype, g.conaffinity = (ENV_CT, ENV_CA) if env else (ROBOT_CT, ROBOT_CA)
    else:
        g.contype = g.conaffinity = 0
    if margin is not None:
        g.margin = margin
    return g


def _box_from_bounds(body, name, lo, hi, rgba, **kw):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return _box(body, name, (lo + hi) / 2, (hi - lo) / 2, rgba, **kw)


def build_spec(cfg: Config, layout: str = "end"):
    """
    layout "end":  printer at one short end of the table, door facing down the 6 ft length
    layout "side": printer at the back long edge, door facing the front edge (3 ft deep)
    """
    P, T, tool, task = cfg.printer, cfg.table, cfg.tooling, cfg.task
    s = mujoco.MjSpec()
    s.modelname = "ur5_x1e_cell"
    s.compiler.degree = False
    s.meshdir = ASSETS
    s.option.timestep = 0.002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10
    s.visual.global_.offwidth = 1600
    s.visual.global_.offheight = 1200
    s.visual.quality.shadowsize = 4096
    s.stat.center = (0, -0.4, 0.3)
    s.stat.extent = 1.4

    # ---------- assets ----------
    tex = s.add_texture()
    tex.name, tex.type, tex.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.rgb1, tex.rgb2, tex.width, tex.height = (0.85, 0.88, 0.92), (0.55, 0.6, 0.68), 512, 3072
    tex2 = s.add_texture()
    tex2.name, tex2.type, tex2.builtin = "grid", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex2.rgb1, tex2.rgb2, tex2.width, tex2.height = (0.42, 0.43, 0.45), (0.38, 0.39, 0.41), 512, 512
    mat = s.add_material()
    mat.name, mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor", "grid"
    mat.texrepeat, mat.reflectance = (8, 8), 0.05
    for nm in ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]:
        m = s.add_mesh()
        m.name, m.file = f"ur5_{nm}", f"ur5/{nm}.stl"
    for i in range(9):
        m = s.add_mesh()
        m.name, m.file = f"d435i_{i}", f"d435i/d435i_{i}.obj"

    wb = s.worldbody
    lt = wb.add_light()
    lt.pos, lt.dir, lt.type = (0.6, -1.4, 2.2), (-0.3, 0.6, -1), mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt.diffuse, lt.castshadow = (0.75, 0.75, 0.75), True
    lt2 = wb.add_light()
    lt2.pos, lt2.dir, lt2.type = (-1.0, -0.5, 1.8), (0.5, 0.2, -1), mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt2.diffuse, lt2.castshadow = (0.35, 0.35, 0.35), False

    cl = wb.add_light()          # chamber LED
    cl.name, cl.pos, cl.dir = "chamber_led", (0, 0.06, 0.41), (0, 0.3, -1)
    cl.diffuse, cl.castshadow, cl.attenuation = (0.45, 0.45, 0.42), False, (1, 0, 0)
    cl.type = mujoco.mjtLightType.mjLIGHT_SPOT
    cl.cutoff = 70

    floor = wb.add_geom()
    floor.name, floor.type, floor.size, floor.material = "floor", mujoco.mjtGeom.mjGEOM_PLANE, (4, 4, 0.05), "floor"
    floor.pos = (0, -0.6, -0.76)
    floor.contype = floor.conaffinity = 0

    # ---------- table (body moved per config so its top sits at min(0, rel_height)) ----------
    table = wb.add_body()
    table.name = "table"
    if layout == "end":
        tl, td = T.depth, T.length          # x extent, y extent in printer frame
        tcy = P.depth + 0.03 - td / 2
    else:
        tl, td = T.length, T.depth
        tcy = P.depth + 0.03 - td / 2
    table.pos = (0, tcy, 0)
    _box(table, "table_top", (0, 0, -T.thickness / 2), (tl / 2, td / 2, T.thickness / 2),
         (0.80, 0.70, 0.55, 1), margin=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(table, f"leg_{sx}_{sy}", (sx * (tl / 2 - 0.04), sy * (td / 2 - 0.04), -0.39),
                 (0.025, 0.025, 0.36), (0.3, 0.3, 0.32, 1), contact=False)

    # ---------- printer (fixed at origin of printer frame) ----------
    pr = wb.add_body()
    pr.name = "printer"
    W, Dp, H = P.width, P.depth, P.height
    dark, frame_c, ams_c = (0.16, 0.17, 0.19, 1), (0.22, 0.23, 0.25, 1), (0.85, 0.85, 0.86, 1)
    c = task.clearance
    # side walls, back wall, floor block, lid
    _box_from_bounds(pr, "pr_wall_L", (-W / 2, 0, 0), (-W / 2 + P.side_wall, Dp, H), dark, margin=c)
    _box_from_bounds(pr, "pr_wall_R", (W / 2 - P.side_wall, 0, 0), (W / 2, Dp, H), dark, margin=c)
    _box_from_bounds(pr, "pr_wall_back", (-W / 2, Dp - P.back_wall, 0), (W / 2, Dp, H), dark, margin=c)
    top = cfg.scenario == "top"
    if top:
        # lid removed: only the top frame ring remains
        _box_from_bounds(pr, "pr_topring_F", (-W / 2, 0, P.lid_inner_z), (W / 2, 0.03, H), frame_c, margin=c)
    else:
        _box_from_bounds(pr, "pr_lid", (-W / 2, 0, P.lid_inner_z), (W / 2, Dp, H), (0.3, 0.32, 0.35, 0.9), margin=c)
    # front frame around the door opening
    _box_from_bounds(pr, "pr_front_L", (-W / 2, -0.004, 0), (-P.door_open_x_half, 0.03, H), frame_c, margin=c)
    _box_from_bounds(pr, "pr_front_R", (P.door_open_x_half, -0.004, 0), (W / 2, 0.03, H), frame_c, margin=c)
    _box_from_bounds(pr, "pr_front_bottom", (-W / 2, -0.004, 0), (W / 2, 0.03, P.door_open_z_min), frame_c, margin=c)
    _box_from_bounds(pr, "pr_front_top", (-W / 2, -0.004, P.door_open_z_max), (W / 2, 0.03, H), frame_c, margin=c)
    # stand under the printer (only visible when the printer is raised: rel_height < 0)
    _box_from_bounds(pr, "pr_stand", (-W / 2 + 0.01, 0.01, -0.30), (W / 2 - 0.01, Dp - 0.01, 0), (0.45, 0.42, 0.38, 1))
    # AMS on top (in the "top" scenario it moves beside the printer, right side)
    if top:
        ax = W / 2 + 0.03 + P.ams_width / 2
        _box_from_bounds(pr, "ams", (ax - P.ams_width / 2, (Dp - P.ams_depth) / 2, 0),
                         (ax + P.ams_width / 2, (Dp + P.ams_depth) / 2, P.ams_height), ams_c, margin=c)
    else:
        _box_from_bounds(pr, "ams", (-P.ams_width / 2, (Dp - P.ams_depth) / 2, H),
                         (P.ams_width / 2, (Dp + P.ams_depth) / 2, H + P.ams_height), ams_c, margin=c)
        _box_from_bounds(pr, "ams_lid", (-P.ams_width / 2 + 0.01, (Dp - P.ams_depth) / 2 + 0.01, H + P.ams_height - 0.004),
                         (P.ams_width / 2 - 0.01, (Dp + P.ams_depth) / 2 - 0.01, H + P.ams_height + 0.002),
                         (0.2, 0.2, 0.22, 0.6), contact=False)
    # parked toolhead + X gantry beam (beam spans the interior width at the toolhead's y)
    tw, tdp, th = P.toolhead_size
    nz = P.nozzle_tip_z
    _box_from_bounds(pr, "toolhead", (P.toolhead_park_x - tw / 2, P.toolhead_park_y - tdp / 2, nz),
                     (P.toolhead_park_x + tw / 2, P.toolhead_park_y + tdp / 2, nz + th), (0.1, 0.1, 0.1, 1), margin=c)
    bz0, bz1 = P.gantry_beam_z
    _box_from_bounds(pr, "gantry_beam", (-W / 2 + P.side_wall, P.toolhead_park_y - 0.02, nz + bz0),
                     (W / 2 - P.side_wall, P.toolhead_park_y + 0.02, nz + bz1), (0.08, 0.08, 0.08, 1), margin=c)
    # side rails of the CoreXY at the top (along y)
    for sx in (-1, 1):
        x0 = sx * (W / 2 - P.side_wall)
        _box_from_bounds(pr, f"rail_{'LR'[sx > 0]}", (min(x0, x0 - sx * 0.03), 0.03, nz + bz0),
                         (max(x0, x0 - sx * 0.03), Dp - P.back_wall, P.lid_inner_z), (0.1, 0.1, 0.1, 1), margin=c)

    # ---------- build plate (moved with bed park Z) ----------
    plate = wb.add_body()
    plate.name = "plate"
    plate.pos = (0, P.plate_front_y + P.plate_size / 2, 0.2)  # z set at runtime: top surface
    ps = P.plate_size
    _box(plate, "plate_top", (0, 0, -P.plate_thickness / 2), (ps / 2, ps / 2, P.plate_thickness / 2),
         (0.72, 0.58, 0.32, 1), margin=c)   # only the gripper may come closer (see Cell.set_touch)
    _box(plate, "plate_tab", (0, -ps / 2 - P.plate_tab_depth / 2, -P.plate_thickness / 2),
         (0.03, P.plate_tab_depth / 2, P.plate_thickness / 2), (0.72, 0.58, 0.32, 1), margin=c)
    # heatbed sits under the plate; the plate tab overhangs its front edge so fingers can pinch it
    _box(plate, "heatbed", (0, 0.01, -0.012), (ps / 2 + 0.005, ps / 2 + 0.01, 0.01), (0.25, 0.25, 0.27, 1), margin=c)
    # bed carrier/leadscrew region under the bed (keeps the arm from going under it)
    _box(plate, "bed_under", (0, 0.01, -0.20), (ps / 2 - 0.02, ps / 2 - 0.02, 0.18), (0.2, 0.2, 0.2, 0.15), margin=c)

    # ---------- door (hinged) ----------
    door = wb.add_body()
    door.name = "door"
    hs = -1 if P.door_hinge_side == "left" else 1
    door.pos = (hs * (P.door_open_x_half + 0.02), -0.008, 0)
    dj = door.add_joint()
    dj.name, dj.type = "door_hinge", mujoco.mjtJoint.mjJNT_HINGE
    dj.axis = (0, 0, -1) if hs < 0 else (0, 0, 1)   # +angle swings the free edge toward the robot
    dj.range = (0, np.deg2rad(120))
    _box(door, "door_glass", (-hs * P.door_panel_w / 2, -0.004, P.door_open_z_min - 0.02 + P.door_panel_h / 2),
         (P.door_panel_w / 2, 0.004, P.door_panel_h / 2), (0.55, 0.65, 0.75, 0.35), margin=c)
    # free edge of the door: where the gripper grabs to open it
    # printed handle on the door's free edge (the robot grabs this to swing the door)
    hx = -hs * (P.door_panel_w - 0.03)
    hz = P.door_open_z_min - 0.02 + P.door_panel_h / 2
    _box(door, "door_handle", (hx, -0.035, hz), (0.008, 0.008, 0.045), (0.95, 0.55, 0.1, 1), contact=False)
    _box(door, "door_handle_stem", (hx, -0.02, hz), (0.006, 0.013, 0.02), (0.95, 0.55, 0.1, 1), contact=False)
    sd = door.add_site()
    sd.name = "door_handle"
    sd.pos = (hx, -0.035, hz)
    sd.size, sd.rgba, sd.group = (0.006, 0, 0), (1, 0.2, 0.2, 1), 4

    # ---------- arm ----------
    mount = wb.add_body()
    mount.name = "ur_mount"           # = UR 'base' DH frame (URDF base_link_inertia)
    mount.pos = (0, -0.5, 0)
    mount.quat = quat_z(cfg.base_yaw_deg)
    riser = mount.add_geom()
    riser.name, riser.type = "riser", mujoco.mjtGeom.mjGEOM_CYLINDER
    riser.size, riser.pos, riser.rgba = (0.09, 0.15, 0), (0, 0, -0.15), (0.35, 0.36, 0.4, 1)
    riser.contype = riser.conaffinity = 0
    bg = mount.add_geom()
    bg.name, bg.type, bg.meshname = "ur5_base", mujoco.mjtGeom.mjGEOM_MESH, "ur5_base"
    bg.quat, bg.rgba = quat_rpy((0, 0, np.pi)), (0.75, 0.77, 0.8, 1)
    bg.contype, bg.conaffinity = ROBOT_CT, ROBOT_CA

    parent = mount
    link_colors = [(0.75, 0.77, 0.8, 1), (0.75, 0.77, 0.8, 1), (0.75, 0.77, 0.8, 1),
                   (0.3, 0.45, 0.65, 1), (0.3, 0.45, 0.65, 1), (0.3, 0.45, 0.65, 1)]
    for i, (bname, xyz, rpy, mesh, mxyz, mrpy) in enumerate(UR5_CHAIN):
        b = parent.add_body()
        b.name, b.pos, b.quat = bname, xyz, quat_rpy(rpy)
        j = b.add_joint()
        j.name, j.type, j.axis = JOINT_NAMES[i], mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1)
        j.range, j.limited = JOINT_RANGE[i], mujoco.mjtLimited.mjLIMITED_TRUE
        j.armature = 0.1
        j.damping = np.array([2.0, 0.0, 0.0])
        g = b.add_geom()
        g.name, g.type, g.meshname = f"ur5_{mesh}", mujoco.mjtGeom.mjGEOM_MESH, f"ur5_{mesh}"
        g.pos, g.quat, g.rgba = mxyz, quat_rpy(mrpy), link_colors[i]
        g.contype, g.conaffinity = ROBOT_CT, ROBOT_CA
        a = s.add_actuator()
        a.name, a.target, a.trntype = f"act_{JOINT_NAMES[i]}", JOINT_NAMES[i], mujoco.mjtTrn.mjTRN_JOINT
        kp = 2000.0 if i < 3 else 500.0
        a.gainprm[0], a.biastype, a.biasprm[:3] = kp, mujoco.mjtBias.mjBIAS_AFFINE, (0, -kp, -50)
        a.ctrlrange, a.ctrllimited = JOINT_RANGE[i], mujoco.mjtLimited.mjLIMITED_TRUE
        parent = b
    wrist3 = parent

    tool0 = wrist3.add_site()
    tool0.name, tool0.quat, tool0.size, tool0.group = "tool0", _tool0_in_wrist3(), (0.006, 0, 0), 4

    # wrist RealSense on its printed mount (frame = tool0)
    wc = wrist3.add_body()
    wc.name, wc.quat = "wrist_cam_mount", _tool0_in_wrist3()
    tilt = np.deg2rad(tool.wrist_cam_tilt_deg)
    # camera sits on the tool -y side (fingers close along tool x), looking along +z, tilted toward the axis
    cam_body = wc.add_body()
    cam_body.name = "wrist_cam_body"
    cam_body.pos = (0, -tool.wrist_cam_offset, tool.wrist_cam_z)
    # orient: camera looks down tool +z, rotated about tool x by the tilt toward the axis
    cam_body.quat = (Rot.from_euler("x", -tilt)).as_quat(scalar_first=True)   # tilt toward the fingers
    _box(cam_body, "wrist_cam_box", (0, 0, 0), (0.045, 0.0125, 0.0125), (0.15, 0.15, 0.17, 1), env=False)
    _box_from_bounds(wc, "wrist_cam_bracket", (-0.02, -tool.wrist_cam_offset + 0.01, 0.0),
                     (0.02, -0.035, 0.012), (0.9, 0.5, 0.15, 1), env=False)
    cam = cam_body.add_camera()
    cam.name = "wrist_cam"
    # MuJoCo cameras look along their -z; point it along tool +z with image "up" = tool -y
    cam.quat = Rot.from_euler("x", np.pi).as_quat(scalar_first=True)
    cam.pos = (0.0, 0.0, 0.013)
    cam.fovy = tool.wrist_cam_fovy_deg

    # ---------- gripper (Menagerie Robotiq 2F-85) ----------
    gspec = mujoco.MjSpec.from_file(os.path.join(ASSETS, "robotiq", "2f85.xml"))
    gspec.meshdir = ""
    for m in gspec.meshes:
        m.file = os.path.join(ASSETS, "robotiq", os.path.basename(m.file))
    # Menagerie gripper geoms: make every collision geom part of the robot group
    for g in gspec.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = ROBOT_CT, ROBOT_CA
    s.attach(gspec, prefix="g_", site=tool0)

    # ---------- third-person camera (placed by the camera study) ----------
    sc = wb.add_camera()
    sc.name = "scene_cam"
    sc.pos = (0.75, -1.0, 0.75)
    sc.fovy = 42.0
    sc.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    sc.targetbody = "plate"
    # fixed viewing cameras for renders
    for nm, pos in [("view_iso", (1.1, -1.55, 1.05)), ("view_side", (1.35, -0.45, 0.55)),
                    ("view_top", (0.0, -0.55, 2.2))]:
        v = wb.add_camera()
        v.name, v.pos, v.fovy = nm, pos, 45.0
        v.mode, v.targetbody = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, "ur_mount" if nm != "view_side" else "plate"

    # ---------- contact exclusions ----------
    for b1, b2 in [("ur_mount", "shoulder_link"), ("ur_mount", "table"), ("shoulder_link", "table"), ("ur_mount", "printer"),
                   ("wrist_3_link", "g_base"), ("wrist_2_link", "g_base_mount"), ("wrist_2_link", "wrist_cam_mount"),
                   ("wrist_cam_mount", "g_base_mount"), ("wrist_cam_mount", "g_base"),
                   ("wrist_cam_body", "g_base_mount"), ("wrist_cam_body", "g_base"),
                   ("wrist_cam_body", "wrist_3_link"), ("wrist_1_link", "wrist_3_link")]:
        e = s.add_exclude()
        e.bodyname1, e.bodyname2 = b1, b2
    return s


class Cell:
    """Compiled model + fast setters used by the sweep and the renders."""

    def __init__(self, cfg: Config, layout: str = "end"):
        self.cfg, self.layout = cfg, layout
        self.spec = build_spec(cfg, layout)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINT_NAMES])
        self.door_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")]
        self.mount_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur_mount")
        self.plate_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.table_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.tool0_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool0")
        self.pinch_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "g_pinch")
        self.table_y = m.body_pos[self.table_id][1]
        robot_body_names = {"ur_mount"} | {c[0] for c in UR5_CHAIN} | {"wrist_cam_mount", "wrist_cam_body"}
        self.robot_bodies = set()
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm in robot_body_names or nm.startswith("g_"):
                self.robot_bodies.add(b)
        self.robot_geom = np.array([m.geom_bodyid[g] in self.robot_bodies for g in range(m.ngeom)])
        self.gripper_geom = np.array([(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith("g_")
                                      for g in range(m.ngeom)])
        self.touch_ids = set()
        # gripper linkage poses: let the real mechanism settle at each command once,
        # then reuse those joint values (the 4-bar linkage needs all 8 joints set)
        self.g_qadr = np.array([m.jnt_qposadr[j] for j in range(m.njnt)
                                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("g_")])
        self.grip_q = {}
        for name, ctrl in (("open", 0.0), ("pinch", 210.0), ("closed", 255.0)):
            dd = mujoco.MjData(m)
            dd.qpos[self.door_qadr] = 0
            act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
            for _ in range(1500):
                dd.ctrl[:6] = 0
                dd.ctrl[act] = ctrl
                dd.qpos[self.qadr] = 0
                dd.qvel[self.qadr] = 0
                mujoco.mj_step(m, dd)
            self.grip_q[name] = dd.qpos[self.g_qadr].copy()
        self.set_grip("open")
        self.base_pose = np.eye(4)

    # ----- configuration -----
    def set_layout(self, base_x, base_dist, rel_height, bed_z_mm, door_deg=None):
        """base_dist = distance of base center in front of the printer face (base y = -base_dist)."""
        m, cfg = self.model, self.cfg
        m.body_pos[self.mount_id] = (base_x, -base_dist, rel_height)
        m.body_pos[self.table_id] = (0, self.table_y, min(0.0, rel_height))
        m.body_pos[self.plate_id][2] = plate_top_z(cfg, bed_z_mm)
        if door_deg is None:
            door_deg = 0.0 if cfg.scenario == "top" else cfg.printer.door_open_angle_deg
        self.data.qpos[self.door_qadr] = np.deg2rad(door_deg)
        # risers are drawn to the right height (render-only geoms; collision uses the table/printer)
        rg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "riser")
        sg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pr_stand")
        h = max(rel_height, 0.0005)
        m.geom_size[rg][1], m.geom_pos[rg][2] = h / 2, -h / 2
        m.geom_rgba[rg][3] = 1.0 if rel_height > 0.001 else 0.0
        m.geom_rgba[sg][3] = 1.0 if rel_height < -0.001 else 0.0
        self.rel_height = rel_height
        R = Rot.from_euler("z", cfg.base_yaw_deg, degrees=True).as_matrix()
        self.base_pose = np.eye(4)
        self.base_pose[:3, :3] = R
        self.base_pose[:3, 3] = (base_x, -base_dist, rel_height)

    def set_grip(self, name):
        self.data.qpos[self.g_qadr] = self.grip_q[name]
        self.grip = name

    def set_door(self, deg):
        self.data.qpos[self.door_qadr] = np.deg2rad(deg)

    def set_q(self, q):
        self.data.qpos[self.qadr] = q

    def set_touch(self, geom_names):
        """Geoms the gripper fingers are meant to be close to (the handle's door, the plate tab)."""
        self.touch_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in geom_names}

    def collides(self, q=None):
        """True if any robot geom is within its clearance margin of the environment or itself.
        Gripper-to-'touch' geom pairs only count on real penetration."""
        if q is not None:
            self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        d = self.data
        if d.ncon == 0:
            return False
        if not self.touch_ids:
            return True
        gg, tid = self.gripper_geom, self.touch_ids
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ((gg[g1] and g2 in tid) or (gg[g2] and g1 in tid)) and c.dist > -0.001:
                continue
            return True
        return False

    def contacts(self):
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), round(float(c.dist), 4)))
        return out

    def min_clearance(self, q, distmax=0.15, held=()):
        """Smallest distance (m) between the robot and the printer/table/door at q.
        Gripper-to-'held' geom pairs (the object being grasped) are skipped."""
        m, d = self.model, self.data
        self.set_q(q)
        mujoco.mj_kinematics(m, d)
        held_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in held}
        best, pair = distmax, None
        ft = np.zeros(6)
        rg = np.where(self.robot_geom & (m.geom_contype > 0))[0]
        eg = np.where(~self.robot_geom & (m.geom_contype == ENV_CT))[0]
        for g1 in rg:
            if m.geom_bodyid[g1] == self.mount_id:
                continue
            for g2 in eg:
                if self.gripper_geom[g1] and g2 in held_ids:
                    continue
                if m.geom_bodyid[g1] in (self.mount_id,) or (
                        m.geom_bodyid[g2] == self.table_id and
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) == "shoulder_link"):
                    continue
                dist = mujoco.mj_geomDistance(m, d, g1, g2, distmax, ft)
                if dist < best:
                    best = dist
                    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or \
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1])
                    pair = (n1, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2))
        return best, pair


def write_scene_xml(cell: Cell, path, q=None):
    """Writes a standalone scene.xml with the current layout baked in (relative mesh paths)."""
    m = cell.model
    spec = cell.spec
    spec.meshdir = "assets"
    mount = spec.body("ur_mount")
    mount.pos = m.body_pos[cell.mount_id].copy()
    spec.body("table").pos = m.body_pos[cell.table_id].copy()
    spec.body("plate").pos = m.body_pos[cell.plate_id].copy()
    spec.joint("door_hinge").ref = 0.0
    rel = getattr(cell, "rel_height", 0.0)
    rg, sg = spec.geom("riser"), spec.geom("pr_stand")
    h = max(rel, 0.0005)
    rg.size = (0.09, h / 2, 0)
    rg.pos = (0, 0, -h / 2)
    rg.rgba = (0.35, 0.36, 0.4, 1.0 if rel > 0.001 else 0.0)
    if rel < -0.001:
        P = cell.cfg.printer
        sg.size = ((P.width - 0.02) / 2, (P.depth - 0.02) / 2, -rel / 2)
        sg.pos = (0, P.depth / 2, rel / 2)
    else:
        sg.rgba = (0.45, 0.42, 0.38, 0.0)
    # keyframe with the door open and the arm at q
    qpos = cell.data.qpos.copy()
    if q is not None:
        qpos[cell.qadr] = q
    for k in list(spec.keys):
        spec.delete(k)
    key = spec.add_key()
    key.name, key.qpos = "layout", qpos
    ctrl = np.zeros(m.nu)
    ctrl[:6] = qpos[cell.qadr]
    key.ctrl = ctrl
    xml = spec.to_xml().replace(os.path.join(ASSETS, "") , "")
    spec.meshdir = ASSETS
    with open(path, "w") as f:
        f.write(xml)
    return path


if __name__ == "__main__":
    cfg = Config()
    cell = Cell(cfg)
    m = cell.model
    print(f"compiled: {m.nbody} bodies, {m.ngeom} geoms, {m.njnt} joints, {m.nu} actuators")
