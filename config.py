"""
All dimensions for the UR5 + Bambu X1E cell, in meters and degrees.

Coordinate frame used everywhere ("printer frame"):
  origin = bottom-front-center of the printer (where it sits on the table)
  +x     = to the right, looking at the printer door
  +y     = INTO the printer (front face is y = 0, so the robot sits at negative y)
  +z     = up

Anything marked  # MEASURE  is an estimate. The X1E outer size and the
build volume are published specs; the interior numbers (door opening,
plate position, nozzle height, toolhead park spot) are not, so measure
them on the real printer and re-run. Each one changes the answer.
"""
from dataclasses import dataclass, field


@dataclass
class Printer:
    # Published: X1E outer 389 x 389 x 457 mm, AMS adds 224 mm on top
    width: float = 0.389
    depth: float = 0.389
    height: float = 0.457
    ams_height: float = 0.224
    ams_width: float = 0.37            # MEASURE (AMS footprint)
    ams_depth: float = 0.29            # MEASURE

    # Front door opening (the hole the arm reaches through), printer frame
    door_open_x_half: float = 0.165    # MEASURE  half-width of the opening
    door_open_z_min: float = 0.045     # MEASURE  bottom sill height above table
    door_open_z_max: float = 0.390     # MEASURE  underside of the top frame bar
    door_hinge_side: str = "left"      # MEASURE  hinge side, looking at the printer
    door_panel_w: float = 0.36         # MEASURE  glass door width
    door_panel_h: float = 0.37         # MEASURE
    door_open_angle_deg: float = 90.0  # how far the door is open during picking (the arm swings it to 90)

    # Interior
    side_wall: float = 0.025           # MEASURE  side wall thickness
    back_wall: float = 0.050           # MEASURE  back wall (filters, electronics)
    lid_inner_z: float = 0.430         # MEASURE  underside of the top lid

    # Build plate. Published: 256 x 256 x 256 mm build volume
    plate_size: float = 0.257
    plate_front_y: float = 0.065       # MEASURE  plate front edge, behind the front face
    plate_tab_depth: float = 0.045     # ASSUMPTION: printed clip-on grip tab. The stock tab (~20 mm, MEASURE)
                                       # is shorter than the 2F-85 pads need, so the lower pad hits the heatbed
    plate_thickness: float = 0.002
    # Plate top surface height = nozzle_tip_z - bed_park_Z
    nozzle_tip_z: float = 0.335        # MEASURE  plate surface height at Z = 0 (bed fully up)

    # Toolhead + X gantry beam after the print. The end G-code decides this;
    # Bambu's default end G-code parks it at the back of the bed.
    toolhead_park_x: float = -0.030    # MEASURE
    toolhead_park_y: float = 0.290     # MEASURE  (toolhead center, near the back)
    toolhead_size: tuple = (0.08, 0.09, 0.09)   # MEASURE  w, d, h above nozzle tip
    gantry_beam_z: tuple = (0.05, 0.085)        # MEASURE  beam bottom/top above nozzle tip


@dataclass
class Table:
    length: float = 1.829   # 6 ft
    depth: float = 0.914    # 3 ft
    thickness: float = 0.03


@dataclass
class Tooling:
    # Robotiq 2F-85 on the Robotiq UR coupling (Menagerie model). Grasp point
    # ("pinch", between the fingertip pads) sits 0.1558 m out from the flange.
    tcp_offset: float = 0.1558
    # Wrist RealSense on the printed mount. Geometry is a guess; it matters
    # because it sticks out and can hit the door frame.
    wrist_cam_offset: float = 0.075    # MEASURE  camera center distance from tool axis
    wrist_cam_z: float = 0.035         # MEASURE  camera center height above the flange
    wrist_cam_tilt_deg: float = 20.0   # MEASURE  tilt of camera toward the fingers
    wrist_cam_fovy_deg: float = 58.0   # D405 color ~58 deg vertical, D435 ~42 deg


@dataclass
class Task:
    # Parts: unknown, so assume a generous envelope
    max_part_height: float = 0.060     # tallest part we plan clearance for
    grasp_height: float = 0.025        # grasp point above plate, top-down-ish grasps
    side_grasp_height: float = 0.040   # grasp point for steep (>= 60 deg) grasps: needs parts >= ~5 cm
    approach_dist: float = 0.080       # pre-grasp: back off this far along the gripper axis
    lift: float = 0.030                # lift the part off the plate before carrying it out
    clearance: float = 0.020           # min gap robot <-> printer/table (2 cm)

    pick_grid_n: int = 5               # 5 x 5 grasp points over the plate
    pick_inset: float = 0.030          # keep grid away from the plate edge
    pick_tilts_deg: tuple = (0.0, 30.0, 50.0, 70.0)   # gripper tilt from vertical, leaning into the printer
    # (yaw is fixed internally: 4 yaws = fingers left-right or front-back, camera on either side)

    plate_pull_dist: float = 0.32      # how far to slide the plate out
    door_angles_deg: tuple = (0, 15, 30, 45, 60, 75, 90)


@dataclass
class Sweep:
    # Base position relative to the printer (printer frame)
    base_x: tuple = tuple(round(-0.45 + 0.05 * i, 3) for i in range(19))   # -0.45..0.45
    base_dist: tuple = tuple(round(0.15 + 0.05 * i, 3) for i in range(15)) # 0.15..0.85 in front
    # Base mounting height MINUS printer bottom height. Positive: arm on a riser.
    # Negative: printer on a stand.
    rel_height: tuple = (-0.20, -0.10, 0.0, 0.10, 0.20)
    # Bed park Z (mm, printer coordinates) after the print
    bed_park_z_mm: tuple = (250, 200, 150, 100)


@dataclass
class Config:
    # "door": AMS on top, lid on, robot works through the front door (as the printer ships)
    # "top" : AMS moved beside the printer and the top lid removed, robot reaches in from above
    scenario: str = "door"
    printer: Printer = field(default_factory=Printer)
    table: Table = field(default_factory=Table)
    tooling: Tooling = field(default_factory=Tooling)
    task: Task = field(default_factory=Task)
    sweep: Sweep = field(default_factory=Sweep)
    # UR5 CB3 base yaw does not change reach (joint 1 turns +/-360 deg).
    # It only sets where the base cable exits; keep the cable away from the printer.
    base_yaw_deg: float = 0.0


def plate_top_z(cfg: Config, bed_z_mm: float) -> float:
    return cfg.printer.nozzle_tip_z - bed_z_mm / 1000.0
