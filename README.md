# UR5 + Bambu X1E workspace planner (MuJoCo)

Finds where to put a **UR5 (CB3) + Robotiq 2F-85 + wrist RealSense** on a **3 ft × 6 ft table** so it can
pull printed parts out of a **Bambu Lab X1E with AMS**, and where to put the **third-person RealSense**
for VLA data collection. Everything is parametric: change a number in `config.py`, re-score, look at it.

## Quick start

Needs Python 3.10 or newer. Use a virtual environment (macOS and recent Linux block system-wide pip):

```bash
python3 --version              # 3.10+; on macOS: brew install python@3.12 if older
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python view.py                 # Linux / Windows
mjpython view.py               # macOS (the passive viewer needs mjpython, installed with mujoco)
```

Next time: `source .venv/bin/activate`, then run as above.

`view.py` opens the active layout from `layout.json` and loops its task motions (door → pick → …).

```bash
python view.py --layout pull                          # the other recommended layouts: pick | pull | both | top
python view.py --x 0.10 --dist 0.50 --rel 0 --bedz 200 # try your own placement
python view.py --static                               # no playback; move joints with the viewer's sliders
python -m mujoco.viewer --mjcf scene_pick.xml         # plain viewer on a saved scene
```

## Results (with the estimated printer interior — re-run after measuring)

Positions in the printer frame (see below): **x** = base center right (+) / left (−) of the printer center,
**dist** = base center in front of the printer's front face. The door hinges on the left.

| strategy | base x | base dist | heights | bed park Z | door (0→90°) | plate points reachable | plate pull | min clearance | score at ±2.5 cm |
|---|---|---|---|---|---|---|---|---|---|
| **pull plate out** | −350 mm | 850 mm | both on the table | 100 mm | full | – | full | 22 mm | 0.985 / 1.0 |
| **both (pick + pull)** | +125 mm | 750 mm | printer on a 200 mm stand | 200 mm | to 75° | 36 % (one finger direction) | full | 26 mm | 0.66 / 0.68 |
| **pick in place** | +350 mm | 600 mm | printer on a 200 mm stand | 200 mm | to 75° | 40 % (one finger direction) | – | 22 mm | 0.47 / 0.47 |
| top access (lid off) | −300 mm | 200 mm | arm on a 100 mm riser | 50 mm | – | 16 % | – | 22 mm | 0.24 |

What the study says (all with 2 cm clearance, gripper fully open, one arm configuration per task):

1. **Straight-down picks through the door don't fit.** With the gripper vertical, the wrist sits ~0.28 m above
   the fingertips; the door opening is ~0.39 m tall (estimate) and the AMS blocks the lid. Picks only work with the
   gripper tilted 50–70° into the printer, fingers closing left-right, over the middle three columns of the plate.
   The back of the plate (under the parked toolhead) is out of reach; the outer columns are limited by the open
   fingers meeting the side walls (less of a problem with smaller parts / a narrower opening).
2. **Pulling the plate out is the robust option.** Door 0→90° and a full 32 cm slide in one arm configuration,
   barely sensitive to ±2.5 cm placement error. It needs a printed grip tab clipped to the plate: the stock tab is
   shorter than the 2F-85 pads, so the lower pad runs into the heatbed (measure the real overhang).
3. **Top access doesn't help** — with the lid off, the gantry beam, parked toolhead and frame still cover most of
   the opening (16 % reachable at 2 cm clearance, 36 % at 1 cm).
4. **Bed park Z is a real lever.** Lowest bed (Z 250) puts the plate tab too close to the door sill for the
   lower finger; Z 100–200 works. Set it in the end G-code.
5. **Door**: needs a printed vertical bar handle on the free edge. In the pick/both layouts the task
   configuration only swings it to 75°; a different wrist configuration reaches 90° (a wrist flip outside the
   printer), or a magnet stop can hold it open.

Third-person camera: see **`CAMERA.md`**. The positions stored in `layout.json` → `camera` come from the
first camera search, which did not score image resolution or how much of the arm stays in view; for the
pick + pull layout it keeps the elbow in frame only 35 % of the time. `CAMERA.md` compares six placements
and explains the base-in-frame vs elbow-outward choice. Figures for each layout are in `results/figures/`.

## The tweak loop

1. Measure the printer and fix the `# MEASURE` values in `config.py` (checklist below).
2. Score a placement: `python evaluate.py --x 0.10 --dist 0.50 --rel 0.0 --bedz 200`
3. Look at it: `python view.py --x 0.10 --dist 0.50 --rel 0.0 --bedz 200`
4. Re-run the full search when the geometry changes:
   ```bash
   python sweep.py --scenario door --stage coarse && python sweep.py --scenario door --stage refine
   python sweep.py --scenario top  --stage coarse && python sweep.py --scenario top  --stage refine
   MUJOCO_GL=egl python report.py      # picks layouts, places the camera, renders, writes layout.json
   ```
   (`MUJOCO_GL=egl` is only for headless Linux; drop it on a desktop.)

## What a "placement" is

All positions are in the **printer frame**: origin at the bottom-front-center of the printer,
+x to the right looking at the door, +y into the printer, +z up.

| knob | meaning |
|---|---|
| `base_x` | arm base center, left (−) / right (+) of the printer center |
| `base_dist` | arm base center distance in front of the printer's front face |
| `rel_height` | arm base height minus printer base height. + = arm on a riser, − = printer on a stand |
| `bed_z` | where the bed parks after the print (printer Z, mm). Set with the end G-code |

Base rotation is not a knob: every UR5 joint turns ±360°, so it only sets where the base cable exits.
Point the cable away from the printer.

## How a placement is scored

Each task is a chain of gripper poses. A chain counts only if **every** waypoint is reachable in **one**
arm configuration (no wrist/elbow flips between waypoints, < 60° per joint step), and the arm, gripper and
wrist camera stay **≥ 2 cm** from the printer, door, table and itself. Fingers may touch the thing they are
holding (door handle, plate tab).

* **door**: grab a printed vertical bar handle on the door's free edge, swing 0 → 105°. The gripper may rotate
  around the bar during the swing (5×5 grip-angle schedules are tried).
* **pick**: 5 × 5 grasp points on the plate × tilt 0/30/50/70° (leaning into the printer) × 4 gripper yaws.
  Chain = pre-grasp (8 cm back along the gripper) → grasp → lift 3 cm → carry out through the door.
  Steep (≥ 60°) grasps grip 4 cm above the plate, so they assume parts ≥ ~5 cm tall.
* **pull**: tool horizontal, pinch the plate's front tab, slide the plate 32 cm out.
* **top** scenario: lid off, AMS beside the printer; pick from above and lift out of the top.

Metrics (0–1): pick coverage (at least one gripper direction / both directions), share of points reachable
within 30° of vertical, door swing achieved, plate pull achieved, and the worst singularity margin over
all motions used (`|sin q5|` wrist, `|sin q3|` elbow, shoulder term in m; the UR Jacobian determinant is
the product of these). Strategy weights are in `evaluate.py → WEIGHTS`; they are a judgment call, change them.

Layout selection takes the **worst** score within ±2.5 cm of each candidate, so the chosen spot tolerates
placement error.

The third-person camera is placed by ray-casting from candidate positions (10 cm grid, 0.35–0.95 m high) to the
plate and gripper: plate visible past the printer frame and open door, not hidden by the arm while it works,
gripper visible, everything inside the D435 color FOV (69° × 42°), and ≥ 15 cm from the arm at all times.

## Measurement checklist (printer interior is estimated)

Published and used as-is: X1E outer 389 × 389 × 457 mm, AMS +224 mm height, 256 mm³ build volume.
Measure these on the real printer (they move the answer):

- [ ] door opening: width, sill height, height of the underside of the top frame bar
- [ ] door hinge side, glass width/height, how far it opens
- [ ] plate front edge distance behind the front face; tab size and how far it overhangs the heatbed
- [ ] plate surface height above the table with the bed fully up (Z = 0) → `nozzle_tip_z`
- [ ] toolhead + gantry beam position after the end G-code runs
- [ ] wrist camera mount: camera offset from the tool axis, height above the flange, tilt (and model: D405 / D435)
- [ ] the real 2F-85 coupling (Menagerie model: grasp point 155.8 mm from the flange)

## Files

| file | what |
|---|---|
| `config.py` | every dimension; `# MEASURE` marks estimates |
| `ur5_kin.py` | UR5 CB3 DH kinematics, closed-form IK (all 8 configurations), singularity terms. `python ur5_kin.py` self-tests |
| `build_scene.py` | builds the MuJoCo model; moves base / bed / door without recompiling |
| `tasks.py` | task waypoint chains (door, pick, pull, top) |
| `evaluate.py` | scores one placement for each strategy |
| `sweep.py` | grid search (coarse 10 cm, refine 2.5 cm) |
| `camera_study.py` | third-person camera search (visibility and keep-out) |
| `camera_compare.py`, `CAMERA.md` | side-by-side camera candidates at VLA resolution, and the reasoning |
| `render.py`, `report.py` | figures, GIFs, `layout.json`, `scene_*.xml` |
| `view.py` | interactive viewer with task playback |
| `results/` | sweep data, `summary.json`, `figures/` |

## Model sources

* UR5 CB3: collision meshes and link frames from ros-industrial `universal_robot` (`ur_description`, BSD);
  kinematics = UR's published DH parameters. The MuJoCo model matches the DH forward kinematics to 1e-15.
* Robotiq 2F-85 and RealSense D435i meshes: MuJoCo Menagerie (licenses in `assets/*/LICENSE`).
* X1E, AMS, table: boxes from the dimensions in `config.py`.

## Limits

Kinematic + clearance study, not a motion planner. Motions between waypoints are checked only at the
waypoints (spaced 1.5–8 cm apart). The parts themselves are not modeled. Printer interior is approximate
until measured. Contact forces (peeling a part off the PEI, flexing the plate) are not simulated.
