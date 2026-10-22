# UR5 + Bambu X1E workspace planner — notes for Claude Code

Workspace-placement study (MuJoCo) for a UR5 **CB3** + Robotiq 2F-85 + wrist RealSense that removes printed
parts from a Bambu Lab X1E with AMS, on a 3 ft × 6 ft table, with a third-person RealSense for VLA data
collection. Vijey is helping a friend who owns the cell; the friend will supply real measurements.
User-facing docs: `README.md` (setup, method, results, measurement checklist), `CAMERA.md` (camera choice).

## Run

```bash
source .venv/bin/activate        # Python 3.10+ (scipy as_quat(scalar_first=True) needs scipy >= 1.14)
mjpython view.py [--layout pick|pull|both|top] [--x .. --dist .. --rel .. --bedz ..]   # macOS viewer
python view.py ... --dry-run     # no window: prints score + motion size
python evaluate.py --x 0.125 --dist 0.75 --rel -0.2 --bedz 200
python ur5_kin.py                # IK/FK self-test, must print "round-trip recovered: 1.0"
```

Full pipeline after any geometry change (≈ 15 min on 2 cores):

```bash
python sweep.py --scenario door --stage coarse && python sweep.py --scenario door --stage refine
python sweep.py --scenario top  --stage coarse && python sweep.py --scenario top  --stage refine
python report.py                 # layout.json, scene_*.xml, results/figures/, results/summary.json
python camera_compare.py --layout both
python results/summarize.py      # text tables from the sweep files
```

Headless Linux: prefix render scripts with `MUJOCO_GL=egl`. An `EGLError` traceback at interpreter exit is
harmless (renderer teardown). The "Attach conflict ... impratio/cone" warning is harmless (handled in spec).

## Conventions

* **Printer frame** everywhere: origin bottom-front-center of the printer, +x right looking at the door,
  +y into the printer (robot sits at negative y), +z up. `base_dist` = −base y.
* `rel_height` = arm base height − printer base height (+ riser under arm, − stand under printer).
  The table top is at `min(0, rel_height)`.
* `bed_z` = printer Z (mm) where the bed parks after the print; plate top = `nozzle_tip_z − bed_z/1000`.
* Tool frame: z = approach (out of the fingers), x = finger closing direction, wrist camera on tool −y.
  TCP = Menagerie `pinch` site, 0.1558 m from the flange.
* IK branch index `b = 4*shoulder + 2*wrist + elbow`; labels in `ur5_kin.BRANCH_LABELS`.
  A task chain is valid only if all waypoints are in ONE branch with < 60° per-joint steps.

## Files

`config.py` (all dimensions; `# MEASURE` = estimate) → `build_scene.py` (MjSpec model + `Cell` setters that
move base/table/bed/door without recompiling) → `tasks.py` (waypoint chains) → `evaluate.py` (scores one
placement for every strategy) → `sweep.py` (grid, multiprocessing) → `report.py` (robust pick at ±2.5 cm,
camera search, renders). `camera_study.py` = ray-cast visibility metrics; `camera_compare.py` = fixed
candidates at 224 px. `render.py` = images/GIF/top-down drawing. `view.py` = interactive playback.

## Verified

* Closed-form IK recovers 20k random configs exactly; |det J| factorization matches numeric Jacobian (1e-16).
* MuJoCo `tool0` site matches the DH FK to 1e-15 m (link frames transcribed from ros-industrial ur_description).
* Door handle site vs `tasks.door_chains` geometry: exact match at every angle.
* Every chosen layout keeps ≥ 22 mm between robot and printer/table/door along all chosen motions
  (fingers vs the object they hold excluded).

## Gotchas already solved (don't re-break)

* `ur_mount`–`shoulder_link` contact is excluded (meshes touch by construction).
* The 2F-85 is a closed 4-bar linkage: set all 8 gripper joints via `Cell.set_grip("open"|"pinch"|"closed")`
  (values settled once by simulation in `Cell.__init__`), never just the driver joint.
* Collision margin on environment geoms = the 2 cm clearance. Gripper-vs-held-object pairs are exempted per
  chain via `Chain.meta["touch"]` → `Cell.set_touch()`; only real penetration counts for those.
* Robotiq meshes are attached with absolute paths; `write_scene_xml` strips them back to relative.
* Wrist camera tilt sign: `Rot.from_euler("x", -tilt)` tilts toward the fingers.
* Door swings 0→90° (105° was unreachable in one branch for most placements).

## Assumptions to replace with measurements (results depend on them)

Door opening size and sill height, hinge side (assumed left), plate front offset, `nozzle_tip_z`, toolhead
and gantry park position, AMS footprint, wrist camera mount offsets, camera model (D405 vs D435 FOV).
Also assumed: a printed ~45 mm grip tab clipped to the plate (stock tab too short for the 2F-85 pads) and a
printed vertical bar handle on the door's free edge.

## Current results (estimated geometry)

| layout | base x | dist | heights | bed Z | notes |
|---|---|---|---|---|---|
| pull | −0.35 | 0.85 | flat | 100 | door 0→90 + full pull, score 0.985 at ±2.5 cm |
| both (active) | +0.125 | 0.75 | printer on 0.2 m stand | 200 | pull 100 %, picks 36 %, door to 75° |
| pick | +0.35 | 0.60 | printer on 0.2 m stand | 200 | picks 40 % (tilted 50–70°, middle columns) |
| top | — | — | — | — | 16 % reachable: not viable |

Straight-down picks through the door don't fit (wrist ~0.28 m above the fingertips vs ~0.39 m opening).

## Open items

1. Replace `# MEASURE` values with the friend's measurements and re-run the pipeline.
2. Camera: `layout.json` cameras come from the first search (keeps the elbow in frame only 35 % for "both").
   Choose E (whole cell, base in view; image-only VLAs like OpenVLA) or F (elbow outward; VLAs that also
   take joint angles + wrist camera) once the VLA is chosen, then write it into `layout.json`. See `CAMERA.md`.
3. Add the drop-off bin to the scene and tasks; it must be inside the third-person frame.
4. Motions are checked at waypoints only (1.5–8 cm apart); add interpolated checks or a planner before
   running on hardware. Parts and contact forces are not modeled.
5. Not yet run on Vijey's Mac (`mjpython view.py`).
