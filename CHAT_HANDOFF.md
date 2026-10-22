# Chat handoff: building the shareable artifact

Context from the Claude chat where this study was built, for Claude Code to use when making the artifact
Vijey will share with his friend. The repo files (`README.md`, `CAMERA.md`, `CLAUDE.md`, `results/`) hold
the method and numbers; this file adds what was only said in chat, flags what is stale, and says how to
frame it.

## Audience and purpose

* **Reader:** Vijey's friend, who owns the cell and will run the robot. Vijey is helping because he has
  MuJoCo experience; the friend supplied the hardware list but not detailed dimensions.
* **Purpose:** show where to place the UR5 and the third-person camera, why, what is still an estimate,
  and what the friend must measure and decide before anything is mounted.
* **Tone:** technical report, plain language. Present options neutrally with their trade-offs; don't pick
  for the friend unless a result is clearly one-sided (e.g. top access).
* **Voice:** don't write a personal note in Vijey's voice. If a greeting or note is wanted, leave a spot for
  Vijey to write it himself.
* **Not in the artifact:** the code. It goes to the friend separately (the zip, or push the repo to GitHub;
  `ur5_x1e_workspace.zip` is in `.gitignore`).

## The cell (as the friend described it)

* UR5 **CB3** (not e-Series), Robotiq **2F-85** with the wrist RealSense on a 3D-printed mount.
* Two Intel RealSense cameras, **model not yet known** (D405 / D435 / D455 change the field of view).
* Bambu Lab **X1E with AMS** on top.
* Everything on one **3 ft × 6 ft table**. Temporary cardboard sheets will cover the background so it is
  clean for the cameras.
* Goal: pick printed parts from the printer and record demonstrations (wrist camera + third-person
  camera) to train a VLA model.
* Still open on the friend's side: pick parts in place vs pull the plate out (he wants to try both),
  part size (assume generous clearance), RealSense model, which VLA, where picked parts go.

## Put this first: nothing is final yet

Every position comes from **estimated printer-interior dimensions**. Published and used as-is: X1E outer
size 389 × 389 × 457 mm, AMS adds 224 mm of height, 256 mm build volume. Estimated (marked `# MEASURE` in
`config.py`): door opening size and sill height, hinge side (assumed left), plate position, plate height at
Z = 0, parked toolhead/gantry position, AMS footprint, wrist camera mount offsets.

So the artifact should say plainly: **don't drill, bolt or mount anything yet.** Measure (checklist in
`README.md`), Vijey re-runs the study, then positions get fixed.

## Coordinate frame (needed to place anything on the real table)

All positions are measured from the **bottom-front-center of the printer** (where its front face meets
the table, at the middle of its width): **x** to the right looking at the door, **distance** straight out
in front of the printer face, heights above the table. The `*_table_layout.png` drawings show this frame
with dimensions.

## Main findings

1. **Straight-down picks through the door don't fit.** With the gripper vertical, the wrist sits about
   0.28 m above the fingertips; the door opening is about 0.39 m tall (estimate) and the AMS blocks the lid.
   Picks only work with the gripper tilted 50–70° into the printer, fingers closing left-right, over the
   middle three columns of the plate. The back row, under the parked toolhead, is out of reach.
2. **Pulling the plate out is the robust option.** Door 0→90° and a full 32 cm slide in one arm
   configuration, barely affected by ±2.5 cm placement error.
3. **Top access (lid off, AMS beside the printer) doesn't help:** gantry, toolhead and frame still block
   most of the opening (16 % of plate points reachable).
4. **Bed park height matters.** Set by the printer's end G-code. Z 100–200 works; Z 250 puts the plate
   tab too close to the door sill for the lower finger.

## Layouts (estimated geometry; all keep ≥ 2 cm robot-to-printer/table clearance)

| layout | base x | distance | heights | bed Z | door | plate points | plate pull | min clearance | score at ±2.5 cm |
|---|---|---|---|---|---|---|---|---|---|
| **Pull plate out** | −350 mm (hinge side) | 850 mm | both on the table | 100 | 0→90° | – | full | 22 mm | 0.985 |
| **Pick + pull** | +125 mm | 750 mm | printer on a 200 mm stand | 200 | to 75° | 36 % | full | 26 mm | 0.658 |
| Pick in place | +350 mm | 600 mm | printer on a 200 mm stand | 200 | to 75° | 40 % | – | 22 mm | 0.467 |
| Top access (lid off) | −300 mm | 200 mm | arm on a 100 mm riser | 50 | – | 16 % | – | 22 mm | 0.238 |

Lead with **Pull** and **Pick + pull**; give Pick in place and Top access one line each.
Plate-point percentages come from a 5 × 5 grid on estimated geometry: present them as comparisons between
layouts, not as final coverage. "Door to 75°": the task's arm configuration stops at 75°; a different
configuration reaches 90° (a wrist flip outside the printer), or a magnet can hold the door open.

## Points that exist only in chat (not yet in the repo)

* **Why every task uses one arm configuration.** If some demonstrations flip the wrist or elbow partway
  through and others don't, the VLA gets inconsistent motions to learn. That's why the study rejects any
  motion needing a flip, even when it would reach more of the plate.
* **The X1E locks its door while printing** (listed feature), so every episode starts after the print
  finishes, with the door closed.
* **Table stiffness.** A wobbly or folding table will shake a camera mounted on it; mount the third-person
  camera off the table, or brace the table.
* **Background.** Keep the cardboard backdrop and lighting identical across every episode; the model
  learns from whatever is in frame.
* **Moving the camera later** doesn't destroy anything, but a trained policy expects the exact viewpoint;
  a shift means collecting new data and fine-tuning. Treat the camera as fixed once mounted.
* **UR5 model.** MuJoCo's ready-made library only has the e-Series UR5e; this study builds the CB3 UR5
  from Universal Robots' published dimensions (verified to match exactly).
* **Mac setup.** `pip` isn't on a Mac's PATH and system-wide installs are blocked; use the virtual
  environment steps in `README.md`.

## Build list for the friend (scattered in the README; give it its own section)

* Printed vertical **bar handle** on the door's free edge (the gripper closes around it to swing the door).
* Printed **clip-on grip tab** for the build plate, about 45 mm long. The stock tab is too short for the
  2F-85 pads; the lower pad hits the heatbed. (Pull layouts only.)
* A **200 mm stand** under the printer (Pick + pull and Pick in place layouts).
* A **clamp arm or floor tripod** for the third-person camera (the good spots are past the table edge).
* Optional: a **magnet** to hold the door open.
* **End G-code change** so the bed parks at the chosen Z after each print.
* Optional: an **AprilTag** to calibrate the camera pose.

## Camera section

Source: `CAMERA.md` and `results/figures/camera_compare_both.png`. Key points:

* At the ~224 × 224 most VLAs use, a 30 mm part is **3–6 px** in the third-person view (any placement)
  but **23 px at pre-grasp / 36 px at grasp** in the wrist camera. The wrist camera carries the part
  detail; the third-person camera carries the task state (door, plate, arm vs door frame, task phase).
* **E (whole cell, base in view)** suits image-only VLAs such as OpenVLA. **F (elbow outward, base out of
  view)** suits VLAs that also take joint angles and the wrist image (π0-style). The choice depends on
  which VLA the friend trains; present both, no pick.
* The arm hides ~30 % of the plate while working inside from every outside viewpoint; the wrist camera
  covers that.
* Settle before mounting: the VLA model, the drop-off bin location (must be in frame), cropping
  (record 640 × 480 or frame for a square crop), fixed exposure and white balance, a rigid mount.

## Figures

Paths are under `results/figures/`.

Use:
* `pull_table_layout.png`, `both_table_layout.png`: dimensioned top views (see the warning below).
* `pull_motion.gif` (2.3 MB), `both_motion.gif` (1.1 MB): door + task motion.
* `both_overview.png`, `pull_overview.png`: 3D views of each layout.
* `camera_compare_both.png`: the six camera candidates, each also shown at 224 × 224.
* Optional: `both_heatmap.png` / `pull_heatmap.png` (score vs base position) to show placement sensitivity.

**Stale, handle with care:** the camera stored in `layout.json` and drawn in every `*_table_layout.png` and
`*_scene_cam.png` is from the first camera search, which loses the elbow from view most of the time
(camera A in `CAMERA.md`). Don't present that spot as the recommendation. Either caption the drawings
("camera position not final, see camera section") or re-render the drawings once E or F is chosen.
Skip `*_wrist_cam.png` for pick/top unless explaining the wrist view, and skip the `top_*` figures beyond
one mention.

## Limits to state

* Kinematics and clearance study, not a motion planner: motions are checked at waypoints 1.5–8 cm apart,
  not continuously.
* Printed parts and contact forces (peeling parts off the plate, flexing it) are not modeled.
* Results change once the printer is measured.

## What the friend needs to do (end of the artifact)

1. Measure the printer (checklist in `README.md`) and send the numbers to Vijey.
2. Decide: pull the plate out, pick in place, or keep both open.
3. Pick the VLA model (decides camera E vs F) and the RealSense models.
4. Choose where picked parts go (bin location must be in the camera frame).
5. Make the printed handle and grip tab; get a stand and camera mount as the chosen layout needs.

## Sources

* Bambu Lab X1E spec sheet (outer size, build volume): https://3dprintergear.com.au/assets/brochures/BL_X1E.pdf
* MatterHackers X1E listing (AMS adds 224 mm height): https://matterhackers.com/store/l/bambu-lab-x1e-3d-printer
* 3DDevice X1E Combo listing (door lock during print): https://3ddevice.com.ua/en/product/3d-printer-bambu-lab-x1e-combo
* MuJoCo Menagerie (Robotiq 2F-85, RealSense D435i models): https://github.com/google-deepmind/mujoco_menagerie
* ros-industrial universal_robot (UR5 CB3 meshes and link frames): https://github.com/ros-industrial/universal_robot
