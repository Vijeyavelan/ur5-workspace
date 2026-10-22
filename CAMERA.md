# Third-person camera placement for VLA data

Fix these before the camera is mounted for good: a trained policy expects the exact viewpoint it was
trained on, so moving the camera later means collecting new data and fine-tuning again.

## What each camera is for

Most VLAs shrink each camera image to about 224 × 224 before the model sees it (OpenVLA, π0).
At that size a 30 mm printed part on the plate is:

| camera | 30 mm part at 224 px |
|---|---|
| third-person, any placement tried | 3–6 px |
| wrist camera, pre-grasp | 23 px |
| wrist camera, at grasp | 36 px |

So the **wrist camera carries the part detail** and the **third-person camera carries the big picture**:
door open or closed, plate in or out, where the arm is relative to the door frame, which step of the task
it is on. The third-person view should be framed around the arm and the opening, not the parts.

## Base in frame, or elbow outward?

Measured with `camera_compare.py` on the pick + pull layout (`results/figures/camera_compare_both.png`):

| | camera (printer frame, m) | 30 mm part | gripper visible | elbow in frame | plate visible through door | base |
|---|---|---|---|---|---|---|
| A: first study pick | 0.30, −0.70, 1.05 above table | 4.3 px | 99 % | 35 % | 96 % | no |
| B: side-on | 0.62, −0.30, 0.75 | 5.7 px | 90 % | 10 % | 16 % | no |
| C: high, looking down | 0.42, −0.55, 1.14 | 4.3 px | 100 % | 3 % | 68 % | no |
| D: beyond the open door | −0.45, −1.05, 0.89 | 3.4 px | 52 % | 99 % | 8 % | no |
| E: whole cell | 0.95, −1.15, 1.28 | 2.7 px | 100 % | 100 % | 56 % | yes |
| **F: elbow outward** | **0.60, −0.80, 0.95** | **3.8 px** | **100 %** | **99 %** | **72 %** | no |

(x = right of printer center, y = into the printer so negative = in front of it, height above the table.
F is aimed 60 % at the door opening and 40 % at the elbow's average position.)

* **Base in view (E)** shows the full arm pose, which helps a model that gets no joint angles
  (OpenVLA takes one image and nothing else). It costs ~35 % of the pixels on the printer and plate.
* **Elbow outward (F)** drops the base, which never moves and carries no information, and keeps
  everything that moves plus the forearm passing through the door (the tightest clearance in the task).
  Good when the model also gets joint angles and the wrist camera (π0-style).
* **A** loses the elbow most of the time; the first camera search did not score image resolution or
  how much of the arm is in view. `camera_compare.py` adds both.

Every layout shares one limit: the arm hides ~30 % of the plate while it works inside, because the door
is the only view into the chamber. No outside camera fixes that; the wrist camera covers it.

F and E sit past the edge of a 3 ft table: use a clamp arm or a floor tripod.

## Settle before mounting

1. **The model you will train.** Input size, whether it takes joint angles, and how many cameras it uses
   decide between E (image only) and F (image + joints + wrist camera).
2. **The drop-off bin.** It is part of the task, so it must be in frame; once the camera is fixed, so is the bin.
3. **Cropping.** If the training pipeline center-crops to a square, the sides of a 16:9 frame are lost.
   Frame for the square, or record at 640 × 480.
4. **Fixed exposure and white balance.** The dark chamber next to a bright table makes auto-exposure
   flicker between episodes.
5. **A rigid mount**, off the robot's table if that table wobbles. Calibrate the camera pose (an AprilTag
   works), save a reference image, and compare against it at the start of every session.

## Try your own spots

Add entries to `candidates()` in `camera_compare.py` (position, look-at target, printer frame) and run

```bash
MUJOCO_GL=egl python camera_compare.py --layout both     # headless Linux; drop MUJOCO_GL on a desktop
```

It renders each candidate at three task moments and at 224 × 224, and writes the numbers above to
`results/camera_compare_<layout>.json`.
