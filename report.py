"""
After the sweeps: pick the layout for each strategy, check placement tolerance,
place the third-person camera, and produce images + layout.json + scene files.

    MUJOCO_GL=egl python report.py
"""
import json
import os
import warnings
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import mujoco

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "egl")

from config import Config, plate_top_z
from evaluate import Evaluator, describe
from camera_study import CameraStudy
from render import Shots, set_scene_cam, motion_gif, top_down_plot
from build_scene import write_scene_xml

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "results", "figures")
TOL = 0.025     # placement tolerance checked around each layout (m)


def load(scenario):
    rows = []
    for stage in ("coarse", "refine"):
        p = os.path.join(RES, f"sweep_{scenario}_{stage}.json")
        if os.path.exists(p):
            with open(p) as f:
                rows += json.load(f)
    return rows


def fits_table(cfg, r, layout="end"):
    """Base (149 mm dia.) must sit fully on the table, with the printer at one end."""
    T, P = cfg.table, cfg.printer
    half_w = (T.depth if layout == "end" else T.length) / 2 - 0.075 - TOL
    max_d = (T.length if layout == "end" else T.depth) - P.depth - 0.03 - 0.075 - TOL
    return abs(r["base_x"]) <= half_w and r["base_dist"] <= max_d


def robust_pick(ev, rows, obj, n_cand=12):
    """Best layout by its worst score within +/- TOL of the nominal spot (placement tolerance)."""
    cand = sorted([r for r in rows if r["objective"] == obj and fits_table(ev.cfg, r)], key=lambda r: -r["score"])
    seen, picked = set(), []
    for r in cand:
        k = (r["base_x"], r["base_dist"], r["rel_height"], r["bed_z"])
        if k not in seen:
            seen.add(k)
            picked.append(r)
        if len(picked) >= n_cand:
            break
    best = None
    for r in picked:
        worst = r["score"]
        for dx, dd in ((TOL, 0), (-TOL, 0), (0, TOL), (0, -TOL)):
            s = ev.evaluate(r["base_x"] + dx, r["base_dist"] + dd, r["rel_height"], r["bed_z"])[obj].score
            worst = min(worst, s)
        if best is None or worst > best[0]:
            best = (worst, r)
    return best[1], best[0]


def heatmap(rows, obj, rel, bz, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["objective"] == obj and r["rel_height"] == rel and r["bed_z"] == bz]
    xs = sorted({round(r["base_x"], 3) for r in rs})
    ds = sorted({round(r["base_dist"], 3) for r in rs})
    if len(xs) < 2 or len(ds) < 2:
        return None
    G = np.full((len(ds), len(xs)), np.nan)
    for r in rs:
        G[ds.index(round(r["base_dist"], 3)), xs.index(round(r["base_x"], 3))] = r["score"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(G, origin="lower", cmap="viridis", vmin=0, vmax=max(0.01, np.nanmax(G)),
                   extent=(xs[0] - 0.05, xs[-1] + 0.05, ds[0] - 0.05, ds[-1] + 0.05), aspect="equal")
    ax.set_xlabel("base x: left (-) / right (+) of printer center (m)")
    ax.set_ylabel("base distance in front of printer (m)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def level_table(rows, obj):
    """Best score for each (base height, bed Z) combination."""
    t = defaultdict(float)
    for r in rows:
        if r["objective"] == obj:
            k = (r["rel_height"], r["bed_z"])
            t[k] = max(t[k], r["score"])
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    layouts, summary = {}, {}
    for scenario, objs in (("door", ("pick", "pull", "both")), ("top", ("top",))):
        rows = load(scenario)
        if not rows:
            continue
        cfg = Config()
        cfg.scenario = scenario
        ev = Evaluator(cfg)
        for obj in objs:
            row, worst = robust_pick(ev, rows, obj)
            res = ev.evaluate(row["base_x"], row["base_dist"], row["rel_height"], row["bed_z"])[obj]
            print(f"\n=== {obj}: x {res.base_x:+.3f}  dist {res.base_dist:.3f}  rel {res.rel_height:+.2f}  "
                  f"bedZ {res.bed_z:.0f}  score {res.score:.3f}  worst within ±{TOL * 100:.1f} cm {worst:.3f}")
            print(describe(res))
            # camera
            cs = CameraStudy(ev, res)
            cams = cs.sweep(table_top_z=min(0.0, res.rel_height))
            cam = cams[0]
            print("camera:", cam)
            # images
            cell = ev.cell
            cell.set_layout(res.base_x, res.base_dist, res.rel_height, res.bed_z)
            set_scene_cam(cell, cam["pos"], cs.target, 42.0)
            shots = Shots(cell)
            segs = ev.trajectory(res)
            # a representative working pose: middle of a pick (grasp) or the pull grasp
            work = [s for s in segs if s[0].startswith("pick")] or [s for s in segs if s[0] == "pull"] or segs
            rep = work[len(work) // 2][1] if work else [(np.zeros(6), 100.0)]
            q_rep, door_rep = rep[min(2, len(rep) - 1)]
            cell.set_grip("pinch" if work and work[0][0] == "pull" else "open")
            lookat = (res.base_x * 0.45, -res.base_dist * 0.45, 0.22)
            iso = shots.shot(q_rep, door_rep, shots.free(lookat, 2.2, 55, -26), label=f"{obj}: overview")
            iso.save(os.path.join(OUT, f"{obj}_overview.png"))
            side = shots.shot(q_rep, door_rep, shots.free((0.0, 0.0, 0.2), 1.25, 168, -14),
                              label=f"{obj}: cutaway, right wall hidden", xray=True)
            side.save(os.path.join(OUT, f"{obj}_closeup.png"))
            sc = shots.shot(q_rep, door_rep, "scene_cam", label="third-person camera (D435 FOV)", w=960, h=540)
            sc.save(os.path.join(OUT, f"{obj}_scene_cam.png"))
            pre = rep[0]
            wc = shots.shot(pre[0], pre[1], "wrist_cam", label="wrist camera at pre-grasp", w=640, h=480)
            wc.save(os.path.join(OUT, f"{obj}_wrist_cam.png"))
            top_down_plot(cfg, dict(base_x=res.base_x, base_dist=res.base_dist), res, cam["pos"],
                          os.path.join(OUT, f"{obj}_table_layout.png"))
            # heatmap at the chosen height / bed Z
            heatmap(rows, obj, res.rel_height, res.bed_z, os.path.join(OUT, f"{obj}_heatmap.png"),
                    f"{obj}: score by base position (height {res.rel_height:+.2f} m, bed Z {res.bed_z:.0f})")
            # true minimum clearance along the chosen motions (fingers vs the object they hold excluded)
            worst_cl, worst_pair = 9.0, None
            for name, seg in segs:
                cell.set_grip("pinch" if name in ("door", "pull") else "open")
                held = {"door": ("door_glass",), "pull": ("heatbed", "plate_top", "plate_tab")}.get(
                    name, ("plate_top", "plate_tab"))
                for q, door in seg:
                    cell.set_door(door)
                    dist, pair = cell.min_clearance(q, held=held)
                    if dist < worst_cl:
                        worst_cl, worst_pair = dist, (pair, name)
            print("min clearance:", round(worst_cl * 1000), "mm", worst_pair)
            # scene file with this layout baked in
            q0 = segs[0][1][0][0] if segs else np.zeros(6)
            write_scene_xml(cell, os.path.join(HERE, f"scene_{obj}.xml"), q=q0)
            layouts[obj] = dict(scenario=scenario, objective=obj, base_x=res.base_x, base_dist=res.base_dist,
                                fits_3ft_side_layout=fits_table(cfg, dict(base_x=res.base_x, base_dist=res.base_dist), "side"),
                                rel_height=res.rel_height, bed_z=res.bed_z, camera=cam["pos"],
                                branch=res.branch_label, score=round(res.score, 3),
                                min_clearance_mm=round(worst_cl * 1000),
                                worst_score_within_2_5cm=round(worst, 3))
            summary[obj] = dict(result={k: v for k, v in asdict(res).items() if k != "chosen"},
                                camera=cam, camera_top5=cams[:5], worst_within_tol=worst,
                                levels={f"{k[0]:+.2f}|{k[1]:.0f}": v for k, v in level_table(rows, obj).items()})
            # motion GIF
            gif_segs = [s for s in segs if s[0] == "door"][:1] + work[len(work) // 2: len(work) // 2 + 1] \
                + [s for s in segs if s[0] == "pull"][:1]
            if gif_segs:
                g = Shots(cell, 640, 420)
                motion_gif(g, gif_segs, g.free(lookat, 2.0, 55, -24), os.path.join(OUT, f"{obj}_motion.gif"))
    active = "both" if "both" in layouts and summary["both"]["result"]["pick_any"] > 0.3 \
        and summary["both"]["result"]["pull"] >= 1.0 else ("pick" if "pick" in layouts else list(layouts)[0])
    with open(os.path.join(HERE, "layout.json"), "w") as f:
        json.dump(dict(active=active, layouts=layouts), f, indent=2)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\nwrote layout.json (active:", active, ")")


if __name__ == "__main__":
    main()
