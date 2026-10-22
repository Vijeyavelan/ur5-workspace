"""
Placement sweep.

    python sweep.py --scenario door --stage coarse     # 10 cm grid over base x / distance, heights, bed Z
    python sweep.py --scenario door --stage refine     # 2.5 cm grid around the best coarse results
    python sweep.py --scenario top  --stage coarse

Writes results/sweep_<scenario>_<stage>.json (one row per placement per strategy).
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

from config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
_ev = None


def _init(scenario):
    global _ev
    import warnings
    warnings.filterwarnings("ignore")
    from evaluate import Evaluator
    cfg = Config()
    cfg.scenario = scenario
    _ev = Evaluator(cfg)


def _run(p):
    out = _ev.evaluate(*p)
    rows = []
    for obj, r in out.items():
        d = asdict(r)
        d.pop("chosen")
        d["chosen"] = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                       for k, v in r.chosen.items()}
        rows.append(d)
    return rows


def grid(scenario, stage):
    if stage == "coarse":
        xs = np.round(np.arange(-0.4, 0.401, 0.1), 3)
        ds = np.round(np.arange(0.2, 0.801, 0.1), 3)
        rels = (-0.2, -0.1, 0.0, 0.1, 0.2)
        bzs = (250, 200, 150, 100) if scenario == "door" else (250, 200, 150, 100, 50)
        return [(float(x), float(d), r, b) for x in xs for d in ds for r in rels for b in bzs]
    # refine around the best coarse rows of each strategy
    with open(os.path.join(RES, f"sweep_{scenario}_coarse.json")) as f:
        rows = json.load(f)
    seeds = set()
    for obj in sorted({r["objective"] for r in rows}):
        rs = sorted([r for r in rows if r["objective"] == obj], key=lambda r: -r["score"])
        taken = 0
        for r in rs:
            key = (r["rel_height"], r["bed_z"], round(r["base_x"], 2), round(r["base_dist"], 2))
            if key in seeds:
                continue
            seeds.add(key)
            taken += 1
            if taken >= 4:
                break
    pts = set()
    for rel, bz, x0, d0 in seeds:
        for dx in np.arange(-0.075, 0.0751, 0.025):
            for dd in np.arange(-0.075, 0.0751, 0.025):
                d = round(d0 + dd, 3)
                if d < 0.15:
                    continue
                pts.add((round(float(x0 + dx), 3), float(d), rel, bz))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="door", choices=["door", "top"])
    ap.add_argument("--stage", default="coarse", choices=["coarse", "refine"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    pts = grid(a.scenario, a.stage)
    print(f"{a.scenario}/{a.stage}: {len(pts)} placements on {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.scenario,)) as pool:
        for i, rr in enumerate(pool.imap_unordered(_run, pts, chunksize=4)):
            rows += rr
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(pts)}  {el:.0f}s elapsed, ~{el / (i + 1) * (len(pts) - i - 1):.0f}s left", flush=True)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f"sweep_{a.scenario}_{a.stage}.json")
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    for obj in sorted({r["objective"] for r in rows}):
        best = max((r for r in rows if r["objective"] == obj), key=lambda r: r["score"])
        print(f"  best [{obj}] score {best['score']:.3f}: x {best['base_x']:+.3f}, dist {best['base_dist']:.3f}, "
              f"rel {best['rel_height']:+.2f}, bedZ {best['bed_z']}, door {best['door']:.0%}, "
              f"pick any {best['pick_any']:.0%}, both {best['pick_both']:.0%}, pull {best['pull']:.0%}")


if __name__ == "__main__":
    main()
