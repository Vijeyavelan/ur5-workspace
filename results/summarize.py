"""Quick text summary of sweep files: best score per strategy for each base height / bed Z."""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

for scen in sys.argv[1:] or ["door", "top"]:
    rows = []
    for st in ("coarse", "refine"):
        p = os.path.join(HERE, f"sweep_{scen}_{st}.json")
        if os.path.exists(p):
            rows += json.load(open(p))
    if not rows:
        continue
    print(f"\n##### {scen}: {len(rows)} rows")
    for obj in sorted({r["objective"] for r in rows}):
        rs = [r for r in rows if r["objective"] == obj]
        t = defaultdict(lambda: None)
        for r in rs:
            k = (r["rel_height"], r["bed_z"])
            if t[k] is None or r["score"] > t[k]["score"]:
                t[k] = r
        bz = sorted({k[1] for k in t})
        print(f"\n[{obj}] best score by base height (rows) x bed Z (cols)")
        print("  rel\\Z " + "".join(f"{b:>8.0f}" for b in bz))
        for rel in sorted({k[0] for k in t}):
            print(f"  {rel:+.2f} " + "".join(f"{t[(rel, b)]['score']:8.3f}" if t[(rel, b)] else "       -" for b in bz))
        top = sorted(rs, key=lambda r: -r["score"])[:6]
        for r in top:
            print(f"   {r['score']:.3f}  x {r['base_x']:+.3f} d {r['base_dist']:.3f} rel {r['rel_height']:+.2f} Z {r['bed_z']:.0f}"
                  f" | door {r['door']:.0%} pick_any {r['pick_any']:.0%} both {r['pick_both']:.0%} low {r['low_tilt']:.0%}"
                  f" pull {r['pull']:.0%} | wrist {r['sing_wrist']:.2f} elbow {r['sing_elbow']:.2f}"
                  f" sh {r['sing_shoulder']:.2f} | {r['branch_label']}")
