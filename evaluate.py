"""
Scores one arm placement for each task strategy. Used by the sweep, and on its
own for hand tweaks:

    python evaluate.py                       # scores the layout in layout.json
    python evaluate.py --x 0.05 --dist 0.45 --rel 0.10 --bedz 250
    python evaluate.py --scenario top --x 0 --dist 0.35 --rel 0.2 --bedz 100

A placement = base x (left/right of the printer center), base distance in front
of the printer face, base height relative to the printer bottom (+ = arm on a
riser, - = printer on a stand), and the bed park Z after the print.

Strategies scored
  pick : open the door, pick parts in place through the door, carry them out
  pull : open the door, slide the plate out (parts get removed outside)
  both : one layout that supports pick and pull, so you can try both
  top  : (scenario "top") AMS beside the printer, lid off, pick from above
"""
import argparse
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

import ur5_kin as K
from config import Config
from tasks import pick_chains, door_chains, pull_chains, plate_fits_door

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_JUMP = np.deg2rad(60)      # max joint change between neighbouring waypoints

# How each strategy's score is weighted (all metrics are 0..1)
WEIGHTS = {
    "pick": dict(door=0.25, pick_any=0.40, pick_both=0.15, low_tilt=0.10, singularity=0.10),
    "pull": dict(door=0.40, pull=0.45, singularity=0.15),
    "both": dict(door=0.20, pick_any=0.30, pick_both=0.05, low_tilt=0.05, pull=0.30, singularity=0.10),
    "top": dict(pick_any=0.30, pick_both=0.40, low_tilt=0.15, singularity=0.15),
}


def _flatten(chains):
    poses = np.array([T for c in chains for T in c.poses])
    starts = np.cumsum([0] + [len(c.poses) for c in chains])[:-1]
    return poses, starts


@dataclass
class Result:
    base_x: float
    base_dist: float
    rel_height: float
    bed_z: float
    scenario: str = "door"
    objective: str = ""
    branch: int = -1
    branch_label: str = ""
    score: float = 0.0
    pick_both: float = 0.0      # share of plate points reachable with fingers both ways
    pick_any: float = 0.0       # share reachable at least one way
    low_tilt: float = 0.0       # share reachable (any way) with the gripper within 30 deg of vertical
    door: float = 0.0           # share of the door swing (0 -> 105 deg) achieved
    pull: float = 0.0           # share of the plate pull achieved
    plate_fits_door: bool = True
    sing_wrist: float = 0.0     # min |sin q5| over the motions used (0 = wrist singular)
    sing_elbow: float = 0.0     # min |sin q3|
    sing_shoulder: float = 0.0  # min shoulder term (m)
    sing_norm: float = 0.0
    kin_pick_any: float = 0.0   # reachability before collision checks
    pick_map: list = field(default_factory=list)   # 5x5: 0 none, 1 one way, 2 both ways
    tilt_map: list = field(default_factory=list)   # smallest tilt (deg) that worked per point, -1 if none
    chosen: dict = field(default_factory=dict)     # chain ids used (for playback)


def objective_score(r: Result, obj: str):
    w = WEIGHTS[obj]
    vals = dict(door=r.door, pick_both=r.pick_both, pick_any=r.pick_any, low_tilt=r.low_tilt,
                pull=r.pull, singularity=r.sing_norm)
    return sum(w[k] * vals[k] for k in w)


class Evaluator:
    def __init__(self, cfg: Config, cell=None, layout="end"):
        self.cfg = cfg
        if cell is None:
            from build_scene import Cell
            cell = Cell(cfg, layout)
        self.cell = cell
        self.T_flange_tcp_inv = np.eye(4)
        self.T_flange_tcp_inv[2, 3] = -cfg.tooling.tcp_offset
        self.top = cfg.scenario == "top"
        self.door = [] if self.top else door_chains(cfg)
        self._cache = {}
        self.objectives = ["top"] if self.top else ["pick", "pull", "both"]

    def chains(self, bed_z):
        if bed_z not in self._cache:
            pk = pick_chains(self.cfg, bed_z)
            pl = [] if self.top else pull_chains(self.cfg, bed_z)
            allc = pk + pl + self.door
            poses, starts = _flatten(allc)
            self._cache[bed_z] = (pk, pl, allc, poses, starts)
        return self._cache[bed_z]

    # ---------- kinematics for all waypoints, all 8 branches ----------
    def kinematics(self, poses, starts, base_pose):
        Tb_inv = np.linalg.inv(base_pose)
        flange = Tb_inv @ poses @ self.T_flange_tcp_inv
        Q, valid = K.ik(flange)
        e, w, s = K.singularity_terms(Q)
        e, w, s = np.nan_to_num(e), np.nan_to_num(w), np.nan_to_num(s)
        n = len(poses)
        dq = np.abs(np.angle(np.exp(1j * (Q[1:] - Q[:-1])))).max(-1)      # (n-1, 8)
        jump_ok = np.nan_to_num(dq, nan=9) < MAX_JUMP
        is_first = np.zeros(n, bool)
        is_first[starts] = True
        pair_ok = np.ones((n, 8), bool)
        pair_ok[1:] = jump_ok | is_first[1:, None]
        ok_pose = valid & pair_ok
        chain_ok = np.logical_and.reduceat(ok_pose, starts, axis=0)          # (nchains, 8)
        sw = np.minimum.reduceat(w, starts, axis=0)
        se = np.minimum.reduceat(e, starts, axis=0)
        ss = np.minimum.reduceat(s, starts, axis=0)
        return Q, chain_ok, (sw, se, ss)

    def chain_reach(self, Q, start, chain, b):
        """Number of waypoints from the start that are reachable, continuous and collision-free."""
        cell = self.cell
        last_door = None
        cell.set_grip(chain.meta.get("gripper", "open"))
        cell.set_touch(chain.meta.get("touch", ()))
        for k in range(len(chain.poses)):
            q = Q[start + k, b]
            if np.isnan(q).any():
                return k
            if k > 0 and np.abs(np.angle(np.exp(1j * (q - Q[start + k - 1, b])))).max() > MAX_JUMP:
                return k
            if chain.door_deg[k] != last_door:
                cell.set_door(chain.door_deg[k])
                last_door = chain.door_deg[k]
            if cell.collides(q):
                return k
        return len(chain.poses)

    def evaluate(self, base_x, base_dist, rel_height, bed_z, kin_gate=0.3, max_branches=3):
        """Returns {objective: Result} with the best branch for each objective."""
        cfg, cell = self.cfg, self.cell
        cell.set_layout(base_x, base_dist, rel_height, bed_z)
        pk, pl, allc, poses, starts = self.chains(bed_z)
        Q, chain_ok, sing = self.kinematics(poses, starts, cell.base_pose)
        base = Result(base_x, base_dist, rel_height, bed_z, scenario=cfg.scenario)
        base.plate_fits_door = self.top or plate_fits_door(cfg, bed_z)
        n_pick = len(pk)
        ng, ntilt = cfg.task.pick_grid_n, len(cfg.task.pick_tilts_deg)
        pk_ok = chain_ok[:n_pick].reshape(ng * ng, ntilt, 4, 8)
        kin_any = pk_ok.any(axis=(1, 2)).mean(0)
        base.kin_pick_any = float(kin_any.max())
        other_ok = chain_ok[n_pick:].any(0) if len(chain_ok) > n_pick else np.zeros(8, bool)
        cand = [int(b) for b in np.argsort(-kin_any)[:max_branches]
                if kin_any[b] >= max(kin_gate, 0.8 * kin_any.max())]
        cand += [int(b) for b in np.where(other_ok)[0] if b not in cand]
        out = {}
        if not cand:
            for obj in self.objectives:
                r = Result(**asdict(base))
                r.objective = obj
                out[obj] = r
            return out
        per_branch = {b: self._metrics(base, Q, chain_ok, starts, sing, pk, pl, b,
                                       do_pick=kin_any[b] >= kin_gate) for b in cand}
        for obj in self.objectives:
            best = None
            for b, r in per_branch.items():
                s = objective_score(r, obj)
                if best is None or s > best[0]:
                    best = (s, b)
            r = Result(**asdict(per_branch[best[1]]))
            r.objective, r.score = obj, best[0]
            out[obj] = r
        return out

    def _metrics(self, base, Q, chain_ok, starts, sing, pk, pl, b, do_pick=True):
        cfg = self.cfg
        r = Result(**asdict(base))
        r.branch, r.branch_label = int(b), K.BRANCH_LABELS[b]
        ng, tilts = cfg.task.pick_grid_n, cfg.task.pick_tilts_deg
        ntilt = len(tilts)
        allc = pk + pl + self.door
        used = []
        pick_map = np.zeros((ng, ng), int)
        tilt_map = -np.ones((ng, ng))
        low = 0
        chosen = {"pick": {}, "pull": None, "door": None}

        def full(ci):
            return chain_ok[ci, b] and self.chain_reach(Q, starts[ci], allc[ci], b) == len(allc[ci].poses)

        if do_pick:
            for p in range(ng * ng):
                found = {}
                for d, yaws in (("A", (0, 2)), ("B", (1, 3))):   # fingers left-right / front-back
                    for t in range(ntilt):                      # prefer the most upright grasp
                        hit = next((((p * ntilt + t) * 4 + yw) for yw in yaws if full((p * ntilt + t) * 4 + yw)), None)
                        if hit is not None:
                            found[d] = (t, hit)
                            break
                iy, ix = divmod(p, ng)
                pick_map[iy, ix] = len(found)
                if found:
                    tmin = min(t for t, _ in found.values())
                    tilt_map[iy, ix] = tilts[tmin]
                    if tilts[tmin] <= 30:
                        low += 1
                    used += [ci for _, ci in found.values()]
                    chosen["pick"][p] = {d: int(ci) for d, (t, ci) in found.items()}
        npts = ng * ng
        r.pick_both = float((pick_map == 2).sum() / npts)
        r.pick_any = float((pick_map >= 1).sum() / npts)
        r.low_tilt = low / npts
        r.pick_map, r.tilt_map = pick_map.tolist(), tilt_map.tolist()

        if self.door:
            d0 = len(pk) + len(pl)
            nd = len(self.door[0].poses)
            best_reach, best_ci = 0, None
            # kinematically complete variants first, then partial ones
            order = sorted(range(len(self.door)), key=lambda k: (not chain_ok[d0 + k, b], k))
            for k in order:
                ci = d0 + k
                reached = self.chain_reach(Q, starts[ci], self.door[k], b)
                if reached > best_reach:
                    best_reach, best_ci = reached, ci
                if reached == nd:
                    break
            r.door = best_reach / nd
            if best_reach == nd:
                used.append(best_ci)
                chosen["door"] = best_ci
        if pl and r.plate_fits_door:
            best = 0.0
            for k in range(len(pl)):
                ci = len(pk) + k
                reached = self.chain_reach(Q, starts[ci], pl[k], b)
                if reached == len(pl[k].poses):
                    best, chosen["pull"] = 1.0, ci
                    used.append(ci)
                    break
                best = max(best, 0.8 * reached / len(pl[k].poses))
            r.pull = best

        sw, se, ss = sing
        if used:
            r.sing_wrist = float(sw[used, b].min())
            r.sing_elbow = float(se[used, b].min())
            r.sing_shoulder = float(ss[used, b].min())
            r.sing_norm = float(min(r.sing_wrist / 0.5, r.sing_elbow / 0.5, r.sing_shoulder / 0.15, 1.0))
        r.chosen = chosen
        return r

    def trajectory(self, r: Result):
        """Joint waypoints for playback of the chosen motions of a result."""
        cell = self.cell
        cell.set_layout(r.base_x, r.base_dist, r.rel_height, r.bed_z)
        pk, pl, allc, poses, starts = self.chains(r.bed_z)
        Q, _, _ = self.kinematics(poses, starts, cell.base_pose)
        b = r.branch
        segs = []
        if r.chosen.get("door") is not None:
            ci = r.chosen["door"]
            segs.append(("door", [(Q[starts[ci] + k, b], allc[ci].door_deg[k]) for k in range(len(allc[ci].poses))]))
        for p, dd in sorted(r.chosen.get("pick", {}).items(), key=lambda kv: int(kv[0])):
            for d, ci in dd.items():
                segs.append((f"pick p{p}{d}", [(Q[starts[ci] + k, b], allc[ci].door_deg[k]) for k in range(len(allc[ci].poses))]))
        if r.chosen.get("pull") is not None:
            ci = r.chosen["pull"]
            segs.append(("pull", [(Q[starts[ci] + k, b], allc[ci].door_deg[k]) for k in range(len(allc[ci].poses))]))
        return segs


def load_layout(path=os.path.join(HERE, "layout.json")):
    with open(path) as f:
        return json.load(f)


def describe(r: Result):
    lines = [f"[{r.objective}] score {r.score:.3f}   arm configuration: {r.branch_label}",
             f"  plate points: both finger directions {r.pick_both:.0%}, at least one {r.pick_any:.0%}, "
             f"within 30 deg of vertical {r.low_tilt:.0%}"]
    if r.scenario == "door":
        lines.append(f"  door swing achieved {r.door:.0%}   plate pull {r.pull:.0%} "
                     f"(plate inside door opening at this bed Z: {r.plate_fits_door})")
    lines.append(f"  singularity margins: wrist |sin q5| {r.sing_wrist:.2f}, elbow |sin q3| {r.sing_elbow:.2f}, "
                 f"shoulder {r.sing_shoulder * 1000:.0f} mm")
    lines.append("  plate map, printer back at top (2 both ways, 1 one way, 0 none) | smallest tilt used (deg):")
    for row, trow in zip(reversed(r.pick_map), reversed(r.tilt_map)):
        lines.append("    " + " ".join(str(v) for v in row) + "   |  " + " ".join(f"{int(t):>3}" if t >= 0 else "  -" for t in trow))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float)
    ap.add_argument("--dist", type=float)
    ap.add_argument("--rel", type=float)
    ap.add_argument("--bedz", type=float)
    ap.add_argument("--scenario", choices=["door", "top"])
    ap.add_argument("--layout", default="end")
    a = ap.parse_args()
    lay = load_layout() if os.path.exists(os.path.join(HERE, "layout.json")) else {}
    cfg = Config()
    cfg.scenario = a.scenario or lay.get("scenario", "door")
    x = a.x if a.x is not None else lay.get("base_x", 0.0)
    dist = a.dist if a.dist is not None else lay.get("base_dist", 0.45)
    rel = a.rel if a.rel is not None else lay.get("rel_height", 0.0)
    bz = a.bedz if a.bedz is not None else lay.get("bed_z", 250)
    ev = Evaluator(cfg, layout=a.layout)
    res = ev.evaluate(x, dist, rel, bz)
    print(f"placement ({cfg.scenario}): x={x:+.3f} m, {dist:.3f} m in front of the printer, "
          f"height {rel:+.2f} m vs printer, bed Z {bz:.0f} mm")
    for obj, r in res.items():
        print(describe(r))


if __name__ == "__main__":
    main()
