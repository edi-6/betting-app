"""Validate every camera: clearance above trees and terrain, never inside a tree canopy, and never in the path
of the swarm (closest flying arrow to the lens, measured by running the real simulation with the real
slow-motion schedule; arrow-cam shots are exempt since riding with the swarm is their point)."""
import numpy as np

import main as M
import scene
import sim as S
import timeline as T

SWARM_MIN = 2.0          # closest a flying arrow may pass a fixed camera


def tree_clearance(eye, trees, H):
    half = H.shape[0] // 2
    d = np.hypot(trees[:, 0] - eye[0], trees[:, 1] - eye[1])
    near = d < 6.0
    c = (eye[2] - (trees[near, 2] + 11.0)).min() if near.any() else 99.0
    ix, iy = int(np.floor(eye[0])) + half, int(np.floor(eye[1])) + half
    ground = H[ix, iy] if 0 <= ix < H.shape[0] and 0 <= iy < H.shape[0] else 0
    c = min(c, eye[2] - ground - 1.0)
    canopy = d < 4.6
    inside = bool(canopy.any() and (eye[2] < trees[canopy, 2] + 11.5).any())
    return c, inside


def swarm_dist(sim, eye):
    f = sim.state == S.FLY
    if not f.any():
        return 999.0
    return float(np.linalg.norm(sim.p[f] - eye, axis=1).min())


def check(verbose=True):
    wd = scene.world_data()
    trees, H = wd['trees'], wd['heightmap']
    bad = 0
    rounds = T.round_defs()
    rmap = {rd['key']: rd for rd in rounds}
    for rd in rounds:
        sim = M.make_sim(rd)
        n = T.sec(rd['duration'])
        scales = T.frame_scales(rd.get('scale'), n)
        stats = {}
        for f in range(n):
            sim.step_frame(scales[f])
            t = f / T.FPS
            si = max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)
            shot = rd['shots'][si]
            if 'follow' in shot:
                continue
            eye, _, _ = T.cam_path(shot['keys'], t)
            c, inside = tree_clearance(eye, trees, H)
            st = stats.setdefault(si, [1e9, False, 1e9])
            st[0] = min(st[0], c)
            st[1] |= inside
            st[2] = min(st[2], swarm_dist(sim, eye))
        for si, (c, inside, sw) in sorted(stats.items()):
            ok = c > 0 and not inside and sw > SWARM_MIN
            bad += not ok
            if verbose:
                print(f"{'OK ' if ok else 'BAD'} {rd['key']} shot{si}: clearance {c:6.2f}  canopy {inside}  "
                      f"closest arrow {sw:6.2f}")
    for hi, hs in enumerate(T.hook_shots()):
        if 'follow' in hs:
            continue
        rd = rmap[hs['round']]
        sim = S.RoundSim(rd['formation'], **rd['sim'])
        for _ in range(T.sec(hs['sim_start'])):
            sim.step_frame(1.0)
        nf = T.sec(hs['dur'])
        scales = T.frame_scales(hs.get('scale'), nf)
        worst, inside, sw = 1e9, False, 1e9
        for f in range(nf):
            sim.step_frame(scales[f])
            eye, _, _ = T.cam_path(hs['keys'], f / T.FPS)
            c, ins = tree_clearance(eye, trees, H)
            worst, inside, sw = min(worst, c), inside or ins, min(sw, swarm_dist(sim, eye))
        ok = worst > 0 and not inside and sw > SWARM_MIN
        bad += not ok
        if verbose:
            print(f"{'OK ' if ok else 'BAD'} hook{hi}: clearance {worst:6.2f}  canopy {inside}  closest arrow {sw:6.2f}")
    return bad


if __name__ == '__main__':
    print('problems:', check())
