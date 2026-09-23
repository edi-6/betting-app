"""Validate every camera path: clearance above trees and terrain, no camera inside a tree canopy, and no camera
inside the flight corridor of a TNT formation (which would put the lens inside the flying blocks)."""
import numpy as np
import scene
import timeline as T


def check(verbose=True):
    wd = scene.world_data()
    trees = wd['trees']            # (x, y, base_h)
    H = wd['heightmap']
    half = H.shape[0] // 2
    rounds = {rd['key']: rd for rd in T.round_defs()}
    paths = []
    for rd in rounds.values():
        for si, s in enumerate(rd['shots']):
            paths.append((f"{rd['key']} shot{si}", s['keys'], s['start'], s['end'], rd))
    for hi, hs in enumerate(T.hook_shots()):
        paths.append((f"hook{hi}", hs['keys'], 0.0, hs['dur'], rounds[hs['round']]))
    bad = 0
    for name, keys, t0, t1, rd in paths:
        worst = 1e9
        problems = []
        p = rd['formation']['pos']
        lo, hi = p.min(0) - 0.6, p.max(0) + 0.6
        for t in np.linspace(t0, t1, 30):
            eye, tgt, fov = T.cam_path(keys, t)
            d = np.hypot(trees[:, 0] - eye[0], trees[:, 1] - eye[1])
            near = d < 6.0
            clear_z = eye[2] - (trees[near, 2] + 11.0)
            c = clear_z.min() if near.any() else 99.0
            ix, iy = int(np.floor(eye[0])) + half, int(np.floor(eye[1])) + half
            ground = H[ix, iy] if 0 <= ix < H.shape[0] and 0 <= iy < H.shape[0] else 0
            c = min(c, eye[2] - ground - 1.0)
            worst = min(worst, c)
            canopy = d < 4.6
            if canopy.any() and (eye[2] < trees[canopy, 2] + 11.5).any():
                problems.append('inside canopy')
            if lo[0] < eye[0] < hi[0] and lo[2] < eye[2] < hi[2] and eye[1] > lo[1]:
                problems.append('in TNT flight corridor')
        ok = worst > 0 and not problems
        bad += not ok
        if verbose:
            print(f"{'OK ' if ok else 'BAD'} {name:12s} min clearance {worst:6.2f} {sorted(set(problems))}")
    return bad


if __name__ == '__main__':
    print('problems:', check())
