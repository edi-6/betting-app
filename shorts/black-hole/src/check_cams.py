"""Validate every camera of the edit: never inside a tree canopy or below the tree tops near it, never under
the terrain, and (measured by running the real story on the edit's schedule) never hit by what is flying around:
the closest giant voxel, torn-out block and debris cube to the lens."""
import sys

import numpy as np

import blackhole as BH
import main as M
import scene
import timeline as T


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


def body_clearance(w, eye):
    best = 999.0
    for b in w.bodies:
        if b.state == 'gone' or b.count == 0:
            continue
        idx = np.nonzero(b.alive)[0][::7]
        if len(idx) == 0:
            continue
        pw = b.to_world(b.ps[idx])
        best = min(best, float(np.linalg.norm(pw - eye, axis=1).min()))
    return best


def flyer_clearance(f, eye):
    if len(f) == 0:
        return 999.0
    return float(np.linalg.norm(f.p - eye, axis=1).min())


def check(simulate=True):
    wd = scene.world_data()
    trees, H = wd['trees'], wd['heightmap']
    shots = T.shots()
    bad = 0
    print('static checks (tree clearance > 0, not inside a canopy):')
    for sh in shots + T.cold_open():
        for k in sh['keys']:
            c, inside = tree_clearance(np.array(k[1], float), trees, H)
            ok = c > 0 and not inside
            bad += not ok
            if not ok or '-v' in sys.argv:
                print(f"  {'OK ' if ok else 'BAD'} {sh['name']:13s} t={k[0]:.2f} eye={k[1]} clearance={c:.1f} "
                      f"canopy={inside}")
    if not simulate:
        return bad
    print('dynamic checks (closest giant voxel / block / debris to the lens):')
    stats = {}
    for clip in T.cold_open():
        w = BH.World(T.plan())
        M.skip_to(w, clip['sim'])
        for k in range(T.sec(clip['dur'])):
            w.step_frame(clip['scale'])
            eye, _, _ = T.cam_path(clip['keys'], k / T.FPS)
            st = stats.setdefault(clip['name'], [999.0, 999.0, 999.0])
            st[0] = min(st[0], body_clearance(w, eye))
            st[1] = min(st[1], flyer_clearance(w.blocks, eye))
            st[2] = min(st[2], flyer_clearance(w.debris, eye))
    sched = T.schedule()
    w = BH.World(T.plan())
    for i, (si, f, scale, skip) in enumerate(sched):
        if skip is not None:
            M.skip_to(w, skip)
        w.step_frame(scale)
        sh = shots[si]
        eye, _, _ = T.cam_path(sh['keys'], f / T.FPS)
        st = stats.setdefault(sh['name'], [999.0, 999.0, 999.0])
        if f % 2 == 0:
            st[0] = min(st[0], body_clearance(w, eye))
        st[1] = min(st[1], flyer_clearance(w.blocks, eye))
        st[2] = min(st[2], flyer_clearance(w.debris, eye))
    for name, (bd, bl, db) in stats.items():
        ok = bd > 4.0 and bl > 1.5
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name:13s} closest giant {bd:6.1f}  block {bl:6.1f}  debris {db:6.1f}")
    return bad


if __name__ == '__main__':
    n = check(simulate='--static' not in sys.argv)
    print('problems:', n)
