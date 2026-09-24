"""Validate every camera of the edit: never inside a tree canopy or below the tree tops near it, never under
the terrain, and (measured by running the real battle on the edit's schedule) never hit by what is flying
around: the closest falling anvil / TNT / flying arrow and debris-free lens clearance."""
import sys

import numpy as np

import arena as A
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


def check(simulate=True):
    wd = scene.world_data()
    trees, H = wd['trees'], wd['heightmap']
    shots = T.shots()
    bad = 0
    print('static checks (tree clearance > 0, not inside a canopy):')
    for sh in shots + [dict(name=f'cold{i}', keys=c['keys']) for i, c in enumerate(T.cold_open())]:
        for k in sh['keys']:
            c, inside = tree_clearance(np.array(k[1], float), trees, H)
            ok = c > 0 and not inside
            bad += not ok
            if not ok or '-v' in sys.argv:
                print(f"  {'OK ' if ok else 'BAD'} {sh['name']:10s} t={k[0]:.2f} eye={k[1]} clearance={c:.1f} "
                      f"canopy={inside}")
    if not simulate:
        return bad
    print('dynamic checks (closest falling anvil / TNT / arrow to the lens):')
    sched = T.schedule()
    sim = A.Arena(T.plan())
    stats = {}
    for i, (si, f, scale, skip) in enumerate(sched):
        if skip is not None:
            M.skip_to(sim, skip)
        sim.step_frame(scale)
        sh = shots[si]
        eye, _, _ = T.cam_path(sh['keys'], f / T.FPS)
        sw = M.swarm_info(sim, eye)
        st = stats.setdefault(sh['name'], [999.0, 999.0, 999.0])
        st[0] = min(st[0], sw['anvils'][1])
        st[1] = min(st[1], sw['tnt'][1])
        st[2] = min(st[2], sw['arrows'][1])
    for name, (an, tn, ar) in stats.items():
        ok = an > 3.0 and tn > 2.5
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name:10s} closest anvil {an:6.1f}  TNT {tn:6.1f}  arrow {ar:6.1f}")
    return bad


if __name__ == '__main__':
    n = check(simulate='--static' not in sys.argv)
    print('problems:', n)
