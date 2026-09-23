"""Validate every camera path against tree canopies and terrain (so no shot starts inside a tree)."""
import numpy as np
import scene, timeline as T


def check(verbose=True):
    wd = scene.world_data()
    trees = wd['trees']            # (x, y, base_h)
    H = wd['heightmap']
    half = H.shape[0] // 2
    paths = []
    for rd in T.round_defs():
        for si, s in enumerate(rd['shots']):
            paths.append((f"{rd['key']} shot{si}", s['keys'], s['start'], s['end']))
    for hi, hs in enumerate(T.hook_shots()):
        paths.append((f"hook{hi}", hs['keys'], 0.0, hs['dur']))
    bad = 0
    for name, keys, t0, t1 in paths:
        worst = 1e9
        for t in np.linspace(t0, t1, 25):
            eye, tgt, fov = T.cam_path(keys, t)
            d = np.hypot(trees[:, 0] - eye[0], trees[:, 1] - eye[1])
            near = d < 6.0
            # canopy tops reach base + ~10.5
            clear_z = eye[2] - (trees[near, 2] + 11.0)
            c = clear_z.min() if near.any() else 99.0
            ix, iy = int(np.floor(eye[0])) + half, int(np.floor(eye[1])) + half
            ground = H[ix, iy] if 0 <= ix < H.shape[0] and 0 <= iy < H.shape[0] else 0
            c = min(c, eye[2] - ground - 1.0)
            worst = min(worst, c)
        flag = 'OK ' if worst > 0 else 'BAD'
        bad += worst <= 0
        if verbose:
            print(f'{flag} {name:12s} min clearance {worst:6.2f}')
    return bad


if __name__ == '__main__':
    print('problems:', check())
