"""The creeper who saves the day: the creeper skin from the earlier videos (own pixel art), one voxel per skin pixel
(1/16 of a block, game size: 26 pixels tall), animated here: it walks in with its legs swinging in diagonal pairs,
stops, turns to look at the camera, and when its fuse is lit it swells in pulses and flashes white, faster and
faster, like in the game, until it goes off.
"""
import numpy as np

from mathutil import quat_rotate

PX = 1.0 / 16.0
# name: (x0, y0, z0, sx, sy, sz) in pixels; the creeper faces -Y
PARTS = {
    'leg_fl': (-4, -6, 0, 4, 4, 6),
    'leg_fr': (0, -6, 0, 4, 4, 6),
    'leg_bl': (-4, 2, 0, 4, 4, 6),
    'leg_br': (0, 2, 0, 4, 4, 6),
    'body': (-4, -2, 6, 8, 4, 12),
    'head': (-4, -4, 18, 8, 8, 8),
}
GREENS = np.array([(76, 168, 58), (66, 152, 50), (88, 184, 70), (58, 138, 44), (98, 192, 82), (50, 124, 40)], float)
LIGHT = np.array([(142, 208, 128), (172, 226, 160), (120, 196, 104)], float)
DARK = np.array([(34, 92, 28), (42, 104, 34)], float)
GREY = np.array([(150, 164, 140), (132, 146, 124)], float)
FACE = np.array([(16, 20, 14), (22, 28, 18), (10, 14, 10)], float)
FACE_ROWS = ["........",
             "........",
             ".EE..EE.",
             ".EE..EE.",
             "...EE...",
             "..EEEE..",
             "..EEEE..",
             "..E..E.."]
VOXEL_DTYPE = np.dtype([('pos', 'f4', 3), ('quat', 'f4', 4), ('scale', 'f4'), ('cx', 'u1', 4),
                        ('cy', 'u1', 4), ('cz', 'u1', 4), ('inner', 'u1', 4)])


def _mottled(h, w, rng, shade=1.0):
    r = rng.random((h, w))
    img = GREENS[rng.integers(len(GREENS), size=(h, w))].copy()
    sel = r < 0.09
    img[sel] = LIGHT[rng.integers(len(LIGHT), size=sel.sum())]
    sel = (r >= 0.09) & (r < 0.18)
    img[sel] = DARK[rng.integers(len(DARK), size=sel.sum())]
    sel = (r >= 0.18) & (r < 0.215)
    img[sel] = GREY[rng.integers(len(GREY), size=sel.sum())]
    img *= (1.0 + (rng.random((h, w, 1)) - 0.5) * 0.08) * shade
    return np.clip(img, 0, 255)


def make_skin(seed=5):
    """{part: {face: HxWx3}}: ny = front (-Y), py = back, nx/px = sides, pz = top, nz = bottom; row 0 = top."""
    rng = np.random.default_rng(seed)
    skin = {}
    for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
        faces = {'ny': _mottled(sz, sx, rng), 'py': _mottled(sz, sx, rng, 0.96),
                 'nx': _mottled(sz, sy, rng, 0.97), 'px': _mottled(sz, sy, rng, 0.97),
                 'pz': _mottled(sy, sx, rng, 1.03), 'nz': _mottled(sy, sx, rng, 0.72)}
        if name.startswith('leg'):
            for f in ('ny', 'py', 'nx', 'px'):
                faces[f][-1] *= 0.82
        if name == 'head':
            for r, row in enumerate(FACE_ROWS):
                for c, ch in enumerate(row):
                    if ch == 'E':
                        faces['ny'][r, c] = FACE[rng.integers(len(FACE))]
        skin[name] = faces
    return skin


class Creeper:
    def __init__(self, seed=5):
        skin = make_skin(seed)
        self.parts = {}
        for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
            ii, jj, kk = np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing='ij')
            surf = (ii == 0) | (ii == sx - 1) | (jj == 0) | (jj == sy - 1) | (kk == 0) | (kk == sz - 1)
            ii, jj, kk = ii[surf], jj[surf], kk[surf]
            n = len(ii)
            tex = skin[name]
            cx = np.zeros((n, 3))
            cy = np.zeros((n, 3))
            cz = np.zeros((n, 3))
            vis = np.zeros(n, np.uint8)
            sel = ii == sx - 1
            cx[sel] = tex['px'][sz - 1 - kk[sel], jj[sel]]
            vis[sel] |= 1
            sel = ii == 0
            cx[sel] = tex['nx'][sz - 1 - kk[sel], sy - 1 - jj[sel]]
            vis[sel] |= 2
            sel = jj == sy - 1
            cy[sel] = tex['py'][sz - 1 - kk[sel], sx - 1 - ii[sel]]
            vis[sel] |= 4
            sel = jj == 0
            cy[sel] = tex['ny'][sz - 1 - kk[sel], ii[sel]]
            vis[sel] |= 8
            sel = kk == sz - 1
            cz[sel] = tex['pz'][sy - 1 - jj[sel], ii[sel]]
            vis[sel] |= 16
            sel = kk == 0
            cz[sel] = tex['nz'][jj[sel], ii[sel]]
            vis[sel] |= 32
            # pixel centres (creeper frame, blocks): feet at z = 0, facing -Y
            pos = np.stack([x0 + ii + 0.5, y0 + jj + 0.5, z0 + kk + 0.5], -1) * PX
            self.parts[name] = dict(pos=pos, cx=cx, cy=cy, cz=cz, vis=vis,
                                    pivot=np.array([x0 + sx / 2, y0 + sy / 2, z0 + sz]) * PX)
        self.center = np.array([0.0, 0.0, 13.0 * PX])

    def instances(self, pos, yaw, walk_phase=0.0, walk_amp=0.0, head_yaw=0.0, swell=1.0, white=0.0):
        """pos: feet position (world); yaw: which way it faces (radians, 0 = -Y... see below);
        walk_phase: leg cycle (radians); walk_amp: leg swing (radians); swell: scale about its middle;
        white: 0..1 flash."""
        out = []
        # the creeper faces -Y in its own frame; yaw turns that towards world direction (sin(yaw), -cos(yaw))
        qy = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        bob = abs(np.sin(walk_phase)) * walk_amp * 0.02
        for name, p in self.parts.items():
            P = p['pos'].copy()
            q = np.array([0.0, 0.0, 0.0, 1.0])
            if name.startswith('leg'):
                ph = walk_phase + (0.0 if name in ('leg_fl', 'leg_br') else np.pi)
                a = walk_amp * np.sin(ph)
                q = np.array([np.sin(a / 2), 0.0, 0.0, np.cos(a / 2)])         # swing about x at the hip
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            elif name == 'head' and head_yaw:
                q = np.array([0.0, 0.0, np.sin(head_yaw / 2), np.cos(head_yaw / 2)])
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            P = self.center + (P - self.center) * swell
            P[:, 2] += bob
            W = quat_rotate(np.tile(qy, (len(P), 1)), P) + np.asarray(pos, float)
            qq = _qmul(qy, q)
            n = len(P)
            inst = np.zeros(n, VOXEL_DTYPE)
            inst['pos'] = W
            inst['quat'] = qq
            inst['scale'] = PX * swell
            for key, col in (('cx', p['cx']), ('cy', p['cy']), ('cz', p['cz'])):
                c = col * (1 - white) + 255.0 * white
                inst[key][:, :3] = np.clip(c, 0, 255).astype(np.uint8)
            inst['inner'][:, :3] = (60, 60, 60)
            inst['cx'][:, 3] = 0b111111
            inst['cy'][:, 3] = p['vis']
            inst['cz'][:, 3] = 0
            out.append(inst)
        return np.concatenate(out)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def fuse(t_lit, t, length=1.5):
    """Swell and white flash of a lit creeper, t seconds in: pulses that speed up, and it grows at the end."""
    a = np.clip((t - t_lit) / length, 0.0, 1.0)
    if t < t_lit:
        return 1.0, 0.0
    # flash frequency rises from 1.2 to 3 Hz (no faster: it fills a lot of the screen)
    ph = 2 * np.pi * (1.2 * (t - t_lit) + 0.9 * (t - t_lit) ** 2 / length)
    white = 0.58 * (0.5 + 0.5 * np.sin(ph)) ** 2 * min(1.0, a * 3)
    swell = 1.0 + 0.06 * (0.5 + 0.5 * np.sin(ph)) + 0.22 * a ** 3
    return swell, white
