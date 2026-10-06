"""Steve, lying on his back in the grass, staring at the sky: the Steve skin from the earlier videos (own pixel art),
one voxel per skin pixel (1/16 of a block, game size: 32 pixels tall). Lying down, his head is to the north and his
face is up; he's placed so that the pupil of his right eye (as you look at him) is exactly at the world's origin:
that's where the zoom is centred, and at the end the whole world turns out to be in it.
"""
import numpy as np

from creeper import VOXEL_DTYPE, _qmul
from mathutil import quat_rotate

PX = 1.0 / 16.0
# name: (x0, y0, z0, sx, sy, sz) in pixels; standing, he faces -Y, feet at z = 0
PARTS = {
    'leg_l': (-4, -2, 0, 4, 4, 12),
    'leg_r': (0, -2, 0, 4, 4, 12),
    'body': (-4, -2, 12, 8, 4, 12),
    'arm_l': (-8, -2, 12, 4, 4, 12),
    'arm_r': (4, -2, 12, 4, 4, 12),
    'head': (-4, -4, 24, 8, 8, 8),
}
HAIR = [(64, 42, 26), (72, 48, 30), (56, 36, 22), (80, 54, 33)]
SKIN = [(200, 146, 104), (194, 140, 99), (206, 152, 110), (189, 136, 96)]
PAL = {'H': HAIR, 'S': SKIN, 's': [(176, 124, 86)], 'N': [(150, 96, 62)], 'M': [(110, 66, 46)],
       'W': [(250, 250, 250)], 'P': [(78, 60, 140)], 'Q': [(66, 96, 101)],
       'T': [(22, 178, 182), (16, 168, 173), (30, 188, 190)],
       't': [(10, 140, 148)], 'J': [(58, 56, 160), (52, 50, 150), (64, 62, 170)], 'O': [(112, 112, 112), (98, 98, 98)]}
FACE = ["HHHHHHHH",
        "HHHHHHHH",
        "HSSSSSSH",
        "SSSSSSSS",
        "SWPSSQWS",                   # Q: the pupil with the world in it (its colour is the world's, seen from afar)
        "SSsNNsSS",
        "SSMMMMSS",
        "SSSSSSSS"]
PUPIL = (4, 5)                    # (row, column) of the face pixel the zoom is centred on
SIDE = ["HHHHHHHH", "HHHHHHHH", "HHHHHHSS", "HHHSSSSS", "HHSSSSSS", "HSSSSSSS", "SSSSSSSS", "SSSSSSSS"]
BACK = ["HHHHHHHH"] * 6 + ["SHHHHHHS", "SSSSSSSS"]
LIFT = 3 * PX                     # lying down, his back is this far above the grass


def _paint(rows, rng, shade=1.0):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            opts = PAL[ch]
            img[r, c] = opts[rng.integers(len(opts))]
    img *= (1.0 + (rng.random((h, w, 1)) - 0.5) * 0.05) * shade
    return np.clip(img, 0, 255)


def make_skin(seed=8):
    rng = np.random.default_rng(seed)
    skin = {}
    for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
        if name == 'head':
            f = {'ny': _paint(FACE, rng), 'py': _paint(BACK, rng, 0.95), 'nx': _paint(SIDE, rng, 0.97),
                 'px': _paint([r[::-1] for r in SIDE], rng, 0.97), 'pz': _paint(["H" * 8] * 8, rng, 1.02),
                 'nz': _paint(["S" * 8] * 8, rng, 0.75)}
        elif name == 'body':
            rows = ["T" * 8] * 11 + ["t" * 8]
            f = {k: _paint(rows if k in ('ny', 'py') else (["T" * 4] * 11 + ["t" * 4] if k in ('nx', 'px')
                                                           else ["T" * 8] * 4), rng, sh)
                 for k, sh in (('ny', 1.0), ('py', 0.95), ('nx', 0.97), ('px', 0.97), ('pz', 1.0), ('nz', 0.8))}
        elif name.startswith('arm'):
            rows = ["TTTT"] * 4 + ["SSSS"] * 8
            f = {k: _paint(rows if k not in ('pz', 'nz') else (["TTTT"] * 4 if k == 'pz' else ["SSSS"] * 4), rng, sh)
                 for k, sh in (('ny', 1.0), ('py', 0.95), ('nx', 0.97), ('px', 0.97), ('pz', 1.0), ('nz', 0.85))}
        else:
            rows = ["JJJJ"] * 10 + ["OOOO"] * 2
            f = {k: _paint(rows if k not in ('pz', 'nz') else (["JJJJ"] * 4 if k == 'pz' else ["OOOO"] * 4), rng, sh)
                 for k, sh in (('ny', 1.0), ('py', 0.95), ('nx', 0.97), ('px', 0.97), ('pz', 1.0), ('nz', 0.8))}
        skin[name] = f
    return skin


def _qx(a):
    return np.array([np.sin(a / 2), 0.0, 0.0, np.cos(a / 2)])


def _qz(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


class Steve:
    def __init__(self, seed=8):
        skin = make_skin(seed)
        self.parts = {}
        for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
            ii, jj, kk = np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing='ij')
            surf = (ii == 0) | (ii == sx - 1) | (jj == 0) | (jj == sy - 1) | (kk == 0) | (kk == sz - 1)
            ii, jj, kk = ii[surf], jj[surf], kk[surf]
            n = len(ii)
            tex = skin[name]
            cx, cy, cz = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
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
            pos = np.stack([x0 + ii + 0.5, y0 + jj + 0.5, z0 + kk + 0.5], -1) * PX
            self.parts[name] = dict(pos=pos, cx=cx, cy=cy, cz=cz, vis=vis)
        # lying on his back: turn -90 degrees about x (his front, -Y, comes to face +Z; his head points +Y)
        self.q_lie = _qx(-np.pi / 2)
        # where the pupil's face ends up, so the whole body can be moved to put it on the origin
        r, c = PUPIL
        x0, y0, z0, sx, sy, sz = PARTS['head']
        p = np.array([x0 + c + 0.5, y0, z0 + sz - 1 - r + 0.5]) * PX        # on the face (y = y0: the front)
        pl = quat_rotate(self.q_lie[None], p[None])[0]
        self.offset = np.array([-pl[0], -pl[1], 0.0])
        self.z_back = 2 * PX + LIFT       # his back (y = +2 px standing) becomes z = -2 px: lift it onto the grass
        self.pupil = np.array([0.0, 0.0, pl[2] + self.z_back])

    def instances(self, ground_z=0.0, breathe=0.0):
        """Voxel instances of Steve lying in the grass with his pupil over the origin; ground_z: the grass top."""
        out = []
        q = self.q_lie
        for name, p in self.parts.items():
            P = p['pos']
            if name == 'body' and breathe:
                P = P.copy()
                P[:, 1] *= 1.0 + 0.02 * breathe        # his chest rises a little
            W = quat_rotate(np.tile(q, (len(P), 1)), P) + self.offset + np.array([0.0, 0.0, ground_z + self.z_back])
            n = len(P)
            inst = np.zeros(n, VOXEL_DTYPE)
            inst['pos'] = W
            inst['quat'] = q
            inst['scale'] = PX
            for key, col in (('cx', p['cx']), ('cy', p['cy']), ('cz', p['cz'])):
                inst[key][:, :3] = np.clip(col, 0, 255).astype(np.uint8)
            inst['inner'][:, :3] = (40, 30, 24)
            inst['cx'][:, 3] = 0b111111
            inst['cy'][:, 3] = p['vis']
            inst['cz'][:, 3] = 0
            out.append(inst)
        return np.concatenate(out)

    def pupil_square(self, ground_z=0.0):
        """The pupil's top face: (centre xyz, half size) in world units."""
        return self.pupil + np.array([0.0, 0.0, ground_z]), PX / 2


if __name__ == '__main__':
    s = Steve()
    v = s.instances()
    print('voxels', len(v), 'pupil', np.round(s.pupil, 4), 'bbox', np.round(v['pos'].min(0), 3), np.round(v['pos'].max(0), 3))
