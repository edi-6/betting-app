"""Herobrine: the Steve skin from the earlier videos (own pixel art) with blank white eyes, one voxel per skin pixel
(1/16 of a block, game size: 32 pixels tall). He stands still and stares; his head can turn and tilt to follow the
camera. His eyes are the game's glowing material, so they light up in the dark.
"""
import numpy as np

from creeper import VOXEL_DTYPE, _qmul
from mathutil import quat_rotate

PX = 1.0 / 16.0
# name: (x0, y0, z0, sx, sy, sz) in pixels; he faces -Y, feet at z = 0
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
SKIN_DK = [(168, 116, 80)]
NOSE = [(150, 96, 62)]
MOUTH = [(96, 56, 38)]
BEARD = [(122, 76, 50), (114, 70, 46)]
EYE = [(255, 255, 255)]
SHIRT = [(22, 178, 182), (16, 168, 173), (30, 188, 190), (12, 158, 165)]
SHIRT_DK = [(10, 140, 148), (6, 132, 140)]
PANTS = [(58, 56, 160), (52, 50, 150), (64, 62, 170), (48, 46, 140)]
SHOES = [(112, 112, 112), (98, 98, 98)]
PAL = {'H': HAIR, 'S': SKIN, 's': SKIN_DK, 'N': NOSE, 'M': MOUTH, 'B': BEARD, 'E': EYE, 'T': SHIRT,
       't': SHIRT_DK, 'P': PANTS, 'O': SHOES}
FACE = ["HHHHHHHH",
        "HHHHHHHH",
        "HSSSSSSH",
        "SEESSEES",           # Steve's eyes, but blank white
        "SSsNNsSS",
        "SBMMMMBS",
        "SSSSSSSS",
        "SSSSSSSS"]
SIDE = ["HHHHHHHH", "HHHHHHHH", "HHHHHHSS", "HHHSSSSS", "HHSSSSSS", "HSSSSSSS", "SSSSSSSS", "SSSSSSSS"]
BACK = ["HHHHHHHH"] * 6 + ["SHHHHHHS", "SSSSSSSS"]


def _paint(rows, rng, shade=1.0):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            opts = PAL[ch]
            img[r, c] = opts[rng.integers(len(opts))]
    img *= (1.0 + (rng.random((h, w, 1)) - 0.5) * 0.06) * shade
    return np.clip(img, 0, 255)


def make_skin(seed=8):
    """{part: {face: HxWx3}}: ny = front (-Y), py = back, nx/px = sides, pz = top, nz = bottom; row 0 = top. Also
    {part: {face: HxW bool}} of the glowing pixels."""
    rng = np.random.default_rng(seed)
    skin, glow = {}, {}
    for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
        g = {}
        if name == 'head':
            f = {'ny': _paint(FACE, rng), 'py': _paint(BACK, rng, 0.95), 'nx': _paint(SIDE, rng, 0.97),
                 'px': _paint([r[::-1] for r in SIDE], rng, 0.97), 'pz': _paint(["H" * 8] * 8, rng, 1.02),
                 'nz': _paint(["S" * 8] * 8, rng, 0.75)}
            g['ny'] = np.array([[ch == 'E' for ch in row] for row in FACE])
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
            rows = ["PPPP"] * 10 + ["OOOO"] * 2
            f = {k: _paint(rows if k not in ('pz', 'nz') else (["PPPP"] * 4 if k == 'pz' else ["OOOO"] * 4), rng, sh)
                 for k, sh in (('ny', 1.0), ('py', 0.95), ('nx', 0.97), ('px', 0.97), ('pz', 1.0), ('nz', 0.8))}
        skin[name] = f
        glow[name] = g
    return skin, glow


class Herobrine:
    def __init__(self, seed=8):
        skin, glow = make_skin(seed)
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
            hot = np.zeros(n, bool)
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
            if 'ny' in glow[name]:
                hot[sel] = glow[name]['ny'][sz - 1 - kk[sel], ii[sel]]
            sel = kk == sz - 1
            cz[sel] = tex['pz'][sy - 1 - jj[sel], ii[sel]]
            vis[sel] |= 16
            sel = kk == 0
            cz[sel] = tex['nz'][jj[sel], ii[sel]]
            vis[sel] |= 32
            pos = np.stack([x0 + ii + 0.5, y0 + jj + 0.5, z0 + kk + 0.5], -1) * PX
            pivot = {'head': (0, 0, 24), 'arm_l': (-6, 0, 22), 'arm_r': (6, 0, 22), 'leg_l': (-2, 0, 12),
                     'leg_r': (2, 0, 12)}.get(name, (0, 0, 12))
            self.parts[name] = dict(pos=pos, cx=cx, cy=cy, cz=cz, vis=vis, hot=hot, pivot=np.array(pivot) * PX)

    def instances(self, pos, yaw, head_yaw=0.0, head_pitch=0.0, arm_swing=0.0, eyes_on=True):
        """pos: feet (world); yaw: he faces world direction (sin(yaw), -cos(yaw)), i.e. yaw 0 faces -Y (south);
        head_yaw / head_pitch: the head turns (radians, + = to his left / up); arm_swing: arms forwards (radians)."""
        out = []
        qy = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        for name, p in self.parts.items():
            P = p['pos'].copy()
            q = np.array([0.0, 0.0, 0.0, 1.0])
            if name == 'head' and (head_yaw or head_pitch):
                q = _qmul(np.array([0.0, 0.0, np.sin(head_yaw / 2), np.cos(head_yaw / 2)]),
                          np.array([np.sin(head_pitch / 2), 0.0, 0.0, np.cos(head_pitch / 2)]))
            elif name.startswith('arm') and arm_swing:
                a = arm_swing
                q = np.array([np.sin(a / 2), 0.0, 0.0, np.cos(a / 2)])
            if q[3] < 1.0:
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            W = quat_rotate(np.tile(qy, (len(P), 1)), P) + np.asarray(pos, float)
            n = len(P)
            inst = np.zeros(n, VOXEL_DTYPE)
            inst['pos'] = W
            inst['quat'] = _qmul(qy, q)
            inst['scale'] = PX
            for key, col in (('cx', p['cx']), ('cy', p['cy']), ('cz', p['cz'])):
                inst[key][:, :3] = np.clip(col, 0, 255).astype(np.uint8)
            inst['inner'][:, :3] = (40, 30, 24)
            inst['cx'][:, 3] = 0b111111
            inst['cy'][:, 3] = p['vis']
            inst['cz'][:, 3] = np.where(p['hot'] & eyes_on, 255, 0)
            out.append(inst)
        return np.concatenate(out)
