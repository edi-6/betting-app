"""The villagers and the iron golem as voxels (one per skin pixel, 1/16 of a block; own pixel art drawn here, in the
game's proportions): a villager in a robe with its arms folded and its big nose, in a few professions' colours; the
iron golem, nearly three blocks tall, cracked iron with vines on it.

Model.instances(pose) gives VOXEL_DTYPE instances for the renderer. pose: pos (feet), yaw (0 faces -y; it faces
(sin yaw, -cos yaw)), tumble (quaternion, swept away), head_yaw, head_pitch, arms (raised, radians), bob.
"""
import numpy as np

from creeper import VOXEL_DTYPE, _qmul
from mathutil import quat_rotate

PX = 1.0 / 16.0


def _q(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(ang / 2)), np.cos(ang / 2)])


def _mottle(base, h, w, rng, var=0.07, shade=1.0):
    img = np.ones((h, w, 3)) * np.array(base, float)
    img *= 1.0 + (rng.random((h, w, 1)) - 0.5) * 2 * var
    return np.clip(img * shade, 0, 255)


VILLAGER = {          # (x0, y0, z0, sx, sy, sz) in pixels; faces -y
    'robe': (-4, -3, 0, 8, 6, 20),
    'head': (-4, -4, 20, 8, 8, 10),
    'nose': (-1, -6, 21, 2, 2, 4),
    'arms': (-4, -7, 12, 8, 4, 4),
}
VILLAGER_HEAD = ('head', 'nose')
VILLAGER_NECK = np.array([0.0, 0.0, 20.0]) * PX

GOLEM = {
    'leg_l': (-7, -2.5, 0, 6, 5, 16),
    'leg_r': (1, -2.5, 0, 6, 5, 16),
    'waist': (-4.5, -3, 16, 9, 6, 5),
    'chest': (-9, -6, 21, 18, 12, 12),
    'head': (-4, -7, 33, 8, 8, 10),
    'nose': (-1, -9, 34, 2, 2, 4),
    'arm_l': (-13, -3, 3, 4, 6, 30),
    'arm_r': (9, -3, 3, 4, 6, 30),
}
GOLEM_HEAD = ('head', 'nose')
GOLEM_NECK = np.array([0.0, -1.0, 33.0]) * PX

PROFESSIONS = [  # robe, trim
    ((110, 82, 52), (150, 118, 70)),      # farmer: brown with a straw hat band
    ((236, 236, 230), (190, 160, 70)),    # librarian: white robe, gold trim
    ((128, 56, 136), (200, 170, 70)),     # cleric: purple
    ((60, 64, 70), (110, 110, 116)),      # smith: dark apron
    ((70, 120, 60), (150, 120, 60)),      # shepherd: green
    ((170, 120, 60), (210, 190, 140)),    # fisherman: tan
]
SKIN = (186, 132, 102)


def villager_skin(prof, seed):
    rng = np.random.default_rng(seed)
    robe, trim = PROFESSIONS[prof % len(PROFESSIONS)]
    s = {}
    for name, (x0, y0, z0, sx, sy, sz) in VILLAGER.items():
        if name in ('head', 'nose'):
            base = SKIN if name == 'head' else (170, 116, 88)
            f = lambda h, w, sh=1.0: _mottle(base, h, w, rng, 0.05, sh)
        elif name == 'arms':
            f = lambda h, w, sh=1.0: _mottle(robe, h, w, rng, 0.08, sh)
        else:
            f = lambda h, w, sh=1.0: _mottle(robe, h, w, rng, 0.08, sh)
        faces = {'ny': f(sz, sx), 'py': f(sz, sx, 0.95), 'nx': f(sz, sy, 0.97), 'px': f(sz, sy, 0.97),
                 'pz': f(sy, sx, 1.03), 'nz': f(sy, sx, 0.8)}
        if name == 'head':
            fr = faces['ny']                                  # row 0 = top
            fr[3, 1:7] = (64, 44, 30)                         # the unibrow
            fr[4, 1] = (236, 236, 236)
            fr[4, 2] = (40, 120, 50)
            fr[4, 5] = (40, 120, 50)
            fr[4, 6] = (236, 236, 236)
            fr[8, 3:5] = (120, 76, 58)                        # mouth under the nose
            for k in ('nx', 'px', 'py'):
                faces[k][:2] = (90, 66, 44)                   # hair line at the back and sides
            faces['pz'][:] = _mottle((96, 70, 46), sy, sx, rng, 0.06)
        if name == 'robe':
            for k in ('ny', 'py', 'nx', 'px'):
                faces[k][-3:] *= 0.75                         # hem
                faces[k][8] = trim                            # a belt
            faces['ny'][9:, 3:5] = np.array(trim) * 0.9       # trim down the front
        if name == 'arms':
            faces['ny'][:, 2:6] = _mottle(SKIN, sz, 4, rng, 0.05)   # the hands, clasped in the sleeves
        s[name] = faces
    return s


def golem_skin(seed=9):
    rng = np.random.default_rng(seed)
    iron = (214, 208, 196)
    s = {}
    for name, (x0, y0, z0, sx, sy, sz) in GOLEM.items():
        def f(h, w, sh=1.0):
            img = _mottle(iron, h, w, rng, 0.05, sh)
            cr = rng.random((h, w)) < 0.05                   # cracks and rust
            img[cr] = (120, 112, 104)
            ru = rng.random((h, w)) < 0.025
            img[ru] = (150, 104, 72)
            return img
        faces = {'ny': f(sz, sx), 'py': f(sz, sx, 0.95), 'nx': f(sz, sy, 0.97), 'px': f(sz, sy, 0.97),
                 'pz': f(sy, sx, 1.03), 'nz': f(sy, sx, 0.8)}
        if name in ('chest', 'leg_l', 'arm_r'):
            for k in ('ny', 'nx', 'px', 'py'):                # vines
                fc = faces[k]
                h, w = fc.shape[:2]
                c = int(rng.integers(0, w))
                for r in range(h):
                    if rng.random() < 0.85:
                        fc[r, c % w] = (62, 116, 44)
                    if rng.random() < 0.3:
                        c += int(rng.integers(-1, 2))
                    if rng.random() < 0.15:
                        fc[r, (c + 1) % w] = (90, 150, 60)
        if name == 'head':
            fr = faces['ny']
            fr[2, 1:7] = (90, 84, 78)                         # brow
            fr[3, 2] = (140, 20, 20)                          # dark red eyes
            fr[3, 5] = (140, 20, 20)
        s[name] = faces
    return s


class Model:
    def __init__(self, parts, skin, head_parts, neck):
        self.parts = {}
        self.head_parts = head_parts
        self.neck = neck
        for name, (x0, y0, z0, sx, sy, sz) in parts.items():
            sx, sy, sz = int(sx), int(sy), int(sz)
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
            pivot = np.array([x0 + sx / 2, y0 + sy / 2, z0 + sz]) * PX
            self.parts[name] = dict(pos=pos, cx=cx, cy=cy, cz=cz, vis=vis, pivot=pivot)
        zs = np.concatenate([p['pos'][:, 2] for p in self.parts.values()])
        self.height = float(zs.max())

    def instances(self, pose):
        yaw = pose.get('yaw', 0.0)
        qbody = _q((0, 0, 1), yaw)
        tumble = pose.get('tumble')
        centre = np.array([0.0, 0.0, self.height * 0.5])
        hy, hp = pose.get('head_yaw', 0.0), pose.get('head_pitch', 0.0)
        qh = _qmul(_q((0, 0, 1), hy), _q((1, 0, 0), -hp))
        arms = pose.get('arms', 0.0)
        bob = pose.get('bob', 0.0)
        out = []
        for name, p in self.parts.items():
            P = p['pos'].copy()
            q = np.array([0.0, 0.0, 0.0, 1.0])
            if name in self.head_parts:
                q = qh
                P = self.neck + quat_rotate(np.tile(q, (len(P), 1)), P - self.neck)
            elif name.startswith('arm') and arms:
                side = -1.0 if name.endswith('_l') else 1.0
                q = _qmul(_q((1, 0, 0), -arms), _q((0, 1, 0), side * arms * 0.25))
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            elif name == 'arms' and arms:
                q = _q((1, 0, 0), -arms)
                piv = np.array([0.0, -3.0, 16.0]) * PX
                P = piv + quat_rotate(np.tile(q, (len(P), 1)), P - piv)
            elif name.startswith('leg') and pose.get('walk', 0.0):
                ph = pose.get('phase', 0.0) + (0.0 if name.endswith('_l') else np.pi)
                a = pose['walk'] * np.sin(ph)
                q = _q((1, 0, 0), a)
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            P[:, 2] += bob
            if tumble is not None:
                qt = np.asarray(tumble, float)
                W = quat_rotate(np.tile(qt, (len(P), 1)), quat_rotate(np.tile(qbody, (len(P), 1)), P - centre)) + centre
                qq = _qmul(qt, _qmul(qbody, q))
            else:
                W = quat_rotate(np.tile(qbody, (len(P), 1)), P)
                qq = _qmul(qbody, q)
            W = W + np.asarray(pose['pos'], float)
            inst = np.zeros(len(P), VOXEL_DTYPE)
            inst['pos'] = W
            inst['quat'] = qq
            inst['scale'] = PX
            for key, col in (('cx', p['cx']), ('cy', p['cy']), ('cz', p['cz'])):
                inst[key][:, :3] = np.clip(col, 0, 255).astype(np.uint8)
            inst['inner'][:, :3] = (60, 60, 60)
            inst['cx'][:, 3] = 0b111111
            inst['cy'][:, 3] = p['vis']
            inst['cz'][:, 3] = 0
            out.append(inst)
        return np.concatenate(out)


def villager(prof, seed):
    return Model(VILLAGER, villager_skin(prof, seed), VILLAGER_HEAD, VILLAGER_NECK)


def golem():
    return Model(GOLEM, golem_skin(), GOLEM_HEAD, GOLEM_NECK)
