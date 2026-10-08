"""The goat that saves the day: a Minecraft-style goat, one voxel per pixel of its skin (own pixel art, drawn here),
about a block and a half to the tips of its horns. It trots in, stops, looks up at the machine, looks at you, backs
up, lowers its head, charges and rams the pedestal, bounces off and shakes its head.

script(t) gives its pose at sim time t for the ram at t_ram; Goat.instances(pose) the voxels.
"""
import numpy as np

from creeper import VOXEL_DTYPE, _qmul
from mathutil import quat_rotate

PX = 1.0 / 16.0
# name: (x0, y0, z0, sx, sy, sz) in pixels; the goat faces -Y
PARTS = {
    'leg_fl': (-4.0, -6.0, 0, 3, 3, 8),
    'leg_fr': (1.0, -6.0, 0, 3, 3, 8),
    'leg_bl': (-4.0, 4.0, 0, 3, 3, 8),
    'leg_br': (1.0, 4.0, 0, 3, 3, 8),
    'body': (-4.5, -7.0, 7, 9, 15, 10),
    'tail': (-1.0, 8.0, 13, 2, 2, 3),
    'neck': (-2.5, -10.0, 13, 5, 4, 6),
    'head': (-2.5, -16.0, 15, 5, 7, 6),
    'beard': (-1.0, -15.0, 11, 2, 2, 4),
    'ear_l': (-5.5, -12.0, 19, 3, 2, 1),
    'ear_r': (2.5, -12.0, 19, 3, 2, 1),
    'horn_l': (-2.5, -11.0, 21, 2, 2, 6),
    'horn_r': (0.5, -11.0, 21, 2, 2, 6),
}
HEAD_PARTS = ('neck', 'head', 'beard', 'ear_l', 'ear_r', 'horn_l', 'horn_r')
NECK = np.array([0.0, -8.5, 15.0]) * PX           # where the head pitches about
FUR = np.array([(234, 230, 216), (226, 221, 206), (240, 238, 228), (216, 210, 194)], float)
HORN = np.array([(196, 186, 168), (210, 202, 186), (182, 172, 154)], float)


def _fur(h, w, rng, shade=1.0):
    img = FUR[rng.integers(len(FUR), size=(h, w))].copy()
    # long shaggy strands: darker streaks running down
    for c in range(w):
        if rng.random() < 0.35:
            r0 = int(rng.integers(0, max(1, h - 2)))
            img[r0:r0 + int(rng.integers(2, 4)), c] *= 0.93
    return np.clip(img * shade, 0, 255)


def make_skin(seed=12):
    rng = np.random.default_rng(seed)
    skin = {}
    for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
        if name.startswith('horn'):
            f = lambda h, w, s=1.0: np.clip(HORN[rng.integers(len(HORN), size=(h, w))] * s, 0, 255)
        else:
            f = lambda h, w, s=1.0: _fur(h, w, rng, s)
        faces = {'ny': f(sz, sx), 'py': f(sz, sx, 0.96), 'nx': f(sz, sy, 0.97), 'px': f(sz, sy, 0.97),
                 'pz': f(sy, sx, 1.03), 'nz': f(sy, sx, 0.78)}
        if name.startswith('leg'):
            for k in ('ny', 'py', 'nx', 'px'):
                faces[k][-2:] = (84, 74, 64)                # hooves
            faces['nz'][:] = (70, 62, 54)
        if name == 'head':
            fr = faces['ny']                                # the face: a grey-pink nose, the mouth
            fr[3, 1:4] = (186, 168, 160)
            fr[4, 1:4] = (170, 150, 144)
            fr[5, 2] = (120, 100, 96)
            for k, c0 in (('nx', 1), ('px', 4)):            # eyes on the sides: yellow with a dark bar
                side = faces[k]
                side[1, c0] = (226, 186, 70)
                side[1, c0 + 1 if k == 'nx' else c0 - 1] = (34, 28, 24)
        if name.startswith('ear'):
            faces['nz'][:] = (214, 168, 160)
        if name.startswith('horn'):
            faces['pz'][:] = (222, 216, 204)
        skin[name] = faces
    return skin


class Goat:
    def __init__(self, seed=12):
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
            self.parts[name] = dict(pos=pos, cx=cx, cy=cy, cz=cz, vis=vis,
                                    pivot=np.array([x0 + sx / 2, y0 + sy / 2, z0 + sz]) * PX)

    def instances(self, pose):
        """pose: dict(pos (feet, world), yaw (0 faces -Y; it faces (sin yaw, -cos yaw)), phase, amp (legs),
        head_pitch (+ = head down), head_yaw, head_roll, squash (along its length, at the impact), lean)."""
        out = []
        yaw = pose['yaw']
        qy = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        lean = pose.get('lean', 0.0)
        ql = np.array([np.sin(lean / 2), 0.0, 0.0, np.cos(lean / 2)])        # pitch of the whole body
        qbody = _qmul(qy, ql)
        ph, amp = pose.get('phase', 0.0), pose.get('amp', 0.0)
        bob = abs(np.sin(ph)) * amp * 0.05
        hp, hy, hr = pose.get('head_pitch', 0.0), pose.get('head_yaw', 0.0), pose.get('head_roll', 0.0)
        qh = _qmul(_qmul(np.array([0.0, 0.0, np.sin(hy / 2), np.cos(hy / 2)]),
                         np.array([np.sin(-hp / 2), 0.0, 0.0, np.cos(-hp / 2)])),
                   np.array([0.0, np.sin(hr / 2), 0.0, np.cos(hr / 2)]))
        sq = pose.get('squash', 1.0)
        for name, p in self.parts.items():
            P = p['pos'].copy()
            q = np.array([0.0, 0.0, 0.0, 1.0])
            if name.startswith('leg'):
                lp = ph + (0.0 if name in ('leg_fl', 'leg_br') else np.pi)
                a = amp * np.sin(lp)
                q = np.array([np.sin(a / 2), 0.0, 0.0, np.cos(a / 2)])
                P = p['pivot'] + quat_rotate(np.tile(q, (len(P), 1)), P - p['pivot'])
            elif name in HEAD_PARTS:
                q = qh
                P = NECK + quat_rotate(np.tile(q, (len(P), 1)), P - NECK)
            P[:, 1] *= sq
            P[:, 0] *= 1.0 + (1.0 - sq) * 0.5
            P[:, 2] += bob
            W = quat_rotate(np.tile(qbody, (len(P), 1)), P) + np.asarray(pose['pos'], float)
            qq = _qmul(qbody, q)
            n = len(P)
            inst = np.zeros(n, VOXEL_DTYPE)
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


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


class Script:
    """The goat's moves, timed to the ram at t_ram (sim seconds). Positions are world blocks; the pedestal's front
    face is at y = face_y."""

    def __init__(self, t_ram, face_y, x=6.5, cam=None):
        self.t_ram = t_ram
        self.face_y = face_y
        self.x = x
        self.cam = cam
        a = t_ram
        self.k = {'enter': a - 4.1, 'stop': a - 2.5, 'look_up': a - 2.4, 'look_cam': a - 1.75, 'turn': a - 1.15,
                  'back': a - 0.95, 'down': a - 0.62, 'charge': a - 0.42, 'hit': a, 'recoil': a + 0.32,
                  'shake': a + 0.6, 'calm': a + 1.6}
        self.stand = np.array([x, face_y - 6.3, 0.0])          # where it stops
        self.start = np.array([x + 16.0, face_y - 7.6, 0.0])   # where it comes in from
        self.hit_at = np.array([x, face_y - 1.15, 0.0])        # its feet when its head meets the stone

    def visible(self, t):
        return t >= self.k['enter']

    def pose(self, t):
        k = self.k
        p = dict(pos=self.start.copy(), yaw=-np.pi / 2, phase=0.0, amp=0.0, head_pitch=0.0, head_yaw=0.0,
                 head_roll=0.0, squash=1.0, lean=0.0)
        trot = 7.5                                              # leg cycle (radians / block)
        if t < k['stop']:
            u = _ss(k['enter'], k['stop'], t)
            p['pos'] = _lerp(self.start, self.stand, u)
            dist = np.linalg.norm(self.start - self.stand) * u
            p['phase'] = dist * trot / 2.0
            p['amp'] = 0.75 * (1.0 - _ss(k['stop'] - 0.2, k['stop'], t))
            p['yaw'] = np.arctan2(-(self.stand - self.start)[0], (self.stand - self.start)[1]) + np.pi
            return p
        p['pos'] = self.stand.copy()
        # facing the way it came in, then turning to face the machine (+y: yaw = pi)
        yaw_in = np.arctan2(-(self.stand - self.start)[0], (self.stand - self.start)[1]) + np.pi
        u = _ss(k['turn'], k['back'], t)
        p['yaw'] = yaw_in * (1 - u) + np.pi * u
        # head: up at the machine, then a look at you (and a bleat), then back to the front
        up = _ss(k['look_up'], k['look_up'] + 0.3, t) * (1 - _ss(k['look_cam'] - 0.1, k['look_cam'] + 0.15, t))
        p['head_pitch'] = -0.55 * up
        cam = _ss(k['look_cam'], k['look_cam'] + 0.2, t) * (1 - _ss(k['turn'] - 0.05, k['turn'] + 0.2, t))
        p['head_yaw'] = 0.9 * cam
        p['head_roll'] = 0.12 * cam
        if t >= k['back']:
            # backs up a step, head down
            u = _ss(k['back'], k['down'], t)
            p['pos'] = self.stand + np.array([0.0, -1.0, 0.0]) * u
            p['phase'] = -u * 2.5
            p['amp'] = 0.35 * np.sin(np.pi * u)
            p['head_pitch'] = 0.55 * _ss(k['back'] + 0.1, k['charge'], t)
            p['lean'] = 0.08 * _ss(k['down'], k['charge'], t)
        if t >= k['charge']:
            # the charge: accelerating flat out at the pedestal
            u = np.clip((t - k['charge']) / (k['hit'] - k['charge']), 0.0, 1.0)
            back = self.stand + np.array([0.0, -1.0, 0.0])
            p['pos'] = _lerp(back, self.hit_at, u * u)
            p['phase'] = 3.0 + u * u * 26.0
            p['amp'] = 1.0
            p['head_pitch'] = 0.62
            p['lean'] = 0.1
        if t >= k['hit']:
            # the impact: squashed for a moment, then bouncing back off, legs splayed
            u = np.clip((t - k['hit']) / (k['recoil'] - k['hit']), 0.0, 1.0)
            p['squash'] = 1.0 - 0.18 * np.sin(np.pi * min(u * 3.0, 1.0)) * (1 - u)
            bounce = np.sin(np.pi * u) * 0.55
            p['pos'] = _lerp(self.hit_at, self.hit_at + np.array([0.0, -1.9, 0.0]), _ss(0.0, 1.0, u)) + \
                np.array([0.0, 0.0, bounce])
            p['phase'] = 0.0
            p['amp'] = 0.0
            p['head_pitch'] = 0.62 * (1 - u) - 0.15 * u
            p['lean'] = -0.15 * np.sin(np.pi * u)
        if t >= k['recoil']:
            p['pos'] = self.hit_at + np.array([0.0, -1.9, 0.0])
            # dizzy: shakes its head
            v = t - k['shake']
            s = np.sin(v * 24.0) * np.exp(-max(v, 0.0) * 2.2) if v > 0 else 0.0
            p['head_yaw'] = 0.35 * s
            p['head_roll'] = 0.25 * s
            p['head_pitch'] = -0.15
            p['lean'] = 0.0
        return p

    def charging(self, t):
        return self.k['charge'] <= t < self.k['hit']
