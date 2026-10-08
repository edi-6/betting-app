"""A recorded round played back at any moment of its simulated time (between the 30 Hz frames): the water for the
shader, the village as it still stands, the loose blocks, the villagers and the iron golem, the spray."""
import os

import numpy as np

import blocks as BL
import characters as CH
import sim as SM
import village as VL
import world as WD

# professions of the villagers, in the order village.py places them (the golem is last)
PROFS = [5, 5, 0, 1, 4, 2, 0, 1]


def _nlerp(q0, q1, a):
    s = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -1.0, 1.0)
    q = q0 * (1 - a) + q1 * s * a
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)


class Playback:
    def __init__(self, n, vil):
        d = SM.round_dir(n)
        self.n = n
        self.cfg = SM.ROUNDS[n]
        self.vil = vil
        self.water = np.load(os.path.join(d, 'water.npy'), mmap_mode='r')
        self.debris = np.load(os.path.join(d, 'debris.npy'), mmap_mode='r')
        self.spray = np.load(os.path.join(d, 'spray.npy'), mmap_mode='r')
        m = np.load(os.path.join(d, 'meta.npz'))
        self.kinds = m['kinds'].astype(np.int64)
        self.agents = m['agents']
        ev = m['events'].reshape(-1, 2)
        self.ev_t = ev[:, 0]                       # in the order the blocks broke (= the debris order)
        self.ev_i = ev[:, 1].astype(np.int64)
        nx, ny, x0, y0, dx = m['grid']
        self.nx, self.ny = int(nx), int(ny)
        self.x0, self.y0, self.dx = float(x0), float(y0), float(dx)
        self.terrain = m['terrain']
        self.n_rec = self.water.shape[0]
        self.t_end = (self.n_rec - 1) / SM.REC_HZ
        depth_edge = max(0.0, -float(self.terrain[-1].mean()))
        A = self.cfg['A']
        self.wave = (A, 0.0, self.cfg['ramp'], float(np.sqrt(9.81 * (depth_edge + A))))
        self.wave2 = (self.cfg['hold'], self.cfg['fall'])
        self.variant = BL.debris_variant(self.kinds).astype(np.float32)
        self.scale = np.where(self.kinds == BL.KIND_ID['glass'], 0.55, 1.0).astype(np.float32)
        self._mesh = (None, None)
        self.models = [CH.villager(p, 100 + i) for i, p in enumerate(PROFS)] + [CH.golem()]

    def grid(self):
        return self.nx, self.ny, self.x0, self.y0, self.nx * self.dx, self.ny * self.dx

    def _frame(self, t):
        f = float(np.clip(t * SM.REC_HZ, 0, self.n_rec - 1))
        k0 = int(np.floor(f))
        k1 = min(k0 + 1, self.n_rec - 1)
        return k0, k1, f - k0

    # -- water -------------------------------------------------------------------------------------------------------
    def water_at(self, t, swirl=None):
        k0, k1, a = self._frame(t)
        w = self.water[k0].astype(np.float32)
        if a > 1e-4 and k1 != k0:
            w = w * (1 - a) + self.water[k1].astype(np.float32) * a
        out = {'state': w[..., :4], 'vel': w[..., 4:], 'sea': 0.0, 'wave': self.wave, 'wave2': self.wave2,
               'time': float(t)}
        if swirl is not None:
            out['swirl'] = swirl
        return out

    def sample(self, t, x, y):
        """Water surface height and depth at world points (nearest recorded frame)."""
        k0, k1, a = self._frame(t)
        k = k1 if a > 0.5 else k0
        i = np.clip(((np.asarray(x) - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip(((np.asarray(y) - self.y0) / self.dx).astype(int), 0, self.ny - 1)
        w = self.water[k]
        return w[j, i, 0].astype(np.float32), w[j, i, 1].astype(np.float32)

    # -- the village -------------------------------------------------------------------------------------------------
    def broken(self, t):
        return int(np.searchsorted(self.ev_t, t, side='right'))

    def village_mesh(self, t):
        n = self.broken(t)
        if self._mesh[0] == n:
            return self._mesh[1]
        K = self.vil.K.copy()
        if n:
            K.ravel()[self.ev_i[:n]] = 0
        mesh = WD.mesh_blocks(K, VL.ORIGIN, tint=self.vil.tint)
        self._mesh = (n, mesh)
        return mesh

    def debris_at(self, t):
        """Instances for the 'block' prop kind: pos3 quat4 scale variant fade flash."""
        n = self.broken(t)
        if n == 0:
            return np.zeros((0, 11), np.float32)
        k0, k1, a = self._frame(t)
        D0 = np.array(self.debris[k0, :n])
        D1 = np.array(self.debris[k1, :n])
        new0 = np.isnan(D0[:, 0])
        D0[new0] = D1[new0]                         # broke since the last frame: it starts where it is next
        new1 = np.isnan(D1[:, 0])
        D1[new1] = D0[new1]
        ok = ~np.isnan(D0[:, 0])
        out = np.zeros((n, 11), np.float32)
        out[:, :3] = D0[:, :3] * (1 - a) + D1[:, :3] * a
        out[:, 3:7] = _nlerp(D0[:, 3:7], D1[:, 3:7], a)
        out[:, 7] = self.scale[:n]
        out[:, 8] = self.variant[:n]
        out[:, 9] = 1.0
        return out[ok]

    # -- people ------------------------------------------------------------------------------------------------------
    def agents_at(self, t):
        k0, k1, a = self._frame(t)
        A0, A1 = self.agents[k0], self.agents[k1]
        pos = A0[:, :3] * (1 - a) + A1[:, :3] * a
        yaw = A0[:, 3] * (1 - a) + A1[:, 3] * a
        state = (A1 if a > 0.5 else A0)[:, 4].astype(int)
        t_in = (A1 if a > 0.5 else A0)[:, 5]
        q = _nlerp(A0[:, 6:10], A1[:, 6:10], a)
        return pos, yaw, state, t_in, q

    def people(self, t, overrides=None, hide=()):
        """Voxel instances of everyone. overrides: {index: pose dict entries} for the shots that direct them."""
        pos, yaw, state, t_in, q = self.agents_at(t)
        out = []
        for i, model in enumerate(self.models):
            if i in hide:
                continue
            golem = i == len(self.models) - 1
            ov = dict(overrides.get(i, {})) if overrides else {}
            st = ov.pop('state', state[i])
            pose = {'pos': pos[i].copy(), 'yaw': float(yaw[i]) + np.pi}
            if st == SM.Agents.IDLE:
                pose['head_yaw'] = 0.35 * np.sin(t * 0.7 + i * 1.7) * (0 if golem else 1)
                pose['head_pitch'] = 0.08 * np.sin(t * 0.5 + i)
                if golem:
                    pose['arms'] = 0.06 * np.sin(t * 1.1)
            elif st == SM.Agents.RUN:
                ph = t * 9.0 + i
                pose['bob'] = abs(np.sin(ph)) * 0.07
                pose['head_pitch'] = -0.1
                pose['walk'] = 0.6
                pose['phase'] = ph
            else:
                pose['tumble'] = q[i]
                pose['arms'] = 1.5 + 0.4 * np.sin(t * 7.0 + i)
                pose['head_pitch'] = 0.3
            pose.update(ov)
            out.append(model.instances(pose))
        if not out:
            return None
        return np.concatenate(out)

    # -- spray -------------------------------------------------------------------------------------------------------
    def spray_at(self, t, alpha=0.6, size=1.0):
        """Puffs for the renderer: pos3 size alpha fire variant age."""
        k0, k1, a = self._frame(t)
        S = np.asarray(self.spray[k1 if a > 0.5 else k0], np.float32)
        S = S[S[:, 4] > 0.01]
        out = np.zeros((len(S), 8), np.float32)
        out[:, :3] = S[:, :3]
        out[:, 3] = S[:, 3] * size
        out[:, 4] = S[:, 4] * alpha
        out[:, 6] = S[:, 5]
        out[:, 7] = t
        return out
