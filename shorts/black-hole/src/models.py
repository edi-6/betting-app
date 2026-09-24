"""The four voxel giants of the arena (Steve, Creeper, Zombie, Warden), built by one generic class.

Every skin pixel of a character is split into K x K x K voxels of size VS = pxu / K. Each voxel stores the
skin colour for its outward x/y/z faces (+ a bit mask telling which of its six faces are on the skin) and an
'inner' colour (flesh, bone, gunpowder, glowing souls) used for faces exposed by damage. Glowing skin voxels
carry 255 in the alpha of their z colour; inner alpha is 255 for bone/powder, 128 for glowing specks.

The characters and their skins are the ones from the earlier videos; here each giant is placed at a world
offset (all facing -Y), so the physics can address them through instance methods.
"""
import numpy as np

import creeper as CR
import steve as ST
import warden as WD
import zombie as ZB

VOXEL_DTYPE = np.dtype([('pos', 'f4', 3), ('quat', 'f4', 4), ('scale', 'f4'), ('cx', 'u1', 4),
                        ('cy', 'u1', 4), ('cz', 'u1', 4), ('inner', 'u1', 4)])
assert VOXEL_DTYPE.itemsize == 48

FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


# ---------------------------------------------------------------------------------------------
# per-character rules: bones / cores and inner colours
# ---------------------------------------------------------------------------------------------
def _limb_bones(name, ii, jj, kk, ni, nj, nk, K):
    c0, c1 = ni // 2 - K // 2 - 1, ni // 2 + K // 2
    d0, d1 = nj // 2 - K // 2 - 1, nj // 2 + K // 2
    return (ii >= c0) & (ii <= c1) & (jj >= d0) & (jj <= d1) & (kk >= 3) & (kk <= nk - 4)


def steve_bones(name, ii, jj, kk, ni, nj, nk, K, rng):
    if name == 'head':
        return np.zeros(ii.shape, bool)
    if name == 'body':
        return (ii >= 14) & (ii <= 17) & (jj >= 9) & (jj <= 12)
    b = (ii >= 6) & (ii <= 9) & (jj >= 6) & (jj <= 9) & (kk >= 3) & (kk <= nk - 4)
    if name.startswith('arm'):
        b &= kk >= 5
    return b


def zombie_bones(name, ii, jj, kk, ni, nj, nk, K, rng):
    if name == 'head':
        return np.zeros(ii.shape, bool)
    if name == 'body':
        return (ii >= 14) & (ii <= 17) & (jj >= 9) & (jj <= 12)
    if name.startswith('arm'):
        return (ii >= 6) & (ii <= 9) & (kk >= 6) & (kk <= 9) & (jj >= 3) & (jj <= nj - 4)
    return (ii >= 6) & (ii <= 9) & (jj >= 6) & (jj <= 9) & (kk >= 3) & (kk <= nk - 4)


def creeper_core(name, ii, jj, kk, ni, nj, nk, K, rng):
    """The creeper is full of gunpowder: everything deeper than ~1.25 units under the skin of head and body."""
    if name not in ('head', 'body'):
        return np.zeros(ii.shape, bool)
    dsurf = np.minimum.reduce([ii, ni - 1 - ii, jj, nj - 1 - jj, kk, nk - 1 - kk])
    depth = 5 + rng.integers(-1, 2, size=dsurf.shape)
    return dsurf >= depth


def warden_bones(name, ii, jj, kk, ni, nj, nk, K, rng):
    c3 = lambda n: (n // 2 - 2, n // 2 + 1)
    if name == 'body':
        a0, a1 = c3(ni)
        spine = (ii >= a0) & (ii <= a1) & (jj >= nj - 9) & (jj <= nj - 6)
        ribs = (jj >= 3) & (jj <= 5) & (kk >= nk - 36) & (kk <= nk - 7) & ((kk // 3) % 3 == 0) \
            & (ii >= 6) & (ii <= ni - 7)
        return spine | ribs
    if name.startswith('arm') or name.startswith('leg'):
        a0, a1 = c3(ni)
        c0, c1 = c3(nj)
        return (ii >= a0) & (ii <= a1) & (jj >= c0) & (jj <= c1) & (kk >= 3) & (kk <= nk - 4)
    return np.zeros(ii.shape, bool)


def red_inner(flesh, flesh_p, bone):
    def f(n, bsel, rng):
        col = flesh[rng.choice(len(flesh), size=n, p=flesh_p)] * (1 + (rng.random((n, 1)) - 0.5) * 0.12)
        nb = int(bsel.sum())
        col[bsel] = bone[rng.integers(len(bone), size=nb)] * (1 + (rng.random((nb, 1)) - 0.5) * 0.06)
        return col, np.where(bsel, 255, 0)
    return f


# muted interiors (the series' earlier videos used bright red; this one is watched by a broad audience):
# Steve is raw-beef pink-brown, the Zombie rotten olive, the Creeper dark leafy green around its gunpowder
STEVE_FLESH = np.array([[178, 98, 82], [160, 86, 72], [192, 112, 94], [146, 78, 66], [132, 70, 60],
                        [200, 124, 104], [154, 90, 76]], float)
ZOMBIE_FLESH = np.array([[92, 96, 56], [78, 84, 48], [104, 108, 64], [66, 72, 42], [58, 62, 38],
                         [112, 112, 70], [84, 80, 52]], float)
CREEPER_FLESH = np.array([[58, 104, 42], [48, 90, 36], [70, 118, 50], [40, 78, 30], [34, 66, 26],
                          [80, 126, 56], [52, 96, 40]], float)


def creeper_inner(n, bsel, rng):
    col = CREEPER_FLESH[rng.choice(len(CREEPER_FLESH), size=n, p=CR.FLESH_P)] * (1 + (rng.random((n, 1)) - 0.5) * 0.12)
    nb = int(bsel.sum())
    col[bsel] = CR.POWDER[rng.choice(len(CR.POWDER), size=nb, p=CR.POWDER_P)] * \
        (1 + (rng.random((nb, 1)) - 0.5) * 0.08)
    return col, np.where(bsel, 255, 0)


def warden_inner(n, bsel, rng):
    col = WD.FLESH[rng.choice(len(WD.FLESH), size=n, p=WD.FLESH_P)] * (1 + (rng.random((n, 1)) - 0.5) * 0.14)
    soul = rng.random(n) < 0.06
    col[soul] = WD.SOUL[rng.integers(len(WD.SOUL), size=int(soul.sum()))]
    nb = int(bsel.sum())
    col[bsel] = WD.BONE[rng.integers(len(WD.BONE), size=nb)] * (1 + (rng.random((nb, 1)) - 0.5) * 0.06)
    return col, np.where(bsel, 255, np.where(soul, 128, 0))


SPECS = {
    'steve': dict(label='STEVE', parts=ST.PARTS, pxu=1.0, K=4, skin=ST.make_skin, bones=steve_bones,
                  inner=red_inner(STEVE_FLESH, ST.FLESH_P, ST.BONE), shape=None),
    'creeper': dict(label='CREEPER', parts=CR.PARTS, pxu=1.0, K=4, skin=CR.make_creeper_skin, bones=creeper_core,
                    inner=creeper_inner, shape=None),
    'zombie': dict(label='ZOMBIE', parts=ZB.PARTS, pxu=1.0, K=4, skin=ZB.make_zombie_skin, bones=zombie_bones,
                   inner=red_inner(ZOMBIE_FLESH, ZB.FLESH_P, ZB.BONE), shape=None),
    'warden': dict(label='WARDEN', parts=WD.PARTS, pxu=WD.PXU, K=WD.K, skin=WD.make_warden_skin,
                   bones=warden_bones, inner=warden_inner, shape=WD._part_shape),
}


def face_portrait(kind):
    """The front of the character's head as an (h, w, 3) uint8 pixel image (for the scoreboard)."""
    skin = SPECS[kind]['skin']()
    f = skin['head']['ny']
    img = f[0] if isinstance(f, tuple) else f
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
class Giant:
    def __init__(self, kind, offset, seed=1):
        spec = SPECS[kind]
        self.kind = kind
        self.label = spec['label']
        self.offset = np.asarray(offset, np.float64)
        K = spec['K']
        pxu = spec['pxu']
        self.K = K
        self.VS = pxu / K
        boxes = np.array(list(spec['parts'].values()), np.int64)
        g0 = boxes[:, :3].min(0)
        g1 = (boxes[:, :3] + boxes[:, 3:]).max(0)
        self.g0p = g0
        self.NX, self.NY, self.NZ = (int(v) for v in (g1 - g0) * K)
        self.origin = self.offset + g0 * pxu                  # world position of the grid corner
        self.lo = self.origin.copy()
        self.hi = self.origin + np.array([self.NX, self.NY, self.NZ]) * self.VS
        self.height = float(self.hi[2])
        rng = np.random.default_rng(seed)
        skin = spec['skin']()
        NX, NY, NZ = self.NX, self.NY, self.NZ
        self.occ = np.zeros((NX, NY, NZ), bool)
        self.part = np.zeros((NX, NY, NZ), np.uint8)
        self.cx = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cy = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cz = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.inner = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.mask = np.zeros((NX, NY, NZ), np.uint8)
        self.bone = np.zeros((NX, NY, NZ), bool)
        self.glow = np.zeros((NX, NY, NZ), bool)
        self.part_names = list(spec['parts'])
        for pid, (name, (x0, y0, z0, sx, sy, sz)) in enumerate(spec['parts'].items(), start=1):
            i0, j0, k0 = (np.array([x0, y0, z0]) - g0) * K
            ni, nj, nk = sx * K, sy * K, sz * K
            sl = (slice(i0, i0 + ni), slice(j0, j0 + nj), slice(k0, k0 + nk))
            shape = spec['shape'](name, ni, nj, nk) if spec['shape'] else np.ones((ni, nj, nk), bool)
            self.occ[sl] |= shape
            self.part[sl] = np.where(shape, pid, self.part[sl])
            ii, jj, kk = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing='ij')
            pi, pj, pk = ii // K, jj // K, kk // K
            tex = skin[name]
            m = np.zeros((ni, nj, nk), np.uint8)
            cx = np.zeros((ni, nj, nk, 3))
            cy = np.zeros((ni, nj, nk, 3))
            cz = np.zeros((ni, nj, nk, 3))
            gl = np.zeros((ni, nj, nk), bool)

            def put(sel, arr, face, r, c, bit):
                f = tex[face]
                img, g = (f if isinstance(f, tuple) else (f, None))
                arr[sel] = img[r, c]
                if g is not None:
                    gl[sel] |= g[r, c]
                m[sel] |= 1 << bit
            pad = np.pad(shape, 1)
            out_px = ~pad[2:, 1:-1, 1:-1] & shape
            out_nx = ~pad[:-2, 1:-1, 1:-1] & shape
            out_py = ~pad[1:-1, 2:, 1:-1] & shape
            out_ny = ~pad[1:-1, :-2, 1:-1] & shape
            out_pz = ~pad[1:-1, 1:-1, 2:] & shape
            out_nz = ~pad[1:-1, 1:-1, :-2] & shape
            put(out_px, cx, 'px', sz - 1 - pk[out_px], pj[out_px], 0)
            put(out_nx, cx, 'nx', sz - 1 - pk[out_nx], sy - 1 - pj[out_nx], 1)
            put(out_py, cy, 'py', sz - 1 - pk[out_py], sx - 1 - pi[out_py], 2)
            put(out_ny, cy, 'ny', sz - 1 - pk[out_ny], pi[out_ny], 3)
            put(out_pz, cz, 'pz', sy - 1 - pj[out_pz], pi[out_pz], 4)
            put(out_nz, cz, 'nz', pj[out_nz], pi[out_nz], 5)
            self.mask[sl] = np.where(shape, m, self.mask[sl])
            for arr, dst in ((cx, self.cx), (cy, self.cy), (cz, self.cz)):
                cur = dst[sl + (slice(0, 3),)]
                dst[sl + (slice(0, 3),)] = np.where(shape[..., None], np.clip(arr, 0, 255).astype(np.uint8), cur)
            self.glow[sl] |= gl & shape
            self.bone[sl] |= spec['bones'](name, ii, jj, kk, ni, nj, nk, K, rng) & shape
        n = int(self.occ.sum())
        idx = np.nonzero(self.occ)
        col, alpha = spec['inner'](n, self.bone[idx], rng)
        inner = np.zeros((n, 4), np.uint8)
        inner[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
        inner[:, 3] = alpha
        self.inner[idx] = inner
        self.cx[..., 3] = self.mask
        self.cz[..., 3] = np.where(self.glow, 255, 0)
        self.total = n
        self.center = (self.lo + self.hi) * 0.5
        self.alive = True

    # -- queries --------------------------------------------------------------------------------
    def world_to_index(self, p):
        p = np.asarray(p, np.float64)
        q = (p - self.origin) / self.VS
        return (np.floor(q[..., 0]).astype(np.int64), np.floor(q[..., 1]).astype(np.int64),
                np.floor(q[..., 2]).astype(np.int64))

    def index_to_world(self, i, j, k):
        return np.stack([self.origin[0] + (i + 0.5) * self.VS, self.origin[1] + (j + 0.5) * self.VS,
                         self.origin[2] + (k + 0.5) * self.VS], -1)

    def inb(self, i, j, k):
        return (i >= 0) & (i < self.NX) & (j >= 0) & (j < self.NY) & (k >= 0) & (k < self.NZ)

    def occupied(self, i, j, k):
        inb = self.inb(i, j, k)
        out = np.zeros(np.shape(i), bool)
        out[inb] = self.occ[i[inb], j[inb], k[inb]]
        return out

    def occupied_at(self, p):
        i, j, k = self.world_to_index(p)
        return self.occupied(i, j, k)

    def near(self, p, pad=0.0):
        """Points within the giant's bounding box (grown by pad)."""
        p = np.asarray(p)
        return np.all((p > self.lo - pad) & (p < self.hi + pad), axis=-1)

    def face_visibility(self, i, j, k):
        m = np.zeros(len(i), np.uint8)
        for bit, (di, dj, dk) in enumerate(FACES):
            ni, nj, nk = i + di, j + dj, k + dk
            empty = ~self.occupied(ni, nj, nk)
            if bit == 5:
                empty &= nk >= 0
            m |= (empty.astype(np.uint8) << bit)
        return m

    def exposed(self):
        o = self.occ
        e = np.zeros_like(o)
        e[1:] |= ~o[:-1]
        e[:-1] |= ~o[1:]
        e[:, 1:] |= ~o[:, :-1]
        e[:, :-1] |= ~o[:, 1:]
        e[:, :, 1:] |= ~o[:, :, :-1]
        e[:, :, :-1] |= ~o[:, :, 1:]
        e[0] = True
        e[-1] = True
        e[:, 0] = True
        e[:, -1] = True
        e[:, :, -1] = True
        return o & e

    def colours(self, i, j, k):
        col = np.zeros(len(i), VOXEL_DTYPE)
        col['cx'] = self.cx[i, j, k]
        col['cy'] = self.cy[i, j, k]
        col['cz'] = self.cz[i, j, k]
        col['inner'] = self.inner[i, j, k]
        return col

    def pack_static(self):
        """Instances for the voxels still attached to the giant (only those with an exposed face)."""
        i, j, k = np.nonzero(self.exposed())
        out = self.colours(i, j, k)
        out['pos'] = self.index_to_world(i, j, k)
        out['quat'] = (0, 0, 0, 1)
        out['scale'] = self.VS
        out['cy'][:, 3] = self.face_visibility(i, j, k)
        return out


# the arena line-up (left to right as seen from the front): zombie, creeper, steve, warden
LINEUP = [('zombie', (-33.0, 0.0, 0.0)), ('creeper', (-16.0, 0.0, 0.0)), ('steve', (1.0, 0.0, 0.0)),
          ('warden', (23.0, 0.0, 0.0))]


def make_giants(seed=1):
    return [Giant(kind, off, seed + n) for n, (kind, off) in enumerate(LINEUP)]


if __name__ == '__main__':
    import time
    t = time.time()
    gs = make_giants()
    for g in gs:
        print(g.label, 'voxels', g.total, 'exposed', int(g.exposed().sum()), 'grid', (g.NX, g.NY, g.NZ),
              'lo', np.round(g.lo, 1), 'hi', np.round(g.hi, 1))
    print(f'{time.time() - t:.1f}s')
