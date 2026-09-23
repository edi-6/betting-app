"""Voxel giant: every skin pixel is split into K x K x K voxels.

Each voxel stores the skin colour for its outward x/y/z faces (+ a bit mask telling which of its six faces
are on the character's skin) and an 'inner' colour (flesh or bone) used for faces exposed by damage.
"""
import numpy as np
from textures import make_skin

K = 4
VS = 1.0 / K
GX0, GY0, GZ0 = -8.0, -4.0, 0.0            # world position of grid corner
NX, NY, NZ = 16 * K, 8 * K, 32 * K

# name: (x0, y0, z0, sx, sy, sz) in skin pixels (world units); the giant faces -Y
PARTS = {
    'leg_r': (-4, -2, 0, 4, 4, 12),
    'leg_l': (0, -2, 0, 4, 4, 12),
    'body': (-4, -2, 12, 8, 4, 12),
    'arm_r': (-8, -2, 12, 4, 4, 12),
    'arm_l': (4, -2, 12, 4, 4, 12),
    'head': (-4, -4, 24, 8, 8, 8),
}
PART_IDS = {n: i + 1 for i, n in enumerate(PARTS)}

FLESH = np.array([[206, 24, 20], [192, 18, 16], [218, 34, 26], [176, 14, 14], [150, 10, 10],
                  [120, 6, 8], [226, 48, 36]], np.float64)
FLESH_P = np.array([0.22, 0.2, 0.14, 0.14, 0.12, 0.08, 0.10])
BONE = np.array([[236, 228, 204], [226, 216, 190], [244, 238, 218], [214, 204, 178]], np.float64)


class Giant:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        skin = make_skin()
        self.occ = np.zeros((NX, NY, NZ), bool)
        self.part = np.zeros((NX, NY, NZ), np.uint8)
        self.cx = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cy = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cz = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.inner = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.mask = np.zeros((NX, NY, NZ), np.uint8)
        self.bone = np.zeros((NX, NY, NZ), bool)
        for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
            i0, j0, k0 = int((x0 - GX0) * K), int((y0 - GY0) * K), int((z0 - GZ0) * K)
            ni, nj, nk = sx * K, sy * K, sz * K
            sl = (slice(i0, i0 + ni), slice(j0, j0 + nj), slice(k0, k0 + nk))
            self.occ[sl] = True
            self.part[sl] = PART_IDS[name]
            ii, jj, kk = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing='ij')
            pi, pj, pk = ii // K, jj // K, kk // K
            tex = skin[name]
            m = np.zeros((ni, nj, nk), np.uint8)
            cx = np.zeros((ni, nj, nk, 3))
            cy = np.zeros((ni, nj, nk, 3))
            cz = np.zeros((ni, nj, nk, 3))
            # +x face (viewer at +X: left = front)
            sel = ii == ni - 1
            cx[sel] = tex['px'][sz - 1 - pk[sel], pj[sel]]
            m[sel] |= 1 << 0
            sel = ii == 0
            cx[sel] = tex['nx'][sz - 1 - pk[sel], sy - 1 - pj[sel]]
            m[sel] |= 1 << 1
            sel = jj == nj - 1
            cy[sel] = tex['py'][sz - 1 - pk[sel], sx - 1 - pi[sel]]
            m[sel] |= 1 << 2
            sel = jj == 0
            cy[sel] = tex['ny'][sz - 1 - pk[sel], pi[sel]]
            m[sel] |= 1 << 3
            sel = kk == nk - 1
            cz[sel] = tex['pz'][sy - 1 - pj[sel], pi[sel]]
            m[sel] |= 1 << 4
            sel = kk == 0
            cz[sel] = tex['nz'][pj[sel], pi[sel]]
            m[sel] |= 1 << 5
            self.mask[sl] = m
            self.cx[sl + (slice(0, 3),)] = np.clip(cx, 0, 255).astype(np.uint8)
            self.cy[sl + (slice(0, 3),)] = np.clip(cy, 0, 255).astype(np.uint8)
            self.cz[sl + (slice(0, 3),)] = np.clip(cz, 0, 255).astype(np.uint8)
            # --- bones ---------------------------------------------------------------------------
            dsurf = np.minimum.reduce([ii, ni - 1 - ii, jj, nj - 1 - jj, kk, nk - 1 - kk])
            b = np.zeros((ni, nj, nk), bool)
            if name == 'head':
                b = np.zeros((ni, nj, nk), bool)
            elif name == 'body':
                b = (ii >= 14) & (ii <= 17) & (jj >= 9) & (jj <= 12)          # spine
            else:
                b = (ii >= 6) & (ii <= 9) & (jj >= 6) & (jj <= 9) & (kk >= 3) & (kk <= nk - 4)
                if name.startswith('arm'):
                    b &= kk >= 5
            self.bone[sl] = b
        # inner colours
        n = self.occ.sum()
        idx = np.nonzero(self.occ)
        choice = rng.choice(len(FLESH), size=n, p=FLESH_P)
        col = FLESH[choice] * (1 + (rng.random((n, 1)) - 0.5) * 0.12)
        bsel = self.bone[idx]
        bc = BONE[rng.integers(len(BONE), size=bsel.sum())] * (1 + (rng.random((bsel.sum(), 1)) - 0.5) * 0.06)
        col[bsel] = bc
        inner = np.zeros((n, 4), np.uint8)
        inner[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
        inner[:, 3] = np.where(bsel, 255, 0)
        self.inner[idx] = inner
        self.cx[..., 3] = self.mask       # mask travels in the alpha of the x colour
        self.total = int(n)
        self.initial_occ = self.occ.copy()

    # -- queries --------------------------------------------------------------------------------
    @staticmethod
    def world_to_index(p):
        p = np.asarray(p)
        i = np.floor((p[..., 0] - GX0) * K).astype(np.int64)
        j = np.floor((p[..., 1] - GY0) * K).astype(np.int64)
        k = np.floor((p[..., 2] - GZ0) * K).astype(np.int64)
        return i, j, k

    @staticmethod
    def index_to_world(i, j, k):
        return np.stack([GX0 + (i + 0.5) * VS, GY0 + (j + 0.5) * VS, GZ0 + (k + 0.5) * VS], -1)

    def occupied(self, i, j, k):
        inb = (i >= 0) & (i < NX) & (j >= 0) & (j < NY) & (k >= 0) & (k < NZ)
        out = np.zeros(i.shape, bool)
        out[inb] = self.occ[i[inb], j[inb], k[inb]]
        return out

    def face_visibility(self, i, j, k):
        """6-bit mask per voxel: bit f set when the neighbour across face f is empty.
        Face order: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z (bottom faces on the ground stay hidden)."""
        m = np.zeros(len(i), np.uint8)
        for bit, (di, dj, dk) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
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


VOXEL_DTYPE = np.dtype([('pos', 'f4', 3), ('quat', 'f4', 4), ('scale', 'f4'), ('cx', 'u1', 4),
                        ('cy', 'u1', 4), ('cz', 'u1', 4), ('inner', 'u1', 4)])
assert VOXEL_DTYPE.itemsize == 48


def pack_static(g, sel=None):
    """Instances for the voxels still attached to the giant (only those with an exposed face)."""
    ex = g.exposed() if sel is None else sel
    i, j, k = np.nonzero(ex)
    out = np.zeros(len(i), VOXEL_DTYPE)
    out['pos'] = Giant.index_to_world(i, j, k)
    out['quat'] = (0, 0, 0, 1)
    out['scale'] = VS
    out['cx'] = g.cx[i, j, k]
    out['cy'] = g.cy[i, j, k]
    out['cz'] = g.cz[i, j, k]
    out['inner'] = g.inner[i, j, k]
    out['cy'][:, 3] = g.face_visibility(i, j, k)
    return out


if __name__ == '__main__':
    import time
    t = time.time()
    g = Giant()
    print('voxels', g.total, 'exposed', g.exposed().sum(), 'bones', g.bone.sum(), 'time', time.time() - t)
