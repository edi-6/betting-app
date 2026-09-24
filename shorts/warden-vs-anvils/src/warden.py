"""Voxel giant Warden: every skin pixel is split into K x K x K voxels.

Own pixel art in the style of the Deep Dark's Warden (skin pixels, 50 tall): long arms hanging to the knees, a
ribcage chest full of glowing souls, two glowing tendrils on the sides of the head. Faces -Y. One skin pixel
is PXU world units, so he stands 30 blocks tall.

Each voxel stores the skin colour for its outward x/y/z faces (+ a bit mask telling which of its six faces are
on the skin) and an 'inner' colour (dark sculk-like flesh with glowing soul specks, pale bone) used for faces
exposed by damage. Glowing voxels carry a flag in the alpha of their z colour (skin glow) or 128 in the alpha
of the inner colour (a glowing speck inside); bone is 255 there.
"""
import numpy as np

PXU = 0.6                     # world units per skin pixel
K = 3                         # voxels per skin pixel (per axis)
VS = PXU / K                  # voxel size in world units
GX0P, GY0P, GZ0P = -17, -7, 0  # grid corner in skin pixels
GX0, GY0, GZ0 = GX0P * PXU, GY0P * PXU, GZ0P * PXU
NX, NY, NZ = 34 * K, 12 * K, 50 * K
HEIGHT = 50 * PXU

# name: (x0, y0, z0, sx, sy, sz) in skin pixels; he faces -Y
PARTS = {
    'leg_r': (-7, -3, 0, 6, 6, 13),
    'leg_l': (1, -3, 0, 6, 6, 13),
    'body': (-9, -5, 13, 18, 10, 21),
    'arm_r': (-16, -4, 8, 7, 7, 26),        # long arms hanging from the shoulders to the knees
    'arm_l': (9, -4, 8, 7, 7, 26),
    'head': (-8, -7, 34, 16, 10, 16),
    'tendril_r': (-17, -3, 41, 9, 2, 9),
    'tendril_l': (8, -3, 41, 9, 2, 9),
}
PART_IDS = {n: i + 1 for i, n in enumerate(PARTS)}


def part_box_world(name):
    x0, y0, z0, sx, sy, sz = PARTS[name]
    return np.array([x0, y0, z0]) * PXU, np.array([x0 + sx, y0 + sy, z0 + sz]) * PXU


FLESH = np.array([[16, 44, 52], [12, 36, 44], [20, 54, 62], [10, 30, 38], [24, 62, 70], [14, 40, 50]], np.float64)
FLESH_P = np.array([0.24, 0.22, 0.16, 0.16, 0.1, 0.12])
SOUL = np.array([[70, 236, 244], [56, 220, 232], [92, 246, 250]], np.float64)
BONE = np.array([[168, 184, 180], [156, 172, 168], [178, 192, 188]], np.float64)

# ---------------------------------------------------------------------------------------------
# skin (own pixel art in the style of the Warden)
# ---------------------------------------------------------------------------------------------
PAL = {
    'B': [(19, 42, 52), (17, 38, 47), (21, 46, 57)],                    # near-black teal base
    'b': [(27, 62, 74), (25, 58, 69)],                                  # dark teal
    't': [(36, 92, 104), (33, 85, 96)],                                 # teal patches
    'l': [(58, 134, 142), (54, 126, 134)],                              # light teal highlight
    'r': [(150, 170, 168), (140, 160, 158), (160, 178, 176)],           # ribs
    'D': [(6, 14, 18), (5, 12, 16)],                                    # deep cavities
    'g': [(62, 232, 242), (72, 242, 250), (52, 218, 230)],              # glowing souls
    'T': [(62, 76, 76), (54, 68, 68)],                                  # teeth
    'h': [(8, 20, 26), (9, 22, 28)],                                    # claws / soles
}
GLOW_CHARS = {'g'}


def _paint(rows, rng, jitter=0.05):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    glow = np.zeros((h, w), bool)
    for r, row in enumerate(rows):
        assert len(row) == w, (row, w)
        for c, ch in enumerate(row):
            opts = PAL[ch]
            img[r, c] = opts[rng.integers(len(opts))]
            glow[r, c] = ch in GLOW_CHARS
    img *= 1.0 + (rng.random((h, w, 1)) - 0.5) * 2 * jitter
    return np.clip(img, 0, 255), glow


def _speckle(w, h, rng, spots=None, base='B'):
    """Mottled hide: a smooth random field thresholded into dark-teal areas and teal patches with a few
    light flecks (coherent blotches, not per-pixel noise)."""
    coarse = rng.random((h // 3 + 3, w // 3 + 3))
    fine = rng.random((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = yy / 3.0, xx / 3.0
    y0, x0 = cy.astype(int), cx.astype(int)
    fy, fx = cy - y0, cx - x0
    field = (coarse[y0, x0] * (1 - fx) * (1 - fy) + coarse[y0, x0 + 1] * fx * (1 - fy)
             + coarse[y0 + 1, x0] * (1 - fx) * fy + coarse[y0 + 1, x0 + 1] * fx * fy)
    field = 0.75 * field + 0.25 * fine
    rows = []
    for y in range(h):
        row = ''
        for x in range(w):
            v = field[y, x]
            if spots is not None:                       # explicit (plain) palette for undersides
                ch = spots[0][0] if fine[y, x] < spots[0][1] else base
            elif v > 0.80:
                ch = 'l' if fine[y, x] > 0.7 else 't'
            elif v > 0.66:
                ch = 'b'
            else:
                ch = base
            row += ch
        rows.append(row)
    return rows


def _overlay(rows, pattern, x0, y0):
    rows = [list(r) for r in rows]
    for dy, prow in enumerate(pattern):
        for dx, ch in enumerate(prow):
            if ch != '.':
                rows[y0 + dy][x0 + dx] = ch
    return [''.join(r) for r in rows]


def make_warden_skin(seed=13):
    """{part: {face: (HxWx3 colours, HxW glow)}}. Face images are as seen from outside the part: ny = front
    (-Y), py = back, nx/px = sides, pz = top, nz = bottom; row 0 = top edge (pz: back edge, nz: front edge).
    For px the first column is the front, for nx the first column is the back."""
    rng = np.random.default_rng(seed)
    P = lambda rows, j=0.05: _paint(rows, rng, j)
    S = lambda w, h, **kw: _speckle(w, h, rng, **kw)

    face = ["BBBBBBBBBBBBBBBB",
            "BbBBBtBBBBtBBBbB",
            "BBtBBBBbbBBBBtBB",
            "BBBtlBBBBBBltBBB",
            "BBBBtBBBBBBtBBBB",
            "BBDDDBBBBBBDDDBB",
            "BDDDDDBBBBDDDDDB",
            "BDDDDBBttBBDDDDB",
            "BBDDBBtlltBBDDBB",
            "BBBBBBBttBBBBBBB",
            "BlDDDDDDDDDDDDlB",
            "BDTDTDTDTDTDTDTB",
            "BDDDDDDDDDDDDDDB",
            "BDTDTDTDTDTDTDTB",
            "BBDDDDDDDDDDDDBB",
            "BBBBbBBBBBBbBBBB"]
    head = {
        'ny': P(face),
        'nx': P(S(10, 16)), 'px': P(S(10, 16)), 'py': P(S(16, 16)),
        'pz': P(_overlay(S(16, 10), ["t..t..t..t..t..t", ".l....l....l...."], 0, 3)),
        'nz': P(S(16, 10, spots=(('b', 0.2),))),
    }
    chest = ["BBBBBBBBBBBBBBBBBB",
             "BBbBBBBBBBBBBBBbBB",
             "BBBBrrrrrBrrrrrBBB",
             "BBBrDggDDrDDggDrBB",
             "BBBrDggDDrDDggDrBB",
             "BBBBrrrrrrrrrrrBBB",
             "BBBrDDgDDrDDgDDrBB",
             "BBBrDgggDrDgggDrBB",
             "BBBBrrrrrrrrrrrBBB",
             "BBBBrDDgDrDgDDrBBB",
             "BBBBrDggDrDggDrBBB",
             "BBBBBrrrrrrrrrBBBB",
             "BBBBBBBBrBBBBBBBBB"]
    front = _overlay(S(18, 21), chest, 0, 0)
    body = {
        'ny': P(front),
        'py': P(_overlay(S(18, 21), ["...t....t....t...", "....b..b..b..b..."], 0, 4)),
        'nx': P(S(10, 21)), 'px': P(S(10, 21)),
        'pz': P(S(18, 10)), 'nz': P(S(18, 10, spots=(('b', 0.15),))),
    }

    def arm():
        side = lambda: S(7, 26)[:21] + ["BBBBBBB", "BhBBBhB", "hBhBhBh", "hhhhhhh", "hhhhhhh"]
        return {'ny': P(side()), 'py': P(side()), 'nx': P(side()), 'px': P(side()),
                'pz': P(S(7, 7)), 'nz': P(["hhhhhhh"] * 7)}

    def leg():
        side = lambda: S(6, 13)[:11] + ["hhhhhh", "hhhhhh"]
        return {'ny': P(side()), 'py': P(side()), 'nx': P(side()), 'px': P(side()),
                'pz': P(S(6, 6)), 'nz': P(["hhhhhh"] * 6)}

    def tendril():
        # the broad faces glow: a cyan core inside a dark rim
        broad = ["BBBBBBBBB",
                 "BtggggggB",
                 "BgggggggB",
                 "BgggggglB",
                 "BtgggggtB",
                 "BBgggggBB",
                 "BBtgggtBB",
                 "BBBtgtBBB",
                 "BBBBBBBBB"]
        return {'ny': P(broad), 'py': P([r[::-1] for r in broad]),
                'nx': P(["BB"] * 9), 'px': P(["BB"] * 9),
                'pz': P(["t" * 9] * 2), 'nz': P(["B" * 9] * 2)}

    return {'head': head, 'body': body, 'arm_r': arm(), 'arm_l': arm(), 'leg_r': leg(), 'leg_l': leg(),
            'tendril_r': tendril(), 'tendril_l': tendril()}


def _part_shape(name, ni, nj, nk):
    """Occupancy inside a part's box (voxel coordinates); the tendrils taper outwards and sweep up."""
    if not name.startswith('tendril'):
        return np.ones((ni, nj, nk), bool)
    ii, jj, kk = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing='ij')
    u = (ii + 0.5) / ni                           # 0 at the outer tip ... 1 at the head (for tendril_r)
    if name == 'tendril_l':
        u = 1.0 - u
    v = (kk + 0.5) / nk
    lo = 0.55 * (1.0 - u)                         # bottom edge rises towards the tip
    return (v >= lo) & (v <= 1.0)


class Giant:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        skin = make_warden_skin()
        self.occ = np.zeros((NX, NY, NZ), bool)
        self.part = np.zeros((NX, NY, NZ), np.uint8)
        self.cx = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cy = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.cz = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.inner = np.zeros((NX, NY, NZ, 4), np.uint8)
        self.mask = np.zeros((NX, NY, NZ), np.uint8)
        self.bone = np.zeros((NX, NY, NZ), bool)
        self.glow = np.zeros((NX, NY, NZ), bool)
        for name, (x0, y0, z0, sx, sy, sz) in PARTS.items():
            i0, j0, k0 = (x0 - GX0P) * K, (y0 - GY0P) * K, (z0 - GZ0P) * K
            ni, nj, nk = sx * K, sy * K, sz * K
            sl = (slice(i0, i0 + ni), slice(j0, j0 + nj), slice(k0, k0 + nk))
            shape = _part_shape(name, ni, nj, nk)
            self.occ[sl] |= shape
            self.part[sl] = np.where(shape, PART_IDS[name], self.part[sl])
            ii, jj, kk = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing='ij')
            pi, pj, pk = ii // K, jj // K, kk // K
            tex = skin[name]
            m = np.zeros((ni, nj, nk), np.uint8)
            cx = np.zeros((ni, nj, nk, 3))
            cy = np.zeros((ni, nj, nk, 3))
            cz = np.zeros((ni, nj, nk, 3))
            gl = np.zeros((ni, nj, nk), bool)

            def put(sel, arr, face, r, c, bit):
                img, g = tex[face]
                arr[sel] = img[r, c]
                gl[sel] |= g[r, c]
                m[sel] |= 1 << bit
            # a voxel is on the skin where its neighbour across that face is outside the part's shape
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
            # --- bones: spine, a ribcage in the chest, limb bones ------------------------------------------
            c3 = lambda n: (n // 2 - 2, n // 2 + 1)
            if name == 'body':
                a0, a1 = c3(ni)
                spine = (ii >= a0) & (ii <= a1) & (jj >= nj - 9) & (jj <= nj - 6)
                ribs = (jj >= 3) & (jj <= 5) & (kk >= nk - 36) & (kk <= nk - 7) & ((kk // 3) % 3 == 0) \
                    & (ii >= 6) & (ii <= ni - 7)
                b = spine | ribs
            elif name.startswith('arm') or name.startswith('leg'):
                a0, a1 = c3(ni)
                c0, c1 = c3(nj)
                b = (ii >= a0) & (ii <= a1) & (jj >= c0) & (jj <= c1) & (kk >= 3) & (kk <= nk - 4)
            else:
                b = np.zeros((ni, nj, nk), bool)
            self.bone[sl] |= b & shape
        n = int(self.occ.sum())
        idx = np.nonzero(self.occ)
        choice = rng.choice(len(FLESH), size=n, p=FLESH_P)
        col = FLESH[choice] * (1 + (rng.random((n, 1)) - 0.5) * 0.14)
        soul = rng.random(n) < 0.06
        col[soul] = SOUL[rng.integers(len(SOUL), size=int(soul.sum()))]
        bsel = self.bone[idx]
        bc = BONE[rng.integers(len(BONE), size=int(bsel.sum()))] * (1 + (rng.random((int(bsel.sum()), 1)) - 0.5) * 0.06)
        col[bsel] = bc
        inner = np.zeros((n, 4), np.uint8)
        inner[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
        inner[:, 3] = np.where(bsel, 255, np.where(soul, 128, 0))
        self.inner[idx] = inner
        self.cx[..., 3] = self.mask       # mask travels in the alpha of the x colour
        self.cz[..., 3] = np.where(self.glow, 255, 0)
        self.total = n
        self.initial_occ = self.occ.copy()

    # -- queries --------------------------------------------------------------------------------
    @staticmethod
    def world_to_index(p):
        p = np.asarray(p)
        i = np.floor((p[..., 0] - GX0) / VS).astype(np.int64)
        j = np.floor((p[..., 1] - GY0) / VS).astype(np.int64)
        k = np.floor((p[..., 2] - GZ0) / VS).astype(np.int64)
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
    import sys
    import time
    t = time.time()
    g = Giant()
    print('voxels', g.total, 'exposed', int(g.exposed().sum()), 'bones', int(g.bone.sum()), 'glow',
          int(g.glow.sum()), f'{time.time() - t:.1f}s', 'height', HEIGHT)
    if len(sys.argv) > 1:
        from PIL import Image
        skin = make_warden_skin()
        tiles = []
        for part in ('head', 'body', 'arm_r', 'tendril_r'):
            for f in ('ny', 'px', 'py', 'pz'):
                img = skin[part][f][0]
                big = np.kron(img, np.ones((12, 12, 1)))
                tiles.append(np.pad(big, ((4, 4), (4, 4), (0, 0)), constant_values=255))
        h = max(t_.shape[0] for t_ in tiles)
        tiles = [np.pad(t_, ((0, h - t_.shape[0]), (0, 0), (0, 0)), constant_values=255) for t_ in tiles]
        Image.fromarray(np.concatenate(tiles, 1).astype(np.uint8)).save(sys.argv[1])
