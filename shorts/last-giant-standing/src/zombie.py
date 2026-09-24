"""Voxel giant zombie: every skin pixel is split into K x K x K voxels.

Minecraft proportions (in skin pixels = world units), 32 units tall, facing -Y, with the classic zombie pose:
both arms held straight forward (horizontal 4x12x4 boxes at shoulder height). Each voxel stores the skin colour
for its outward x/y/z faces (+ a bit mask telling which of its six faces are on the skin) and an 'inner'
colour (dark red flesh or bone) used for faces exposed by damage.
"""
import numpy as np

K = 4
VS = 1.0 / K
GX0, GY0, GZ0 = -8.0, -10.0, 0.0            # world position of grid corner
NX, NY, NZ = 16 * K, 14 * K, 32 * K

# name: (x0, y0, z0, sx, sy, sz) in skin pixels (world units); the zombie faces -Y
PARTS = {
    'leg_r': (-4, -2, 0, 4, 4, 12),
    'leg_l': (0, -2, 0, 4, 4, 12),
    'body': (-4, -2, 12, 8, 4, 12),
    'arm_r': (-8, -10, 20, 4, 12, 4),        # arms stick straight out in front
    'arm_l': (4, -10, 20, 4, 12, 4),
    'head': (-4, -4, 24, 8, 8, 8),
}
PART_IDS = {n: i + 1 for i, n in enumerate(PARTS)}

FLESH = np.array([[134, 24, 26], [120, 20, 22], [146, 34, 32], [104, 16, 20], [92, 14, 16],
                  [156, 46, 38], [112, 38, 30]], np.float64)
FLESH_P = np.array([0.22, 0.2, 0.14, 0.14, 0.1, 0.1, 0.1])
BONE = np.array([[232, 224, 196], [222, 212, 184], [240, 234, 212], [210, 200, 172]], np.float64)

# ---------------------------------------------------------------------------------------------
# skin (own pixel art in the classic zombie style)
# ---------------------------------------------------------------------------------------------
PAL = {
    'S': [(84, 132, 62), (78, 124, 58), (92, 140, 68), (72, 116, 54)],      # skin
    's': [(64, 104, 48), (60, 98, 46)],                                      # skin shadow
    'H': [(62, 100, 47), (58, 95, 44), (66, 104, 50)],                       # dark green "hair"
    'd': [(40, 66, 32)],                                                     # sunken eye sockets
    'E': [(14, 16, 14), (20, 22, 18)],                                       # eyes
    'n': [(52, 84, 40)],                                                     # nose shadow
    'M': [(34, 52, 28), (30, 46, 26)],                                       # mouth
    'T': [(24, 158, 162), (18, 148, 154), (30, 168, 170), (14, 140, 148)],   # shirt
    't': [(10, 116, 124), (8, 108, 116)],                                    # shirt shadow / torn edge
    'P': [(56, 50, 150), (50, 46, 140), (62, 56, 160), (46, 42, 130)],       # pants
    'G': [(72, 72, 78), (64, 64, 70), (80, 80, 86)],                         # shoes
}


def _paint(rows, rng, jitter=0.05):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        assert len(row) == w, (row, w)
        for c, ch in enumerate(row):
            opts = PAL[ch]
            img[r, c] = opts[rng.integers(len(opts))]
    img *= 1.0 + (rng.random((h, w, 1)) - 0.5) * 2 * jitter
    return np.clip(img, 0, 255)


def make_zombie_skin(seed=9):
    """Return {part: {face: HxWx3}}. Face images are as seen from outside the part:
    ny = front (-Y), py = back, nx/px = sides, pz = top, nz = bottom; row 0 = top edge (pz: back edge,
    nz: front edge). For px the first column is the front, for nx the first column is the back."""
    rng = np.random.default_rng(seed)
    P = lambda rows, j=0.05: _paint(rows, rng, j)
    head = {
        'ny': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "sSSSSSSs",
                 "SddSSddS",
                 "SEESSEES",
                 "SSSnnSSS",
                 "SSMMMMSS",
                 "SSSSSSSS"]),
        'nx': P(["HHHHHHHH",          # first column = back of the head
                 "HHHHHHHH",
                 "HHHHSSSS",
                 "HHHSSSSS",
                 "HHSSSSSS",
                 "HSSSSSSS",
                 "HSSSSSSS",
                 "SSSSSSSS"]),
        'px': P(["HHHHHHHH",          # first column = front of the head
                 "HHHHHHHH",
                 "SSSSHHHH",
                 "SSSSSHHH",
                 "SSSSSSHH",
                 "SSSSSSSH",
                 "SSSSSSSH",
                 "SSSSSSSS"]),
        'py': P(["HHHHHHHH"] * 5 + ["HsHHHHsH", "sSsSSsSs", "SSSSSSSS"]),
        'pz': P(["HHHHHHHH"] * 8),
        'nz': P(["ssssssss"] * 8, 0.03),
    }
    body_front = ["TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTtTT",
                  "TTTTTTTT",
                  "TtTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTSS",
                  "tSTTTtSt",
                  "PPPPPPPP",
                  "PPPPPPPP",
                  "PPPPPPPP"]
    body_back = ["TTTTTTTT",
                 "TTTTTTTT",
                 "TTTtTTTT",
                 "TTTTTTTT",
                 "TTTTTTTT",
                 "TTTTTTtT",
                 "TTTTTTTT",
                 "SSTTTTTT",
                 "tSSTtTTt",
                 "PPPPPPPP",
                 "PPPPPPPP",
                 "PPPPPPPP"]
    body = {
        'ny': P(body_front),
        'py': P(body_back),
        'nx': P(["TTTT"] * 7 + ["TTTS", "tSTt"] + ["PPPP"] * 3),
        'px': P(["TTTT"] * 7 + ["STTT", "tTSt"] + ["PPPP"] * 3),
        'pz': P(["TTTTTTTT"] * 4),
        'nz': P(["PPPPPPPP"] * 4),
    }
    for f in ('ny', 'py', 'nx', 'px'):
        g = np.linspace(1.03, 0.95, body[f].shape[0])[:, None, None]
        body[f] = np.clip(body[f] * g, 0, 255)

    def arm():
        # forward arm box: 4 wide (x) x 12 long (y) x 4 tall (z); the 4 units nearest the body are sleeve
        side_px = ["SSSSSSSSTTTT", "SSSSSSStTTTT", "SSSSSSSStTTT", "sSSSSSSSStTT"]   # first column = hand
        side_nx = [r[::-1] for r in side_px]                                         # first column = shoulder
        top = ["TTTT", "TTTT", "TtTT", "StTS"] + ["SSSS"] * 7 + ["SSSS"]              # first row = shoulder
        bottom = ["ssss"] + ["SsSS"] * 7 + ["tTTt", "TTTT", "TTTT", "TTTT"]         # first row = hand
        return {
            'ny': P(["SSSS", "SsSS", "SSsS", "ssss"]),        # hand end
            'py': P(["TTTT"] * 4),                           # shoulder end (mostly against the body)
            'px': P(side_px), 'nx': P(side_nx), 'pz': P(top), 'nz': P(bottom),
        }

    leg_rows = ["PPPP"] * 10 + ["GGGG"] * 2
    leg = lambda: {
        'ny': P(leg_rows), 'py': P(leg_rows), 'nx': P(leg_rows), 'px': P(leg_rows),
        'pz': P(["PPPP"] * 4), 'nz': P(["GGGG"] * 4),
    }
    return {'head': head, 'body': body, 'arm_r': arm(), 'arm_l': arm(), 'leg_r': leg(), 'leg_l': leg()}


class Giant:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        skin = make_zombie_skin()
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
            # --- bones: spine and limb bones (the arms' bones run forward along y) ------------------------
            if name == 'head':
                b = np.zeros((ni, nj, nk), bool)
            elif name == 'body':
                b = (ii >= 14) & (ii <= 17) & (jj >= 9) & (jj <= 12)
            elif name.startswith('arm'):
                b = (ii >= 6) & (ii <= 9) & (kk >= 6) & (kk <= 9) & (jj >= 3) & (jj <= nj - 4)
            else:
                b = (ii >= 6) & (ii <= 9) & (jj >= 6) & (jj <= 9) & (kk >= 3) & (kk <= nk - 4)
            self.bone[sl] = b
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
