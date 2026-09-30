"""The voxel world: a grid of block ids (+ a state byte per voxel), a vectorised mesher and Minecraft's two light
fields (sky light and block light, 0..15, spreading one level less per block) plus a third for soul fire.

Mesh vertex layout (12 floats): pos3 normal3 uv2 layer1 tint3; uv in blocks (textures repeat), layer = texture array
layer. Output is split into buckets ('solid', 'cutout') and 16^3 sections with bounding boxes for culling.
"""
import numpy as np

import blocks as BL
import textures as TX

SECTION = 16
FACE_DIRS = {'px': (1, 0, 0), 'nx': (-1, 0, 0), 'py': (0, 1, 0), 'ny': (0, -1, 0), 'pz': (0, 0, 1), 'nz': (0, 0, -1)}
FACE_ORDER = ['px', 'nx', 'py', 'ny', 'pz', 'nz']


# ---------------------------------------------------------------------------------------------
# textures -> texture array layers
# ---------------------------------------------------------------------------------------------
class Atlas:
    def __init__(self, seed=7):
        T, E = TX.make_all(seed)
        self.names = sorted(T)
        self.layer = {n: i for i, n in enumerate(self.names)}
        self.rgba = np.stack([T[n] for n in self.names])                       # (L,16,16,4) uint8
        self.emit = np.stack([E.get(n, np.zeros((16, 16))) for n in self.names]).astype(np.float32)
        self.has_alpha = np.array([(T[n][..., 3] < 255).any() for n in self.names])

    def __getitem__(self, name):
        return self.layer[name]


ATLAS = None


def atlas():
    global ATLAS
    if ATLAS is None:
        ATLAS = Atlas()
    return ATLAS


def face_layer_table(at):
    """(NBLOCKS, 4 facings, 6 faces) -> texture layer of cube faces."""
    tab = np.zeros((BL.NBLOCKS, 4, 6), np.int32)
    for b in BL.REG:
        if b.shape not in ('cube', 'liquid'):
            continue
        for f in range(4):
            for k, face in enumerate(FACE_ORDER):
                name = b.face_tex(face, f)
                if b.shape == 'liquid':
                    name = 'water'
                tab[b.id, f, k] = at[name]
    return tab


# ---------------------------------------------------------------------------------------------
# quads
# ---------------------------------------------------------------------------------------------
def face_corners(face, x0, y0, z0, x1, y1, z1):
    """Corner positions (4, N, 3) of axis aligned faces, counter-clockwise seen from outside, starting bottom-left;
    and the uv at those corners (4, N, 2) (texture row 0 at the top of side faces, north on top faces)."""
    if face == 'px':
        P = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        UV = [(y0, -z0), (y1, -z0), (y1, -z1), (y0, -z1)]
    elif face == 'nx':
        P = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
        UV = [(-y1, -z0), (-y0, -z0), (-y0, -z1), (-y1, -z1)]
    elif face == 'py':
        P = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
        UV = [(-x1, -z0), (-x0, -z0), (-x0, -z1), (-x1, -z1)]
    elif face == 'ny':
        P = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        UV = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
    elif face == 'pz':
        P = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        UV = [(x0, -y0), (x1, -y0), (x1, -y1), (x0, -y1)]
    else:
        P = [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)]
        UV = [(-x1, -y0), (-x0, -y0), (-x0, -y1), (-x1, -y1)]
    P = np.stack([np.stack(np.broadcast_arrays(*p), -1) for p in P])
    UV = np.stack([np.stack(np.broadcast_arrays(*u), -1) for u in UV])
    return P.astype(np.float32), UV.astype(np.float32)


def quads_to_arrays(P, UV, nrm, layer, tint):
    """P (4, N, 3), UV (4, N, 2), nrm (N, 3) or (3,), layer (N,), tint (N, 3) or (3,) -> vertices (4N, 12) and
    indices (6N,) (local)."""
    n = P.shape[1]
    V = np.zeros((n, 4, 12), np.float32)
    V[:, :, 0:3] = np.transpose(P, (1, 0, 2))
    nrm = np.asarray(nrm, np.float32)
    V[:, :, 3:6] = nrm if nrm.ndim == 1 else nrm[:, None, :]
    V[:, :, 6:8] = np.transpose(UV, (1, 0, 2))
    V[:, :, 8] = np.asarray(layer, np.float32).reshape(-1, 1)
    tint = np.asarray(tint, np.float32)
    V[:, :, 9:12] = tint if tint.ndim == 1 else tint[:, None, :]
    base = (np.arange(n) * 4)[:, None]
    I = (base + np.array([0, 1, 2, 0, 2, 3])[None, :]).astype(np.uint32)
    return V.reshape(-1, 12), I.reshape(-1)


# ---------------------------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------------------------
class World:
    def __init__(self, size=(384, 384, 160), origin=(-192, -192, -100)):
        self.size = tuple(size)
        self.origin = np.array(origin, np.int64)
        self.ids = np.zeros(size, np.uint16)
        self.state = np.zeros(size, np.uint8)
        self.hidden = np.zeros(size, bool)            # drawn as props (doors that open, blocks that break)
        self.tint = {}                                # optional per-voxel tints: (i, j, k) -> rgb
        self.signs = []                               # (x, y, z, facing, wall, lines)
        self.frames = []                              # item frames: (x, y, z, facing, content)

    # -- indexing ----------------------------------------------------------------------------
    def ix(self, x, y, z):
        return int(x) - self.origin[0], int(y) - self.origin[1], int(z) - self.origin[2]

    def inside(self, x, y, z):
        i, j, k = self.ix(x, y, z)
        return 0 <= i < self.size[0] and 0 <= j < self.size[1] and 0 <= k < self.size[2]

    def set(self, x, y, z, name, state=0):
        i, j, k = self.ix(x, y, z)
        if 0 <= i < self.size[0] and 0 <= j < self.size[1] and 0 <= k < self.size[2]:
            self.ids[i, j, k] = BL.B[name] if isinstance(name, str) else name
            self.state[i, j, k] = state

    def get(self, x, y, z):
        i, j, k = self.ix(x, y, z)
        if 0 <= i < self.size[0] and 0 <= j < self.size[1] and 0 <= k < self.size[2]:
            return BL.REG[self.ids[i, j, k]].name
        return 'air'

    def fill(self, x0, y0, z0, x1, y1, z1, name, state=0):
        """Inclusive box, like /fill."""
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        za, zb = sorted((z0, z1))
        i0, j0, k0 = self.ix(xa, ya, za)
        i1, j1, k1 = self.ix(xb, yb, zb)
        i0, j0, k0 = max(i0, 0), max(j0, 0), max(k0, 0)
        i1, j1, k1 = min(i1, self.size[0] - 1), min(j1, self.size[1] - 1), min(k1, self.size[2] - 1)
        if i1 < i0 or j1 < j0 or k1 < k0:
            return
        self.ids[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = BL.B[name] if isinstance(name, str) else name
        self.state[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = state

    def hide(self, x, y, z, flag=True):
        i, j, k = self.ix(x, y, z)
        self.hidden[i, j, k] = flag

    def column_top(self, x, y):
        """z of the top face of the highest opaque block in the column (the ground a player stands on)."""
        i, j, _ = self.ix(x, y, 0)
        col = BL.OPAQUE[self.ids[i, j]]
        ks = np.nonzero(col)[0]
        return int(ks[-1] + 1 + self.origin[2]) if len(ks) else int(self.origin[2])

    def copy_box(self, src0, src1, dst0, rot=0):
        """Copy the inclusive box src0..src1 to dst0 (its min corner), rotated by rot quarter turns about z
        (states' facings rotate with it)."""
        (x0, y0, z0), (x1, y1, z1) = src0, src1
        i0, j0, k0 = self.ix(x0, y0, z0)
        i1, j1, k1 = self.ix(x1, y1, z1)
        ids = self.ids[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1].copy()
        st = self.state[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1].copy()
        for _ in range(rot % 4):
            ids = np.rot90(ids, 1, (0, 1))
            st = np.rot90(st, 1, (0, 1))
            st = (st & ~np.uint8(3)) | ((st & 3) + 1) % 4
        di, dj, dk = self.ix(*dst0)
        a, b, c = ids.shape
        self.ids[di:di + a, dj:dj + b, dk:dk + c] = ids
        self.state[di:di + a, dj:dj + b, dk:dk + c] = st

    # -- light --------------------------------------------------------------------------------
    def light(self, iters=15):
        """(X, Y, Z, 4) uint8: sky light, block light, soul light (0..15 scaled to 0..255), 0."""
        opq = BL.OPAQUE[self.ids]
        # sky: full down every column until the first opaque block
        blocked = np.flip(np.maximum.accumulate(np.flip(opq, 2), axis=2), 2)
        sky = np.where(blocked, 0, 15).astype(np.uint8)
        blk = BL.LIGHT[self.ids].astype(np.uint8)
        soul = BL.SOUL[self.ids].astype(np.uint8)
        free = ~opq
        out = []
        for L in (sky, blk, soul):
            if L.max() == 0:
                out.append(L)
                continue
            # only propagate inside the bounding box of what can change (sources grown by 15)
            nz = np.nonzero(L)
            lo = np.maximum(np.array([a.min() for a in nz]) - iters - 1, 0)
            hi = np.minimum(np.array([a.max() for a in nz]) + iters + 2, np.array(L.shape))
            if L is sky:
                lo = np.zeros(3, int)
                hi = np.array(L.shape)
            sub = L[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
            fr = free[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            for _ in range(iters):
                m = sub.copy()
                for ax in range(3):
                    for s in (1, -1):
                        sh = np.roll(sub, s, ax)
                        idx = [slice(None)] * 3
                        idx[ax] = 0 if s == 1 else -1
                        sh[tuple(idx)] = 0
                        np.maximum(m, sh, out=m)
                m = np.where(m > 0, m - 1, 0).astype(np.uint8)
                new = np.where(fr, np.maximum(sub, m), sub)
                if np.array_equal(new, sub):
                    break
                sub = new
            L = L.copy()
            L[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub
            out.append(L)
        vol = np.zeros(self.size + (4,), np.uint8)
        for c, L in enumerate(out):
            vol[..., c] = (L.astype(np.uint16) * 17).astype(np.uint8)
        return vol

    # -- meshing ------------------------------------------------------------------------------
    def mesh(self, at=None, region=None):
        """Build the static mesh. Returns dict bucket -> (vertices, indices, sections) where sections is a list of
        (first_index, count, lo, hi)."""
        at = at or atlas()
        tab = face_layer_table(at)
        ids = np.where(self.hidden, 0, self.ids).astype(np.int64)
        st = self.state
        X, Y, Z = self.size
        ox, oy, oz = self.origin
        shape = BL.SHAPE[ids]
        cube = BL.IS_CUBE[ids]
        opq = BL.OPAQUE[ids]
        liquid = shape == 'liquid'
        parts = {'solid': [], 'cutout': [], 'plants': []}

        def nb(arr, d, fill):
            out = np.full_like(arr, fill)
            dx, dy, dz = d
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            for ax, s in enumerate((dx, dy, dz)):
                if s == 1:
                    src[ax] = slice(1, None)
                    dst[ax] = slice(0, -1)
                elif s == -1:
                    src[ax] = slice(0, -1)
                    dst[ax] = slice(1, None)
            out[tuple(dst)] = arr[tuple(src)]
            return out

        facing = (st & 3).astype(np.int64)
        for fk, face in enumerate(FACE_ORDER):
            d = FACE_DIRS[face]
            nb_ids = nb(ids, d, 0 if face != 'nz' else BL.B['stone'])
            nb_opq = BL.OPAQUE[nb_ids]
            same_see = (nb_ids == ids) & ~opq
            show = cube & ~nb_opq & ~same_see
            layer = tab[ids, facing, fk]
            # merge runs along an in-plane axis
            ax = 0 if face in ('pz', 'nz', 'py', 'ny') else 1
            cut = BL.CUTOUT[ids]
            for bucket, sel in (('solid', show & ~cut), ('cutout', show & cut)):
                if not sel.any():
                    continue
                key = np.where(sel, layer, -1)
                self._runs(parts[bucket], sel, key, ax, face, d)
            # water: top surface a little below the block top, sides against air
            wsel = liquid & (nb_ids == 0) if face != 'pz' else liquid & ~(BL.SHAPE[nb_ids] == 'liquid')
            if face == 'nz':
                wsel = np.zeros_like(wsel)
            if wsel.any():
                self._runs(parts['solid'], wsel, np.where(wsel, at['water'], -1), ax, face, d, water=True)
        self._models(parts, ids, st, at)
        self._plants(parts, ids, at)
        out = {}
        for bucket, lst in parts.items():
            out[bucket] = self._assemble(lst)
        return out

    def _runs(self, sink, sel, key, ax, face, d, water=False):
        L = self.size[ax]
        S = np.moveaxis(sel, ax, -1)
        K = np.moveaxis(key, ax, -1)
        shp = S.shape
        S2 = S.reshape(-1, L)
        K2 = K.reshape(-1, L)
        pos = np.arange(L)
        prev_same = np.zeros_like(S2)
        prev_same[:, 1:] = S2[:, :-1] & (K2[:, :-1] == K2[:, 1:])
        prev_same[:, (pos % SECTION) == 0] = False
        start = S2 & ~prev_same
        next_same = np.zeros_like(S2)
        next_same[:, :-1] = S2[:, 1:] & (K2[:, 1:] == K2[:, :-1])
        next_same[:, ((pos + 1) % SECTION) == 0] = False
        end = S2 & ~next_same
        rs, cs = np.nonzero(start)
        re_, ce = np.nonzero(end)
        assert len(rs) == len(re_)
        other = np.unravel_index(rs, shp[:-1])
        coords = [None, None, None]
        oth_axes = [a for a in range(3) if a != ax]
        coords[oth_axes[0]] = other[0]
        coords[oth_axes[1]] = other[1]
        lo = [None] * 3
        hi = [None] * 3
        for a in range(3):
            if a == ax:
                lo[a] = cs + self.origin[a]
                hi[a] = ce + 1 + self.origin[a]
            else:
                lo[a] = coords[a] + self.origin[a]
                hi[a] = coords[a] + 1 + self.origin[a]
        layer = K2[rs, cs]
        x0, y0, z0 = [np.asarray(v, np.float32) for v in lo]
        x1, y1, z1 = [np.asarray(v, np.float32) for v in hi]
        if water and face == 'pz':
            z1 = z1 - 0.12                   # the surface sits a little low; falling columns stay whole
        P, UV = face_corners(face, x0, y0, z0, x1, y1, z1)
        V, I = quads_to_arrays(P, UV, np.array(d, np.float32), layer, np.ones(3, np.float32))
        sink.append((V, I))

    def _models(self, parts, ids, st, at):
        model_ids = [b.id for b in BL.REG if b.shape == 'model']
        sel = np.isin(ids, model_ids)
        pos = np.argwhere(sel)
        if not len(pos):
            return
        buckets = {'solid': [], 'cutout': []}
        conn = BL.CONNECTS
        for (i, j, k) in pos:
            b = BL.REG[ids[i, j, k]]
            s = int(st[i, j, k])
            nbm = {}
            if b.connects:
                for (dx, dy) in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    ii, jj = i + dx, j + dy
                    if 0 <= ii < self.size[0] and 0 <= jj < self.size[1]:
                        o = ids[ii, jj, k]
                        nbm[(dx, dy)] = bool(conn[o] or BL.OPAQUE[o])
            boxes = b.model(s, nbm)
            base = np.array([i + self.origin[0], j + self.origin[1], k + self.origin[2]], np.float32)
            for bx in boxes:
                for face, spec in bx['faces'].items():
                    tex, uv = spec
                    lay = at[tex]
                    q = model_face(bx, face, uv, s & BL.FACING)
                    if q is None:
                        continue
                    P, UVq, nrm = q
                    ax = int(np.argmax(np.abs(nrm)))
                    if abs(nrm[ax]) > 0.999:
                        c = P[:, ax]
                        sgn = 1 if nrm[ax] > 0 else -1
                        edge = 1.0 if sgn > 0 else 0.0
                        if np.all(np.abs(c - edge) < 1e-4):
                            d = [0, 0, 0]
                            d[ax] = sgn
                            ii, jj, kk = i + d[0], j + d[1], k + d[2]
                            if 0 <= ii < self.size[0] and 0 <= jj < self.size[1] and 0 <= kk < self.size[2]:
                                o = ids[ii, jj, kk]
                                if BL.OPAQUE[o] or (o == ids[i, j, k] and st[ii, jj, kk] == s):
                                    continue
                    bucket = 'cutout' if at.has_alpha[lay] else 'solid'
                    buckets[bucket].append((P + base, UVq, nrm, lay))
        for bucket, lst in buckets.items():
            if not lst:
                continue
            P = np.stack([q[0] for q in lst], 1)            # (4, N, 3)
            UV = np.stack([q[1] for q in lst], 1)
            nrm = np.stack([q[2] for q in lst])
            lay = np.array([q[3] for q in lst])
            V, I = quads_to_arrays(P, UV, nrm, lay, np.ones(3, np.float32))
            parts[bucket].append((V, I))

    def _plants(self, parts, ids, at):
        cross_ids = [b.id for b in BL.REG if b.shape == 'cross']
        pos = np.argwhere(np.isin(ids, cross_ids))
        if not len(pos):
            return
        rng = np.random.default_rng(3)
        n = len(pos)
        lay = np.array([at[BL.REG[ids[i, j, k]].name] for (i, j, k) in pos])
        is_grass = np.array([BL.REG[ids[i, j, k]].name in ('tall_grass', 'poppy', 'dandelion', 'cornflower')
                             for (i, j, k) in pos])
        off = rng.uniform(-0.2, 0.2, (n, 2)) * is_grass[:, None]
        hscale = np.where(is_grass, rng.uniform(0.75, 1.0, n), 1.0)
        c = pos[:, :2] + self.origin[:2] + 0.5 + off
        z0 = pos[:, 2] + self.origin[2]
        z1 = z0 + hscale
        s = 0.45
        quads = []
        for (ax, ay) in ((1, 1), (1, -1)):
            ex, ey = ax * s, ay * s
            a = np.stack([c[:, 0] - ex, c[:, 1] - ey], -1)
            b = np.stack([c[:, 0] + ex, c[:, 1] + ey], -1)
            for (p, q) in ((a, b), (b, a)):
                P = np.stack([np.c_[p, z0], np.c_[q, z0], np.c_[q, z1], np.c_[p, z1]]).astype(np.float32)
                UV = np.stack([np.c_[np.zeros(n), -np.zeros(n) - 1 + (1 - hscale) * 0],
                               np.c_[np.ones(n), -np.ones(n) * 0 - 1],
                               np.c_[np.ones(n), np.zeros(n) - 1 + hscale * 0 - hscale + 1 - 1 + 0],
                               np.c_[np.zeros(n), np.zeros(n)]]).astype(np.float32)
                # u 0..1 across, v from the bottom row (1) to the top row (0) (texture row 0 = top)
                UV[0, :, 1] = 1.0
                UV[1, :, 1] = 1.0
                UV[2, :, 1] = 1.0 - hscale
                UV[3, :, 1] = 1.0 - hscale
                UV[2, :, 1] = 0.0
                UV[3, :, 1] = 0.0
                quads.append((P, UV))
        for (P, UV) in quads:
            V, I = quads_to_arrays(P, UV, np.array((0, 0, 1), np.float32), lay, np.ones(3, np.float32))
            parts['plants'].append((V, I))

    @staticmethod
    def _assemble(lst):
        if not lst:
            return np.zeros((0, 12), np.float32), np.zeros(0, np.uint32), []
        V = np.concatenate([v for v, _ in lst])
        # re-index and sort quads by section
        nq = len(V) // 4
        Q = V.reshape(nq, 4, 12)
        cen = Q[:, :, 0:3].mean(1)
        sec = np.floor(cen / SECTION).astype(np.int64)
        key = (sec[:, 0] + 512) * 1024 * 1024 + (sec[:, 1] + 512) * 1024 + (sec[:, 2] + 512)
        order = np.argsort(key, kind='stable')
        Q = Q[order]
        key = key[order]
        V = Q.reshape(-1, 12)
        I = ((np.arange(nq) * 4)[:, None] + np.array([0, 1, 2, 0, 2, 3])[None, :]).astype(np.uint32).reshape(-1)
        bounds = np.nonzero(np.diff(key))[0] + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [nq]])
        sections = []
        for a, b in zip(starts, ends):
            p = Q[a:b, :, 0:3].reshape(-1, 3)
            sections.append((int(a * 6), int((b - a) * 6), p.min(0), p.max(0)))
        return V, I, sections


# ---------------------------------------------------------------------------------------------
# model faces
# ---------------------------------------------------------------------------------------------
def _rot_axis(axis, ang):
    a = np.radians(ang)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def model_face(bx, face, uv, facing):
    """Corners (4,3) in block units, uv (4,2) in texture units, normal (3,) of one face of a model box."""
    (x0, y0, z0), (x1, y1, z1) = bx['from'], bx['to']
    P, _ = face_corners(face, np.float32(x0), np.float32(y0), np.float32(z0), np.float32(x1), np.float32(y1),
                        np.float32(z1))
    P = P.reshape(4, 3).astype(np.float64)
    n = np.array(FACE_DIRS[face], float)
    if bx.get('rot'):
        axis, ang, origin = bx['rot']
        R = _rot_axis(axis, ang)
        o = np.array(origin, float)
        P = (P - o) @ R.T + o
        n = R @ n
    # facing: rotate about the block's vertical centre line
    if facing:
        R = _rot_axis('z', 90.0 * facing)
        o = np.array([8.0, 8.0, 0.0])
        P = (P - o) @ R.T + o
        n = R @ n
    u0, v0, u1, v1 = uv
    UV = np.array([(u0, v1), (u1, v1), (u1, v0), (u0, v0)], float) / 16.0
    return (P / 16.0).astype(np.float32), UV.astype(np.float32), n.astype(np.float32)


if __name__ == '__main__':
    import time
    w = World((64, 64, 32), (-32, -32, -16))
    w.fill(-32, -32, -16, 31, 31, -1, 'stone')
    w.fill(-32, -32, -1, 31, 31, -1, 'grass_block')
    w.fill(-3, -3, 0, 3, 3, 3, 'oak_planks')
    w.fill(-2, -2, 0, 2, 2, 3, 'air')
    w.set(0, 0, 0, 'torch')
    w.set(1, 1, 0, 'chest')
    w.set(-1, 1, 0, 'red_bed', BL.HALF_TOP)
    w.set(-1, 0, 0, 'red_bed', 0)
    for x in range(5, 12):
        w.set(x, 5, 0, 'oak_fence')
    w.set(8, 8, 0, 'tall_grass')
    t = time.time()
    m = w.mesh()
    for k, (V, I, S) in m.items():
        print(k, len(V), 'verts', len(I) // 3, 'tris', len(S), 'sections')
    t2 = time.time()
    L = w.light()
    print('mesh %.2fs light %.2fs' % (t2 - t, time.time() - t2), L[..., 1].max(), L[32, 32, 16:20, 1])
