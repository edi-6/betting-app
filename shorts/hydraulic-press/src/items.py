"""Voxel models of what goes under the press: a Minecraft block is 16 x 16 x 16 voxels (one per texture pixel),
each surface voxel carrying the colour of its texture pixel on every outward face and an 'inner' colour for
faces exposed when it breaks. Also the press parts that can break (ram plate and rod) and the chest's loot
(extruded 16 x 16 item sprites, the way the game draws items)."""
import numpy as np

import pixelart as PA

VOXEL_DTYPE = np.dtype([('pos', 'f4', 3), ('quat', 'f4', 4), ('scale', 'f4'), ('cx', 'u1', 4),
                        ('cy', 'u1', 4), ('cz', 'u1', 4), ('inner', 'u1', 4)])

B = 4.0                      # a block is 4 world units
RES = 16
VS = B / RES                 # voxel edge


class VoxelSet:
    """A fixed set of voxels: rest positions, colours, face masks, part/chunk ids."""

    def __init__(self, ijk, origin, vs, cx, cy, cz, inner, mask, name):
        self.ijk = ijk                        # (N, 3) grid indices
        self.origin = np.asarray(origin, float)
        self.vs = vs
        self.pos = self.origin + (ijk + 0.5) * vs
        self.cx, self.cy, self.cz, self.inner = cx, cy, cz, inner
        self.mask = mask                      # bit per face (px nx py ny pz nz) that is on the skin
        self.name = name
        self.n = len(ijk)
        self.crack = np.full(self.n, 2.0)     # crack stage at which the voxel's skin darkens
        self.base_cols = (cx.copy(), cy.copy(), cz.copy())
        self.glow = np.zeros(self.n, bool)

    def visibility(self, alive=None, group=None):
        """Face bits (px nx py ny pz nz) of each voxel that are not covered by a neighbour: an alive voxel of the
        same group (chunk) covers a face; voxels of other chunks don't (the cracks show the inside)."""
        alive = np.ones(self.n, bool) if alive is None else alive
        grp = np.zeros(self.n, np.int64) if group is None else np.asarray(group)
        lo = self.ijk.min(0) - 1
        shape = self.ijk.max(0) - lo + 2
        key = np.zeros(tuple(shape), np.int64)
        q = self.ijk - lo
        a = q[alive]
        key[a[:, 0], a[:, 1], a[:, 2]] = grp[alive] + 1
        vis = np.zeros(self.n, np.uint8)
        me = grp + 1
        for bit, d in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            nb = q + np.array(d)
            vis |= (key[nb[:, 0], nb[:, 1], nb[:, 2]] != me).astype(np.uint8) << bit
        return vis

    def instances(self, sel=None, vis=None):
        s = np.arange(self.n) if sel is None else sel
        if vis is None:
            vis = self.visibility()
        out = np.zeros(len(s), VOXEL_DTYPE)
        out['pos'] = self.pos[s]
        out['quat'] = (0, 0, 0, 1)
        out['scale'] = self.vs
        out['cx'][:, :3] = self.cx[s]
        out['cx'][:, 3] = self.mask[s]
        out['cy'][:, 3] = vis[s]
        out['cy'][:, :3] = self.cy[s]
        out['cz'][:, :3] = self.cz[s]
        out['cz'][:, 3] = np.where(self.glow[s], 255, 0)
        out['inner'][:, :3] = self.inner[s]
        out['inner'][:, 3] = 0
        return out

    def set_crack(self, stage, dark=0.32):
        """Minecraft's destroy stages: skin pixels whose order is below the stage turn into dark cracks."""
        c = self.crack < stage
        for arr, base in zip((self.cx, self.cy, self.cz), self.base_cols):
            arr[:] = base
            arr[c] = (base[c] * dark).astype(np.uint8)


def _face_pixel(face, u, v):
    return face[np.clip(v, 0, RES - 1), np.clip(u, 0, RES - 1)]


def make_block(kind, items, center_xy=(0.0, 0.0), z0=0.0, seed=1, hollow=0):
    """Voxelise one of the test blocks, sitting on z0, centred on center_xy."""
    it = items[kind]
    top, side, bottom = it['faces'][:3]
    front = it['faces'][3] if len(it['faces']) > 3 else side
    rng = np.random.default_rng(seed)
    g = np.arange(RES)
    I, J, K = np.meshgrid(g, g, g, indexing='ij')
    occ = np.ones((RES, RES, RES), bool)
    if hollow:
        h = hollow
        occ[h:-h, h:-h, h:-h] = False
    i, j, k = I[occ], J[occ], K[occ]
    n = len(i)
    cx = np.zeros((n, 3), np.uint8)
    cy = np.zeros((n, 3), np.uint8)
    cz = np.zeros((n, 3), np.uint8)
    mask = np.zeros(n, np.uint8)
    crack = np.full(n, 2.0)
    order = PA.crack_order(seed + 3)
    # x faces (side textures), u along the face, v down from the top
    for sel, img, u, bit in ((i == RES - 1, side, RES - 1 - j, 0), (i == 0, side, j, 1)):
        cx[sel] = _face_pixel(img, u[sel], (RES - 1 - k)[sel])
        mask[sel] |= 1 << bit
        crack[sel] = np.minimum(crack[sel], order[(RES - 1 - k)[sel], u[sel]])
    # y faces: -y is the front (towards the camera)
    for sel, img, u, bit in ((j == RES - 1, side, i, 2), (j == 0, front, i, 3)):
        cy[sel] = _face_pixel(img, u[sel], (RES - 1 - k)[sel])
        mask[sel] |= 1 << bit
        crack[sel] = np.minimum(crack[sel], order[(RES - 1 - k)[sel], u[sel]])
    for sel, img, bit in ((k == RES - 1, top, 4), (k == 0, bottom, 5)):
        cz[sel] = _face_pixel(img, i[sel], (RES - 1 - j)[sel])
        mask[sel] |= 1 << bit
        crack[sel] = np.minimum(crack[sel], order[(RES - 1 - j)[sel], i[sel]])
    if 'alpha' in it:
        # see-through (glass): only the voxels under an opaque pixel of one of their faces exist
        al = it['alpha']
        keep = np.zeros(n, bool)
        for sel, u, v in ((i == RES - 1, RES - 1 - j, RES - 1 - k), (i == 0, j, RES - 1 - k),
                          (j == RES - 1, i, RES - 1 - k), (j == 0, i, RES - 1 - k),
                          (k == RES - 1, i, RES - 1 - j), (k == 0, i, RES - 1 - j)):
            keep[sel] |= al[v[sel], u[sel]]
        i, j, k = i[keep], j[keep], k[keep]
        cx, cy, cz, mask, crack = cx[keep], cy[keep], cz[keep], mask[keep], crack[keep]
        # faces that now look inwards show the texture too
        mask[:] = 0b111111
        cx[cx.sum(1) == 0] = (206, 232, 240)
        cy[cy.sum(1) == 0] = (206, 232, 240)
        cz[cz.sum(1) == 0] = (206, 232, 240)
        n = len(i)
    pal = np.array(it['inner'], float)
    inner = pal[rng.choice(len(pal), size=n, p=it['inner_p'])] * (1 + (rng.random((n, 1)) - 0.5) * 0.1)
    if 'rind' in it:
        d = np.minimum.reduce([i, RES - 1 - i, j, RES - 1 - j, k, RES - 1 - k])
        inner[d <= 1] = np.array(it['rind']) * (1 + (rng.random(((d <= 1).sum(), 1)) - 0.5) * 0.08)
    inner = np.clip(inner, 0, 255).astype(np.uint8)
    origin = np.array([center_xy[0] - B / 2, center_xy[1] - B / 2, z0])
    vox = VoxelSet(np.stack([i, j, k], -1), origin, VS, cx, cy, cz, inner, mask, kind)
    vox.crack = crack
    return vox


def make_box(name, size, origin, faces, vs=VS, inner=(90, 92, 98), seed=2):
    """A textured box of voxels (press parts): faces = (top, side, bottom) 16x16 textures, tiled every 4 units."""
    nx, ny, nz = (int(round(s / vs)) for s in size)
    g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    i, j, k = (a.ravel() for a in g)
    surf = (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1) | (k == 0) | (k == nz - 1)
    i, j, k = i[surf], j[surf], k[surf]
    n = len(i)
    top, side, bottom = faces
    cx = np.zeros((n, 3), np.uint8)
    cy = np.zeros((n, 3), np.uint8)
    cz = np.zeros((n, 3), np.uint8)
    mask = np.zeros(n, np.uint8)
    for sel, bit in ((i == nx - 1, 0), (i == 0, 1)):
        cx[sel] = _face_pixel(side, j[sel] % RES, (nz - 1 - k[sel]) % RES)
        mask[sel] |= 1 << bit
    for sel, bit in ((j == ny - 1, 2), (j == 0, 3)):
        cy[sel] = _face_pixel(side, i[sel] % RES, (nz - 1 - k[sel]) % RES)
        mask[sel] |= 1 << bit
    for sel, img, bit in ((k == nz - 1, top, 4), (k == 0, bottom, 5)):
        cz[sel] = _face_pixel(img, i[sel] % RES, (ny - 1 - j[sel]) % RES)
        mask[sel] |= 1 << bit
    rng = np.random.default_rng(seed)
    inn = np.clip(np.array(inner, float) * (1 + (rng.random((n, 1)) - 0.5) * 0.12), 0, 255).astype(np.uint8)
    return VoxelSet(np.stack([i, j, k], -1), origin, vs, cx, cy, cz, inn, mask, name)


def make_sprite_item(name, scale=1.6):
    """An item as the game draws it: its 16x16 sprite extruded one pixel thick (standing in the x-z plane)."""
    spr = PA.sprite(name)
    vs = scale / RES
    rr, cc = np.nonzero(spr[..., 3] > 0)
    n = len(rr)
    col = spr[rr, cc, :3]
    i = cc
    k = RES - 1 - rr
    j = np.zeros(n, int)
    dark = (col.astype(float) * 0.7).astype(np.uint8)
    mask = np.full(n, 0b001100, np.uint8)             # the flat faces are skin
    filled = spr[..., 3] > 0
    for bit, (dr, dc) in ((0, (0, 1)), (1, (0, -1)), (4, (-1, 0)), (5, (1, 0))):
        r2, c2 = rr + dr, cc + dc
        inb = (r2 >= 0) & (r2 < RES) & (c2 >= 0) & (c2 < RES)
        edge = np.ones(n, bool)
        edge[inb] = ~filled[r2[inb], c2[inb]]
        mask[edge] |= 1 << bit
    origin = np.array([-scale / 2, -vs / 2, 0.0])
    vox = VoxelSet(np.stack([i, j, k], -1), origin, vs, dark, col, dark, dark, mask, name)
    return vox


def voronoi_chunks(vox, n_chunks, rng, jitter=0.35, sel=None):
    """Split voxels into rigid chunks (Voronoi cells around random seeds, with a little noise so the cracks are
    ragged). Returns a chunk id per voxel (only for the selected voxels if given)."""
    idx = np.arange(vox.n) if sel is None else sel
    p = vox.pos[idx]
    lo, hi = p.min(0), p.max(0)
    seeds = lo + (hi - lo) * rng.random((n_chunks, 3))
    d = np.linalg.norm(p[:, None, :] - seeds[None, :, :], axis=-1)
    d *= 1.0 + jitter * rng.random(d.shape)
    return np.argmin(d, axis=1)
