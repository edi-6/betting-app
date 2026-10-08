"""The hidden picture: a Minecraft sunset. A creeper stands in a meadow of grass and flowers, turned a little so its
lit left side shows, looking straight out of the picture; behind it blocky hills with oak trees, purple mountains,
a square sun going down into them, pink clouds and the first stars. And, for a second watch: a tiny figure standing
in front of the sun with two white eyes.

The picture is painted at PPB pixels a block over the tank (x across from -HALF to HALF, z up from the floor), and
every marble takes the colour under its final resting place (colors()). The creeper is drawn by ray casting its voxel
model (creeper.py's skin, one voxel per pixel of the skin) in a three-quarter view.
"""
import numpy as np

import creeper as CR
import machine as M
from noise import fbm2d

PPB = 8                            # picture pixels a block
TOP = 66.0                         # the picture's height (blocks); the fill ends around 64
W_PX = int(2 * M.HALF * PPB)
H_PX = int(TOP * PPB)

HORIZON = 24.0
SUN = (-8.6, 30.0, 3.8)            # centre x, centre z, half size (a square sun, like the game's)
HERO = (-9.0, 25.4)                # the figure's feet: on the ridge, in front of the sun
CREEPER_X = 6.4                    # its middle
CREEPER_FEET = 7.0
CREEPER_PX = 1.72                  # blocks a skin pixel (26 pixels tall: about 53 blocks)

SKY = [(HORIZON, (255, 214, 120)), (29.0, (255, 166, 92)), (36.0, (246, 120, 96)), (44.0, (214, 92, 124)),
       (52.0, (138, 70, 140)), (59.0, (70, 46, 112)), (66.0, (36, 28, 78))]


def _grid():
    xs = (np.arange(W_PX) + 0.5) / PPB - M.HALF
    zs = TOP - (np.arange(H_PX) + 0.5) / PPB
    return np.meshgrid(xs, zs)


def _lerp_stops(z, stops):
    zs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(z, zs, cs[:, k]) for k in range(3)], -1)


def _blocky(x, scale, amp, seed, base, step=1.0):
    """A stepped skyline: the height of each 1-block column."""
    xc = np.floor(x) + 0.5
    h = base + amp * fbm2d(xc / scale, np.full_like(xc, 0.37 * seed), octaves=3, seed=seed)
    return np.floor(h / step) * step


# ---------------------------------------------------------------------------------------------
# the creeper, ray cast
# ---------------------------------------------------------------------------------------------
def creeper_sprite(px_per_vox=8, yaw=24.0, pitch=7.0, seed=5):
    """RGBA image (rows top to bottom) of the creeper in a three-quarter view, px_per_vox pixels a skin pixel, and
    a label image (0 none, 1 skin, 2 face (eyes and mouth)). Its feet are at the bottom row, centred."""
    skin = CR.make_skin(seed)
    lo = np.array([-4, -6, 0])
    dims = np.array([8, 12, 26])
    part = np.zeros(dims, np.int8)
    names = list(CR.PARTS)
    for k, n in enumerate(names):
        x0, y0, z0, sx, sy, sz = CR.PARTS[n]
        part[x0 - lo[0]:x0 - lo[0] + sx, y0 - lo[1]:y0 - lo[1] + sy, z0 - lo[2]:z0 - lo[2] + sz] = k + 1
    a, p = np.radians(yaw), np.radians(pitch)
    d = np.array([np.sin(a) * np.cos(p), np.cos(a) * np.cos(p), -np.sin(p)])
    right = np.cross(d, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    # the image covers the model's projected bounds
    corners = np.array([[x, y, z] for x in (-4, 4) for y in (-6, 6) for z in (0, 26)], float)
    pu, pv = corners @ right, corners @ up
    W = int(np.ceil((pu.max() - pu.min()) * px_per_vox)) + 2
    H = int(np.ceil((pv.max() - pv.min()) * px_per_vox)) + 2
    uu = pu.min() + (np.arange(W) + 0.5) / px_per_vox
    vv = pv.max() - (np.arange(H) + 0.5) / px_per_vox
    U, V = np.meshgrid(uu, vv)
    o = U[..., None] * right + V[..., None] * up - 40.0 * d
    o = o.reshape(-1, 3) - lo                      # grid coordinates
    n = len(o)
    step = np.sign(d).astype(int)
    tdelta = np.abs(1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d))
    vox = np.floor(o).astype(int)
    tmax = np.where(step > 0, (vox + 1 - o), (o - vox)) * tdelta
    hit = np.zeros(n, bool)
    hit_axis = np.full(n, -1)
    hit_vox = np.zeros((n, 3), int)
    last_axis = np.full(n, -1)
    for _ in range(200):
        inside = np.all((vox >= 0) & (vox < dims), axis=1)
        idx = np.where(inside & ~hit)[0]
        if len(idx):
            occ = part[vox[idx, 0], vox[idx, 1], vox[idx, 2]] > 0
            j = idx[occ]
            hit[j] = True
            hit_axis[j] = last_axis[j]
            hit_vox[j] = vox[j]
        if hit.all():
            break
        ax = np.argmin(tmax, axis=1)
        rows = np.arange(n)
        vox[rows, ax] += step[ax]
        tmax[rows, ax] += tdelta[ax]
        last_axis = ax
    rgba = np.zeros((n, 4))
    label = np.zeros(n, np.int8)
    light = np.array([-0.80, 0.45, 0.40])           # the low sun, behind it on the left
    light /= np.linalg.norm(light)
    j = np.where(hit)[0]
    hv = hit_vox[j] + lo                            # model coordinates of the voxel hit
    ax = hit_axis[j]
    pid = part[hit_vox[j, 0], hit_vox[j, 1], hit_vox[j, 2]] - 1
    col = np.zeros((len(j), 3))
    lab = np.ones(len(j), np.int8)
    nrm = np.zeros((len(j), 3))
    for k, name in enumerate(names):
        x0, y0, z0, sx, sy, sz = CR.PARTS[name]
        faces = skin[name]
        sel = pid == k
        for axis, face, nv in ((0, 'nx', (-1, 0, 0)), (1, 'ny', (0, -1, 0)), (2, 'pz', (0, 0, 1))):
            s2 = sel & (ax == axis)
            if not s2.any():
                continue
            v = hv[s2]
            if face == 'ny':
                r_, c_ = (z0 + sz - 1) - v[:, 2], v[:, 0] - x0
            elif face == 'nx':
                r_, c_ = (z0 + sz - 1) - v[:, 2], (y0 + sy - 1) - v[:, 1]
            else:
                r_, c_ = (y0 + sy - 1) - v[:, 1], v[:, 0] - x0
            img = faces[face]
            col[s2] = img[np.clip(r_, 0, img.shape[0] - 1), np.clip(c_, 0, img.shape[1] - 1)]
            nrm[s2] = nv
            if name == 'head' and face == 'ny':
                fr = CR.FACE_ROWS
                e = np.array([fr[int(np.clip(rr, 0, 7))][int(np.clip(cc, 0, 7))] == 'E' for rr, cc in zip(r_, c_)])
                lab[np.where(s2)[0][e]] = 2
    ndl = np.clip(nrm @ light, 0.0, 1.0)
    warm = np.array([1.30, 0.98, 0.72])
    cool = np.array([0.66, 0.70, 0.86])
    shade = cool[None, :] * 0.86 + warm[None, :] * ndl[:, None] * 0.5
    # the front face reads clearly: a little extra fill light on it
    shade += (ax == 1)[:, None] * np.array([0.13, 0.13, 0.11])
    col = col * shade
    rgba[j, :3] = col
    rgba[j, 3] = 1.0
    label[j] = lab
    return rgba.reshape(H, W, 4), label.reshape(H, W)


# ---------------------------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------------------------
def paint(seed=3, star=None):
    """RGB float image (H_PX, W_PX, 3), rows from the top (z = TOP) down, and masks: creeper, face, figure.
    star: (x, z) of the bright star at the top (where the last marble ends up)."""
    X, Z = _grid()
    rng = np.random.default_rng(seed)
    img = _lerp_stops(Z, SKY)
    # stars in the top of the sky
    st = (rng.random(Z.shape) < 0.0016) & (Z > 52.0)
    img[st] = np.maximum(img[st], (230, 220, 255))
    if star is not None:
        sx_, sz_ = star
        dx, dz = np.abs(X - sx_), np.abs(Z - sz_)
        arms = ((dx < 0.12) & (dz < 1.1)) | ((dz < 0.12) & (dx < 1.1))
        img[arms] = img[arms] * 0.7 + np.array([200, 170, 255]) * 0.3
        img[np.hypot(dx, dz) < 0.32] = (255, 244, 196)
    # the square sun, with a glow
    sx, sz, sh = SUN
    dist = np.maximum(np.abs(X - sx), np.abs(Z - sz))
    glow = np.exp(-np.maximum(np.hypot(X - sx, Z - sz) - sh, 0.0) / 6.0)
    img = img + (np.array([255, 190, 110]) - img) * (0.55 * glow)[..., None]
    sun = dist < sh
    core = dist < sh * 0.62
    img[sun] = (255, 222, 120)
    img[core] = (255, 246, 196)
    # blocky clouds, lit pink from below
    for (cx, cz, w, h) in ((-12.0, 47.0, 9.0, 1.4), (-4.0, 49.5, 7.0, 1.2), (9.0, 55.0, 11.0, 1.3),
                           (-14.0, 56.5, 6.0, 1.0), (14.5, 44.0, 5.0, 1.1)):
        cl = (np.abs(X - cx) < w / 2) & (np.abs(Z - cz) < h / 2)
        lip = cl & (Z < cz - h / 2 + 0.5)
        img[cl] = img[cl] * 0.35 + np.array([255, 196, 206]) * 0.65
        img[lip] = img[lip] * 0.3 + np.array([255, 160, 120]) * 0.7
    # purple mountains far away (in front of the sun's lower part)
    mtn = _blocky(X, 7.0, 7.0, 11, HORIZON + 1.5)
    m = Z < mtn
    haze = np.clip((Z - (HORIZON - 4)) / 10.0, 0, 1)[..., None]
    img[m] = (np.array([112, 82, 150]) * (1 - haze) + np.array([170, 110, 150]) * haze)[m]
    # the figure: on the ridge in front of the sun, dark, two white eyes
    hx, hz = HERO
    fig = np.zeros_like(X, bool)
    fig |= (np.abs(X - (hx + 0.0)) < 0.55) & (Z >= hz) & (Z < hz + 2.0)          # legs and body
    fig |= (np.abs(X - hx) < 0.62) & (Z >= hz + 2.0) & (Z < hz + 3.2)            # head
    fig |= (np.abs(X - hx) < 0.95) & (Z >= hz + 1.0) & (Z < hz + 2.0)            # arms
    img[fig] = (40, 30, 44)
    eyes = ((np.abs(X - (hx - 0.27)) < 0.2) | (np.abs(X - (hx + 0.27)) < 0.2)) & (np.abs(Z - (hz + 2.62)) < 0.2)
    img[eyes] = (255, 255, 255)
    # hills with oak trees, blue-green in the evening light
    hill = _blocky(X, 9.0, 4.0, 23, HORIZON - 3.5)
    hm = Z < hill
    img[hm] = (np.array([58, 96, 84]) * (1 - 0.35 * np.clip((hill - Z) / 8, 0, 1))[..., None])[hm]
    for tx in (-15.0, -11.5, -6.0, 1.0, 7.5, 12.5, 15.5):
        base = float(_blocky(np.array([tx]), 9.0, 4.0, 23, HORIZON - 3.5)[0])
        trunk = (np.abs(X - tx) < 0.5) & (Z >= base - 0.5) & (Z < base + 2.0)
        crown = (np.abs(X - tx) < 2.0) & (Z >= base + 1.6) & (Z < base + 4.2)
        crown |= (np.abs(X - tx) < 1.0) & (Z >= base + 4.2) & (Z < base + 5.0)
        img[trunk] = (70, 54, 48)
        img[crown] = (44, 84, 58)
    # the meadow, in two steps, with grass blocks' fringe and flowers
    near = _blocky(X, 6.0, 1.5, 31, 10.0)
    gm = Z < near
    tex = 0.86 + 0.14 * fbm2d(np.floor(X * 2) / 3.0, np.floor(Z * 2) / 3.0, octaves=2, seed=41)
    grass = np.array([96, 168, 62]) * tex[..., None]
    img[gm] = grass[gm] * np.clip(0.78 + 0.22 * (Z / 10.0), 0.6, 1.0)[gm][:, None]
    flowers = rng.random(Z.shape) < 0.012
    flowers &= gm & (Z < near - 0.8)
    fk = rng.integers(3, size=Z.shape)
    for k, c in enumerate(((228, 40, 36), (250, 214, 40), (90, 110, 240))):
        f = flowers & (fk == k)
        # a flower is a little cross of 3 x 3 picture pixels
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx and dy:
                    continue
                img[np.roll(np.roll(f, dy, 0), dx, 1)] = c
    # the creeper (and its long shadow on the grass, towards the right)
    spr, lab = creeper_sprite(px_per_vox=int(round(CREEPER_PX * PPB)))
    h, w = spr.shape[:2]
    x0 = int(round((CREEPER_X + M.HALF) * PPB - w / 2))
    y0 = int(round((TOP - CREEPER_FEET) * PPB - h))
    shadow = np.zeros_like(X, bool)
    shadow |= (Z < CREEPER_FEET + 0.8) & (Z > CREEPER_FEET - 2.2) & (X > CREEPER_X - 2) & \
        (X < CREEPER_X + 14.0 - (CREEPER_FEET - Z) * 1.5)
    img[shadow & gm] *= np.array([0.62, 0.62, 0.78])
    creeper = np.zeros_like(X, bool)
    face = np.zeros_like(X, bool)
    ys, xs = np.where(spr[..., 3] > 0.5)
    yy, xx = ys + y0, xs + x0
    ok = (yy >= 0) & (yy < H_PX) & (xx >= 0) & (xx < W_PX)
    img[yy[ok], xx[ok]] = spr[ys[ok], xs[ok], :3]
    creeper[yy[ok], xx[ok]] = True
    face[yy[ok], xx[ok]] = lab[ys[ok], xs[ok]] == 2
    return np.clip(img, 0, 255), {'creeper': creeper, 'face': face, 'figure': fig | eyes}


def sample(img, x, z):
    """The picture's colour at block coordinates (x across the tank, z up from its floor)."""
    c = np.clip(((np.asarray(x) + M.HALF) * PPB).astype(int), 0, W_PX - 1)
    r = np.clip(((TOP - np.asarray(z)) * PPB).astype(int), 0, H_PX - 1)
    return img[r, c]


def colors(final_xz, star=None):
    """Each marble's colour (uint8 RGB) from where it ends up, and its role: 0 background, 1 creeper, 2 its face,
    3 the figure."""
    img, masks = paint(star=star)
    x, z = final_xz[:, 0], final_xz[:, 1]
    col = sample(img, x, z)
    role = np.zeros(len(x), np.int8)
    role[sample(masks['creeper'].astype(np.int8), x, z) > 0] = 1
    role[sample(masks['face'].astype(np.int8), x, z) > 0] = 2
    role[sample(masks['figure'].astype(np.int8), x, z) > 0] = 3
    return np.clip(col, 0, 255).astype(np.uint8), role


if __name__ == '__main__':
    from PIL import Image
    img, masks = paint()
    Image.fromarray(img.astype(np.uint8)).save('/tmp/picture.png')
    print(img.shape)
