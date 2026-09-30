"""Procedural pixel art for the film: 16x16 block textures (RGBA, alpha = cut-out), their emission masks, item icons
and the block-breaking cracks. Everything is drawn here from noise and a few hand-placed pixels; no image assets.

make_all() -> (textures: dict name -> (16,16,4) uint8, emission: dict name -> (16,16) float 0..1)
item_icons() -> dict name -> (16,16,4) uint8
"""
import numpy as np

N = 16


def _rng(seed):
    return np.random.default_rng(seed)


def speckle(base, var, rng, coarse=0.4, n=N):
    fine = rng.random((n, n)) - 0.5
    blob = np.kron(rng.random((n // 4, n // 4)) - 0.5, np.ones((4, 4)))
    k = fine * (1 - coarse) + blob * coarse
    return np.clip(np.array(base, float)[None, None, :] * (1 + var * 2 * k[..., None]), 0, 255)


def rgba(rgb, alpha=None):
    out = np.zeros((N, N, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = 255 if alpha is None else alpha
    return out


def shade_px(img, rng, amount=0.06):
    j = 1.0 + (rng.random(img.shape[:2]) - 0.5) * 2 * amount
    return np.clip(img * j[..., None], 0, 255)


def quant(img, step=6):
    """Snap colours to a coarse grid so textures read as hand-picked palettes, not smooth noise."""
    return np.round(img / step) * step


# ---------------------------------------------------------------------------------------------
# natural blocks
# ---------------------------------------------------------------------------------------------
def grass_top(rng):
    g = speckle((104, 150, 62), 0.13, rng, 0.3)
    d = rng.random((N, N))
    g[d < 0.10] *= 0.82
    g[d > 0.93] *= 1.12
    return quant(g, 4)


def dirt(rng):
    d = speckle((128, 92, 64), 0.11, rng, 0.3)
    r = rng.random((N, N))
    d[r < 0.10] = (100, 70, 48)
    d[r > 0.92] = (152, 112, 80)
    d[(r > 0.5) & (r < 0.53)] = (116, 116, 112)       # a few pebbles
    return quant(d, 4)


def grass_side(rng, top, dirt_):
    s = dirt_.copy()
    lip = 3 + (rng.random(N) < 0.5).astype(int) + (rng.random(N) < 0.25).astype(int)
    for c in range(N):
        s[:lip[c], c] = top[:lip[c], c] * 0.96
        if rng.random() < 0.3:
            s[lip[c], c] = top[lip[c] - 1, c] * 0.8
    return s


def dirt_path(rng):
    p = speckle((148, 122, 74), 0.10, rng, 0.35)
    r = rng.random((N, N))
    p[r < 0.12] = (124, 100, 60)
    p[r > 0.94] = (170, 144, 92)
    return quant(p, 4)


def farmland(rng):
    f = speckle((82, 56, 36), 0.12, rng, 0.2)
    for r in range(N):
        if r % 4 == 0:
            f[r] *= 0.72
    return quant(f, 4)


def stone(rng, base=(126, 126, 128)):
    s = speckle(base, 0.09, rng, 0.55)
    r = rng.random((N, N))
    s[r < 0.07] *= 0.8
    s[r > 0.95] *= 1.15
    return quant(s, 4)


def cobble(rng, base=(122, 122, 122), moss=False):
    """Rounded stones separated by dark mortar (a small Voronoi)."""
    pts = rng.uniform(0, N, (9, 2))
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    d = []
    for (px, py) in pts:
        dx = np.minimum(abs(xx - px), N - abs(xx - px))
        dy = np.minimum(abs(yy - py), N - abs(yy - py))
        d.append(np.hypot(dx, dy))
    d = np.stack(d)
    o = np.sort(d, 0)
    idx = np.argmin(d, 0)
    edge = (o[1] - o[0]) < 1.1
    shades = rng.uniform(0.78, 1.18, len(pts))
    img = np.array(base, float)[None, None, :] * shades[idx][..., None]
    # light top-left of each stone, darker bottom-right
    img *= (1.0 + 0.10 * np.clip(1.5 - o[0], 0, 1.5))[..., None]
    img[edge] = np.array(base) * 0.52
    img = shade_px(img, rng, 0.06)
    if moss:
        m = rng.random((N, N)) < 0.35
        blob = np.kron(rng.random((4, 4)) < 0.55, np.ones((4, 4), bool))
        mm = m & blob
        img[mm] = speckle((82, 112, 52), 0.1, rng)[mm]
    return quant(img, 4)


def stone_bricks(rng, base=(122, 122, 124), moss=False, cracked=False):
    img = speckle(base, 0.06, rng, 0.5)
    for r in range(N):
        for c in range(N):
            row = r // 4
            off = 0 if row % 2 == 0 else 4
            if r % 4 == 3 or (c + off) % 8 == 7:
                img[r, c] = np.array(base) * 0.6
            elif r % 4 == 0 or (c + off) % 8 == 0:
                img[r, c] *= 1.1
    if moss:
        m = (rng.random((N, N)) < 0.4) & np.kron(rng.random((4, 4)) < 0.5, np.ones((4, 4), bool))
        img[m] = speckle((78, 106, 50), 0.1, rng)[m]
    if cracked:
        x, y = rng.integers(2, 13), 0
        while y < N:
            img[y, x] = np.array(base) * 0.45
            y += 1
            x = int(np.clip(x + rng.integers(-1, 2), 0, N - 1))
    return quant(img, 4)


def gravel(rng):
    g = speckle((132, 124, 120), 0.18, rng, 0.1)
    r = rng.random((N, N))
    g[r < 0.18] = (96, 88, 86)
    g[r > 0.85] = (162, 156, 152)
    return quant(g, 4)


def sand(rng):
    return quant(speckle((214, 200, 152), 0.06, rng, 0.3), 4)


def deepslate(rng, top=False):
    d = speckle((78, 78, 84), 0.10, rng, 0.4)
    if not top:
        for r in range(0, N, 3):
            d[r] *= 0.86                       # layered look on the sides
    return quant(d, 4)


def deepslate_tiles(rng):
    img = speckle((62, 62, 68), 0.07, rng, 0.5)
    for r in range(N):
        for c in range(N):
            if r % 8 == 7 or c % 8 == 7:
                img[r, c] = (36, 36, 40)
    return quant(img, 4)


def obsidian(rng):
    o = speckle((24, 16, 36), 0.25, rng, 0.2)
    r = rng.random((N, N))
    o[r > 0.93] = (70, 44, 100)
    return quant(o, 3)


def log_side(rng, bark=(106, 82, 50), dark=(78, 58, 34)):
    img = np.zeros((N, N, 3))
    cols = rng.random(N)
    for c in range(N):
        img[:, c] = bark if cols[c] > 0.35 else dark
    img = shade_px(img, rng, 0.09)
    for _ in range(5):
        c, r = rng.integers(0, N), rng.integers(0, N - 3)
        img[r:r + rng.integers(2, 5), c] = np.array(dark) * 0.8
    return quant(img, 4)


def log_top(rng, ring=(170, 136, 86), ring2=(146, 114, 70), bark=(96, 74, 44)):
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.maximum(abs(xx - 7.5), abs(yy - 7.5))
    img = np.where((rr.astype(int) % 3 == 0)[..., None], ring2, ring).astype(float)
    img[rr >= 7] = bark
    return quant(shade_px(img, rng, 0.05), 4)


def birch_side(rng):
    img = speckle((214, 212, 204), 0.04, rng, 0.2)
    for _ in range(7):
        r, c = rng.integers(0, N), rng.integers(0, N - 3)
        img[r, c:c + rng.integers(2, 5)] = (44, 44, 40)
    return quant(img, 4)


def leaves(rng, col=(64, 122, 44), holes=0.18):
    lv = speckle(col, 0.22, rng, 0.25)
    r = rng.random((N, N))
    lv[r < 0.18] = np.array(col) * 0.55
    lv[r > 0.88] = np.array(col) * 1.3
    alpha = np.where(rng.random((N, N)) < holes, 0, 255).astype(np.uint8)
    return quant(lv, 4), alpha


def planks(rng, base=(160, 128, 78), line=None):
    line = line if line is not None else np.array(base) * 0.66
    img = speckle(base, 0.07, rng, 0.2)
    img *= (1 + 0.1 * (rng.random((N, 1, 1)) - 0.5))
    for r in range(N):
        if r % 4 == 3:
            img[r] = line
    for band in range(4):
        c = (band * 5 + rng.integers(0, 16)) % N
        img[band * 4:band * 4 + 3, c] = line * 1.08
    grain = rng.random((N, N)) < 0.08
    img[grain] *= 0.88
    return quant(img, 4)


def glass(rng):
    img = np.full((N, N, 3), 220.0)
    a = np.zeros((N, N), np.uint8)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 255
    img[0, :] = img[:, 0] = (236, 244, 248)
    img[-1, :] = img[:, -1] = (178, 196, 206)
    for k in range(3):                         # glint streaks
        a[3 + k, 2 + k] = 150
        a[4 + k, 2 + k] = 120
    a[11, 12] = a[12, 11] = 140
    img[a == 150] = img[a == 120] = img[a == 140] = (240, 248, 255)
    return img, a


def wool(rng, col):
    w = speckle(col, 0.07, rng, 0.2)
    for r in range(N):
        for c in range(N):
            if (r + c * 3) % 5 == 0:
                w[r, c] *= 0.93
    return quant(w, 3)


def concrete(rng, col):
    return quant(speckle(col, 0.025, rng, 0.5), 2)


def water(rng):
    w = speckle((52, 86, 168), 0.10, rng, 0.45)
    for r in range(N):
        for c in range(N):
            if (r * 2 + c) % 7 == 0:
                w[r, c] *= 1.12
    return quant(w, 3)


def hay(rng):
    top = speckle((186, 150, 40), 0.10, rng, 0.3)
    side = speckle((196, 160, 44), 0.09, rng, 0.2)
    for c in range(N):
        if c % 3 == 0:
            side[:, c] *= 0.85
    side[4:6] = (120, 70, 30)
    side[10:12] = (120, 70, 30)
    return quant(top, 4), quant(side, 4)


def pumpkin(rng):
    side = speckle((206, 120, 24), 0.08, rng, 0.3)
    for c in range(N):
        if c % 4 == 0:
            side[:, c] *= 0.82
    top = speckle((200, 116, 24), 0.08, rng)
    top[6:10, 6:10] = (90, 110, 40)
    return quant(side, 4), quant(top, 4)


# ---------------------------------------------------------------------------------------------
# furniture
# ---------------------------------------------------------------------------------------------
def bookshelf(rng, planks_):
    img = planks_.copy()
    spines = [(142, 38, 32), (46, 70, 140), (48, 110, 52), (156, 116, 44), (104, 50, 110), (70, 60, 50),
              (164, 150, 120), (40, 40, 44)]
    for top in (1, 9):
        img[top:top + 6, 1:15] = (38, 26, 16)
        c = 1
        while c < 15:
            w = int(rng.integers(1, 3))
            h = int(rng.integers(4, 7))
            col = np.array(spines[rng.integers(len(spines))], float) * rng.uniform(0.85, 1.1)
            img[top + 6 - h:top + 6, c:c + w] = col
            img[top + 6 - h, c:c + w] = col * 1.2
            if rng.random() < 0.5:
                img[top + 6 - h + 2, c:c + w] = col * 0.6       # a band on the spine
            c += w
    img[7:9] = planks_[7:9] * 0.9
    return quant(img, 4)


def crafting_table(rng, planks_):
    top = planks_.copy() * 1.02
    top[0, :] = top[-1, :] = top[:, 0] = top[:, -1] = (84, 60, 36)
    top[3:13, 3:13] = speckle((150, 110, 66), 0.08, rng)[3:13, 3:13]
    for k in range(3, 13, 3):
        top[k, 3:13] = (98, 70, 42)
        top[3:13, k] = (98, 70, 42)
    side = planks_.copy()
    side[:3] = (110, 80, 50)
    side[1, 3:6] = (170, 170, 176)           # a saw on the side
    side[2, 3:10] = (140, 140, 146)
    side[4:14, 10:13] = (90, 64, 38)
    side[4:6, 9:14] = (150, 150, 156)        # a hammer head
    front = planks_.copy()
    front[:3] = (110, 80, 50)
    front[5:14, 3] = (80, 56, 34)
    front[5:14, 12] = (80, 56, 34)
    front[6:8, 5:11] = (150, 150, 156)       # tongs
    return quant(top, 4), quant(side, 4), quant(front, 4)


def furnace(rng, lit):
    side = cobble(rng, (116, 116, 118))
    top = stone(rng, (112, 112, 114))
    front = cobble(rng, (116, 116, 118))
    front[0:2, :] = (92, 92, 94)
    front[2:6, 3:13] = (56, 56, 58)          # the upper slot
    front[3, 4:12] = (40, 40, 42)
    front[8:15, 3:13] = (34, 34, 36)         # the fire box
    front[8, 3:13] = (70, 70, 72)
    emit = np.zeros((N, N))
    if lit:
        fire = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 1], [1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                         [1, 2, 3, 2, 1, 2, 3, 2, 2, 1], [1, 2, 3, 3, 2, 3, 3, 3, 2, 1], [2, 3, 3, 3, 3, 3, 3, 3, 3, 2]])
        pal = {1: (180, 60, 10), 2: (244, 140, 30), 3: (255, 222, 110)}
        for r in range(6):
            for c in range(10):
                v = fire[r, c]
                if v:
                    front[9 + r, 3 + c] = pal[v]
                    emit[9 + r, 3 + c] = 0.45 + 0.18 * v
    return quant(front, 2), quant(side, 4), quant(top, 4), emit


def chest(rng):
    wood = (160, 108, 40)
    front = speckle(wood, 0.07, rng, 0.2)
    for r in range(N):
        if r % 3 == 1:
            front[r] *= 0.9
    front[0, :] = front[-1, :] = front[:, 0] = front[:, -1] = (72, 44, 16)
    front[5, :] = (72, 44, 16)                            # the lid's edge
    front[4:9, 7:9] = (190, 190, 196)                      # the latch
    front[8, 7:9] = (120, 120, 126)
    side = front.copy()
    side[4:9, 7:9] = speckle(wood, 0.07, rng)[4:9, 7:9]
    top = speckle(wood, 0.07, rng, 0.2)
    top[0, :] = top[-1, :] = top[:, 0] = top[:, -1] = (72, 44, 16)
    return quant(front, 4), quant(side, 4), quant(top, 4)


def door(rng, planks_, part):
    img = planks_.copy()
    img[:, 0] = img[:, -1] = np.array((96, 70, 40))
    if part == 'top':
        img[0, :] = (96, 70, 40)
        img[2:7, 3:7] = (56, 40, 24)                   # two window panes (dark: see-through feel)
        img[2:7, 9:13] = (56, 40, 24)
        a = np.full((N, N), 255, np.uint8)
        a[3:6, 4:6] = 0
        a[3:6, 10:12] = 0
        return quant(img, 4), a
    img[-1, :] = (96, 70, 40)
    img[3:5, 11:13] = (60, 60, 64)                     # handle
    return quant(img, 4), None


def trapdoor(rng, planks_):
    img = planks_.copy()
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (96, 70, 40)
    a = np.full((N, N), 255, np.uint8)
    for (r, c) in ((3, 3), (3, 12), (12, 3), (12, 12)):
        a[r:r + 2, c:c + 2] = 0
    return quant(img, 4), a


def ladder(rng):
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    for c in (2, 3, 12, 13):
        img[:, c] = (120, 88, 48)
        a[:, c] = 255
    for r in (1, 5, 9, 13):
        img[r:r + 2, 2:14] = (140, 104, 58)
        a[r:r + 2, 2:14] = 255
    return quant(shade_px(img, rng, 0.08), 4), a


def torch(rng, soul=False):
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    emit = np.zeros((N, N))
    img[6:16, 7:9] = (110, 80, 42)
    img[6:16, 7] = (128, 96, 52)
    a[6:16, 7:9] = 255
    flame = [(255, 244, 180), (255, 214, 90), (244, 150, 40)] if not soul else \
        [(200, 255, 255), (90, 230, 240), (40, 160, 200)]
    img[5:7, 7:9] = flame[1]
    img[5, 7] = flame[0]
    img[4, 7:9] = flame[2]
    a[4:7, 7:9] = 255
    emit[4:7, 7:9] = 1.0
    img[6, 7:9] = (60, 44, 24)                    # coal band under the flame
    emit[6, 7:9] = 0.0
    return img, a, emit


def lantern(rng, soul=False):
    """Laid out for the lantern model: frame top (rows 0-1), glass body (2-8), base (9-10), handle (12-15)."""
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    emit = np.zeros((N, N))
    iron = (60, 62, 70)
    img[0:2, 5:11] = iron
    a[0:2, 5:11] = 255
    glow = (255, 206, 120) if not soul else (120, 230, 240)
    glow2 = (255, 170, 70) if not soul else (60, 190, 220)
    img[2:9, 5:11] = glow
    img[2:9, 5] = img[2:9, 10] = iron
    img[5, 6:10] = glow2
    a[2:9, 5:11] = 255
    emit[2:9, 6:10] = 1.0
    img[9:11, 5:11] = iron
    a[9:11, 5:11] = 255
    img[12:16, 7:9] = (44, 46, 52)
    a[12:16, 7:9] = 255
    return img, a, emit


def chain(rng):
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    for r in range(N):
        if r % 4 in (0, 1):
            img[r, 7:9] = (54, 56, 64)
            a[r, 7:9] = 255
        else:
            img[r, 6] = img[r, 9] = (70, 72, 82)
            a[r, 6] = a[r, 9] = 255
    return img, a


def bed(rng, col=(160, 40, 36)):
    wool_ = wool(rng, col)
    pillow = speckle((230, 228, 222), 0.03, rng)
    head_top = wool_.copy()
    head_top[2:8, 2:14] = pillow[2:8, 2:14]
    head_top[2, 2:14] *= 0.9
    foot_top = wool_.copy()
    side = np.zeros((N, N, 3))
    side[:] = wool_
    side[7:, :] = speckle((150, 112, 66), 0.06, rng)[7:, :]      # the wooden frame under the blanket
    side[12:, 3:13] = 0
    a_side = np.full((N, N), 255, np.uint8)
    a_side[12:, 3:13] = 0                                        # gap between the legs
    return quant(head_top, 3), quant(foot_top, 3), quant(side, 3), a_side


def lectern(rng, planks_):
    top = planks_.copy()
    top[2:14, 2:14] = speckle((176, 140, 90), 0.05, rng)[2:14, 2:14]
    top[2:14, 8] = (110, 80, 50)
    side = planks_.copy() * 0.95
    base = planks_.copy() * 0.9
    return quant(top, 4), quant(side, 4), quant(base, 4)


def sign_wood(rng, planks_):
    s = planks_.copy() * 1.03
    return quant(s, 4)


def item_frame(rng):
    back = speckle((150, 104, 64), 0.06, rng)
    back[0:2, :] = back[-2:, :] = back[:, 0:2] = back[:, -2:] = (120, 86, 48)
    back[2:14, 2:14] = speckle((116, 78, 48), 0.05, rng)[2:14, 2:14]   # leather
    return quant(back, 4)


def cobweb(rng):
    img = np.full((N, N, 3), 225.0)
    a = np.zeros((N, N), np.uint8)
    c = 7.5
    for ang in np.linspace(0, np.pi, 5, endpoint=False):
        for t in np.linspace(-8, 8, 40):
            x, y = int(c + t * np.cos(ang)), int(c + t * np.sin(ang))
            if 0 <= x < N and 0 <= y < N:
                a[y, x] = 200
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - c, yy - c)
    for r0 in (2.5, 5.0, 7.2):
        a[(abs(rr - r0) < 0.5) & (rng.random((N, N)) < 0.8)] = 190
    return img, a


def cross_plant(rng, kind):
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    if kind == 'tall_grass':
        for _ in range(9):
            c = int(rng.integers(1, 15))
            h = int(rng.integers(6, 15))
            col = np.array((92, 150, 56)) * rng.uniform(0.8, 1.15)
            lean = rng.uniform(-0.25, 0.25)
            for k in range(h):
                cc = int(np.clip(round(c + lean * k), 0, N - 1))
                img[N - 1 - k, cc] = col * (0.8 + 0.4 * k / h)
                a[N - 1 - k, cc] = 255
    elif kind in ('poppy', 'dandelion', 'cornflower'):
        petal = {'poppy': (212, 36, 30), 'dandelion': (250, 220, 50), 'cornflower': (80, 110, 220)}[kind]
        img[8:16, 7] = (60, 120, 40)
        a[8:16, 7] = 255
        img[11, 8] = img[12, 6] = (70, 130, 44)
        a[11, 8] = a[12, 6] = 255
        for (r, c) in ((4, 7), (5, 6), (5, 8), (6, 7), (4, 6), (4, 8), (6, 6), (6, 8), (3, 7), (5, 5), (5, 9)):
            img[r, c] = np.array(petal) * rng.uniform(0.85, 1.1)
            a[r, c] = 255
        img[5, 7] = (60, 20, 16) if kind == 'poppy' else (220, 180, 40)
        a[5, 7] = 255
    elif kind in ('wheat', 'wheat_young'):
        ripe = kind == 'wheat'
        for c in (1, 4, 7, 10, 13):
            h = int(rng.integers(11, 15)) if ripe else int(rng.integers(5, 8))
            for k in range(h):
                img[N - 1 - k, c] = (180, 150, 60) if ripe else (100, 160, 50)
                a[N - 1 - k, c] = 255
            if ripe:
                img[N - h:N - h + 4, c - 1:c + 2] = (206, 170, 70)
                a[N - h:N - h + 4, c - 1:c + 2] = 255
    elif kind == 'dead_bush':
        for (r, c) in [(15, 7), (14, 7), (13, 7), (12, 6), (11, 5), (10, 4), (12, 8), (11, 9), (10, 10), (9, 11),
                       (13, 8), (11, 7), (10, 7), (9, 6), (8, 6)]:
            img[r, c] = (120, 84, 44)
            a[r, c] = 255
    elif kind == 'sugar_cane':
        for c in (3, 8, 12):
            img[:, c] = (136, 190, 90)
            a[:, c] = 255
            img[::4, c] = (100, 150, 70)
    return quant(img, 4), a


def crack(stage, rng):
    """Block-breaking overlay stage 0..9: dark crack lines spreading from the centre."""
    a = np.zeros((N, N), np.uint8)
    img = np.zeros((N, N, 3))
    r2 = _rng(99)
    paths = []
    for k in range(8):
        ang = k / 8 * 2 * np.pi + r2.uniform(-0.3, 0.3)
        x, y = 7.5, 7.5
        pts = []
        for s in range(14):
            ang += r2.uniform(-0.5, 0.5)
            x += np.cos(ang)
            y += np.sin(ang)
            pts.append((int(np.clip(x, 0, 15)), int(np.clip(y, 0, 15))))
        paths.append(pts)
    L = int(round((stage + 1) / 10 * 13))
    for pts in paths[:2 + (stage * 6) // 9]:
        for (x, y) in pts[:L]:
            a[y, x] = 230
    return img, a



# ---------------------------------------------------------------------------------------------
# the coaster's blocks: badlands, rails, the Nether, the End, the minecart
# ---------------------------------------------------------------------------------------------
def terracotta(rng, base):
    img = speckle(base, 0.045, rng, 0.5)
    r = rng.random((N, N))
    img[r < 0.06] *= 0.93
    img[r > 0.95] *= 1.05
    return quant(img, 3)


def cactus(rng, top=False):
    img = speckle((74, 120, 44), 0.08, rng, 0.3)
    if top:
        img[1:15, 1:15] = speckle((96, 146, 56), 0.06, rng, 0.2)[1:15, 1:15]
        img[6:10, 6:10] = (70, 112, 42)
        return quant(img, 3)
    for c in (1, 5, 10, 14):
        img[:, c] *= 0.72
    for (y, x) in ((2, 3), (6, 8), (11, 3), (13, 12), (4, 12), (9, 7)):
        img[y, x] = (228, 222, 170)                      # spines
    return quant(img, 3)


def rail(rng, powered=False, on=False):
    """The rail tile: two rails along v on wooden ties, see-through between them."""
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    tie = speckle((110, 84, 50), 0.10, rng, 0.2)
    for r0 in (1, 5, 9, 13):
        img[r0:r0 + 2, 1:15] = tie[r0:r0 + 2, 1:15]
        img[r0 + 1, 1:15] *= 0.8
        a[r0:r0 + 2, 1:15] = 255
    col, hi, lo = ((196, 160, 52), (250, 222, 110), (140, 104, 28)) if powered else \
        ((150, 150, 156), (206, 206, 212), (96, 96, 102))
    for c in (2, 13):
        img[:, c] = col
        img[:, c + (1 if c == 2 else -1)] = lo
        img[::4, c] = hi
        a[:, c] = 255
        a[:, c + (1 if c == 2 else -1)] = 255
    em = None
    if powered:
        # the redstone down the middle, dark when off, glowing when on
        rc = (240, 40, 20) if on else (96, 20, 16)
        img[:, 7:9] = rc
        img[::3, 7] = (255, 110, 70) if on else (120, 30, 20)
        a[:, 7:9] = 255
        if on:
            em = np.zeros((N, N))
            em[:, 7:9] = 0.9
    return img, a, em


def gold_block(rng):
    img = speckle((246, 208, 62), 0.05, rng, 0.3)
    img[0, :] = img[:, 0] = (255, 246, 150)
    img[-1, :] = img[:, -1] = (190, 140, 30)
    img[3:13, 3:13] *= 1.03
    return quant(img, 4)


def iron_block(rng):
    img = speckle((210, 210, 214), 0.03, rng, 0.3)
    img[0, :] = img[:, 0] = (236, 236, 240)
    img[-1, :] = img[:, -1] = (160, 160, 166)
    return quant(img, 4)


def crying_obsidian(rng):
    o = obsidian(rng)
    em = np.zeros((N, N))
    for (y, x) in ((3, 4), (4, 4), (9, 11), (10, 11), (11, 11), (6, 13), (12, 3), (13, 3), (2, 9)):
        o[y, x] = (160, 60, 250)
        em[y, x] = 0.9
    return o, em


def netherrack(rng):
    img = speckle((110, 48, 48), 0.12, rng, 0.35)
    r = rng.random((N, N))
    img[r < 0.18] = (84, 32, 34)
    img[r > 0.90] = (132, 62, 60)
    return quant(img, 4)


def nether_bricks(rng):
    img = speckle((48, 24, 30), 0.10, rng, 0.3)
    for r in range(N):
        for c in range(N):
            off = 0 if (r // 4) % 2 == 0 else 4
            if r % 4 == 3 or (c + off) % 8 == 7:
                img[r, c] = (26, 12, 16)
            elif r % 4 == 0:
                img[r, c] *= 1.18
    return quant(img, 3)


def lava(rng):
    """The still frame; the shader makes it flow."""
    img = speckle((214, 96, 18), 0.10, rng, 0.5)
    r = rng.random((N, N))
    img[r > 0.70] = (242, 150, 40)
    img[r > 0.88] = (254, 204, 80)
    img[r < 0.10] = (178, 58, 12)
    return quant(img, 4), np.full((N, N), 1.0)


def magma(rng):
    img = speckle((70, 26, 18), 0.12, rng, 0.3)
    em = np.zeros((N, N))
    for (y0, x0) in ((1, 1), (1, 9), (5, 5), (9, 1), (9, 11), (13, 6)):
        for k in range(4):
            y, x = y0 + (k % 2), x0 + k
            if 0 <= y < N and 0 <= x < N:
                img[y, x] = (240, 110, 30)
                em[y, x] = 0.8
    return quant(img, 3), em


def basalt(rng, top=False):
    img = speckle((78, 78, 84), 0.10, rng, 0.4)
    if top:
        for r in range(N):
            for c in range(N):
                d = max(abs(r - 7.5), abs(c - 7.5))
                if int(d) % 3 == 0:
                    img[r, c] *= 0.8
    else:
        for c in range(0, N, 3):
            img[:, c] *= 0.82
    return quant(img, 3)


def blackstone(rng):
    img = speckle((44, 38, 44), 0.14, rng, 0.35)
    r = rng.random((N, N))
    img[r > 0.9] = (64, 58, 66)
    return quant(img, 3)


def ore(rng, base, col, count=10):
    img = base.copy()
    for _ in range(count):
        y, x = rng.integers(1, 14, 2)
        img[y, x] = col
        img[y, x + 1] = np.array(col) * 0.85
    return img


def soul_sand(rng):
    img = speckle((84, 64, 50), 0.12, rng, 0.35)
    for (y, x) in ((4, 4), (4, 7), (9, 10), (9, 13)):
        img[y:y + 2, x:x + 2] = (52, 38, 30)
    img[6, 5:7] = (52, 38, 30)
    img[11, 11:13] = (52, 38, 30)
    return quant(img, 3)


def nylium(rng, side=False, rack=None):
    top = speckle((148, 26, 36), 0.12, rng, 0.35)
    if not side:
        return quant(top, 3)
    img = rack.copy()
    img[:4] = top[:4]
    for c in range(N):
        if rng.random() < 0.5:
            img[4, c] = top[4, c]
    return quant(img, 3)


def wart_block(rng):
    img = speckle((128, 6, 10), 0.14, rng, 0.35)
    r = rng.random((N, N))
    img[r > 0.88] = (170, 30, 30)
    return quant(img, 3)


def stem(rng, top=False):
    if top:
        img = speckle((92, 26, 46), 0.08, rng, 0.2)
        img[3:13, 3:13] = speckle((150, 56, 70), 0.06, rng, 0.2)[3:13, 3:13]
        return quant(img, 3)
    img = speckle((96, 28, 52), 0.10, rng, 0.3)
    for c in range(0, N, 4):
        img[:, c] *= 0.75
    r = rng.random((N, N))
    img[r > 0.93] = (30, 160, 160)          # the stem's teal glints
    return quant(img, 3)


def shroomlight(rng):
    img = speckle((246, 150, 76), 0.1, rng, 0.3)
    r = rng.random((N, N))
    img[r > 0.85] = (255, 212, 130)
    return quant(img, 4), np.full((N, N), 0.85)


def end_stone(rng):
    img = speckle((222, 224, 164), 0.06, rng, 0.35)
    r = rng.random((N, N))
    img[r < 0.12] = (196, 196, 138)
    img[r > 0.94] = (236, 238, 190)
    return quant(img, 4)


def end_stone_bricks(rng):
    img = speckle((220, 222, 162), 0.05, rng, 0.3)
    for r in range(N):
        for c in range(N):
            off = 0 if (r // 8) % 2 == 0 else 8
            if r % 8 == 7 or (c + off) % 16 == 15:
                img[r, c] = (170, 170, 118)
    return quant(img, 4)


def bedrock(rng):
    img = speckle((84, 84, 84), 0.25, rng, 0.5)
    r = rng.random((N, N))
    img[r < 0.2] = (40, 40, 40)
    img[r > 0.88] = (140, 140, 140)
    return quant(img, 6)


def iron_bars(rng):
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    for c in (1, 5, 9, 13):
        img[:, c:c + 2] = (110, 110, 114)
        img[:, c] = (160, 160, 166)
        a[:, c:c + 2] = 255
    for r in (0, 15):
        img[r, :] = (120, 120, 124)
        a[r, :] = 255
    return img, a


def end_frame(rng, part):
    base = speckle((68, 104, 96), 0.08, rng, 0.3) if part != 'top' else speckle((72, 110, 100), 0.06, rng, 0.3)
    if part == 'side':
        base[:3] = speckle((222, 224, 164), 0.05, rng, 0.3)[:3]
        base[3] *= 0.7
        return quant(base, 3)
    img = base
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (46, 74, 68)
    if part == 'eye':
        img[4:12, 4:12] = (22, 90, 60)
        img[5:11, 5:11] = (40, 160, 110)
        img[6:10, 6:10] = (10, 30, 20)
        img[7:9, 7:9] = (0, 0, 0)
    return quant(img, 3)


def dragon_egg(rng):
    img = speckle((14, 8, 18), 0.4, rng, 0.3)
    r = rng.random((N, N))
    img[r > 0.9] = (110, 40, 150)
    return quant(img, 3)


def minecart(rng, part):
    if part == 'inside':
        img = speckle((62, 62, 68), 0.08, rng, 0.3)
        img[::4, :] *= 0.8
        return quant(img, 3)
    img = speckle((126, 126, 134), 0.06, rng, 0.3)
    img[0, :] = img[:, 0] = (170, 170, 176)
    img[-1, :] = img[:, -1] = (70, 70, 76)
    if part == 'side':
        img[5:11, 1:15] *= 0.86
        for (y, x) in ((2, 2), (2, 13), (13, 2), (13, 13)):
            img[y, x] = (200, 200, 206)             # rivets
    return quant(img, 3)


# ---------------------------------------------------------------------------------------------
# mobs (drawn as boxes of 16x16 faces, like block models)
# ---------------------------------------------------------------------------------------------
def ghast(rng, part):
    base = speckle((236, 236, 236), 0.03, rng, 0.4)
    base[rng.random((N, N)) < 0.12] *= 0.93
    if part == 'side':
        return quant(base, 3), None
    if part == 'tentacle':
        img = speckle((226, 226, 226), 0.04, rng, 0.4)
        img[:, 0] *= 0.9
        return quant(img, 3), None
    img = base
    em = np.zeros((N, N))
    if part == 'face':
        for ex in (3, 10):
            img[5, ex:ex + 3] = (70, 70, 74)                     # closed eyes
            img[6:10, ex + 1] = (170, 170, 176)                  # tear streaks
        img[11, 6:10] = (80, 80, 84)
    else:                                                        # shooting: eyes and mouth wide open
        for ex in (3, 10):
            img[4:7, ex:ex + 3] = (30, 30, 34)
            img[5, ex + 1] = (200, 30, 30)
            em[5, ex + 1] = 0.6
            img[7:10, ex + 1] = (170, 170, 176)
        img[10:14, 5:11] = (60, 10, 14)
        img[11:13, 6:10] = (150, 20, 24)
    return quant(img, 3), em


def dragon(rng, part):
    em = np.zeros((N, N))
    if part == 'scale':
        img = speckle((26, 26, 30), 0.25, rng, 0.3)
        for r in range(0, N, 4):
            for c in range(0, N, 4):
                img[r + ((c // 4) % 2) * 2, c:c + 3] = (44, 44, 50)
        return quant(img, 2), em
    if part == 'belly':
        img = speckle((52, 52, 60), 0.12, rng, 0.3)
        img[::3, :] *= 0.8
        return quant(img, 2), em
    if part == 'wing':
        img = speckle((20, 20, 24), 0.2, rng, 0.3)
        for c in (1, 7, 13):
            img[:, c] = (58, 58, 64)                             # the wing's bones
        return quant(img, 2), em
    if part == 'eye':
        img = speckle((26, 26, 30), 0.25, rng, 0.3)
        img[2:4, 1:6] = (214, 110, 250)
        img[2:4, 10:15] = (214, 110, 250)
        img[2, 2:4] = (250, 220, 255)
        img[2, 12:14] = (250, 220, 255)
        em[2:4, 1:6] = 1.0
        em[2:4, 10:15] = 1.0
        return quant(img, 2), em
    if part == 'tooth':
        return quant(speckle((210, 210, 214), 0.05, rng, 0.3), 3), em
    if part == 'mouth':
        img = speckle((90, 20, 60), 0.2, rng, 0.3)
        return quant(img, 2), em
    return quant(speckle((30, 30, 34), 0.2, rng, 0.3), 2), em


def enderman(rng, part):
    img = speckle((14, 14, 16), 0.35, rng, 0.3)
    img[rng.random((N, N)) > 0.9] = (34, 34, 40)
    em = np.zeros((N, N))
    if part == 'face':
        for ex in (2, 10):
            img[9, ex:ex + 4] = (224, 121, 250)
            img[9, ex + 1:ex + 3] = (250, 214, 255)
            em[9, ex:ex + 4] = 1.0
    return quant(img, 2), em


def crystal(rng, part):
    em = np.zeros((N, N))
    img = np.zeros((N, N, 3))
    a = np.zeros((N, N), np.uint8)
    if part == 'frame':
        img[:] = (250, 200, 240)
        m = np.zeros((N, N), bool)
        m[:2, :] = m[-2:, :] = m[:, :2] = m[:, -2:] = True
        a[m] = 255
        em[m] = 0.7
        return img, a, em
    img = speckle((210, 120, 220), 0.15, rng, 0.4)
    img[4:12, 4:12] = (255, 190, 250)
    a[:] = 255
    em[:] = 0.8
    return quant(img, 3), a, em

# ---------------------------------------------------------------------------------------------
# the whole set
# ---------------------------------------------------------------------------------------------
def make_all(seed=7):
    rng = _rng(seed)
    T = {}
    E = {}

    def put(name, img, alpha=None, emit=None):
        T[name] = rgba(img, alpha)
        if emit is not None:
            E[name] = emit

    gt = grass_top(rng)
    dt = dirt(rng)
    put('grass_top', gt)
    put('dirt', dt)
    put('grass_side', grass_side(rng, gt, dt))
    put('dirt_path_top', dirt_path(rng))
    ds = dt.copy()
    ds[:2] = dirt_path(rng)[:2]
    put('dirt_path_side', ds)
    put('farmland', farmland(rng))
    put('stone', stone(rng))
    put('smooth_stone', quant(speckle((160, 160, 162), 0.03, rng, 0.6), 3))
    put('cobblestone', cobble(rng))
    put('mossy_cobblestone', cobble(rng, moss=True))
    put('stone_bricks', stone_bricks(rng))
    put('mossy_stone_bricks', stone_bricks(rng, moss=True))
    put('cracked_stone_bricks', stone_bricks(rng, cracked=True))
    put('gravel', gravel(rng))
    put('sand', sand(rng))
    put('deepslate', deepslate(rng))
    put('deepslate_top', deepslate(rng, top=True))
    put('cobbled_deepslate', cobble(rng, (70, 70, 76)))
    put('deepslate_tiles', deepslate_tiles(rng))
    put('obsidian', obsidian(rng))
    put('oak_log', log_side(rng))
    put('oak_log_top', log_top(rng))
    put('spruce_log', log_side(rng, (70, 48, 26), (50, 34, 18)))
    put('spruce_log_top', log_top(rng, (130, 96, 56), (110, 80, 46), (60, 40, 22)))
    put('birch_log', birch_side(rng))
    put('birch_log_top', log_top(rng, (200, 180, 130), (180, 160, 110), (220, 218, 210)))
    put('dead_log', log_side(rng, (84, 76, 66), (60, 54, 48)))
    lv, la = leaves(rng)
    put('oak_leaves', lv, la)
    lv, la = leaves(rng, (46, 88, 52), 0.14)
    put('spruce_leaves', lv, la)
    lv, la = leaves(rng, (104, 140, 64))
    put('birch_leaves', lv, la)
    pk = planks(rng)
    put('oak_planks', pk)
    sp = planks(rng, (112, 82, 50))
    put('spruce_planks', sp)
    put('dark_oak_planks', planks(rng, (72, 50, 28)))
    g, a = glass(rng)
    put('glass', g, a)
    for name, col in (('white_wool', (226, 226, 222)), ('red_wool', (164, 40, 36)), ('gray_wool', (78, 78, 80)),
                      ('blue_wool', (54, 72, 150)), ('light_gray_wool', (152, 152, 150)),
                      ('brown_wool', (110, 74, 44)), ('green_wool', (84, 110, 36))):
        put(name, wool(rng, col))
    for name, col in (('white_concrete', (210, 214, 216)), ('light_gray_concrete', (132, 132, 126)),
                      ('gray_concrete', (56, 60, 64)), ('black_concrete', (10, 12, 16)),
                      ('blue_concrete', (46, 50, 130)), ('quartz', (234, 228, 220))):
        put(name, concrete(rng, col))
    put('water', water(rng))
    ht, hs = hay(rng)
    put('hay_top', ht)
    put('hay_side', hs)
    ps, pt = pumpkin(rng)
    put('pumpkin_side', ps)
    put('pumpkin_top', pt)
    put('bookshelf', bookshelf(rng, pk))
    ct, cs, cf = crafting_table(rng, pk)
    put('crafting_table_top', ct)
    put('crafting_table_side', cs)
    put('crafting_table_front', cf)
    for lit in (False, True):
        f, s, t, e = furnace(rng, lit)
        put('furnace_front_on' if lit else 'furnace_front', f, emit=e if lit else None)
        if not lit:
            put('furnace_side', s)
            put('furnace_top', t)
    cf, cs, ctop = chest(rng)
    put('chest_front', cf)
    put('chest_side', cs)
    put('chest_top', ctop)
    for part in ('top', 'bottom'):
        img, a = door(rng, pk, part)
        put('oak_door_' + part, img, a)
        img, a = door(rng, sp, part)
        put('spruce_door_' + part, img, a)
    img, a = trapdoor(rng, pk)
    put('oak_trapdoor', img, a)
    img, a = ladder(rng)
    put('ladder', img, a)
    for soul in (False, True):
        img, a, e = torch(rng, soul)
        put('soul_torch' if soul else 'torch', img, a, e)
        img, a, e = lantern(rng, soul)
        put('soul_lantern' if soul else 'lantern', img, a, e)
    img, a = chain(rng)
    put('chain', img, a)
    for name, col in (('red', (160, 40, 36)), ('blue', (48, 66, 150))):
        ht_, ft_, sd_, asd = bed(rng, col)
        put(f'{name}_bed_head', ht_)
        put(f'{name}_bed_foot', ft_)
        put(f'{name}_bed_side', sd_, asd)
    lt, ls, lb = lectern(rng, pk)
    put('lectern_top', lt)
    put('lectern_side', ls)
    put('lectern_base', lb)
    put('sign', sign_wood(rng, pk))
    put('item_frame', item_frame(rng))
    img, a = cobweb(rng)
    put('cobweb', img, a)
    for kind in ('tall_grass', 'poppy', 'dandelion', 'cornflower', 'wheat', 'wheat_young', 'dead_bush', 'sugar_cane'):
        img, a = cross_plant(rng, kind)
        put(kind, img, a)
    for s in range(10):
        img, a = crack(s, rng)
        put(f'crack_{s}', img, a)
    # glowing blocks
    gl = speckle((230, 190, 110), 0.2, rng, 0.2)
    put('glowstone', gl, emit=np.full((N, N), 0.8))
    rl = speckle((120, 70, 40), 0.1, rng)
    rl[2:14, 2:14] = (250, 200, 130)
    e = np.zeros((N, N))
    e[2:14, 2:14] = 0.9
    put('redstone_lamp_on', rl, emit=e)
    put('screen', np.full((N, N, 3), 20.0))                  # replaced per frame by the screen prop
    put('white', np.full((N, N, 3), 255.0))
    put('black', np.full((N, N, 3), 6.0))
    # the coaster
    for nm, base in (('terracotta', (150, 92, 66)), ('orange_terracotta', (164, 86, 40)),
                     ('yellow_terracotta', (188, 134, 38)), ('red_terracotta', (146, 62, 48)),
                     ('white_terracotta', (212, 180, 162)), ('brown_terracotta', (80, 52, 38)),
                     ('light_gray_terracotta', (138, 108, 98))):
        put(nm, terracotta(rng, base))
    put('red_sand', quant(speckle((194, 104, 36), 0.07, rng, 0.3), 4))
    put('cactus_side', cactus(rng))
    put('cactus_top', cactus(rng, top=True))
    img, a, _ = rail(rng)
    put('rail', img, a)
    img, a, _ = rail(rng, powered=True)
    put('powered_rail', img, a)
    img, a, em = rail(rng, powered=True, on=True)
    put('powered_rail_on', img, a, emit=em)
    put('gold_block', gold_block(rng))
    put('iron_block', iron_block(rng))
    img, em = crying_obsidian(rng)
    put('crying_obsidian', img, emit=em)
    rtimg, rta, _ = torch(rng)
    rtimg = np.array(rtimg, float)
    head = rta > 0
    rtimg[:6][head[:6]] = (230, 40, 24)
    put('redstone_torch', rtimg, rta, emit=np.where(np.arange(N)[:, None] < 6, 0.9, 0.0) * head)
    nr = netherrack(rng)
    put('netherrack', nr)
    put('nether_bricks', nether_bricks(rng))
    img, em = lava(rng)
    put('lava', img, emit=em)
    img, em = magma(rng)
    put('magma_block', img, emit=em)
    put('basalt_side', basalt(rng))
    put('basalt_top', basalt(rng, top=True))
    put('blackstone', blackstone(rng))
    put('nether_gold_ore', ore(rng, nr, (250, 214, 70), 12))
    put('nether_quartz_ore', ore(rng, nr, (236, 230, 222), 10))
    put('soul_sand', soul_sand(rng))
    put('crimson_nylium', nylium(rng))
    put('crimson_nylium_side', nylium(rng, side=True, rack=nr))
    put('nether_wart_block', wart_block(rng))
    put('crimson_stem', stem(rng))
    put('crimson_stem_top', stem(rng, top=True))
    img, em = shroomlight(rng)
    put('shroomlight', img, emit=em)
    put('end_stone', end_stone(rng))
    put('end_stone_bricks', end_stone_bricks(rng))
    put('bedrock', bedrock(rng))
    img, a = iron_bars(rng)
    put('iron_bars', img, a)
    put('end_portal_frame_top', end_frame(rng, 'top'))
    eye_em = np.zeros((N, N))
    eye_em[5:11, 5:11] = 0.5
    eye_em[6:10, 6:10] = 0.0
    put('end_portal_frame_eye', end_frame(rng, 'eye'), emit=eye_em)
    put('end_portal_frame_side', end_frame(rng, 'side'))
    put('dragon_egg', dragon_egg(rng))
    put('minecart_side', minecart(rng, 'side'))
    put('minecart_inside', minecart(rng, 'inside'))
    put('minecart_bottom', minecart(rng, 'bottom'))
    # mobs
    for part in ('side', 'tentacle', 'face', 'face_shoot'):
        img, em = ghast(rng, part)
        put('ghast_' + part, img, emit=em)
    for part in ('scale', 'belly', 'wing', 'eye', 'tooth', 'mouth'):
        img, em = dragon(rng, part)
        put('dragon_' + part, img, emit=em)
    for part in ('body', 'face'):
        img, em = enderman(rng, part)
        put('enderman_' + part, img, emit=em)
    for part in ('frame', 'core'):
        img, a, em = crystal(rng, part)
        put('crystal_' + part, img, a, emit=em)
    return T, E


# ---------------------------------------------------------------------------------------------
# items (16x16 icons, also extruded into 3D for held items and item frames)
# ---------------------------------------------------------------------------------------------
def _icon(rows, pal):
    img = np.zeros((N, N, 4), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != '.':
                img[r, c, :3] = pal[ch]
                img[r, c, 3] = 255
    return img


ICONS = {
    'written_book': ([
        "................",
        "...kkkkkkkkkk...",
        "..kBBBBBBBBBBk..",
        "..kBbbbbbbbbWk..",
        "..kBbGGGGGbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbGGGGbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBBBBBBBBBBk..",
        "...kkkkkkkkkk...",
        "................",
        "................"], {'k': (40, 22, 14), 'B': (104, 58, 30), 'b': (132, 76, 40), 'W': (230, 226, 210),
                              'G': (214, 176, 60)}),
    'book': ([
        "................",
        "...kkkkkkkkkk...",
        "..kBBBBBBBBBBk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBbbbbbbbbWk..",
        "..kBBBBBBBBBBk..",
        "...kkkkkkkkkk...",
        "................",
        "................"], {'k': (40, 22, 14), 'B': (96, 52, 26), 'b': (120, 68, 36), 'W': (230, 226, 210)}),
    'filled_map': ([
        "................",
        ".kkkkkkkkkkkkkk.",
        ".kPPPPPPPPPPPPk.",
        ".kPggggwwggggPk.",
        ".kPgGGgwwgGggPk.",
        ".kPgggwwggGGgPk.",
        ".kPbbgwwgggggPk.",
        ".kPbbbwwwgrggPk.",
        ".kPbbgggwwgggPk.",
        ".kPggGggwwggGPk.",
        ".kPgggggwwgggPk.",
        ".kPggGgggwwggPk.",
        ".kPPPPPPPPPPPPk.",
        ".kkkkkkkkkkkkkk.",
        "................",
        "................"], {'k': (60, 44, 24), 'P': (218, 200, 150), 'g': (126, 150, 80), 'G': (84, 110, 56),
                              'w': (190, 170, 120), 'b': (70, 100, 170), 'r': (200, 30, 30)}),
    'torch': ([
        "................",
        "................",
        "................",
        ".......yY.......",
        ".......YO.......",
        ".......kk.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        "................",
        "................"], {'y': (255, 244, 180), 'Y': (255, 214, 90), 'O': (244, 150, 40), 'k': (60, 44, 24),
                              's': (120, 88, 46)}),
    'bread': ([
        "................",
        "................",
        "................",
        "................",
        ".....kkkkkk.....",
        "...kkBBBBBBkk...",
        "..kBBbBBbBBbBk..",
        ".kBbBBbBBbBBbBk.",
        ".kBBBBBBBBBBBBk.",
        ".kDBBBBBBBBBBDk.",
        "..kDDDDDDDDDDk..",
        "...kkkkkkkkkk...",
        "................",
        "................",
        "................",
        "................"], {'k': (70, 40, 14), 'B': (196, 140, 60), 'b': (236, 196, 110), 'D': (150, 96, 36)}),
    'apple': ([
        "................",
        "........s.......",
        ".......sL.......",
        "......sLL.......",
        "....kkksskk.....",
        "...kRRRRRRRk....",
        "..kRRWRRRRRRk...",
        "..kRWRRRRRRRk...",
        "..kRRRRRRRRRk...",
        "..kRRRRRRRRDk...",
        "..kRRRRRRRDDk...",
        "...kRRRRRDDk....",
        "....kkRkRkk.....",
        "......k.k.......",
        "................",
        "................"], {'k': (60, 10, 10), 'R': (210, 30, 34), 'W': (255, 200, 200), 'D': (150, 16, 20),
                              's': (90, 60, 30), 'L': (80, 150, 50)}),
    'clock': ([
        "................",
        ".....kkkkkk.....",
        "...kkGGGGGGkk...",
        "..kGGssssssGGk..",
        "..kGssssssssGk..",
        ".kGsssssssssGGk.",
        ".kGssssnnsssssk.",
        ".kGssssnnssssGk.",
        ".kGsssmmmmsssGk.",
        ".kGsmmmmmmmmsGk.",
        ".kGmmmmmmmmmmGk.",
        "..kGmmmmmmmmGk..",
        "..kGGmmmmmmGGk..",
        "...kkGGGGGGkk...",
        ".....kkkkkk.....",
        "................"], {'k': (70, 50, 10), 'G': (230, 190, 60), 's': (40, 50, 120), 'n': (240, 240, 250),
                              'm': (90, 150, 60)}),
    'paper': ([
        "................",
        "................",
        "...kkkkkkkkkk...",
        "...kWWWWWWWWk...",
        "...kWggggggWk...",
        "...kWWWWWWWWk...",
        "...kWggggggWk...",
        "...kWWWWWWWWk...",
        "...kWgggggWWk...",
        "...kWWWWWWWWk...",
        "...kWggggWWWk...",
        "...kWWWWWWWk....",
        "...kkWWkWWk.....",
        ".....kk.kk......",
        "................",
        "................"], {'k': (120, 110, 90), 'W': (236, 232, 218), 'g': (150, 146, 136)}),
    'wooden_pickaxe': ([
        "................",
        "...kkkkkkk......",
        "..kPPPPPPPkk....",
        "...kkkkkkPPk....",
        "........kSPPk...",
        ".......kSk.kPk..",
        "......kSk...kPk.",
        ".....kSk.....kk.",
        "....kSk.........",
        "...kSk..........",
        "..kSk...........",
        ".kSk............",
        ".kk.............",
        "................",
        "................",
        "................"], {'k': (40, 28, 14), 'P': (170, 134, 80), 'S': (120, 88, 46)}),
    'iron_axe': ([
        "................",
        "......kkk.......",
        ".....kIIIk......",
        "....kIIIIk......",
        "....kIIIkSk.....",
        ".....kIkSk......",
        "......kSk.......",
        ".....kSk........",
        "....kSk.........",
        "...kSk..........",
        "..kSk...........",
        ".kSk............",
        ".kk.............",
        "................",
        "................",
        "................"], {'k': (40, 40, 44), 'I': (210, 210, 216), 'S': (120, 88, 46)}),
    'stone_sword': ([
        "..............kk",
        ".............kIk",
        "............kIIk",
        "...........kIIk.",
        "..........kIIk..",
        ".........kIIk...",
        "........kIIk....",
        "...kk..kIIk.....",
        "...kSkkIIk......",
        "....kSSIk.......",
        ".....kSSk.......",
        "....kSkkSk......",
        "...kSk..kk......",
        "..kSk...........",
        "..kk............",
        "................"], {'k': (40, 40, 44), 'I': (140, 140, 140), 'S': (120, 88, 46)}),
    'coal': ([
        "................",
        "................",
        ".....kkkk.......",
        "....kCCCCk......",
        "...kCcCCCCkk....",
        "..kCCCCCcCCCk...",
        "..kCCcCCCCCCk...",
        "..kCCCCCCcCCk...",
        "...kCCCCCCCk....",
        "....kkkkkkk.....",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................"], {'k': (10, 10, 12), 'C': (40, 40, 44), 'c': (90, 90, 96)}),
}


def item_icons():
    icons = {k: _icon(r, p) for k, (r, p) in ICONS.items()}
    # the torn page: a paper with a ragged top edge
    p = icons['paper'].copy()
    for c in range(3, 13):
        if c % 2 == 0:
            p[2, c, 3] = 0
            p[3, c, :3] = (200, 196, 180)
    icons['torn_page'] = p
    return icons


if __name__ == '__main__':
    import sys
    from PIL import Image
    T, E = make_all()
    names = sorted(T)
    cols = 12
    rows = (len(names) + cols - 1) // cols
    k = 6
    sheet = np.zeros((rows * (N + 2) * k, cols * (N + 2) * k, 3), np.uint8)
    sheet[:] = (60, 30, 60)
    for i, nme in enumerate(names):
        r, c = divmod(i, cols)
        t = T[nme].astype(float)
        a = t[..., 3:4] / 255
        img = t[..., :3] * a + np.array((60, 30, 60)) * (1 - a)
        big = np.kron(img, np.ones((k, k, 1)))
        sheet[r * (N + 2) * k:r * (N + 2) * k + N * k, c * (N + 2) * k:c * (N + 2) * k + N * k] = big
    Image.fromarray(sheet).save(sys.argv[1] if len(sys.argv) > 1 else 'tex_sheet.png')
    ic = item_icons()
    sheet2 = np.zeros((N * k, len(ic) * (N + 2) * k, 3), np.uint8)
    for i, (nme, t) in enumerate(ic.items()):
        a = t[..., 3:4] / 255
        img = t[..., :3] * a + np.array((60, 30, 60)) * (1 - a)
        sheet2[:, i * (N + 2) * k:i * (N + 2) * k + N * k] = np.kron(img, np.ones((k, k, 1)))
    Image.fromarray(sheet2).save((sys.argv[1] if len(sys.argv) > 1 else 'tex_sheet.png')[:-4] + '_items.png')
    print(len(T), 'textures,', len(E), 'emissive,', len(ic), 'items')
