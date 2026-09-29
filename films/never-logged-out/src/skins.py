"""Procedural 64 x 64 skins in the game's box-UV layout: the two players (the protagonist, "YOU", and NOAH_404), the
copies, the villagers and the animals. All original designs, painted from rectangles and noise.

Box UV layout for a box of w x h x d pixels at texture offset (u, v):
  top (u+d, v, w, d)   bottom (u+d+w, v, w, d)
  right side (u, v+d, d, h)   front (u+d, v+d, w, h)   left side (u+d+w, v+d, d, h)   back (u+2d+w, v+d, w, h)
"""
import numpy as np

SKIN_NAMES = ['you', 'noah', 'you_wrong', 'villager_farmer', 'villager_librarian', 'villager_cartographer',
              'villager_cleric', 'villager_smith', 'villager_shepherd', 'villager_noah', 'cow', 'sheep', 'pig',
              'chicken', 'hand_debug']


def region(u, v, w, h, d):
    return {'top': (u + d, v, w, d), 'bottom': (u + d + w, v, w, d), 'right': (u, v + d, d, h),
            'front': (u + d, v + d, w, h), 'left': (u + d + w, v + d, d, h), 'back': (u + 2 * d + w, v + d, w, h)}


class Canvas:
    def __init__(self, seed=0):
        self.img = np.zeros((64, 64, 4), np.uint8)
        self.rng = np.random.default_rng(seed)

    def rect(self, x, y, w, h, col, var=0.06, alpha=255):
        if w <= 0 or h <= 0:
            return
        c = np.array(col, float)[None, None, :] * (1 + var * 2 * (self.rng.random((h, w, 1)) - 0.5))
        self.img[y:y + h, x:x + w, :3] = np.clip(np.round(c / 3) * 3, 0, 255)
        self.img[y:y + h, x:x + w, 3] = alpha

    def px(self, x, y, col, alpha=255):
        self.img[y, x, :3] = col
        self.img[y, x, 3] = alpha

    def box(self, u, v, w, h, d, col, var=0.06, faces=None):
        r = region(u, v, w, h, d)
        for f, (x, y, ww, hh) in r.items():
            if faces is None or f in faces:
                self.rect(x, y, ww, hh, col if not isinstance(col, dict) else col.get(f, col['all']), var)
        return r


# ---------------------------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------------------------
def player(spec, seed=1):
    """spec: skin, hair, hair_style, eyes, pupil, brow, mouth, shirt, shirt2, pants, shoes, extra."""
    c = Canvas(seed)
    s = spec
    skin = s['skin']
    # head (0,0) 8x8x8
    r = c.box(0, 0, 8, 8, 8, skin, 0.04)
    hair = s['hair']
    # hair: top, back, upper sides, fringe on the face
    c.rect(*r['top'], hair, 0.1)
    bx, by, bw, bh = r['back']
    c.rect(bx, by, bw, bh - 2 if s.get('long') else 6, hair, 0.1)
    for side in ('right', 'left'):
        x, y, w, h = r[side]
        c.rect(x, y, w, 3, hair, 0.1)
        c.rect(x + (0 if side == 'left' else 5), y + 3, 3, 2 + (2 if s.get('long') else 0), hair, 0.1)
    fx, fy, fw, fh = r['front']
    c.rect(fx, fy, fw, 2, hair, 0.1)
    fringe = s.get('fringe', [1, 1, 0, 1, 0, 0, 1, 1])
    for k in range(8):
        if fringe[k]:
            c.px(fx + k, fy + 2, tuple(int(v) for v in np.array(hair) * 0.9))
    # eyes at row 4, brows at row 3
    eyes = s['eyes']
    pupil = s['pupil']
    for (ex, pup_on_left) in ((1, False), (5, True)):
        c.px(fx + ex, fy + 4, eyes)
        c.px(fx + ex + 1, fy + 4, eyes)
        c.px(fx + ex + (0 if pup_on_left else 1), fy + 4, pupil)
    if s.get('brow'):
        for ex in (1, 5):
            c.px(fx + ex, fy + 3, s['brow'])
            c.px(fx + ex + 1, fy + 3, s['brow'])
    # nose shade and mouth
    c.px(fx + 3, fy + 5, tuple(int(v) for v in np.array(skin) * 0.86))
    c.px(fx + 4, fy + 5, tuple(int(v) for v in np.array(skin) * 0.86))
    for mx in (3, 4):
        c.px(fx + mx, fy + 6, s['mouth'])
    if s.get('mouth_w'):
        c.px(fx + 2, fy + 6, s['mouth'])
        c.px(fx + 5, fy + 6, s['mouth'])
    # hat layer (32,0): stray hair over the ears, headphones band for the protagonist
    rh = region(32, 0, 8, 8, 8)
    if s.get('headphones'):
        for side in ('right', 'left'):
            x, y, w, h = rh[side]
            c.rect(x + 2, y + 3, 4, 4, (26, 26, 30), 0.05)
            c.rect(x + 3, y + 4, 2, 2, (170, 34, 34), 0.05)
    # body (16,16) 8x12x4
    shirt = s['shirt']
    rb = c.box(16, 16, 8, 12, 4, shirt, 0.07)
    x, y, w, h = rb['front']
    if s.get('hoodie'):
        c.rect(x, y, w, 1, tuple(int(v) for v in np.array(shirt) * 0.8))          # hood collar
        c.rect(x + 2, y + 1, 1, 4, s['shirt2'])                                     # drawstrings
        c.rect(x + 5, y + 1, 1, 4, s['shirt2'])
        c.rect(x + 1, y + 7, 6, 3, tuple(int(v) for v in np.array(shirt) * 0.85))  # pocket
        c.rect(x + 1, y + 7, 6, 1, tuple(int(v) for v in np.array(shirt) * 0.72))
        bxx, byy, bww, bhh = rb['back']
        c.rect(bxx + 1, byy, 6, 3, tuple(int(v) for v in np.array(shirt) * 0.8))   # the hood, down
    if s.get('jacket'):
        c.rect(x + 2, y, 4, 12, s['shirt2'], 0.04)                                  # the shirt under it
        c.rect(x + 1, y, 1, 3, tuple(int(v) for v in np.array(shirt) * 1.12))       # lapels
        c.rect(x + 6, y, 1, 3, tuple(int(v) for v in np.array(shirt) * 1.12))
        c.rect(x, y + 11, 8, 1, tuple(int(v) for v in np.array(shirt) * 0.8))
    if s.get('headphones'):
        c.rect(x + 1, y, 6, 1, (26, 26, 30))
        c.rect(x + 1, y, 1, 2, (170, 34, 34))
        c.rect(x + 6, y, 1, 2, (170, 34, 34))
    # arms: (40,16) right, (32,48) left; 4x12x4: sleeve to the wrist, hands of skin
    for (u, v) in ((40, 16), (32, 48)):
        ra = c.box(u, v, 4, 12, 4, shirt, 0.07)
        for f in ('right', 'front', 'left', 'back'):
            x, y, w, h = ra[f]
            c.rect(x, y + 9, w, 3, skin, 0.04)
            if s.get('hoodie'):
                c.rect(x, y + 8, w, 1, tuple(int(v) for v in np.array(shirt) * 0.8))
        c.rect(*ra['bottom'], skin)
    # legs: (0,16) right, (16,48) left
    for (u, v) in ((0, 16), (16, 48)):
        rl = c.box(u, v, 4, 12, 4, s['pants'], 0.08)
        for f in ('right', 'front', 'left', 'back'):
            x, y, w, h = rl[f]
            c.rect(x, y + 10, w, 2, s['shoes'], 0.05)
            if s.get('sneakers'):
                c.rect(x, y + 11, w, 1, (60, 60, 64))
        c.rect(*rl['bottom'], (50, 50, 54))
    return c.img


YOU = dict(skin=(198, 146, 112), hair=(34, 28, 26), eyes=(236, 236, 236), pupil=(52, 34, 22), brow=(40, 32, 28),
           mouth=(150, 90, 72), shirt=(46, 86, 60), shirt2=(222, 222, 212), pants=(52, 56, 68), shoes=(226, 226, 222),
           hoodie=True, headphones=True, sneakers=True, fringe=[1, 1, 1, 0, 1, 1, 0, 1])
NOAH = dict(skin=(226, 176, 142), hair=(104, 66, 36), eyes=(236, 236, 236), pupil=(52, 96, 170), brow=(84, 52, 30),
            mouth=(170, 106, 90), shirt=(66, 98, 150), shirt2=(226, 224, 216), pants=(156, 136, 96),
            shoes=(92, 60, 36), jacket=True, long=True, fringe=[1, 1, 1, 1, 1, 0, 0, 1])


def you_wrong():
    s = dict(YOU)
    s['pupil'] = (8, 8, 10)
    s['eyes'] = (8, 8, 10)            # all dark: no whites
    s['mouth'] = (120, 70, 60)
    img = player(s, seed=1)
    return img


# ---------------------------------------------------------------------------------------------
# villagers (head 8x10x8 at (0,0), nose 2x4x2 at (24,0), body 8x12x6 at (16,20), robe 8x18x6 at (0,38),
# arms 8x4x4 at (44,22) + 4x8x4 at (40,38), legs 4x12x4 at (0,22))
# ---------------------------------------------------------------------------------------------
def villager(robe, trim, hat=None, seed=3, apron=None):
    c = Canvas(seed)
    skin = (170, 124, 96)
    r = c.box(0, 0, 8, 10, 8, skin, 0.05)
    fx, fy, fw, fh = r['front']
    c.rect(fx, fy + 3, 8, 1, (64, 44, 34))              # the unibrow
    for ex in (1, 5):
        c.px(fx + ex, fy + 4, (236, 236, 236))
        c.px(fx + ex + 1, fy + 4, (30, 90, 50))
        c.px(fx + ex, fy + 4, (236, 236, 236))
    c.px(fx + 2, fy + 4, (30, 90, 50))
    c.px(fx + 5, fy + 4, (30, 90, 50))
    c.rect(fx + 2, fy + 8, 4, 1, (110, 70, 56))
    c.rect(*r['top'], (150, 108, 84), 0.06)
    c.box(24, 0, 2, 4, 2, (160, 112, 88), 0.04)          # nose
    if hat is not None:
        c.rect(*r['top'], hat, 0.08)
        for side in ('right', 'left', 'back'):
            x, y, w, h = r[side]
            c.rect(x, y, w, 2, hat, 0.08)
        c.rect(fx, fy, 8, 2, hat, 0.08)
    rb = c.box(16, 20, 8, 12, 6, robe, 0.07)
    x, y, w, h = rb['front']
    c.rect(x + 3, y, 2, 12, trim, 0.05)
    if apron is not None:
        c.rect(x, y + 3, 8, 9, apron, 0.05)
    c.box(0, 38, 8, 18, 6, robe, 0.07)
    rr = region(0, 38, 8, 18, 6)
    x, y, w, h = rr['front']
    c.rect(x + 3, y, 2, 18, trim, 0.05)
    c.rect(x, y + 16, w, 2, tuple(int(v) for v in np.array(robe) * 0.75))
    c.box(44, 22, 8, 4, 4, robe, 0.07)
    ra = c.box(40, 38, 4, 8, 4, robe, 0.07)
    c.rect(*region(44, 22, 8, 4, 4)['front'][:2], 3, 4, skin)       # hands in the middle of the crossed arms
    c.rect(region(44, 22, 8, 4, 4)['front'][0] + 5, region(44, 22, 8, 4, 4)['front'][1], 3, 4, skin)
    c.box(0, 22, 4, 12, 4, (70, 56, 44), 0.08)
    return c.img


# ---------------------------------------------------------------------------------------------
# animals
# ---------------------------------------------------------------------------------------------
def cow(seed=5):
    c = Canvas(seed)
    brown, white = (84, 58, 40), (220, 214, 204)
    c.box(0, 0, 8, 8, 6, brown, 0.06)                     # head 8x8x6
    r = region(0, 0, 8, 8, 6)
    fx, fy, fw, fh = r['front']
    c.rect(fx + 2, fy + 5, 4, 3, (200, 170, 160))          # muzzle
    c.px(fx + 3, fy + 6, (40, 30, 30))
    c.px(fx + 4, fy + 6, (40, 30, 30))
    c.rect(fx + 1, fy + 2, 1, 1, (20, 20, 20))
    c.rect(fx + 6, fy + 2, 1, 1, (20, 20, 20))
    c.rect(fx + 3, fy, 2, 3, white)
    c.box(22, 0, 1, 3, 1, (220, 216, 200))                 # horn
    c.box(18, 4, 12, 18, 10, brown, 0.06)                  # body 12 x 18 x 10 (drawn lying)
    rb = region(18, 4, 12, 18, 10)
    rng = np.random.default_rng(seed)
    for f, (x, y, w, h) in rb.items():
        for _ in range(3):
            px, py = rng.integers(x, x + max(w - 3, 1)), rng.integers(y, y + max(h - 3, 1))
            c.rect(int(px), int(py), int(rng.integers(2, 5)), int(rng.integers(2, 5)), white, 0.03)
    c.box(0, 16, 4, 12, 4, brown, 0.06)                    # leg
    rl = region(0, 16, 4, 12, 4)
    for f in ('right', 'front', 'left', 'back'):
        x, y, w, h = rl[f]
        c.rect(x, y + 10, w, 2, (50, 40, 34))
    return c.img


def sheep(seed=6):
    c = Canvas(seed)
    c.box(0, 0, 6, 6, 8, (190, 170, 150), 0.05)            # head
    r = region(0, 0, 6, 6, 8)
    fx, fy, fw, fh = r['front']
    c.px(fx + 1, fy + 2, (30, 30, 30))
    c.px(fx + 4, fy + 2, (30, 30, 30))
    c.rect(fx + 2, fy + 4, 2, 1, (140, 110, 100))
    c.box(28, 8, 8, 16, 6, (232, 230, 224), 0.05)          # wool body
    c.box(0, 16, 4, 12, 4, (190, 170, 150), 0.05)          # leg
    c.box(0, 34, 6, 6, 6, (232, 230, 224), 0.05)           # wool cap on the head
    return c.img


def pig(seed=7):
    c = Canvas(seed)
    pink = (236, 160, 158)
    c.box(0, 0, 8, 8, 8, pink, 0.04)
    r = region(0, 0, 8, 8, 8)
    fx, fy, fw, fh = r['front']
    c.px(fx + 1, fy + 3, (250, 250, 250))
    c.px(fx + 2, fy + 3, (30, 30, 30))
    c.px(fx + 5, fy + 3, (30, 30, 30))
    c.px(fx + 6, fy + 3, (250, 250, 250))
    c.box(16, 16, 4, 3, 1, (220, 130, 130))                 # snout
    x, y, w, h = region(16, 16, 4, 3, 1)['front']
    c.px(x + 1, y + 1, (120, 60, 60))
    c.px(x + 2, y + 1, (120, 60, 60))
    c.box(28, 8, 10, 16, 8, pink, 0.05)
    c.box(0, 16, 4, 6, 4, pink, 0.05)
    return c.img


def chicken(seed=8):
    c = Canvas(seed)
    white = (236, 236, 232)
    c.box(0, 0, 4, 6, 3, white, 0.03)                       # head
    r = region(0, 0, 4, 6, 3)
    fx, fy, fw, fh = r['front']
    c.px(fx, fy + 1, (20, 20, 20))
    c.px(fx + 3, fy + 1, (20, 20, 20))
    c.box(14, 0, 4, 2, 2, (240, 180, 40))                   # beak
    c.box(14, 4, 2, 2, 2, (200, 30, 30))                    # wattle
    c.box(0, 9, 6, 8, 6, white, 0.03)                       # body
    c.box(24, 13, 1, 4, 6, (226, 226, 222), 0.03)           # wing
    c.box(26, 0, 3, 5, 3, (240, 180, 40))                   # leg
    return c.img


def debug_skin():
    """Every face region labelled with a flat colour, to check the box mapping."""
    c = Canvas(9)
    cols = {'top': (255, 255, 255), 'bottom': (40, 40, 40), 'right': (220, 40, 40), 'front': (40, 200, 40),
            'left': (40, 40, 220), 'back': (220, 220, 40)}
    for (u, v, w, h, d) in ((0, 0, 8, 8, 8), (16, 16, 8, 12, 4), (40, 16, 4, 12, 4), (32, 48, 4, 12, 4),
                            (0, 16, 4, 12, 4), (16, 48, 4, 12, 4)):
        for f, (x, y, ww, hh) in region(u, v, w, h, d).items():
            c.rect(x, y, ww, hh, cols[f], 0.0)
            c.px(x, y, (0, 0, 0))                   # the top-left texel marked
    return c.img


def make_skins():
    S = {}
    S['you'] = player(YOU, 1)
    S['noah'] = player(NOAH, 2)
    S['you_wrong'] = you_wrong()
    S['villager_farmer'] = villager((110, 84, 52), (160, 130, 80), hat=(212, 186, 96), seed=3)
    S['villager_librarian'] = villager((210, 206, 196), (170, 40, 40), seed=4)
    S['villager_cartographer'] = villager((200, 196, 170), (60, 90, 150), seed=5)
    S['villager_cleric'] = villager((110, 60, 130), (200, 170, 60), seed=6)
    S['villager_smith'] = villager((90, 70, 56), (70, 70, 70), apron=(40, 36, 34), seed=7)
    S['villager_shepherd'] = villager((150, 120, 90), (230, 226, 216), seed=8)
    S['villager_noah'] = villager(NOAH['shirt'], (226, 224, 216), seed=9)      # dressed in Noah's colours
    S['cow'] = cow()
    S['sheep'] = sheep()
    S['pig'] = pig()
    S['chicken'] = chicken()
    S['hand_debug'] = debug_skin()
    return S


if __name__ == '__main__':
    from PIL import Image
    S = make_skins()
    row = np.concatenate([np.kron(S[k], np.ones((5, 5, 1), np.uint8)) for k in SKIN_NAMES[:8]], 1)
    row2 = np.concatenate([np.kron(S[k], np.ones((5, 5, 1), np.uint8)) for k in SKIN_NAMES[8:]] +
                          [np.zeros((320, 320, 4), np.uint8)] * (16 - len(SKIN_NAMES)), 1)
    img = np.concatenate([row, row2], 0)
    bg = np.zeros_like(img)
    bg[..., :3] = (70, 40, 70)
    a = img[..., 3:4] / 255.0
    out = (img[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.uint8)
    Image.fromarray(out).save('/tmp/skins.png')
