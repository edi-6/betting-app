"""Steve-style giant: parts, flesh colours and the skin (own pixel art, from the Steve vs Swords video)."""
import numpy as np

RNG = np.random.default_rng


def _hex(c):
    return np.array(c, dtype=np.float64)


def _shade(img, rng, amount=0.05):
    """Subtle per-pixel brightness jitter, like hand-shaded skins."""
    j = 1.0 + (rng.random(img.shape[:2]) - 0.5) * 2 * amount
    return np.clip(img * j[..., None], 0, 255)


def _paint(rows, palette, rng, jitter=0.045):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        assert len(row) == w, (row, w)
        for c, ch in enumerate(row):
            opts = palette[ch]
            if isinstance(opts, list):
                img[r, c] = opts[rng.integers(len(opts))]
            else:
                img[r, c] = opts
    return _shade(img, rng, jitter)


# ---------------------------------------------------------------------------------------------
# Character skin
# ---------------------------------------------------------------------------------------------
HAIR = [_hex((64, 42, 26)), _hex((72, 48, 30)), _hex((56, 36, 22)), _hex((80, 54, 33))]
SKIN = [_hex((200, 146, 104)), _hex((194, 140, 99)), _hex((206, 152, 110)), _hex((189, 136, 96))]
SKIN_DK = _hex((168, 116, 80))
NOSE = _hex((150, 96, 62))
MOUTH = _hex((96, 56, 38))
BEARD = [_hex((122, 76, 50)), _hex((114, 70, 46))]
EYE_W = _hex((246, 246, 246))
IRIS = _hex((84, 62, 172))
SHIRT = [_hex((22, 178, 182)), _hex((16, 168, 173)), _hex((30, 188, 190)), _hex((12, 158, 165))]
SHIRT_DK = [_hex((10, 140, 148)), _hex((6, 132, 140))]
PANTS = [_hex((58, 54, 176)), _hex((52, 48, 164)), _hex((64, 60, 186)), _hex((48, 44, 156))]
SHOE = [_hex((108, 108, 112)), _hex((96, 96, 100)), _hex((118, 118, 122))]

SKIN_PAL = {
    'H': HAIR, 'S': SKIN, 's': SKIN_DK, 'N': NOSE, 'M': MOUTH, 'B': BEARD, 'W': EYE_W, 'I': IRIS,
    'T': SHIRT, 't': SHIRT_DK, 'P': PANTS, 'G': SHOE,
}


def make_skin(seed=7):
    """Return {part: {face: HxWx3 float array}} for a Steve-like blocky character (own design).

    Face image convention (as seen from outside the part):
      ny = front (-Y), py = back (+Y), nx = -X side, px = +X side, pz = top, nz = bottom.
      Row 0 is the top edge (for pz: the +Y edge), column 0 is the viewer's left.
    """
    rng = RNG(seed)
    P = lambda rows, j=0.045: _paint(rows, SKIN_PAL, rng, j)
    head = {
        'ny': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "HSSSSSSH",
                 "SWISSIWS",
                 "SSsNNsSS",
                 "SBMMMMBS",
                 "SSSSSSSS",
                 "SSSSSSSS"]),
        # viewer at -X: image left = back (+Y), right = front (-Y)
        'nx': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "HHHHSSSS",
                 "HHHSSSSS",
                 "HHSSSSSS",
                 "HSSSSSSS",
                 "HSSSSSSS",
                 "SSSSSSSS"]),
        'px': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "SSSSHHHH",
                 "SSSSSHHH",
                 "SSSSSSHH",
                 "SSSSSSSH",
                 "SSSSSSSH",
                 "SSSSSSSS"]),
        'py': P(["HHHHHHHH"] * 7 + ["HHHHHHHH"]),
        'pz': P(["HHHHHHHH"] * 8),
        'nz': P(["ssssssss"] * 8, 0.03),
    }
    shirt_rows = ["TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "tTTTTTTt",
                  "TtTTTtTT",
                  "PPPPPPPP",
                  "PPPPPPPP",
                  "PPPPPPPP"]
    body = {
        'ny': P(shirt_rows),
        'py': P(shirt_rows),
        'nx': P(["TTTT"] * 8 + ["tTTt"] + ["PPPP"] * 3),
        'px': P(["TTTT"] * 8 + ["tTTt"] + ["PPPP"] * 3),
        'pz': P(["TTTTTTTT"] * 4),
        'nz': P(["PPPPPPPP"] * 4),
    }
    arm_rows = ["TTTT", "TTTT", "TTTT", "tTTt"] + ["SSSS"] * 8
    arm = lambda: {
        'ny': P(arm_rows), 'py': P(arm_rows), 'nx': P(arm_rows), 'px': P(arm_rows),
        'pz': P(["TTTT"] * 4), 'nz': P(["SSSS"] * 4),
    }
    leg_rows = ["PPPP"] * 10 + ["GGGG"] * 2
    leg = lambda: {
        'ny': P(leg_rows), 'py': P(leg_rows), 'nx': P(leg_rows), 'px': P(leg_rows),
        'pz': P(["PPPP"] * 4), 'nz': P(["GGGG"] * 4),
    }
    # vertical shading gradient on shirt / pants so large flat areas don't look plastic
    for part in (body,):
        for f in ('ny', 'py', 'nx', 'px'):
            h = part[f].shape[0]
            g = np.linspace(1.03, 0.95, h)[:, None, None]
            part[f] = np.clip(part[f] * g, 0, 255)
    return {'head': head, 'body': body, 'arm_r': arm(), 'arm_l': arm(), 'leg_r': leg(), 'leg_l': leg()}



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


