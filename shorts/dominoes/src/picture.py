"""The picture hidden in the field: Herobrine. The Steve skin from the earlier videos (own pixel art), but with blank,
white eyes, as a bust on a night-sky background. One domino is one pixel; a skin pixel is SCALE x SCALE dominoes.

picture(nrow, ncol) -> uint8 (nrow, ncol, 3), row 0 = the top of the picture (the north end of the field). Also a
mask of the eye pixels, which glow at the end.
"""
import numpy as np

SCALE = 8
HAIR = [(64, 42, 26), (72, 48, 30), (56, 36, 22), (80, 54, 33)]
SKIN = [(200, 146, 104), (194, 140, 99), (206, 152, 110), (189, 136, 96)]
SKIN_DK = (168, 116, 80)
NOSE = (150, 96, 62)
MOUTH = (96, 56, 38)
BEARD = [(122, 76, 50), (114, 70, 46)]
EYE = (255, 255, 255)
SHIRT = [(22, 178, 182), (16, 168, 173), (30, 188, 190), (12, 158, 165)]
SHIRT_DK = [(10, 140, 148), (6, 132, 140)]
PAL = {'H': HAIR, 'S': SKIN, 's': SKIN_DK, 'N': NOSE, 'M': MOUTH, 'B': BEARD, 'E': EYE, 'T': SHIRT,
       't': SHIRT_DK}
FACE = ["HHHHHHHH",
        "HHHHHHHH",
        "HSSSSSSH",
        "SEESSEES",           # Steve's eyes, but blank white
        "SSsNNsSS",
        "SBMMMMBS",
        "SSSSSSSS",
        "SSSSSSSS"]
TOP_MARGIN = 9                    # rows of night sky above the head


def _paint(rows, rng):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            opts = PAL[ch]
            img[r, c] = opts[rng.integers(len(opts))] if isinstance(opts, list) else opts
    return img


def picture(nrow, ncol, seed=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((nrow, ncol, 3))
    eyes = np.zeros((nrow, ncol), bool)
    # night sky: deep blue at the top to near black, a few stars
    v = np.linspace(0, 1, nrow)[:, None]
    sky = np.array([22, 26, 58]) * (1 - v[..., None]) + np.array([8, 9, 20]) * v[..., None]
    out[:] = sky
    stars = rng.random((nrow, ncol)) < 0.012
    out[stars] = (200, 204, 230)
    # the head: 8 x 8 skin pixels
    face = _paint(FACE, rng)
    x0 = (ncol - 8 * SCALE) // 2
    y0 = TOP_MARGIN
    for r in range(8):
        for c in range(8):
            out[y0 + r * SCALE:y0 + (r + 1) * SCALE, x0 + c * SCALE:x0 + (c + 1) * SCALE] = face[r, c]
            if FACE[r][c] == 'E':
                eyes[y0 + r * SCALE:y0 + (r + 1) * SCALE, x0 + c * SCALE:x0 + (c + 1) * SCALE] = True
    # the shoulders: the shirt under the head, the arms' sleeves at the sides, the rest of the height
    yb = y0 + 8 * SCALE
    nb = (nrow - yb + SCALE - 1) // SCALE
    body = _paint(["T" * 8] * nb, rng)
    for r in range(nb):
        for c in range(8):
            out[yb + r * SCALE:yb + (r + 1) * SCALE, x0 + c * SCALE:x0 + (c + 1) * SCALE] = body[r, c]
    sleeve = _paint(["tT"] * nb, rng)
    for r in range(nb):
        out[yb + r * SCALE:yb + (r + 1) * SCALE, :x0] = sleeve[r, 1] * 0.92
        out[yb + r * SCALE:yb + (r + 1) * SCALE, x0 + 8 * SCALE:] = sleeve[r, 1] * 0.92
    # the arms' inner edge: a darker line against the body
    out[yb:, x0 - 1] *= 0.8
    out[yb:, x0 + 8 * SCALE] *= 0.8
    # each domino a little different, like real painted dominoes
    out *= 1.0 + (rng.random((nrow, ncol, 1)) - 0.5) * 0.07
    out[eyes] = EYE
    return np.clip(out, 0, 255).astype(np.uint8), eyes


if __name__ == '__main__':
    import sys
    from PIL import Image
    img, eyes = picture(132, 72)
    Image.fromarray(img).resize((72 * 8, 132 * 8), Image.NEAREST).save(sys.argv[1] if len(sys.argv) > 1 else 'pic.png')
    print(img.shape, eyes.sum())
