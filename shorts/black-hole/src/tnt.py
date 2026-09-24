"""TNT block: 16x16 pixel-art faces drawn in code (side with the TNT label band, top with stick ends and a
fuse, bottom), packed as a 3-layer texture array for the instanced TNT shader."""
import numpy as np

SIZE = 1.0          # one TNT block = one giant-skin pixel = 1 world unit

RED = np.array((205, 42, 30), float)
RED_HI = np.array((236, 72, 52), float)
RED_DK = np.array((150, 26, 20), float)
RED_SEP = np.array((96, 16, 12), float)
PAPER = np.array((236, 234, 226), float)
PAPER_DK = np.array((198, 196, 188), float)
INK = np.array((24, 22, 22), float)

LETTER_T = ["XXX",
            ".X.",
            ".X.",
            ".X."]
LETTER_N = ["X..X",
            "XX.X",
            "X.XX",
            "X..X"]


def _jitter(img, rng, amount=0.05):
    return np.clip(img * (1.0 + (rng.random(img.shape[:2] + (1,)) - 0.5) * 2 * amount), 0, 255)


def side(rng):
    img = np.zeros((16, 16, 3))
    for c in range(16):
        k = c % 4                       # four dynamite sticks across the face
        img[:, c] = (RED_SEP, RED_HI, RED, RED_DK)[k]
    # paper band with the label
    img[5:11, :] = PAPER
    img[5, :] = PAPER_DK
    img[10, :] = PAPER_DK
    x = 2
    for letter in (LETTER_T, LETTER_N, LETTER_T):
        for r, row in enumerate(letter):
            for c, ch in enumerate(row):
                if ch == 'X':
                    img[6 + r, x + c] = INK
        x += len(letter[0]) + 1
    # slight wear: darker top/bottom rows
    img[0] *= 0.85
    img[15] *= 0.8
    return _jitter(img, rng)


def top(rng, bottom=False):
    """Tops of a 4x4 bundle of dynamite sticks: red sticks, dark gaps, small paper caps, fuse in the middle."""
    img = np.zeros((16, 16, 3))
    img[:] = RED
    for k in (0, 4, 8, 12):
        img[k, :] = RED_SEP              # gaps between the sticks
        img[:, k] = RED_SEP
    for cy in (1, 5, 9, 13):
        for cx in (1, 5, 9, 13):
            img[cy + 1, cx + 1] = RED_HI
            img[cy:cy + 2, cx:cx + 2] = img[cy:cy + 2, cx:cx + 2] * 0.5 + np.array((210, 200, 186)) * 0.5
            img[cy, cx] = (224, 216, 204)
    if not bottom:
        img[7:9, 7:9] = (26, 24, 24)     # fuse
        img[6:8, 8] = (84, 78, 72)
    else:
        img *= 0.8
    return _jitter(img, rng)


def texture_layers(seed=4):
    """RGBA layers: side, top, bottom."""
    rng = np.random.default_rng(seed)
    out = []
    for img in (side(rng), top(rng), top(rng, bottom=True)):
        out.append(np.concatenate([np.clip(img, 0, 255), np.full((16, 16, 1), 255.0)], -1).astype(np.uint8))
    return out


def build_mesh():
    """A TNT block centred on the origin: triangles with pos3 normal3 uv2 layer1 (0 side, 1 top, 2 bottom)."""
    h = SIZE / 2
    v = []
    faces = [
        ((1, 0, 0), [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)], 0),
        ((-1, 0, 0), [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)], 0),
        ((0, 1, 0), [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)], 0),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)], 0),
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)], 1),
        ((0, 0, -1), [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)], 2),
    ]
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)]
    for nrm, quad, layer in faces:
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            for k in (a, b, c):
                v.append((*quad[k], *nrm, *uv[k], float(layer)))
    return np.array(v, np.float32)


if __name__ == '__main__':
    import sys
    from PIL import Image
    layers = [l[..., :3] for l in texture_layers()]
    big = [np.kron(l, np.ones((16, 16, 1))) for l in layers]
    Image.fromarray(np.concatenate(big, 1).astype(np.uint8)).save(sys.argv[1])
