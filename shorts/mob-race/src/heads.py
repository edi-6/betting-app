"""The racers' heads, drawn from scratch as 8 x 8 pixel art in the spirit of the game's mob heads (own designs):
a face, and the colours of the rest of the head. The marbles are balls with these heads mapped onto them.

Each head: 'face' rows (8 strings of 8 palette keys), 'rest' (the key the rest of the ball is made of, noise-shaded)
and optional 'top' rows (hair, ears) for the upper part of the ball.
"""
import numpy as np

N = 8

HEADS = {
    'creeper': dict(
        label='Creeper', color=(98, 196, 74),
        pal={'g': (94, 176, 70), 'G': (126, 206, 96), 'd': (64, 128, 48), 'k': (18, 20, 18), 'K': (40, 44, 40)},
        face=["gGgdgGgg",
              "gdgGggdg",
              "gkkggkkG",
              "gkkGgkkg",
              "gGgkkgdg",
              "gdkkkkgg",
              "gGkKKkGg",
              "ggkgGkgd"],
        rest=['g', 'G', 'd', 'g']),
    'pig': dict(
        label='Pig', color=(240, 150, 160),
        pal={'p': (240, 160, 164), 'P': (250, 186, 188), 'd': (214, 124, 132), 'w': (250, 250, 250),
             'k': (24, 20, 22), 's': (232, 122, 136), 'n': (150, 60, 76)},
        face=["pPpppPpp",
              "pppPpppP",
              "wkpppPkw",
              "pppppppp",
              "pssssssp",
              "psnssnsp",
              "pssssssp",
              "pPpppppd"],
        rest=['p', 'P', 'p', 'd']),
    'enderman': dict(
        label='Enderman', color=(44, 30, 60),
        pal={'b': (22, 20, 26), 'B': (36, 32, 42), 'm': (220, 90, 255), 'M': (250, 190, 255), 'v': (160, 40, 200)},
        face=["bBbbbBbb",
              "bbbBbbbB",
              "bBbbbBbb",
              "bbbbbbbb",
              "vmMvvMmv",
              "bbBbbBbb",
              "bBbbbbbb",
              "bbbBbbBb"],
        rest=['b', 'B', 'b', 'b'], glow='mMv'),
    'blaze': dict(
        label='Blaze', color=(250, 196, 40),
        pal={'y': (250, 206, 46), 'Y': (255, 236, 120), 'o': (230, 140, 20), 'k': (60, 30, 10), 'r': (180, 60, 20)},
        face=["yYyoyYyo",
              "YyyyYyyy",
              "ykkyykky",
              "yYyyyyYy",
              "ooyYyyoo",
              "yyrrrryy",
              "yoyyyyoy",
              "oyYyoyYy"],
        rest=['y', 'Y', 'o', 'y'], glow='yYo'),
    'skeleton': dict(
        label='Skeleton', color=(214, 214, 210),
        pal={'w': (206, 206, 202), 'W': (234, 234, 230), 'g': (170, 170, 166), 'k': (40, 40, 40), 'K': (80, 80, 78)},
        face=["wWwwgwWw",
              "wwwWwwww",
              "wkkwwkkw",
              "wkKwwKkw",
              "wwwkkwww",
              "wgwwwwgw",
              "wkkkkkkw",
              "wwgwwgww"],
        rest=['w', 'W', 'g', 'w']),
    'warden': dict(
        label='Warden', color=(24, 88, 104),
        pal={'t': (18, 54, 66), 'T': (28, 76, 90), 'c': (40, 220, 230), 'C': (160, 255, 255), 'k': (8, 20, 26),
             'b': (70, 60, 50)},
        face=["tTtcttTt",
              "tctTtTct",
              "tttttttt",
              "tkttttkt",
              "tTtCCtTt",
              "tttccttt",
              "tbkkkkbt",
              "tTtkktTt"],
        rest=['t', 'T', 't', 'k'], glow='cC'),
    'fox': dict(
        label='Fox', color=(236, 124, 44),
        pal={'o': (230, 120, 40), 'O': (250, 154, 70), 'd': (190, 84, 26), 'w': (250, 244, 236), 'k': (26, 20, 18),
             'n': (40, 28, 24)},
        face=["oOoooOoo",
              "oooOoooO",
              "okwooowk",
              "oooooooo",
              "wwwnnwww",
              "wwwwwwww",
              "owwwwwwo",
              "oowwwwoo"],
        rest=['o', 'O', 'd', 'o']),
    'villager': dict(
        label='Villager', color=(170, 120, 80),
        pal={'s': (190, 140, 104), 'S': (206, 160, 124), 'h': (96, 64, 40), 'e': (40, 170, 90), 'w': (246, 246, 246),
             'n': (160, 110, 80), 'N': (136, 90, 62), 'b': (70, 46, 30)},
        face=["hhhhhhhh",
              "sSssssSs",
              "bbbbbbbb",
              "wessesew",
              "sssnnsss",
              "sssnNsss",
              "sSsnNsSs",
              "ssbbbbss"],
        rest=['s', 'S', 's', 'n'], top='h'),
}
ORDER = ('creeper', 'pig', 'enderman', 'blaze', 'skeleton', 'warden', 'fox', 'villager')


def face_image(name):
    """8 x 8 RGB of the face, and a glow mask."""
    h = HEADS[name]
    img = np.zeros((N, N, 3), np.uint8)
    glow = np.zeros((N, N), bool)
    for r, row in enumerate(h['face']):
        for c, ch in enumerate(row[:N]):
            img[r, c] = h['pal'][ch]
            glow[r, c] = ch in h.get('glow', '')
    return img, glow


def ball_colors(name, dirs, rng):
    """Colour (and glow flag) of a ball's voxels from their unit directions (x right, y away from the camera, z up):
    the face on the front (-y) cap, the head's colours elsewhere (hair on top for some)."""
    h = HEADS[name]
    img, glow = face_image(name)
    n = len(dirs)
    col = np.zeros((n, 3))
    gl = np.zeros(n, bool)
    pal = h['pal']
    rest = np.array([pal[k] for k in h['rest']], float)
    pick = rng.integers(0, len(rest), n)
    col[:] = rest[pick] * (1 + (rng.random((n, 1)) - 0.5) * 0.08)
    if 'top' in h:
        top = dirs[:, 2] > 0.55
        col[top] = np.array(pal[h['top']], float) * (1 + (rng.random((top.sum(), 1)) - 0.5) * 0.1)
    # the face: a square projected onto the front of the ball
    front = dirs[:, 1] < -0.3
    u = np.clip((dirs[front, 0] / 0.78 + 1.0) * 0.5, 0.0, 0.9999)
    v = np.clip((-dirs[front, 2] / 0.78 + 1.0) * 0.5, 0.0, 0.9999)
    inside = (np.abs(dirs[front, 0]) < 0.78) & (np.abs(dirs[front, 2]) < 0.78)
    fi = np.nonzero(front)[0][inside]
    cc = (u[inside] * N).astype(int)
    rr = (v[inside] * N).astype(int)
    col[fi] = img[rr, cc]
    gl[fi] = glow[rr, cc]
    return np.clip(col, 0, 255).astype(np.uint8), gl


def icon(name, size=64):
    """A round portrait of the ball for the HUD (face on a disc of the head colour), RGBA."""
    from PIL import Image
    img, _ = face_image(name)
    big = np.array(Image.fromarray(img).resize((size, size), Image.NEAREST))
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2.0
    r = np.hypot(xx, yy) / (size / 2.0)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = big
    out[..., 3] = (np.clip((1.0 - r) * size / 2.0, 0, 1) * 255).astype(np.uint8)
    return out
