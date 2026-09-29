"""The maps on the wall of the map house (and in his hand): 128 x 128 top-down views of the world in the muted
colours of the game's maps, with the relief shaded against the block to the north, on a paper border.

Variants: TODAY (the world as it is), BEFORE (only the house, the land untouched), ABANDONED (overgrown, faded, the
fields gone), BURNED (black ruins, embers, and a crater where the player spawns), OLD_n (the village half built,
scribbled over). Each carries a tiny date in its corner, matching a date sign in the cavern.
"""
import numpy as np

import blocks as BL
import pixelfont as PF
import worldgen as WG

MAP_BOX = (-60, -78, 128)          # x0, y0, size (blocks): the village, the spawn and the lake's edge
PAL = {
    'grass_block': (110, 150, 62), 'dirt': (140, 104, 70), 'dirt_path': (160, 130, 84), 'farmland': (102, 74, 48),
    'water': (60, 86, 170), 'sand': (206, 196, 146), 'stone': (118, 118, 118), 'cobblestone': (112, 112, 112),
    'stone_bricks': (120, 120, 122), 'oak_planks': (150, 120, 74), 'spruce_planks': (104, 78, 48),
    'spruce_stairs': (104, 78, 48), 'oak_stairs': (150, 120, 74), 'cobblestone_stairs': (112, 112, 112),
    'spruce_slab': (104, 78, 48), 'cobblestone_slab': (112, 112, 112), 'oak_log': (100, 80, 50),
    'birch_log': (200, 196, 180), 'spruce_log': (70, 50, 30), 'oak_leaves': (50, 104, 36),
    'birch_leaves': (86, 122, 52), 'spruce_leaves': (44, 76, 48), 'wheat': (184, 160, 70),
    'wheat_young': (96, 140, 50), 'hay_block': (184, 150, 44), 'dark_oak_planks': (70, 50, 30),
    'gravel': (128, 122, 118), 'white_concrete': (210, 210, 210), 'gray_concrete': (60, 62, 66),
    'dead_log': (84, 76, 66), 'oak_fence': (150, 120, 74), 'lantern': (200, 170, 90), 'glass_pane': (180, 190, 200),
}


def top_view(w, box=MAP_BOX):
    """(size, size) block names and heights of the top visible blocks (plants ignored)."""
    x0, y0, n = box
    X0, Y0, Z0 = w.origin
    i0, j0 = x0 - X0, y0 - Y0
    ids = w.ids[i0:i0 + n, j0:j0 + n, :]
    plants = np.isin(ids, [BL.B[p] for p in ('tall_grass', 'poppy', 'dandelion', 'cornflower', 'torch', 'chain',
                                               'dead_bush', 'cobweb', 'oak_sign', 'item_frame')])
    solid = (ids != 0) & ~plants
    zi = np.where(solid.any(2), w.size[2] - 1 - np.argmax(solid[:, :, ::-1], axis=2), 0)
    top = np.take_along_axis(ids, zi[..., None], 2)[..., 0]
    return top, zi + Z0


def colourise(top, h, pal=PAL, default=(120, 120, 120)):
    names = np.array([b.name for b in BL.REG])
    lut = np.array([pal.get(nm, default) for nm in names], float)
    img = lut[top]
    # relief: brighter where higher than the block to the north, darker where lower (like the game's maps)
    hn = np.roll(h, -1, axis=1)
    shade = np.where(h > hn, 1.12, np.where(h < hn, 0.82, 1.0))
    img = img * shade[..., None]
    return img


def to_map_image(img_xy, rng, date=None, age=0.0, marks=()):
    """img_xy: (x, y) indexed colours -> (128,128,4) uint8 as seen on the map (north up), on paper."""
    img = np.transpose(img_xy, (1, 0, 2))[::-1]             # rows = -y (north at the top), cols = x
    n = img.shape[0]
    paper = np.array((220, 204, 160), float)
    img = img * 0.82 + paper * 0.18                            # the game's maps look inked on paper
    grain = 1 + 0.06 * (rng.random((n, n)) - 0.5)
    img *= grain[..., None]
    if age > 0:
        yellow = np.array((196, 170, 110), float)
        img = img * (1 - 0.45 * age) + yellow * 0.45 * age
        stains = rng.random((n // 8, n // 8)) < 0.1 * age
        img[np.kron(stains, np.ones((8, 8), bool))] *= 0.85
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = np.clip(img, 0, 255)
    out[..., 3] = 255
    # paper border
    b = 4
    out[:b] = out[-b:] = (206, 186, 140, 255)
    out[:, :b] = out[:, -b:] = (206, 186, 140, 255)
    out[0] = out[-1] = (120, 96, 60, 255)
    out[:, 0] = out[:, -1] = (120, 96, 60, 255)
    if date:
        m = PF.text_mask(date)
        hh, ww = m.shape
        y0, x0 = n - b - hh - 1, n - b - ww - 2
        sub = out[y0:y0 + hh, x0:x0 + ww]
        sub[m] = (70, 50, 30, 255)
    for (mx, my, col) in marks:
        cx, cy = world_to_map(mx, my)
        cx, cy = int(cx), int(cy)
        out[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = (*col, 255)
    return out


def world_to_map(x, y, box=MAP_BOX):
    x0, y0, n = box
    return (x - x0), (n - 1 - (y - y0))


def make_maps(w_today, rng=None):
    """All the map images, keyed by name."""
    rng = rng or np.random.default_rng(5)
    top, h = top_view(w_today)
    base = colourise(top, h)
    names = np.array([b.name for b in BL.REG])
    built = np.isin(names[top], ['oak_planks', 'spruce_planks', 'spruce_stairs', 'oak_stairs', 'cobblestone_stairs',
                                 'spruce_slab', 'cobblestone_slab', 'cobblestone', 'stone_bricks', 'dark_oak_planks',
                                 'oak_fence', 'lantern', 'glass_pane', 'oak_log', 'spruce_log', 'birch_log'])
    trees = np.isin(names[top], ['oak_leaves', 'birch_leaves', 'spruce_leaves'])
    x0, y0, n = MAP_BOX
    xs = np.arange(n) + MAP_BOX[0] + 0.5
    ys = np.arange(n) + MAP_BOX[1] + 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    hx0, hy0 = WG.HOUSE_O
    house = (X >= hx0 - 1) & (X <= hx0 + 10) & (Y >= hy0 - 1) & (Y <= hy0 + 12)
    grass = np.array(PAL['grass_block'], float)
    out = {}
    out['TODAY'] = to_map_image(base, rng, WG.TODAY)
    out['TODAY_SMALL'] = out['TODAY']
    # before: only the house (and the trees)
    bef = base.copy()
    manmade = (built | np.isin(names[top], ['dirt_path', 'farmland', 'wheat', 'wheat_young', 'hay_block', 'water'])) & \
        ~house & ~trees
    lake = np.isin(names[top], ['water']) & (X < -50)
    manmade &= ~lake
    bef[manmade] = grass
    out['BEFORE'] = to_map_image(bef, rng, '2018-06-14', age=0.6)
    # abandoned: fields and paths grown over, roofs holed, darker greens
    ab = base.copy()
    fields = np.isin(names[top], ['farmland', 'wheat', 'wheat_young', 'dirt_path', 'hay_block'])
    ab[fields] = grass * 0.8
    holes = built & (rng.random(built.shape) < 0.25)
    ab[holes] = (50, 40, 30)
    ab[~built & ~trees] *= np.array((0.8, 0.9, 0.75))
    out['ABANDONED'] = to_map_image(ab, rng, '2020-11-03', age=0.4)
    # burned: black ruins, ash, embers, and a crater where the player spawns
    bu = base * np.array((0.55, 0.5, 0.45))
    bu[built] = (30, 26, 24)
    ember = built & (rng.random(built.shape) < 0.12)
    bu[ember] = (220, 90, 30)
    bu[trees] = (40, 36, 30)
    cx, cy = WG.SPAWN[0], WG.SPAWN[1] + 6
    dc = np.hypot(X - cx, Y - cy)
    bu[dc < 30] = (22, 20, 20)
    ring = (dc >= 30) & (dc < 32)
    bu[ring] = (70, 60, 50)
    out['BURNED'] = to_map_image(bu, rng, '2025-07-01', age=0.3)
    # old maps: the village half built, one of them scribbled over
    for k in range(1, 7):
        o = base.copy()
        keep = rng.random() * 0.8
        drop = built & (np.kron(rng.random((n // 16, n // 16)), np.ones((16, 16))) > keep)
        o[drop] = grass
        img = to_map_image(o, rng, None, age=0.3 + 0.1 * k)
        if k == 4:
            # scribbles: a spiral of red over the house
            hx, hy = world_to_map(hx0 + 4, hy0 + 5)
            for t in np.linspace(0, 6 * np.pi, 400):
                rr = 1.5 + t * 1.2
                px, py = int(hx + rr * np.cos(t)), int(hy + rr * np.sin(t))
                if 4 <= px < n - 4 and 4 <= py < n - 4:
                    img[py, px] = (150, 20, 20, 255)
        out[f'OLD_{k}'] = img
    return out


if __name__ == '__main__':
    from PIL import Image
    w = WG.build('village')
    maps = make_maps(w)
    keys = ['BEFORE', 'ABANDONED', 'BURNED', 'TODAY', 'OLD_1', 'OLD_4']
    sheet = np.concatenate([np.kron(maps[k], np.ones((3, 3, 1), np.uint8)) for k in keys], 1)
    Image.fromarray(sheet).save('/tmp/maps.png')
    print('ok')
