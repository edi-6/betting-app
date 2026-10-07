"""The 2D renderer: the cross-section drawn at its own pixel size (16 texels a block) and scaled up with nearest
neighbour, so it stays crisp pixel art at any zoom.

Per frame: the blocks in view (the cut face where there's a block, the wall behind it, darker, where there's air, the
sky above the ground), bevels where blocks meet air, decorations, the cracks on the block being mined; then the light
(sky light, warm light from torches and lava, cold light from soul lanterns and sculk, flood-filled through the air
and smoothed between blocks like the game's smooth lighting) with an x-ray floor so the rock always shows; glowing
texels (lava, ores, crystals); the characters, items and particles; a bloom, then the scale-up.
"""
import cv2
import numpy as np

import art as A
import world as WD

TX = 16                     # texels a block
PX = 6                      # output pixels a texel at zoom 1 (96 px a block)
W, H = 1080, 1920
B = A.B
SKYC = np.array([0.95, 0.97, 1.0])
WARM = np.array([1.0, 0.80, 0.56])
SOUL = np.array([0.45, 0.95, 1.0])
SOUL_SRC = {'lantern', 'sensor', 'shrieker'}


def _bilinear_up(grid, k):
    """Upsample a per-block grid to per-texel with block centres at texel centres (smooth lighting)."""
    h, w = grid.shape[:2]
    return cv2.resize(grid.astype(np.float32), (w * k, h * k), interpolation=cv2.INTER_LINEAR)


class Renderer:
    def __init__(self):
        self.T, self.G = A.block_textures()
        self.lava = [f.astype(np.float32) for f in A.lava_frames(32)]
        self.water = [f.astype(np.float32) for f in A.water_frames(32)]
        names = ['torch', 'lantern', 'cobweb', 'cluster', 'candle', 'sensor', 'rail', 'fence', 'lichen', 'chest',
                 'shrieker']
        self.spr = {n: A.sprite(n) for n in names}
        self.cracks = A.cracks()
        self.solid = np.ones(len(A.BLOCKS), bool)
        self.solid[B['air']] = False
        self.solid[B['lava']] = False
        self.solid[B['water']] = False

    # -- light ----------------------------------------------------------------------------------------
    def light(self, fg, bg, lights):
        """Per-block light, three channels (sky, warm, soul), 0..15."""
        ny, nx = fg.shape
        air = ~self.solid[fg]
        sky = np.zeros((ny, nx), np.float32)
        # sky light comes straight down through open air, fading below the surface (a narrow shaft is gloomy)
        open_col = np.cumprod(air, axis=0).astype(bool)
        depth = np.maximum(0, (WD.SURFACE - (WD.Y_TOP - np.arange(ny)))[:, None] + np.zeros((1, nx)))
        sky[open_col] = np.clip(15 - depth[open_col] / 2.6, 0, 15)
        warm = np.zeros((ny, nx), np.float32)
        soul = np.zeros((ny, nx), np.float32)
        for (x, y), (lv, kind) in lights.items():
            r, c = WD.row(y), WD.col(x)
            if 0 <= r < ny and 0 <= c < nx:
                if kind == 'soul':
                    soul[r, c] = max(soul[r, c], lv)
                else:
                    warm[r, c] = max(warm[r, c], lv)
        warm[fg == B['lava']] = 15
        passable = air | (fg == B['lava'])
        out = []
        for src in (sky, warm, soul):
            L = src.copy()
            for _ in range(15):
                n = L.copy()
                n[1:] = np.maximum(n[1:], L[:-1] - 1)
                n[:-1] = np.maximum(n[:-1], L[1:] - 1)
                n[:, 1:] = np.maximum(n[:, 1:], L[:, :-1] - 1)
                n[:, :-1] = np.maximum(n[:, :-1], L[:, 1:] - 1)
                L = np.maximum(np.where(passable, n, 0), src)
            out.append(L)
        return out

    # -- one frame ------------------------------------------------------------------------------------
    def frame(self, fg, bg, lights, deco, cam, t, crack=None, draw_entities=None, particles=None, xray=0.55,
              dark=0.0, water_cells=(), size=(W, H)):
        """cam: (cx, cy, zoom): the world point at the frame's centre and the zoom. Returns float RGB (H, W, 3)."""
        cx, cy, zoom = cam
        W, H = size
        s = PX * zoom                                  # output px per texel
        vw, vh = W / (s * TX), H / (s * TX)            # view size in blocks
        xa, xb = int(np.floor(cx - vw / 2)) - 1, int(np.ceil(cx + vw / 2)) + 1
        ya, yb = int(np.floor(cy - vh / 2)) - 1, int(np.ceil(cy + vh / 2)) + 1
        ra, rb = WD.row(yb), WD.row(ya)               # rows top..bottom
        ca, cb = WD.col(xa), WD.col(xb)
        ny, nx = fg.shape
        rr = np.clip(np.arange(ra, rb + 1), 0, ny - 1)
        cc = np.clip(np.arange(ca, cb + 1), 0, nx - 1)
        F = fg[np.ix_(rr, cc)]
        Bg = bg[np.ix_(rr, cc)]
        h, w = F.shape
        T = self.T
        solid = self.solid[F]
        # cut face / wall behind
        tid = np.where(solid | (F == B['lava']), F, np.maximum(Bg, 0))
        tex = T[tid]                                                   # (h, w, 16, 16, 3)
        lf = self.lava[int(t * 12) % len(self.lava)]
        tex[F == B['lava']] = lf
        img = tex.transpose(0, 2, 1, 3, 4).reshape(h * TX, w * TX, 3).copy()
        glow = self.G[np.where(solid | (F == B['lava']), F, 0)].transpose(0, 2, 1, 3).reshape(h * TX, w * TX).copy()
        air_px = np.repeat(np.repeat(~solid & (F != B['lava']), TX, 0), TX, 1)
        sky_px = np.repeat(np.repeat((~solid) & (Bg < 0), TX, 0), TX, 1)
        # sky
        if sky_px.any():
            yy = (ya + (h * TX - 1 - np.arange(h * TX)) / TX + 0.0)
            top = np.array([0.36, 0.56, 0.95]) * 255
            hor = np.array([0.70, 0.84, 1.0]) * 255
            k = np.clip((yy - WD.SURFACE) / 18.0, 0, 1)[:, None]
            skyc = hor[None] * (1 - k) + top[None] * k
            img[sky_px] = np.broadcast_to(skyc[:, None, :], img.shape)[sky_px]
        # bevels: block edges that face air
        sh = np.ones((h, w, TX, TX), np.float32)
        def nb(dr, dc):
            p = np.pad(solid, 1, constant_values=True)
            return p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        up, dn, lt, rt = ~nb(-1, 0), ~nb(1, 0), ~nb(0, -1), ~nb(0, 1)
        for m, sl, f in ((up, (slice(None), slice(0, 2), slice(None)), 1.16),
                         (dn, (slice(None), slice(TX - 2, TX), slice(None)), 0.62),
                         (lt, (slice(None), slice(None), slice(0, 2)), 0.82),
                         (rt, (slice(None), slice(None), slice(TX - 2, TX)), 0.82)):
            mm = m & solid
            sub = sh[mm]
            sub[(slice(None),) + sl[1:]] *= f
            sh[mm] = sub
        img *= sh.transpose(0, 2, 1, 3).reshape(h * TX, w * TX)[..., None]
        # water
        for (x, y) in water_cells:
            r, c = WD.row(y) - ra, WD.col(x) - ca
            if 0 <= r < h and 0 <= c < w:
                wf = self.water[int(t * 20 + y) % len(self.water)]
                sl = img[r * TX:(r + 1) * TX, c * TX:(c + 1) * TX]
                sl[:] = sl * 0.25 + wf * 0.75
        # cracks
        if crack is not None:
            (x, y), stage = crack
            r, c = WD.row(y) - ra, WD.col(x) - ca
            if 0 <= r < h and 0 <= c < w and stage >= 0:
                m = self.cracks[min(9, int(stage))]
                img[r * TX:(r + 1) * TX, c * TX:(c + 1) * TX] *= (1 - m)[..., None]
        # decorations (on the wall behind)
        dglow = np.zeros((h * TX, w * TX), np.float32)
        for (x, y), name in deco.items():
            r, c = WD.row(y) - ra, WD.col(x) - ca
            if not (0 <= r < h and 0 <= c < w):
                continue
            spr = self.spr[name]
            sh_, sw_ = spr.shape[:2]
            y0 = r * TX + TX - sh_
            x0 = c * TX + (TX - sw_) // 2
            if name in ('lantern',):
                y0 = r * TX
            if name == 'torch':
                y0 = r * TX + 3
            self._blit(img, spr, x0, y0, dglow, A.GLOWING_SPRITES.get(name, 0.0))
        # light
        Ls = self.light(fg, bg, lights)
        solid_all = self.solid[fg]
        lv = []
        for L in Ls:
            # a block's cut face takes the light of the air next to it (less a little), so walls glow
            f = L.copy()
            f[1:] = np.maximum(f[1:], L[:-1] - 1.5)
            f[:-1] = np.maximum(f[:-1], L[1:] - 1.5)
            f[:, 1:] = np.maximum(f[:, 1:], L[:, :-1] - 1.5)
            f[:, :-1] = np.maximum(f[:, :-1], L[:, 1:] - 1.5)
            L = np.where(solid_all, np.maximum(f, 0), L)
            sub = L[np.ix_(rr, cc)]
            lv.append(_bilinear_up(sub, TX))
        def bright(l):
            return (np.clip(l, 0, 15) / 15.0) ** 1.5
        lrgb = (bright(lv[0])[..., None] * SKYC + bright(lv[1])[..., None] * WARM * 1.05 +
                bright(lv[2])[..., None] * SOUL * 0.9)
        lrgb = np.minimum(lrgb, 1.25)
        self.last_light = (lrgb, ra, ca)
        amb = 0.20
        mult = np.where(air_px[..., None], 0.42 * (amb + lrgb), xray + (1 - xray) * np.minimum(lrgb, 1.0) * 1.1)
        mult[sky_px] = 1.0                                                          # the sky isn't darkened
        img = img * mult
        img += self._glow_add(img, glow, dglow)
        # entities, items, particles (lit by the light where they are)
        to_px = lambda x, y: ((x - xa) * TX, (yb + 1 - y) * TX)
        if draw_entities is not None:
            draw_entities(img, to_px, self)
        if particles is not None:
            for (x, y, col, size) in particles:
                px, py = to_px(x, y)
                px, py = int(px), int(py)
                if 0 <= px < img.shape[1] - size and 0 <= py < img.shape[0] - size:
                    img[py:py + size, px:px + size] = col
        # bloom
        br = np.clip(img - 200, 0, None)
        if br.max() > 0:
            bl = cv2.GaussianBlur(br, (0, 0), 5) * 0.8 + cv2.GaussianBlur(br, (0, 0), 20) * 1.1
            img = img + bl
        # to the screen
        ox = (cx - xa) * TX * s - W / 2
        oy = (yb + 1 - cy) * TX * s - H / 2
        src = np.clip(img, 0, 255).astype(np.float32)
        if s < 2.0:
            # zoomed far out: average the texels down first, so the pixel art doesn't alias into stripes
            sw, sh_ = max(1, int(round(src.shape[1] * s / 2.0))), max(1, int(round(src.shape[0] * s / 2.0)))
            k = sw / src.shape[1]
            src = cv2.resize(src, (sw, sh_), interpolation=cv2.INTER_AREA)
            M = np.array([[s / k, 0, -ox], [0, s / k, -oy]], np.float32)
            out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            M = np.array([[s, 0, -ox], [0, s, -oy]], np.float32)
            out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        if dark > 0:
            out *= 1.0 - dark
        return out

    def light_at(self, x, y):
        """Light colour at a world point (after a frame)."""
        lrgb, ra, ca = self.last_light
        r = (WD.row(np.floor(y)) - ra) * TX + int((1 - (y - np.floor(y))) * TX)
        c = (WD.col(np.floor(x)) - ca) * TX + int((x - np.floor(x)) * TX)
        r = int(np.clip(r, 0, lrgb.shape[0] - 1))
        c = int(np.clip(c, 0, lrgb.shape[1] - 1))
        return lrgb[r, c]

    def _glow_add(self, img, glow, dglow):
        g = np.maximum(glow * 0.55, dglow)
        return np.zeros_like(img) if g.max() <= 0 else img * g[..., None] * 0.9

    @staticmethod
    def _blit(img, spr, x0, y0, glow_map=None, glow=0.0, light=None):
        sh, sw = spr.shape[:2]
        H_, W_ = img.shape[:2]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W_, x0 + sw), min(H_, y0 + sh)
        if xb <= xa or yb <= ya:
            return
        s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
        a = s[..., 3:4] / 255.0
        col = s[..., :3]
        if light is not None:
            col = col * light
        img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + col * a
        if glow_map is not None and glow > 0:
            glow_map[ya:yb, xa:xb] = np.maximum(glow_map[ya:yb, xa:xb], a[..., 0] * glow)
