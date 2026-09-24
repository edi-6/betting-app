"""Minecraft-style overlay for the hydraulic press, drawn from scratch in the pixel font at GUI scales:

* the boss bar at the top: the block under test is the 'boss', its bar is its integrity (a second bar for the
  press itself joins it in the finale and drains);
* the pressure in tons as the experience level over the XP bar (log scale, 1 t .. 100,000 t);
* the hotbar with the nine blocks: the selector moves along, crushed blocks are greyed and crossed out, the one that
  survives gets the enchantment glint;
* /title-style result titles with a subtitle, the advancement toast, a VCR rewind mark, flashes and grading.
"""
import numpy as np
from PIL import Image

import pixelart as PA
import pixelfont as PF

MC = {'black': (0, 0, 0), 'dark_red': (170, 0, 0), 'gold': (255, 170, 0), 'gray': (170, 170, 170),
      'dark_gray': (85, 85, 85), 'green': (85, 255, 85), 'aqua': (85, 255, 255), 'red': (255, 85, 85),
      'light_purple': (255, 85, 255), 'yellow': (255, 255, 85), 'white': (255, 255, 255)}
XP_GREEN = np.array([128, 255, 32], float)
XP_RED = np.array([255, 64, 48], float)
TOAST_PINK = (255, 136, 255)
# boss bar colours: fill, top highlight, empty
BAR_COLS = {
    'pink': ((236, 0, 168), (255, 128, 226), (70, 12, 56)),
    'blue': ((0, 156, 255), (120, 214, 255), (8, 40, 80)),
    'red': ((232, 18, 18), (255, 118, 104), (72, 8, 8)),
    'green': ((26, 214, 26), (136, 255, 116), (8, 62, 8)),
    'yellow': ((240, 222, 0), (255, 255, 136), (72, 64, 4)),
    'purple': ((142, 42, 246), (204, 146, 255), (42, 14, 78)),
    'white': ((226, 226, 226), (255, 255, 255), (72, 72, 72)),
}


# ---------------------------------------------------------------------------------------------
# compositing helpers
# ---------------------------------------------------------------------------------------------
def over(dst, src, x, y, opacity=1.0):
    """Alpha-composite an RGBA uint8 sprite onto a float RGB frame at integer position (x, y)."""
    if opacity <= 0.0:
        return
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def add(dst, src, x, y, gain=1.0):
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    dst[y0:y1, x0:x1] += s[..., :3] * (s[..., 3:4] / 255.0) * gain


def blit(dst, src, x, y):
    """Composite RGBA onto RGBA (uint8, in place; for building sprites)."""
    h, w = src.shape[:2]
    d = dst[y:y + h, x:x + w].astype(np.float32)
    s = src.astype(np.float32)
    a = s[..., 3:4] / 255.0
    d[..., :3] = d[..., :3] * (1 - a) + s[..., :3] * a
    d[..., 3:4] = s[..., 3:4] + d[..., 3:4] * (1 - a)
    dst[y:y + h, x:x + w] = np.clip(d, 0, 255).astype(np.uint8)


def up(img, s):
    """Nearest-neighbour upscale of a GUI-resolution sprite by a (possibly fractional) GUI scale."""
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                                  Image.NEAREST))


def scaled(sprite, s):
    if abs(s - 1.0) < 1e-3:
        return sprite
    return up(sprite, s)


def grey(sprite, k=1.0, dark=0.55):
    out = sprite.astype(np.float32)
    g = out[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    out[..., :3] = out[..., :3] * (1 - k) + (g[..., None] * dark) * k
    return np.clip(out, 0, 255).astype(np.uint8)


def ease_out_back(t, s=2.2):
    t = float(np.clip(t, 0.0, 1.0)) - 1.0
    return 1.0 + (s + 1.0) * t ** 3 + s * t ** 2


def smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def rect(img, x0, y0, x1, y1, col):
    """Fill [x0, x1) x [y0, y1) of an RGBA GUI image with an RGBA colour."""
    img[y0:y1, x0:x1] = col


def glow_sprite(w, h, col, soft=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    a = np.clip(1.0 - r, 0, 1) ** (1.0 / soft) if soft > 0 else (r < 1)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = col
    out[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------
# GUI pieces (built at GUI resolution, one texel per GUI pixel)
# ---------------------------------------------------------------------------------------------
def hotbar_texture():
    """182 x 22: nine 20 px slots on a translucent dark strip."""
    t = np.zeros((22, 182, 4), np.uint8)
    rect(t, 0, 0, 182, 22, (12, 12, 14, 150))
    rect(t, 0, 0, 182, 1, (0, 0, 0, 235))
    rect(t, 0, 21, 182, 22, (0, 0, 0, 235))
    rect(t, 0, 0, 1, 22, (0, 0, 0, 235))
    rect(t, 181, 0, 182, 22, (0, 0, 0, 235))
    for k in range(9):
        x0 = 1 + 20 * k
        rect(t, x0, 1, x0 + 20, 21, (104, 104, 108, 245))              # slot frame
        rect(t, x0 + 1, 2, x0 + 19, 20, (150, 150, 154, 245))          # light bevel (top/left)
        rect(t, x0 + 2, 3, x0 + 19, 20, (70, 70, 74, 245))             # dark bevel (bottom/right)
        rect(t, x0 + 2, 3, x0 + 18, 19, (26, 26, 30, 150))             # the see-through inside
    return t


def selector_texture():
    """24 x 24: the hotbar's selection frame."""
    t = np.zeros((24, 24, 4), np.uint8)
    rect(t, 0, 0, 24, 24, (0, 0, 0, 255))
    rect(t, 1, 1, 23, 23, (255, 255, 255, 255))
    rect(t, 2, 2, 23, 23, (208, 208, 208, 255))
    rect(t, 3, 3, 21, 21, (0, 0, 0, 255))
    rect(t, 4, 4, 20, 20, (0, 0, 0, 0))
    return t


def bar_texture(frac, fill, hi, empty, notches=10, w=182):
    """w x 5 boss / XP bar: a highlight row on top, a dark row at the bottom, notches."""
    t = np.zeros((5, w, 4), np.uint8)
    n = int(round(np.clip(frac, 0.0, 1.0) * (w - 2)))
    dark = tuple(int(c * 0.55) for c in fill)
    rect(t, 0, 0, w, 5, (0, 0, 0, 255))
    rect(t, 1, 1, w - 1, 4, tuple(empty) + (255,))
    if n > 0:
        rect(t, 1, 1, 1 + n, 2, tuple(hi) + (255,))
        rect(t, 1, 2, 1 + n, 3, tuple(fill) + (255,))
        rect(t, 1, 3, 1 + n, 4, dark + (255,))
    if notches:
        for k in range(1, notches):
            x = int(round(k * (w - 1) / notches))
            col = t[1:4, x, :3].astype(np.float32)
            t[1:4, x, :3] = (col * 0.55).astype(np.uint8)
    return t


def cross_texture():
    """16 x 16 red X with a dark outline, laid over a crushed block's icon."""
    n = 16
    yy, xx = np.mgrid[0:n, 0:n]
    core = ((yy == xx) | (yy + xx == n - 1)) & (yy >= 3) & (yy <= n - 4)
    grown = core.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(core, dy, 0), dx, 1)
    t = np.zeros((n, n, 4), np.uint8)
    t[grown] = (40, 6, 6, 200)
    t[core] = (255, 58, 48, 255)
    return t


def rewind_texture():
    """The VCR rewind mark: two left-pointing triangles, 11 x 9."""
    t = np.zeros((9, 11, 4), np.uint8)
    for x0 in (0, 6):
        for r in range(9):
            half = 4 - abs(r - 4)
            rect(t, x0 + 4 - half, r, x0 + 5, r + 1, (255, 255, 255, 255))
    return t


def toast_texture(icon16, title, desc, title_col=TOAST_PINK):
    """160 x 32 advancement toast at GUI resolution (text at one GUI pixel per font pixel)."""
    t = np.zeros((32, 160, 4), np.uint8)
    rect(t, 1, 0, 159, 32, (0, 0, 0, 255))
    rect(t, 0, 1, 160, 31, (0, 0, 0, 255))
    rect(t, 1, 1, 159, 31, (84, 84, 92, 255))
    rect(t, 2, 2, 158, 30, (24, 22, 28, 250))
    blit(t, icon16, 8, 8)
    tt = PF.render(title, 1, title_col, shadow=True)
    blit(t, tt[:min(tt.shape[0], 30 - 5)], 30, 5)
    dd = PF.render(desc, 1, (255, 255, 255), shadow=True)
    blit(t, dd[:min(dd.shape[0], 30 - 17)], 30, 17)
    return t


# ---------------------------------------------------------------------------------------------
class HUD:
    S_BOSS = 5.0          # GUI scale of the boss bars
    S_HOT = 4.5           # GUI scale of the hotbar and XP bar
    S_TOAST = 4.0

    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        self.k = k = W / 1080.0
        self._cache = {}
        items = PA.make_items()
        self.labels = {kd: items[kd]['label'] for kd in PA.ORDER}
        # hotbar
        S = self.S_HOT * k
        self.S = S
        self.hotbar = up(hotbar_texture(), S)
        self.hb_x = (W - self.hotbar.shape[1]) // 2
        self.hb_y = int(round(1499 * k)) - self.hotbar.shape[0]
        self.selector = up(selector_texture(), S)
        isz = int(round(16 * S))
        icons = PA.item_icons(items, isz)
        self.icons = [icons[kd] for kd in PA.ORDER]
        self.icons_x = [grey(ic, 0.75, 0.8) for ic in self.icons]
        cross = up(cross_texture(), S)
        for ic in self.icons_x:
            blit(ic, cross[:ic.shape[0], :ic.shape[1]], 0, 0)
        # 16 px icons for the toast
        self.icons16 = PA.item_icons(items, 16)
        # glint pattern for the survivor
        self.glint_mask = self.icons[-1][..., 3].astype(np.float32) / 255.0
        self.vig = None

    # ------------------------------------------------------------------------------------------
    def _text(self, key, text, px, col, **kw):
        kk = ('txt', key, text, px, tuple(col), tuple(sorted(kw.items())))
        if kk not in self._cache:
            self._cache[kk] = PF.render(text, px, col, **kw)
        return self._cache[kk]

    def slot_x(self, k):
        """Screen x of the left edge of slot k's 16 px icon."""
        return self.hb_x + (3 + 20 * k) * self.S

    # ------------------------------------------------------------------------------------------
    def draw(self, frame, st):
        """frame: HxWx3 uint8. st (all optional):
        boss: [dict(name, frac, col, alpha, pop, shake)] ; xp: dict(tons, alpha, danger, pulse, label) ;
        hotbar: dict(sel, alpha, crossed {slot: seconds}, glint seconds or None, lift) ;
        title: dict(text, sub, col, t, dur) ; toast: seconds ; rewind: seconds ;
        flash ; dark ; red ; vignette ; desat."""
        out = frame.astype(np.float32)
        if st.get('desat', 0.0) > 0:
            g = out @ np.array([0.299, 0.587, 0.114], np.float32)
            out = out * (1 - st['desat']) + g[..., None] * st['desat']
        if st.get('red', 0.0) > 0:
            out = out * (1.0 - 0.35 * st['red']) + np.array([90.0, 0.0, 0.0]) * st['red'] * 0.35
        if st.get('dark', 0.0) > 0:
            out *= 1.0 - st['dark']
        if st.get('vignette', 0.0) > 0:
            out *= (1.0 - st['vignette'] * self._vig())[..., None]
        for i, b in enumerate(st.get('boss', [])):
            self._boss(out, i, **b)
        if st.get('hotbar') is not None:
            self._hotbar(out, **st['hotbar'])
        if st.get('xp') is not None:
            self._xp(out, **st['xp'])
        if st.get('title') is not None:
            self._title(out, **st['title'])
        if st.get('toast') is not None:
            self._toast(out, st['toast'], st.get('toast_kind', 'bedrock'))
        if st.get('rewind') is not None:
            self._rewind(out, st['rewind'])
        if st.get('flash', 0.0) > 0:
            out = out * (1 - st['flash']) + 255.0 * st['flash']
        return np.clip(out, 0, 255).astype(np.uint8)

    def _vig(self):
        if self.vig is None:
            ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            r = np.hypot((xs - self.W / 2) / (self.W / 2), (ys - self.H / 2) / (self.H / 2)) / np.sqrt(2)
            self.vig = np.clip((r - 0.3) / 0.7, 0, 1) ** 1.5
        return self.vig

    # ------------------------------------------------------------------------------------------
    def _boss(self, out, i, name, frac, col='pink', alpha=1.0, pop=None, shake=0.0, notches=10):
        """Boss bar i (stacked 19 GUI px apart like the game's): the name above a 182 x 5 bar."""
        if alpha <= 0:
            return
        k, S = self.k, self.S_BOSS * self.k
        fill, hi, empty = BAR_COLS[col]
        key = ('bar', col, int(round(np.clip(frac, 0, 1) * 180)), notches)
        if key not in self._cache:
            self._cache[key] = up(bar_texture(frac, fill, hi, empty, notches), S)
        bar = self._cache[key]
        y_bar = int(round(214 * k + i * 19 * S))
        rng = np.random.default_rng(int(shake * 1e4) % 100000 + i)
        sx, sy = (rng.normal(0, 1, 2) * shake * 6 * k) if shake > 0 else (0.0, 0.0)
        x = (self.W - bar.shape[1]) // 2
        over(out, bar, x + sx, y_bar + sy, alpha)
        txt = self._text('boss', name, int(round(S)), (255, 255, 255), shadow=True)
        s = 1.0
        if pop is not None and pop < 0.25:
            s = 1.0 + 0.35 * (1.0 - ease_out_back(pop / 0.25, 1.6))
        t2 = scaled(txt, s)
        over(out, t2, (self.W - t2.shape[1]) / 2 + sx, y_bar - 2 * S - t2.shape[0] + sy, alpha)

    def _hotbar(self, out, sel=0.0, alpha=1.0, crossed=None, glint=None, lift=0.0, bump=None):
        if alpha <= 0:
            return
        crossed = crossed or {}
        S = self.S
        y = self.hb_y + lift
        over(out, self.hotbar, self.hb_x, y, alpha)
        isz = self.icons[0].shape[0]
        for kk in range(9):
            x = self.slot_x(kk)
            ic = self.icons[kk]
            s = 1.0
            if bump is not None and kk == int(round(sel)) and bump < 0.2:
                s = 1.0 + 0.25 * np.sin(np.pi * bump / 0.2)
            if kk in crossed:
                tc = crossed[kk]
                if tc < 0.12:
                    s = 1.0 + 0.5 * (1.0 - tc / 0.12)
                    ic = self.icons_x[kk]
                else:
                    ic = self.icons_x[kk]
            icn = scaled(ic, s)
            off = (isz - icn.shape[0]) / 2
            over(out, icn, x + off, y + 3 * S + off, alpha)
            if kk == 8 and glint is not None:
                self._glint(out, x, y + 3 * S, glint, alpha)
        sx = self.hb_x + (-1 + 20 * sel) * S
        over(out, self.selector, sx, y - 1 * S, alpha)

    def _glint(self, out, x, y, t, alpha):
        """The enchantment glint: purple light sweeping diagonally over the icon."""
        m = self.glint_mask
        h, w = m.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (xx + yy) / (h + w)
        ph = (u * 2.2 - t * 0.9) % 1.0
        band = np.exp(-((ph - 0.5) / 0.12) ** 2) + 0.35 * np.exp(-((((u * 3.1 + t * 0.6) % 1.0) - 0.5) / 0.1) ** 2)
        a = np.clip(band * m * min(1.0, t / 0.3), 0, 1)
        spr = np.zeros((h, w, 4), np.uint8)
        spr[..., :3] = (180, 110, 255)
        spr[..., 3] = (a * 255).astype(np.uint8)
        add(out, spr, x, y, 1.1 * alpha)

    def _xp(self, out, tons=0.0, alpha=1.0, danger=0.0, pulse=0.0, label=None, frac=None):
        """The pressure as the XP level: log10(tons) / 5 fills the bar; the number goes red when it's about to
        give."""
        if alpha <= 0:
            return
        k, S = self.k, self.S
        f = np.clip(np.log10(max(tons, 1.0)) / 5.0, 0.0, 1.0) if frac is None else frac
        col = XP_GREEN * (1 - danger) + XP_RED * danger
        hi = np.minimum(255, col * 0.6 + 110)
        key = ('xp', int(round(f * 180)), int(round(danger * 20)))
        if key not in self._cache:
            self._cache[key] = up(bar_texture(f, tuple(int(c) for c in col), tuple(int(c) for c in hi),
                                              (34, 34, 38), notches=0), S)
        bar = self._cache[key]
        y_bar = self.hb_y - 7 * S
        over(out, bar, (self.W - bar.shape[1]) / 2, y_bar, alpha)
        txt = label if label is not None else f'{int(round(tons)):,} TONS'
        px = int(round(7 * k))
        ckey = tuple(int(c) for c in (np.round(col / 8) * 8))
        spr = self._text('xp', txt, px, ckey, shadow=False, outline=1, outline_col=(0, 0, 0))
        s = 1.0 + 0.22 * pulse
        spr = scaled(spr, s)
        rng = np.random.default_rng(int(tons) % 9973)
        sh = rng.normal(0, 1, 2) * danger * 3.0 * k if danger > 0.5 else np.zeros(2)
        over(out, spr, (self.W - spr.shape[1]) / 2 + sh[0], y_bar + 1 * S - spr.shape[0] + sh[1], alpha)

    def _title(self, out, text, sub=None, col='red', t=0.0, dur=1.2, y=None):
        """/title: the big text pops in, holds, and fades; the subtitle under it."""
        if t < 0 or t > dur:
            return
        k = self.k
        a = min(1.0, t / 0.04) * (1.0 - smooth((t - (dur - 0.22)) / 0.22))
        s = 1.0 + 0.7 * (1.0 - ease_out_back(t / 0.16, 1.8))
        c = MC[col] if isinstance(col, str) else col
        px = min(int(round(16 * k)), int((930 * k) // (PF.text_mask(text).shape[1] + 2)))
        big = self._text('title', text, px, c, shadow=True)
        cy = int(round((y if y is not None else 610) * k))
        rng = np.random.default_rng(int(t * 1000))
        shake = rng.normal(0, 6 * k, 2) * max(0.0, 1.0 - t / 0.25)
        glow = self._cache.get(('tglow', c))
        if glow is None:
            glow = glow_sprite(int(900 * k), int(300 * k), c, 0.5)
            self._cache[('tglow', c)] = glow
        add(out, glow, (self.W - glow.shape[1]) / 2, cy - glow.shape[0] / 2, 0.22 * a)
        b = scaled(big, s)
        over(out, b, (self.W - b.shape[1]) / 2 + shake[0], cy - b.shape[0] / 2 + shake[1], a)
        if sub:
            sm = self._text('sub', sub, int(round(7 * k)), (255, 255, 255), shadow=True)
            a2 = a * smooth((t - 0.1) / 0.12)
            over(out, sm, (self.W - sm.shape[1]) / 2, cy + big.shape[0] / 2 + 18 * k, a2)

    def _toast(self, out, t, kind='bedrock', dur=3.2):
        if t < 0 or t > dur:
            return
        k = self.k
        key = ('toast', kind)
        if key not in self._cache:
            icon = self.icons16[kind]
            self._cache[key] = up(toast_texture(icon, 'Challenge Complete!', 'Unbreakable'), self.S_TOAST * k)
        spr = self._cache[key]
        slide = smooth(t / 0.3) * (1.0 - smooth((t - (dur - 0.35)) / 0.35))
        x = self.W - (spr.shape[1] + 24 * k) * slide
        over(out, spr, x, 262 * k)

    def _rewind(self, out, t):
        k = self.k
        if int(t * 6) % 2 == 1:
            return
        spr = self._cache.get('rew')
        if spr is None:
            spr = up(rewind_texture(), 9 * k)
            sh = np.zeros_like(spr)
            sh[..., 3] = spr[..., 3]
            base = np.zeros((spr.shape[0] + int(9 * k), spr.shape[1] + int(9 * k), 4), np.uint8)
            blit(base, sh, int(9 * k), int(9 * k))
            blit(base, spr, 0, 0)
            spr = base
            self._cache['rew'] = spr
        over(out, spr, 90 * k, 330 * k)
