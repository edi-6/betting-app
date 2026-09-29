"""Screen-space layers drawn over the 3D frames (and the 2D-only scenes), at 1920 x 1080.

The game's interface is drawn in its own pixel art at GUI scale 3 (one GUI pixel = 3 screen pixels) with the pixel
font: hotbar, hearts, hunger, crosshair, chat, the action-bar message, the player list, the book, the chest, the
world selection and loading screens, toasts and name tags. The old computer's desktop (wallpaper, icons, taskbar,
the launcher, the file explorer, the cursor) uses a plain system font. Subtitles for his lines and the keystroke
overlay sit on top of everything.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import pixelfont as PF
import skins as SK
import textures as TX

W, H = 1920, 1080
GS = 4                          # GUI scale (the game's own default at 1080p)
FONT_DIR = '/usr/share/fonts/truetype/dejavu'


def font(size, bold=False, mono=False):
    name = 'DejaVuSansMono' if mono else 'DejaVuSans'
    if bold:
        name += '-Bold'
    return ImageFont.truetype(os.path.join(FONT_DIR, name + '.ttf'), size)


_ICONS = None


def icons():
    global _ICONS
    if _ICONS is None:
        _ICONS = TX.item_icons()
    return _ICONS


# ---------------------------------------------------------------------------------------------
# compositing helpers
# ---------------------------------------------------------------------------------------------
def blit(dst, src, x, y, alpha=1.0):
    """Alpha-blend an RGBA uint8 sprite onto an RGB uint8 frame (in place), clipped."""
    h, w = src.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * alpha
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(d * (1 - a) + s[..., :3] * a, 0, 255).astype(np.uint8)


def up(sprite, k=GS):
    return np.kron(sprite, np.ones((k, k, 1), np.uint8))


def rect(dst, x, y, w, h, col, alpha=1.0):
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), dst.shape[1]), min(int(y + h), dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(d * (1 - alpha) + np.array(col, np.float32) * alpha, 0, 255).astype(np.uint8)


def mc_text(text, color=(255, 255, 255), scale=GS, shadow=True):
    return PF.render(text, px=scale, color=color, shadow=shadow)


def text_width(text, scale=GS):
    return PF.text_mask(text).shape[1] * scale


def draw_mc_text(dst, text, x, y, color=(255, 255, 255), scale=GS, shadow=True, alpha=1.0, align='left'):
    spr = mc_text(text, color, scale, shadow)
    if align == 'center':
        x = x - spr.shape[1] / 2
    elif align == 'right':
        x = x - spr.shape[1]
    blit(dst, spr, x, y, alpha)
    return spr.shape[1]


def pil_text(dst, text, x, y, fnt, fill=(255, 255, 255, 255), anchor='la', stroke=0, stroke_fill=(0, 0, 0, 255),
             alpha=1.0):
    """Draw anti-aliased text with PIL onto a numpy RGB frame region."""
    bbox = fnt.getbbox(text, anchor=anchor, stroke_width=stroke)
    w, h = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((2 - bbox[0], 2 - bbox[1]), text, font=fnt, fill=fill, anchor=anchor, stroke_width=stroke,
           stroke_fill=stroke_fill)
    blit(dst, np.asarray(im), x + bbox[0] - 2, y + bbox[1] - 2, alpha)
    return bbox[2] - bbox[0]


# ---------------------------------------------------------------------------------------------
# the game's pixel-art GUI pieces (drawn at 1x, shown at GS)
# ---------------------------------------------------------------------------------------------
def panel(w, h):
    """The light grey window with a bevel and rounded black outline."""
    p = np.zeros((h, w, 4), np.uint8)
    p[..., :3] = (198, 198, 198)
    p[..., 3] = 255
    p[1:3, 1:w - 2, :3] = 255
    p[1:h - 2, 1:3, :3] = 255
    p[h - 3:h - 1, 2:w - 1, :3] = 85
    p[2:h - 1, w - 3:w - 1, :3] = 85
    p[0, :, :3] = 0
    p[-1, :, :3] = 0
    p[:, 0, :3] = 0
    p[:, -1, :3] = 0
    for (y, x) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        p[y, x, 3] = 0
    return p


def slot():
    s = np.zeros((18, 18, 4), np.uint8)
    s[..., :3] = (139, 139, 139)
    s[..., 3] = 255
    s[0, :17, :3] = 55
    s[:17, 0, :3] = 55
    s[17, 1:, :3] = 255
    s[1:, 17, :3] = 255
    return s


def button(w, h, hover=False, disabled=False):
    b = np.zeros((h, w, 4), np.uint8)
    base = (111, 111, 111) if not hover else (124, 134, 190)
    if disabled:
        base = (44, 44, 44)
    b[..., :3] = base
    b[..., 3] = 255
    rng = np.random.default_rng(w * 31 + h)
    b[..., :3] = np.clip(b[..., :3] * (1 + 0.05 * (rng.random((h, w, 1)) - 0.5)), 0, 255)
    b[1, 1:w - 1, :3] = np.array(base) + 45
    b[1:h - 1, 1, :3] = np.array(base) + 30
    b[h - 2, 1:w - 1, :3] = np.array(base) * 0.55
    b[0, :, :3] = 0
    b[-1, :, :3] = 0
    b[:, 0, :3] = 0
    b[:, -1, :3] = 0
    if hover:
        b[0, :, :3] = 255
        b[-1, :, :3] = 255
        b[:, 0, :3] = 255
        b[:, -1, :3] = 255
    return b


HEART_ROWS = [
    ".OO...OO.",
    "ORRO.ORRO",
    "ORWRORRRO",
    "ORRRRRRRO",
    "ORRRRRRDO",
    ".ORRRRDO.",
    "..ORRDO..",
    "...ODO...",
    "....O....",
]


def heart(state='full'):
    """(9, 9, 4) heart icon: full | half | empty."""
    pal = {'O': (12, 6, 6), 'R': (236, 22, 22), 'W': (255, 214, 214), 'D': (170, 0, 0),
           'E': (40, 38, 38), 'e': (66, 62, 62), 'k': (30, 28, 28)}
    img = np.zeros((9, 9, 4), np.uint8)
    for r, row in enumerate(HEART_ROWS):
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            filled = state == 'full' or (state == 'half' and c <= 4)
            if ch == 'O':
                col = pal['O']
            elif filled:
                col = pal[ch]
            else:
                col = pal['e'] if ch == 'W' else (pal['k'] if ch == 'D' else pal['E'])
            img[r, c, :3] = col
            img[r, c, 3] = 255
    return img


HUNGER_ROWS = [
    "....OOOO.",
    "...OBBBBO",
    "..OBbbbBO",
    ".OBbbbBO.",
    "OBbbbBO..",
    "OBbBBO...",
    ".OWO.....",
    "OWWO.....",
    ".OO......",
]


def drumstick(full=True):
    pal = {'O': (38, 14, 4), 'B': (178, 98, 46) if full else (48, 44, 40), 'b': (220, 136, 70) if full else (64, 60, 56),
           'W': (236, 226, 210)}
    img = np.zeros((9, 9, 4), np.uint8)
    for r, row in enumerate(HUNGER_ROWS):
        for c, ch in enumerate(row):
            if ch != '.':
                img[r, c, :3] = pal[ch]
                img[r, c, 3] = 255
    return img[:, ::-1]


def hotbar_sprite(selected=0):
    w = 182
    hb = np.zeros((22, w, 4), np.uint8)
    hb[..., :3] = (60, 60, 60)
    hb[..., 3] = 170
    hb[0, :, :3] = 16
    hb[-1, :, :3] = 16
    hb[:, 0, :3] = 16
    hb[:, -1, :3] = 16
    hb[0, :, 3] = hb[-1, :, 3] = hb[:, 0, 3] = hb[:, -1, 3] = 230
    for k in range(9):
        x = 1 + k * 20
        hb[1:21, x + 19, :3] = 30
        hb[1:21, x + 19, 3] = 200
        hb[2:20, x + 1:x + 19, :3] = (84, 84, 84)
        hb[2:20, x + 1:x + 19, 3] = 150
    sel = np.zeros((24, 24, 4), np.uint8)
    sel[..., 3] = 0
    for d in range(2):
        sel[d, d:24 - d, :3] = 240
        sel[23 - d, d:24 - d, :3] = 240
        sel[d:24 - d, d, :3] = 240
        sel[d:24 - d, 23 - d, :3] = 240
        sel[d, d:24 - d, 3] = sel[23 - d, d:24 - d, 3] = sel[d:24 - d, d, 3] = sel[d:24 - d, 23 - d, 3] = 255
    return hb, sel


def crosshair():
    c = np.zeros((9, 9, 4), np.uint8)
    c[4, :, :] = (235, 235, 235, 220)
    c[:, 4, :] = (235, 235, 235, 220)
    return c


class HUD:
    """The in-game overlay."""

    def __init__(self):
        self.hb, self.sel = hotbar_sprite()
        self.heart_full = up(heart('full'))
        self.heart_empty = up(heart('empty'))
        self.food = up(drumstick(True))
        self.cross = up(crosshair())
        self.icons = {k: up(v) for k, v in icons().items()}

    def draw(self, img, items=(), selected=0, hearts=20, food=20, xp=0.0, level=0, cross=True, alpha=1.0,
             counts=None):
        cx = W // 2
        hbw = 182 * GS
        x0 = cx - hbw // 2
        y0 = H - 22 * GS - 2
        blit(img, up(self.hb), x0, y0, alpha)
        for k, it in enumerate(items[:9]):
            if it and it in self.icons:
                blit(img, self.icons[it], x0 + (3 + k * 20) * GS, y0 + 3 * GS, alpha)
                n = (counts or {}).get(k)
                if n and n > 1:
                    draw_mc_text(img, str(n), x0 + (19 + k * 20) * GS, y0 + 11 * GS, align='right', alpha=alpha)
        blit(img, up(self.sel), x0 + (selected * 20 - 1) * GS, y0 - 1 * GS, alpha)
        # xp bar
        yb = y0 - 7 * GS
        bar = np.zeros((5, 182, 4), np.uint8)
        bar[..., :3] = (20, 20, 20)
        bar[..., 3] = 220
        bar[1:4, 1:181, :3] = (40, 60, 20)
        fill = int(180 * np.clip(xp, 0, 1))
        bar[1:4, 1:1 + fill, :3] = (128, 255, 32)
        blit(img, up(bar), x0, yb, alpha)
        # hearts (left) and hunger (right)
        yh = yb - 10 * GS
        for k in range(10):
            spr = self.heart_full if hearts >= (k + 1) * 2 else self.heart_empty
            blit(img, spr, x0 + k * 8 * GS, yh, alpha)
            spr = self.food
            blit(img, spr, x0 + hbw - (k + 1) * 8 * GS - 1 * GS, yh, alpha if food >= (k + 1) * 2 else 0.35 * alpha)
        if cross:
            blit(img, self.cross, cx - self.cross.shape[1] // 2, H // 2 - self.cross.shape[0] // 2, 0.85 * alpha)
        return yh


def action_bar(img, text, alpha=1.0, color=(255, 255, 255)):
    """The message above the hotbar (e.g. "This bed is occupied")."""
    draw_mc_text(img, text, W // 2, H - 22 * GS - 2 - 7 * GS - 10 * GS - 14 * GS, color, align='center', alpha=alpha)


# ---------------------------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------------------------
YELLOW = (255, 255, 85)
GRAY = (170, 170, 170)


def chat(img, lines, t, open_=False, hide_after=10.0):
    """lines: list of (t_added, text, colour). Newest at the bottom, fading 10 s after they appear."""
    x = 2 * GS
    y = H - 40 * GS - 10
    shown = []
    for (ta, text, col) in lines:
        if ta > t:
            continue
        age = t - ta
        a = 1.0 if open_ else float(np.clip((hide_after - age) / 1.0, 0.0, 1.0))
        a *= float(np.clip(age / 0.08, 0.0, 1.0))
        if a <= 0:
            continue
        shown.append((text, col, a))
    shown = shown[-12:]
    lh = 9 * GS
    for i, (text, col, a) in enumerate(reversed(shown)):
        yy = y - (i + 1) * lh
        rect(img, x - 2, yy - 2, 320 * GS / 1.0, lh, (0, 0, 0), 0.45 * a)
        draw_mc_text(img, text, x + 2, yy, col, alpha=a)


def player_list(img, names, alpha=1.0):
    """Tab list: the players online, centred at the top."""
    ws = max(text_width(n) for n in names) + 20 * GS
    x = W // 2 - ws // 2
    y = 12 * GS
    rect(img, x - 2 * GS, y - 2 * GS, ws + 4 * GS, (len(names) * 9 + 3) * GS, (0, 0, 0), 0.45 * alpha)
    for i, n in enumerate(names):
        rect(img, x, y + i * 9 * GS, ws, 8 * GS, (255, 255, 255), 0.12 * alpha)
        draw_mc_text(img, n, x + 10 * GS, y + i * 9 * GS, alpha=alpha)
        # ping bars
        for b in range(5):
            hh = (b + 1) * GS
            rect(img, x + ws - (12 - 2 * b) * GS, y + i * 9 * GS + 7 * GS - hh, GS, hh, (60, 220, 60), alpha)


# ---------------------------------------------------------------------------------------------
# book, chest
# ---------------------------------------------------------------------------------------------
def book_page_sprite():
    """The written book's page (146 x 180 GUI px): cream paper with a leather edge on the left."""
    w, h = 146, 180
    p = np.zeros((h, w, 4), np.uint8)
    rng = np.random.default_rng(12)
    paper = np.array((244, 236, 212), float)
    p[..., :3] = np.clip(paper * (1 + 0.025 * (rng.random((h, w, 1)) - 0.5)), 0, 255)
    p[..., 3] = 255
    p[:, :10, :3] = (120, 72, 38)
    p[:, 10:12, :3] = (90, 52, 26)
    p[0, :, :3] = p[-1, :, :3] = (60, 36, 18)
    p[:, -1, :3] = (200, 190, 160)
    p[1:4, 12:, :3] = (226, 216, 190)
    return p


def wrap(text, maxw):
    words = text.split(' ')
    lines, cur = [], ''
    for wd in words:
        test = (cur + ' ' + wd).strip()
        if PF.text_mask(test).shape[1] > maxw and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def book(img, page_text, page, n_pages, alpha=1.0, torn=False, title=None, turn=0.0, dim=0.55):
    """The book GUI over the (dimmed) world: one page of text, the page counter, the arrows."""
    if dim > 0:
        img[:] = (img.astype(np.float32) * (1 - dim * alpha)).astype(np.uint8)
    spr = book_page_sprite()
    w, h = spr.shape[1] * GS, spr.shape[0] * GS
    x0, y0 = W // 2 - w // 2, 60
    if torn:
        # the torn page: a ragged edge down the middle and nothing on it
        for yy in range(spr.shape[0]):
            cut = 60 + int(8 * np.sin(yy * 0.7) + 5 * np.sin(yy * 0.23))
            spr[yy, cut:, 3] = 0
            spr[yy, cut - 1, :3] = (200, 188, 160)
    blit(img, up(spr), x0, y0, alpha)
    ink = (20, 14, 10)
    cnt = f'Page {page + 1} of {n_pages}'
    draw_mc_text(img, cnt, x0 + w - 16 * GS, y0 + 14 * GS, ink, shadow=False, align='right', alpha=alpha)
    y = y0 + 30 * GS
    if title:
        draw_mc_text(img, title, x0 + w // 2 + 5 * GS, y, ink, shadow=False, align='center', alpha=alpha)
        y += 14 * GS
    if not torn:
        for para in page_text.split('\n'):
            for ln in (wrap(para, 114) if para else ['']):
                draw_mc_text(img, ln, x0 + 20 * GS, y, ink, shadow=False, alpha=alpha)
                y += 10 * GS
    # arrows
    for (dx, flip) in ((w - 40 * GS, False), (22 * GS, True)):
        if (not flip and page < n_pages - 1) or (flip and page > 0):
            arr = np.zeros((10, 18, 4), np.uint8)
            for k in range(9):
                arr[4 - k // 2:5 + k // 2, 17 - k, :] = (130, 40, 30, 255)
            arr[3:7, 0:10, :] = (130, 40, 30, 255)
            if flip:
                arr = arr[:, ::-1]
            blit(img, up(arr), x0 + dx, y0 + h - 22 * GS, alpha)
    if turn > 0:
        # a page flip: a quick bright sweep
        xx = int(x0 + w * (1 - turn))
        rect(img, xx, y0, 8 * GS, h, (255, 250, 236), 0.35 * alpha)


def chest(img, contents, hover=None, alpha=1.0, title='Chest', dim=0.55, inventory=None):
    """The chest GUI: 27 slots above the player's inventory. contents: dict slot -> (item, count)."""
    if dim > 0:
        img[:] = (img.astype(np.float32) * (1 - dim * alpha)).astype(np.uint8)
    pw, ph = 176, 166
    p = panel(pw, ph)
    s = slot()
    for r in range(3):
        for c in range(9):
            p[17 + r * 18:17 + r * 18 + 18, 7 + c * 18:7 + c * 18 + 18] = s
    for r in range(3):
        for c in range(9):
            p[83 + r * 18:83 + r * 18 + 18, 7 + c * 18:7 + c * 18 + 18] = s
    for c in range(9):
        p[141:159, 7 + c * 18:7 + c * 18 + 18] = s
    x0, y0 = W // 2 - pw * GS // 2, H // 2 - ph * GS // 2
    blit(img, up(p), x0, y0, alpha)
    draw_mc_text(img, title, x0 + 8 * GS, y0 + 6 * GS, (64, 64, 64), shadow=False, alpha=alpha)
    draw_mc_text(img, 'Inventory', x0 + 8 * GS, y0 + 72 * GS, (64, 64, 64), shadow=False, alpha=alpha)
    ic = icons()
    for k, (item, n) in (contents or {}).items():
        r, c = divmod(k, 9)
        xx, yy = x0 + (8 + c * 18) * GS, y0 + (18 + r * 18) * GS
        blit(img, up(ic[item]), xx, yy, alpha)
        if n > 1:
            draw_mc_text(img, str(n), xx + 17 * GS, yy + 9 * GS, align='right', alpha=alpha)
    for k, (item, n) in (inventory or {}).items():
        r, c = divmod(k, 9)
        yy = y0 + ((84 + r * 18) if r < 3 else 142) * GS
        xx = x0 + (8 + c * 18) * GS
        blit(img, up(ic[item]), xx, yy, alpha)
    if hover is not None:
        k, lines = hover
        r, c = divmod(k, 9)
        xx, yy = x0 + (8 + c * 18) * GS, y0 + (18 + r * 18) * GS
        rect(img, xx, yy, 16 * GS, 16 * GS, (255, 255, 255), 0.35 * alpha)
        tooltip(img, lines, xx + 18 * GS, yy - 12 * GS, alpha)
    return x0, y0


def tooltip(img, lines, x, y, alpha=1.0):
    """lines: list of (text, colour)."""
    wmax = max(text_width(t) for t, _ in lines)
    hh = (len(lines) * 10 + 4) * GS
    rect(img, x, y, wmax + 8 * GS, hh, (16, 0, 16), 0.94 * alpha)
    rect(img, x + GS, y + GS, wmax + 6 * GS, GS, (80, 0, 255), 0.5 * alpha)
    rect(img, x + GS, y + hh - 2 * GS, wmax + 6 * GS, GS, (40, 0, 127), 0.5 * alpha)
    for i, (t, col) in enumerate(lines):
        draw_mc_text(img, t, x + 4 * GS, y + (3 + i * 10) * GS, col, alpha=alpha)


# ---------------------------------------------------------------------------------------------
# name tags (over the 3D frame)
# ---------------------------------------------------------------------------------------------
def name_tag(img, text, x, y, dist, alpha=1.0, through_wall=False):
    """A name floating over a head at screen (x, y), sized by distance."""
    if dist <= 0.2:
        return
    scale = float(np.clip(14.0 / dist, 0.8, 5.0))
    k = max(1, int(round(scale)))
    spr = mc_text(text, (255, 255, 255) if not through_wall else (190, 190, 190), k, shadow=False)
    pad = k * 2
    bw, bh = spr.shape[1] + 2 * pad, spr.shape[0] + pad
    rect(img, x - bw / 2, y - bh, bw, bh, (0, 0, 0), 0.28 * alpha)
    blit(img, spr, x - spr.shape[1] / 2, y - bh + pad // 2, alpha * (0.5 if through_wall else 1.0))


# ---------------------------------------------------------------------------------------------
# title screens: world select, loading
# ---------------------------------------------------------------------------------------------
def dirt_background(dim=0.25):
    from textures import dirt, _rng
    d = dirt(_rng(3))
    tile = np.kron(d, np.ones((4, 4, 1))) * dim
    reps = (H // tile.shape[0] + 1, W // tile.shape[1] + 1, 1)
    return np.tile(tile, reps)[:H, :W].astype(np.uint8)


def world_icon(village_img=None):
    if village_img is not None:
        im = Image.fromarray(village_img).resize((64, 64), Image.BILINEAR)
        return np.asarray(im.convert('RGBA'))
    a = np.zeros((64, 64, 4), np.uint8)
    a[..., :3] = (90, 140, 70)
    a[..., 3] = 255
    return a


def world_select(img, icon=None, hover=None, selected=True, cursor=None, date='04/03/2026 04:04'):
    img[:] = dirt_background()
    rect(img, 0, 32 * GS, W, H - 96 * GS, (0, 0, 0), 0.45)
    draw_mc_text(img, 'Select World', W // 2, 11 * GS, align='center')
    # the list entry
    ex, ey = W // 2 - 110 * GS, 42 * GS
    ew, eh = 220 * GS, 36 * GS
    if selected:
        rect(img, ex - GS, ey - GS, ew + 2 * GS, eh + 2 * GS, (128, 128, 128), 1.0)
        rect(img, ex, ey, ew, eh, (0, 0, 0), 1.0)
    if icon is not None:
        blit(img, np.asarray(Image.fromarray(icon).resize((32 * GS, 32 * GS), Image.NEAREST)), ex + 2 * GS, ey + 2 * GS)
    tx = ex + 38 * GS
    draw_mc_text(img, 'NOAH_FINAL', tx, ey + 3 * GS)
    draw_mc_text(img, 'NOAH_FINAL (2) (' + date + ')', tx, ey + 14 * GS, (128, 128, 128))
    draw_mc_text(img, 'Survival Mode, Version: 1.12.2', tx, ey + 25 * GS, (128, 128, 128))
    # buttons: two rows
    bw1 = 150 * GS
    by1 = H - 52 * GS
    labels = [('Play Selected World', W // 2 - 154 * GS, by1, 150), ('Create New World', W // 2 + 4 * GS, by1, 150),
              ('Edit', W // 2 - 154 * GS, by1 + 24 * GS, 72), ('Delete', W // 2 - 76 * GS, by1 + 24 * GS, 72),
              ('Re-Create', W // 2 + 4 * GS, by1 + 24 * GS, 72), ('Cancel', W // 2 + 82 * GS, by1 + 24 * GS, 72)]
    del bw1
    for (lab, bx, by, bw) in labels:
        hv = hover == lab
        blit(img, up(button(bw, 20, hv)), bx, by)
        draw_mc_text(img, lab, bx + bw * GS // 2, by + 6 * GS, (255, 255, 160) if hv else (224, 224, 224),
                     align='center')
    return {lab: (bx + bw * GS // 2, by + 10 * GS) for (lab, bx, by, bw) in labels}, (ex + ew // 2, ey + eh // 2)


def last_played_hint(img, text, alpha=1.0):
    """The hover text on the world: 'Last played: 1 day ago'."""
    tooltip(img, [(text, (255, 255, 255))], W // 2 - 30 * GS, 80 * GS, alpha)


def loading(img, text='Loading terrain', progress=0.0, sub=None):
    img[:] = dirt_background()
    draw_mc_text(img, text, W // 2, H // 2 - 20 * GS, align='center')
    bw = 100 * GS
    x0 = W // 2 - bw // 2
    y0 = H // 2 - 4 * GS
    rect(img, x0, y0, bw, 2 * GS, (128, 128, 128))
    rect(img, x0, y0, bw * np.clip(progress, 0, 1), 2 * GS, (128, 255, 128))
    if sub:
        draw_mc_text(img, sub, W // 2, H // 2 + 6 * GS, (190, 190, 190), align='center')


def toast(img, title, line2, icon_skin=None, slide=1.0, x_right=W - 8, y=8):
    """The game's toast (top right): a dark panel with a player head, a yellow title and a white line."""
    tw, th = 160, 32
    p = np.zeros((th, tw, 4), np.uint8)
    p[..., :3] = (32, 22, 40)
    p[..., 3] = 245
    p[0, :, :3] = p[-1, :, :3] = (140, 110, 170)
    p[:, 0, :3] = p[:, -1, :3] = (140, 110, 170)
    x = x_right - tw * GS * slide
    blit(img, up(p), x, y)
    if icon_skin is not None:
        face = icon_skin[8:16, 8:16]
        blit(img, up(face, 2 * GS), x + 8 * GS, y + 8 * GS)
    draw_mc_text(img, title, x + 30 * GS, y + 7 * GS, YELLOW)
    draw_mc_text(img, line2, x + 30 * GS, y + 18 * GS, (255, 255, 255))


# ---------------------------------------------------------------------------------------------
# subtitles, keystrokes
# ---------------------------------------------------------------------------------------------
_SUB_FONT = None


def subtitle(img, text, alpha=1.0, y=None, italic=False):
    """His voice: white text with a black edge, bottom centre (above the hotbar when it is shown)."""
    global _SUB_FONT
    if _SUB_FONT is None:
        _SUB_FONT = font(38)
    if not text:
        return
    yy = y if y is not None else H - 150
    lines = text.split('\n')
    for i, ln in enumerate(lines):
        pil_text(img, ln, W // 2, yy + (i - len(lines) + 1) * 48, _SUB_FONT, anchor='ms', stroke=3,
                 stroke_fill=(0, 0, 0, 255), alpha=alpha)


def keystrokes(img, pressed, alpha=1.0, x0=W - 250, y0=H - 330):
    """The streamer's key overlay: W A S D, the mouse buttons and ESC; pressed keys light up."""
    f = font(26, bold=True)
    keys = [('ESC', 0, -70, 70), ('W', 80, 0, 70), ('A', 0, 80, 70), ('S', 80, 80, 70), ('D', 160, 80, 70),
            ('LMB', 0, 160, 110), ('RMB', 120, 160, 110)]
    for (k, dx, dy, wk) in keys:
        on = k in pressed
        rect(img, x0 + dx, y0 + dy, wk, 70, (255, 255, 255) if on else (0, 0, 0), (0.85 if on else 0.45) * alpha)
        pil_text(img, k, x0 + dx + wk // 2, y0 + dy + 35, f, fill=(0, 0, 0, 255) if on else (255, 255, 255, 255),
                 anchor='mm', alpha=alpha)


# ---------------------------------------------------------------------------------------------
# the old computer
# ---------------------------------------------------------------------------------------------
def wallpaper(seed=4):
    """A faded photo-like landscape: dusk gradient, dark hills, a lake."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, H)[:, None]
    sky = (np.array((32, 48, 84)) * (1 - y) + np.array((150, 120, 120)) * y)[:, None, :] * np.ones((1, W, 1))
    img = sky.copy()
    xs = np.arange(W)
    for k, (base, amp, col) in enumerate(((0.62, 0.06, (40, 50, 60)), (0.7, 0.05, (26, 34, 40)),
                                          (0.8, 0.03, (16, 20, 24)))):
        ridge = base + amp * np.sin(xs / 190.0 + k * 2.1) + amp * 0.5 * np.sin(xs / 61.0 + k)
        mask = (np.arange(H)[:, None] / H) > ridge[None, :]
        img[mask] = col
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def icon_grass_block(n=64):
    """A small drawing of a grass block for the launcher's desktop icon (our own art)."""
    im = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = n // 2
    top = [(c, 4), (n - 6, n // 4), (c, n // 2 - 2), (6, n // 4)]
    left = [(6, n // 4), (c, n // 2 - 2), (c, n - 4), (6, n - n // 4 - 2)]
    right = [(c, n // 2 - 2), (n - 6, n // 4), (n - 6, n - n // 4 - 2), (c, n - 4)]
    d.polygon(top, fill=(110, 170, 70, 255))
    d.polygon(left, fill=(130, 92, 60, 255))
    d.polygon(right, fill=(104, 72, 46, 255))
    d.polygon([(6, n // 4), (c, n // 2 - 2), (c, n // 2 + 6), (6, n // 4 + 8)], fill=(92, 150, 60, 255))
    d.polygon([(c, n // 2 - 2), (n - 6, n // 4), (n - 6, n // 4 + 8), (c, n // 2 + 6)], fill=(80, 130, 52, 255))
    return np.asarray(im)


def icon_folder(n=64):
    im = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((6, 14, n - 6, n - 10), 4, fill=(232, 192, 88, 255), outline=(170, 130, 40, 255))
    d.rounded_rectangle((6, 8, n // 2, 20), 3, fill=(222, 178, 70, 255))
    d.rectangle((6, 22, n - 6, n - 10), fill=(244, 206, 104, 255))
    return np.asarray(im)


def icon_file(n=64, ext='mp4'):
    im = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(14, 4), (40, 4), (52, 16), (52, n - 4), (14, n - 4)], fill=(246, 246, 246, 255),
              outline=(140, 140, 150, 255))
    d.polygon([(40, 4), (40, 16), (52, 16)], fill=(210, 210, 216, 255))
    if ext == 'mp4':
        d.rounded_rectangle((20, 26, 46, 48), 3, fill=(60, 110, 200, 255))
        d.polygon([(29, 31), (29, 43), (39, 37)], fill=(255, 255, 255, 255))
    else:
        for k in range(5):
            d.line((20, 24 + k * 6, 46, 24 + k * 6), fill=(150, 150, 160, 255), width=2)
    return np.asarray(im)


def icon_bin(n=64):
    im = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(16, 16), (48, 16), (44, n - 6), (20, n - 6)], fill=(200, 210, 220, 200), outline=(120, 130, 140, 255))
    for k in range(3):
        d.line((24 + k * 8, 22, 23 + k * 8, n - 12), fill=(150, 160, 170, 255), width=2)
    d.rectangle((13, 11, 51, 16), fill=(170, 180, 190, 255))
    return np.asarray(im)


class Desktop:
    """The old PC: wallpaper, icons, taskbar with a 4:04 AM clock, windows, the cursor."""

    def __init__(self, clock='4:04 AM', date='4/3/2026'):
        self.wall = wallpaper()
        self.clock = clock
        self.date = date
        self.f = font(15)
        self.fb = font(15, bold=True)
        self.icons = {'Recycle Bin': icon_bin(), 'Minecraft Launcher': icon_grass_block(), 'old_saves': icon_folder(),
                      'readme.txt': icon_file(ext='txt')}
        self.cursor = self._cursor()

    def _cursor(self):
        rows = ["X...........", "XX..........", "XWX.........", "XWWX........", "XWWWX.......", "XWWWWX......",
                "XWWWWWX.....", "XWWWWWWX....", "XWWWWWWWX...", "XWWWWWWWWX..", "XWWWWWXXXXX.", "XWWXWWX.....",
                "XWX.XWWX....", "XX..XWWX....", "X....XWWX...", ".....XWWX...", "......XX...."]
        c = np.zeros((len(rows), 12, 4), np.uint8)
        for r, row in enumerate(rows):
            for k, ch in enumerate(row):
                if ch == 'X':
                    c[r, k] = (0, 0, 0, 255)
                elif ch == 'W':
                    c[r, k] = (255, 255, 255, 255)
        return up(c, 2)

    def base(self, icons=('Recycle Bin', 'Minecraft Launcher', 'old_saves', 'readme.txt'), extra_icons=()):
        img = self.wall.copy()
        y = 24
        for name in list(icons) + list(extra_icons):
            ic = self.icons.get(name) if isinstance(name, str) else None
            if isinstance(name, tuple):
                name, ic = name
            blit(img, ic, 40, y)
            pil_text(img, name, 72, y + 70, self.f, anchor='mt', stroke=2, stroke_fill=(0, 0, 0, 200))
            y += 110
        # taskbar
        rect(img, 0, H - 44, W, 44, (20, 26, 36), 0.94)
        rect(img, 0, H - 44, W, 1, (90, 110, 140), 1.0)
        rect(img, 8, H - 38, 54, 32, (46, 110, 60), 1.0)
        pil_text(img, 'start', 35, H - 22, self.fb, anchor='mm')
        pil_text(img, self.clock, W - 60, H - 30, self.f, anchor='mm')
        pil_text(img, self.date, W - 60, H - 12, self.f, anchor='mm')
        return img

    def window(self, img, x, y, w, h, title, dark=False):
        rect(img, x + 6, y + 8, w, h, (0, 0, 0), 0.35)
        rect(img, x, y, w, h, (36, 36, 40) if dark else (240, 240, 240), 1.0)
        rect(img, x, y, w, 32, (28, 28, 32) if dark else (212, 222, 236), 1.0)
        pil_text(img, title, x + 12, y + 16, self.f, fill=(230, 230, 230, 255) if dark else (20, 20, 20, 255),
                 anchor='lm')
        for k, c in enumerate(('x', '□', '_')):
            pil_text(img, c, x + w - 20 - k * 36, y + 16, self.f, fill=(200, 200, 200, 255) if dark else
                     (40, 40, 40, 255), anchor='mm')

    def launcher(self, img, account, skin, banner=None, hover_play=False, dropdown=False, accounts=(), x=360, y=150):
        w, h = 1200, 720
        self.window(img, x, y, w, h, 'Minecraft Launcher', dark=True)
        # sidebar
        rect(img, x, y + 32, 250, h - 32, (26, 26, 30), 1.0)
        face = skin[8:16, 8:16]
        blit(img, up(face, 5), x + 20, y + 52)
        pil_text(img, account, x + 70, y + 62, self.fb, fill=(240, 240, 240, 255), anchor='lm')
        pil_text(img, 'Switch account  v', x + 70, y + 84, self.f, fill=(140, 140, 150, 255), anchor='lm')
        for k, it in enumerate(('Play', 'Installations', 'Skins', 'Patch Notes')):
            pil_text(img, it, x + 24, y + 150 + k * 40, self.f, fill=(210, 210, 210, 255), anchor='lm')
        # banner
        bx, by, bw, bh = x + 250, y + 32, w - 250, 470
        if banner is not None:
            b = np.asarray(Image.fromarray(banner).resize((bw, bh), Image.BILINEAR))
            img[by:by + bh, bx:bx + bw] = b
        else:
            rect(img, bx, by, bw, bh, (60, 90, 60), 1.0)
        rect(img, bx, by + bh - 80, bw, 80, (0, 0, 0), 0.35)
        # the skin preview (flat front view) on the banner
        doll = front_doll(skin, 7)
        blit(img, doll, bx + 70, by + bh - doll.shape[0] - 30)
        # bottom bar with the play button
        rect(img, bx, by + bh, bw, h - 32 - bh, (32, 32, 36), 1.0)
        pil_text(img, 'Latest release  1.12.2', bx + 30, by + bh + 50, self.f, fill=(200, 200, 200, 255), anchor='lm')
        px, py = bx + bw // 2 - 150, by + bh + 90
        rect(img, px, py, 300, 80, (60, 170, 70) if not hover_play else (80, 200, 90), 1.0)
        pil_text(img, 'PLAY', px + 150, py + 40, font(34, bold=True), anchor='mm')
        if dropdown:
            dy = y + 100
            rect(img, x + 60, dy, 260, 40 * len(accounts) + 10, (50, 50, 56), 1.0)
            for k, (name, blurred) in enumerate(accounts):
                pil_text(img, name, x + 76, dy + 26 + k * 40, self.fb, fill=(240, 240, 240, 255), anchor='lm')
                if blurred:
                    reg = img[dy + 10 + k * 40:dy + 42 + k * 40, x + 70:x + 300]
                    im = Image.fromarray(reg).filter(ImageFilter.GaussianBlur(7))
                    img[dy + 10 + k * 40:dy + 42 + k * 40, x + 70:x + 300] = np.asarray(im)
        return (px + 150, py + 40)

    def explorer(self, img, title, files, x=420, y=190, hover=None):
        w, h = 1080, 560
        self.window(img, x, y, w, h, title)
        rect(img, x, y + 32, 220, h - 32, (232, 236, 242), 1.0)
        for k, it in enumerate(('Desktop', 'Documents', 'Downloads', 'Videos', 'This PC')):
            pil_text(img, it, x + 24, y + 64 + k * 30, self.f, fill=(30, 30, 30, 255), anchor='lm')
        cols = [('Name', 250), ('Date modified', 420), ('Type', 600), ('Size', 720), ('Length', 820)]
        for (c, cx) in cols:
            pil_text(img, c, x + cx, y + 60, self.fb, fill=(60, 60, 70, 255), anchor='lm')
        rect(img, x + 240, y + 76, w - 250, 1, (200, 200, 210), 1.0)
        for k, f in enumerate(files):
            yy = y + 100 + k * 34
            if hover == k:
                rect(img, x + 240, yy - 15, w - 250, 30, (204, 228, 247), 1.0)
            blit(img, np.asarray(Image.fromarray(icon_file(ext='mp4')).resize((24, 24))), x + 244, yy - 12)
            vals = [f['name'], f['date'], f['type'], f['size'], f['length']]
            for (c, cx), v in zip(cols, vals):
                pil_text(img, v, x + cx + (30 if c == 'Name' else 0), yy, self.f, fill=(20, 20, 20, 255),
                         anchor='lm')

    def notification(self, img, title, text, slide=1.0):
        w, h = 420, 110
        x = W - (w + 16) * slide
        y = H - 44 - h - 16
        rect(img, x, y, w, h, (30, 32, 38), 0.97)
        rect(img, x, y, 4, h, (60, 170, 70), 1.0)
        blit(img, np.asarray(Image.fromarray(icon_grass_block()).resize((40, 40))), x + 18, y + 18)
        pil_text(img, title, x + 72, y + 30, self.fb, fill=(240, 240, 240, 255), anchor='lm')
        pil_text(img, text, x + 72, y + 62, self.f, fill=(200, 200, 205, 255), anchor='lm')

    def draw_cursor(self, img, x, y):
        blit(img, self.cursor, x, y)


def front_doll(skin, k=6):
    """A flat front view of a player skin (head, body, arms, legs) for the launcher preview."""
    s = skin
    doll = np.zeros((32, 16, 4), np.uint8)
    doll[0:8, 4:12] = s[8:16, 8:16]
    hat = s[8:16, 40:48]
    m = hat[..., 3] > 0
    doll[0:8, 4:12][m] = hat[m]
    doll[8:20, 4:12] = s[20:32, 20:28]
    doll[8:20, 0:4] = s[20:32, 44:48]
    doll[8:20, 12:16] = s[52:64, 36:40]
    doll[20:32, 4:8] = s[20:32, 4:8]
    doll[20:32, 8:12] = s[52:64, 20:24]
    return up(doll, k)


# ---------------------------------------------------------------------------------------------
# the video player (on the monitor in the room, showing this video)
# ---------------------------------------------------------------------------------------------
def video_player(frame, t, total, title='THE PLAYER WHO NEVER LOGGED OUT', size=(512, 288)):
    """An (h, w, 4) screen image: the frame, a progress bar and the time, in a generic dark player."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    fh = h - 34
    fr = np.asarray(Image.fromarray(frame).resize((w, fh), Image.BILINEAR))
    img[:fh] = fr
    rect(img, 0, fh, w, 34, (16, 16, 16), 1.0)
    rect(img, 8, fh + 6, w - 16, 3, (80, 80, 80), 1.0)
    rect(img, 8, fh + 6, (w - 16) * np.clip(t / total, 0, 1), 3, (230, 30, 30), 1.0)
    f = font(12)

    def mmss(s):
        return f'{int(s // 60)}:{int(s % 60):02d}'
    pil_text(img, f'{mmss(t)} / {mmss(total)}', 10, fh + 22, f, anchor='lm')
    pil_text(img, title, w - 10, fh + 22, f, fill=(200, 200, 200, 255), anchor='rm')
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = img
    out[..., 3] = 255
    return out


# ---------------------------------------------------------------------------------------------
# the hook: join / leave messages and the server log
# ---------------------------------------------------------------------------------------------
def big_lines(img, lines, alpha=1.0, y=None):
    """Centred white terminal-style lines on black (the hook)."""
    f = font(44, mono=True)
    yy = y if y is not None else H // 2 - (len(lines) - 1) * 34
    for i, (text, col) in enumerate(lines):
        pil_text(img, text, W // 2, yy + i * 68, f, fill=(*col, 255), anchor='mm', alpha=alpha)


def server_log(img, entries, alpha=1.0, x=120, y_bottom=H - 90, size=26):
    """A scrolling console: entries = list of (text, colour), newest last."""
    f = font(size, mono=True)
    lh = int(size * 1.35)
    n = (y_bottom - 60) // lh
    for i, (text, col) in enumerate(entries[-n:]):
        pil_text(img, text, x, y_bottom - (len(entries[-n:]) - i) * lh, f, fill=(*col, 255), anchor='ls', alpha=alpha)


def title_card(img, text='THE PLAYER WHO NEVER LOGGED OUT', alpha=1.0, glitch=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    f = font(64, mono=True)
    layer = np.zeros_like(img)
    pil_text(layer, text, W // 2, H // 2, f, anchor='mm', alpha=1.0)
    if glitch > 0:
        # a few rows of the title slip sideways and split into colour channels
        for _ in range(int(3 + 8 * glitch)):
            y0 = int(rng.integers(H // 2 - 40, H // 2 + 40))
            hh = int(rng.integers(2, 10))
            dx = int(rng.normal(0, 30 * glitch))
            layer[y0:y0 + hh] = np.roll(layer[y0:y0 + hh], dx, axis=1)
        r = np.roll(layer[..., 0], int(6 * glitch), axis=1)
        b = np.roll(layer[..., 2], -int(6 * glitch), axis=1)
        layer[..., 0] = r
        layer[..., 2] = b
    img[:] = np.maximum(img, (layer.astype(np.float32) * alpha).astype(np.uint8))
