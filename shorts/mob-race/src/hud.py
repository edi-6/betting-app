"""The race's broadcast graphics: the header (round, stage, how many are left), the live standings strip with the
racers' faces sliding into race order (the eliminated greyed out and crossed), the countdown, round cards,
ELIMINATED / SAVED banners and the winner card. What happens when is read off the recorded race (story()); sprites
are built once with PIL and composited in numpy."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import heads as H

FONT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')
FPS = 30
INK = (10, 12, 24)
PANEL = (12, 16, 34, 205)
YELLOW = (255, 208, 40)
RED = (240, 52, 52)
GOLD = (255, 196, 40)
GREEN = (84, 226, 110)
WHITE = (255, 255, 255)
STAGE_NAMES = {0: 'PLINKO', 1: 'SLIME & ICE', 2: 'PISTONS & TNT', 3: 'THE FINAL'}


def font(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, f'Montserrat-{weight}.ttf'), size)


def _blur_alpha(a, radius):
    return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))


def with_shadow(sprite, blur=6, offset=(0, 5), strength=0.5, pad=18):
    h, w = sprite.shape[:2]
    Hh, Ww = h + 2 * pad, w + 2 * pad
    sh = np.zeros((Hh, Ww), np.uint8)
    sh[pad + offset[1]:pad + offset[1] + h, pad + offset[0]:pad + offset[0] + w] = sprite[..., 3]
    sa = _blur_alpha(sh, blur).astype(np.float32) / 255.0 * strength
    sp = np.zeros((Hh, Ww, 4), np.float32)
    sp[pad:pad + h, pad:pad + w] = sprite
    a = sp[..., 3] / 255.0
    out_a = a + sa * (1 - a)
    out = np.zeros((Hh, Ww, 4), np.float32)
    out[..., :3] = sp[..., :3] * (a / np.maximum(out_a, 1e-6))[..., None]
    out[..., 3] = out_a * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def text_sprite(text, fnt, fill=WHITE, stroke=0, stroke_fill=INK, tracking=0, shadow=True, gradient=None):
    widths = [fnt.getlength(ch) for ch in text]
    total = int(sum(widths) + tracking * (len(text) - 1)) + 2 * stroke + 8
    asc, desc = fnt.getmetrics()
    hgt = asc + desc + 2 * stroke + 8
    img = Image.new('RGBA', (total, hgt), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = stroke + 4
    for ch, w in zip(text, widths):
        d.text((x, stroke + 4), ch, font=fnt, fill=tuple(fill) + (255,), stroke_width=stroke,
               stroke_fill=tuple(stroke_fill) + (255,))
        x += w + tracking
    arr = np.array(img)
    if gradient is not None:
        m = Image.new('L', (total, hgt), 0)
        dm = ImageDraw.Draw(m)
        x = stroke + 4
        for ch, w in zip(text, widths):
            dm.text((x, stroke + 4), ch, font=fnt, fill=255)
            x += w + tracking
        m = np.array(m).astype(np.float32) / 255.0
        ys = np.nonzero(m.max(1) > 0)[0]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, hgt - 1)
        u = np.clip((np.arange(hgt) - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
        top, bot = np.array(gradient[0], np.float32), np.array(gradient[1], np.float32)
        col = top * (1 - u) + bot * u
        arr = arr.astype(np.float32)
        arr[..., :3] = arr[..., :3] * (1 - m[..., None]) + col * m[..., None]
        arr = arr.astype(np.uint8)
    ys, xs = np.nonzero(arr[..., 3] > 0)
    if len(ys):
        arr = arr[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    return with_shadow(arr, blur=5, offset=(0, 4), strength=0.55, pad=14) if shadow else arr


def rounded(w, h, r, fill, border=None, bw=3):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=fill, outline=border,
                                          width=bw if border else 0)
    return np.array(img)


def over(dst, src, x, y, opacity=1.0):
    if opacity <= 0.0:
        return
    h, w = src.shape[:2]
    Hh, Ww = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(Ww, x + w), min(Hh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def scaled(sprite, s, resample=Image.BICUBIC):
    if abs(s - 1.0) < 1e-3:
        return sprite
    img = Image.fromarray(sprite)
    return np.array(img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), resample))


def ease_out_back(t, s=1.9):
    t = float(np.clip(t, 0.0, 1.0)) - 1.0
    return 1.0 + (s + 1.0) * t ** 3 + s * t ** 2


def smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def ring_icon(name, size, ring=(255, 255, 255), ring_w=5, dead=False, crown=False):
    """The racer's face on a disc with a coloured ring (grey and crossed out when eliminated)."""
    S = size * 4
    ic = H.icon(name, S - 2 * ring_w * 4)
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, S - 1, S - 1), fill=tuple(ring) + (255,))
    face = Image.fromarray(ic)
    if dead:
        g = np.array(face).astype(np.float32)
        lum = g[..., :3] @ np.array([0.3, 0.59, 0.11])
        g[..., :3] = (lum[..., None] * 0.55)
        face = Image.fromarray(g.astype(np.uint8))
    img.alpha_composite(face, (ring_w * 4, ring_w * 4))
    if dead:
        w = int(S * 0.09)
        m = int(S * 0.2)
        d.line((m, m, S - m, S - m), fill=(20, 0, 0, 255), width=w + 10)
        d.line((m, S - m, S - m, m), fill=(20, 0, 0, 255), width=w + 10)
        d.line((m, m, S - m, S - m), fill=RED + (255,), width=w)
        d.line((m, S - m, S - m, m), fill=RED + (255,), width=w)
    out = np.array(img.resize((size, size), Image.LANCZOS))
    return out


def crown_sprite(w):
    img = Image.new('RGBA', (w * 4, int(w * 2.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    W4, Hc = w * 4, int(w * 2.6)
    pts = [(0, Hc), (0, Hc * 0.3), (W4 * 0.25, Hc * 0.62), (W4 * 0.5, 0), (W4 * 0.75, Hc * 0.62), (W4, Hc * 0.3),
           (W4, Hc)]
    d.polygon(pts, fill=(255, 204, 40, 255), outline=(120, 70, 0, 255))
    d.rectangle((0, Hc * 0.82, W4, Hc), fill=(230, 160, 20, 255))
    for cx in (W4 * 0.25, W4 * 0.5, W4 * 0.75):
        d.ellipse((cx - W4 * 0.05, Hc * 0.6, cx + W4 * 0.05, Hc * 0.76), fill=(230, 40, 60, 255))
    return np.array(img.resize((w, int(w * 0.65)), Image.LANCZOS))


# ---------------------------------------------------------------------------------------------
# the story: when the graphics do what, from the recorded race
# ---------------------------------------------------------------------------------------------
def story(rec, course, winner):
    n = len(rec)
    names = [m['name'] for m in rec[0]['marbles']]
    go = next((i for i, st in enumerate(rec) if st['go']), 0)
    rounds = [(go, 0)]                               # (frame, round index) when each round starts
    opens = {}
    banners = []                                     # (frame, kind, name, extra)
    cues = {}
    saved_at = set()                                 # trapdoors someone got over as they dropped
    win = None
    for i, st in enumerate(rec):
        for e in st['events']:
            if e[0] == 'release':
                rounds.append((i, e[1] + 1))
                cues.setdefault(i, {})['round'] = e[1] + 1
            elif e[0] == 'open':
                opens[e[1]] = st['t']
                if e[1] == len(course.traps) - 1 and win is None:
                    win = i
                    cues.setdefault(i, {})['winner'] = winner
            elif e[0] == 'lava':
                banners.append((i, 'out', e[1], None))
                cues.setdefault(i, {})['out'] = e[1]
            elif e[0] == 'saved':
                saved_at.add(e[1])
                gap = st['t'] - opens.get(e[1], st['t'])
                banners.append((i, 'saved', e[2], gap))
                cues.setdefault(i, {})['saved'] = e[2]
    # the last one through each trapdoor before it dropped, when the next one went into the lava right after
    # (unless someone was saved there: that's the story of that trapdoor)
    for k, t_open in opens.items():
        if k == len(course.traps) - 1 or k in saved_at:
            continue
        f = next(i for i, st in enumerate(rec) if st['t'] >= t_open - 1e-9)
        st = rec[f]
        last = [e for e in st['events'] if e[0] == 'through' and e[1] == k]
        if last:
            nxt = [b for b in banners if b[1] == 'out' and 0 < b[0] - f < 60]
            if nxt:
                banners.append((f, 'justin', last[-1][2], None))
                cues.setdefault(f, {})['justin'] = last[-1][2]
    banners.sort(key=lambda b: b[0])
    # banners play one after another, each for at least a moment
    sched = []
    t_free = 0
    for (f, kind, name, extra) in banners:
        start = max(f, t_free)
        dur = int(1.15 * FPS) if kind != 'justin' else int(1.0 * FPS)
        sched.append((start, dur, kind, name, extra))
        t_free = start + int(0.55 * FPS)
    # standings: each racer's slot, eased
    slots = np.zeros((n, len(names)))
    order_prev = None
    for i, st in enumerate(rec):
        order = st['standings'] if st['go'] else names
        pos = {nm: k for k, nm in enumerate(order)}
        slots[i] = [pos[nm] for nm in names]
        order_prev = order
    disp = slots.copy()
    for i in range(1, n):
        disp[i] = disp[i - 1] + (slots[i] - disp[i - 1]) * 0.22
    out_at = {}
    for i, st in enumerate(rec):
        for m in st['marbles']:
            if m['out'] is not None and m['name'] not in out_at:
                out_at[m['name']] = i
    return dict(go=go, rounds=rounds, banners=sched, cues=cues, win=win, names=names, disp=disp, out_at=out_at,
                n=n, order_last=order_prev, winner=winner)


def state_at(i, st, rec):
    s = {'i': i}
    fr = rec[i]
    s['left'] = sum(1 for m in fr['marbles'] if m['out'] is None)
    cur = max([k for f, k in st['rounds'] if f <= i], default=0)
    s['round'] = cur
    s['countdown'] = (i - st['go']) / FPS if i < st['go'] + int(0.8 * FPS) else None      # GO is at 2.5 s
    for f, k in st['rounds']:
        if k > 0 and 0 <= i - f < int(1.4 * FPS):
            s['card'] = (k, (i - f) / FPS)
    s['banner'] = None
    for (f0, dur, kind, name, extra) in st['banners']:
        if f0 <= i < f0 + dur:
            s['banner'] = (kind, name, extra, (i - f0) / FPS, dur / FPS)
    if st['win'] is not None and i >= st['win']:
        s['winner'] = (st['winner'], (i - st['win']) / FPS)
    s['slots'] = st['disp'][i]
    s['dead'] = {nm: (i - f) / FPS for nm, f in st['out_at'].items() if i >= f}
    s['names'] = st['names']
    return s


# ---------------------------------------------------------------------------------------------
class HUD:
    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        k = self.k = W / 1080.0
        self._cache = {}
        self.icon = int(round(96 * k))
        self.f_head = font(900, int(round(44 * k)))
        self.f_left = font(800, int(round(38 * k)))
        self.f_count = font(900, int(round(300 * k)))
        self.f_card1 = font(900, int(round(70 * k)))
        self.f_card2 = font(900, int(round(96 * k)))
        self.f_ban = font(900, int(round(64 * k)))
        self.f_ban2 = font(800, int(round(40 * k)))
        self.f_win = font(900, int(round(128 * k)))
        self.f_pos = font(900, int(round(26 * k)))
        self.panel = rounded(int(1000 * k), int(112 * k), int(26 * k), PANEL)
        self.crown = crown_sprite(int(110 * k))

    def _c(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def draw(self, frame, s):
        out = frame.astype(np.float32)
        self._header(out, s)
        self._standings(out, s)
        if s.get('countdown') is not None:
            self._countdown(out, s['countdown'])
        if s.get('card') is not None:
            self._card(out, *s['card'])
        if s.get('banner') is not None:
            self._banner(out, *s['banner'])
        if s.get('winner') is not None:
            self._winner(out, *s['winner'])
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------------------------------
    def _header(self, out, s):
        k = self.k
        x0, y0 = int(40 * k), int(150 * k)
        over(out, self.panel, x0, y0, 1.0)
        rnd = s['round']
        lab = 'FINAL' if rnd == 3 else f'ROUND {rnd + 1}'
        t1 = self._c(('h1', lab), lambda: text_sprite(lab, self.f_head, WHITE, shadow=False))
        t2 = self._c(('h2', rnd), lambda: text_sprite(STAGE_NAMES[rnd] if rnd < 3 else 'WINNER TAKES ALL',
                                                          self.f_head, YELLOW, shadow=False))
        yc = y0 + int(56 * k)
        over(out, t1, x0 + int(30 * k), yc - t1.shape[0] // 2)
        over(out, t2, x0 + int(46 * k) + t1.shape[1], yc - t2.shape[0] // 2)
        left = f"{s['left']} LEFT"
        t3 = self._c(('left', left), lambda: text_sprite(left, self.f_left, WHITE, shadow=False))
        pill = self._c(('pill', t3.shape[1]), lambda: rounded(t3.shape[1] + int(36 * k), int(62 * k), int(31 * k),
                                                              RED + (255,)))
        px = x0 + self.panel.shape[1] - pill.shape[1] - int(24 * k)
        over(out, pill, px, yc - pill.shape[0] // 2)
        over(out, t3, px + int(18 * k), yc - t3.shape[0] // 2)

    def _standings(self, out, s):
        k = self.k
        n = len(s['names'])
        size = self.icon
        gap = int(10 * k)
        x0 = (self.W - (n * size + (n - 1) * gap)) // 2
        y = int(284 * k)
        for j, name in enumerate(s['names']):
            slot = s['slots'][j]
            dead = name in s['dead']
            lead = (not dead) and slot < 0.5 and s['i'] > 0
            col = H.HEADS[name]['color']
            ring = (255, 208, 40) if lead else (tuple(int(c) for c in col) if not dead else (90, 90, 96))
            ic = self._c(('ic', name, dead, lead), lambda: ring_icon(name, size, ring, 5, dead))
            x = x0 + slot * (size + gap)
            pop = 1.0
            if dead:
                a = s['dead'][name]
                if a < 0.3:
                    pop = 1.0 + 0.35 * np.sin(np.pi * a / 0.3)
            icn = scaled(ic, pop) if pop != 1.0 else ic
            off = (size - icn.shape[0]) / 2
            over(out, icn, x + off, y + off, 0.55 if dead else 1.0)
            if not dead:
                num = str(int(round(slot)) + 1)
                badge = self._c(('pos', num, lead), lambda: self._badge(num, lead))
                over(out, badge, x + size - badge.shape[1] + int(8 * k), y + size - badge.shape[0] + int(6 * k))
            if lead:
                cr = self.crown
                over(out, scaled(cr, 0.5), x + size / 2 - cr.shape[1] * 0.25, y - cr.shape[0] * 0.5 + int(4 * k))

    def _badge(self, num, lead):
        k = self.k
        d = int(40 * k)
        img = Image.new('RGBA', (d * 4, d * 4), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((0, 0, d * 4 - 1, d * 4 - 1), fill=(YELLOW if lead else (20, 24, 44)) + (255,),
                                    outline=(255, 255, 255, 255), width=10)
        b = np.array(img.resize((d, d), Image.LANCZOS))
        t = text_sprite(num, self.f_pos, INK if lead else WHITE, shadow=False)
        over_rgba(b, t, (d - t.shape[1]) // 2, (d - t.shape[0]) // 2 + 1)
        return b

    def _countdown(self, out, t):
        k = self.k
        # t: seconds relative to GO (negative before); 3 - 2 - 1 share the time before GO
        if t > 0.8:
            return
        if t < 0:
            step = 2.5 / 3.0                   # on the beat: two beats of the 144 BPM track per number
            q = min(3, int(np.ceil(-t / step - 1e-9)))
            num = str(q)
            u = 1.0 - (-t - (q - 1) * step) / step
            if q == 3:
                u = max(u, 0.12)               # already there on the very first frame
            col, grad = WHITE, ((255, 255, 255), (200, 220, 255))
        else:
            num, u = 'GO!', t / 0.8
            col, grad = GREEN, ((170, 255, 150), (60, 200, 90))
        spr = self._c(('cd', num), lambda: text_sprite(num, self.f_count, col, stroke=int(12 * k),
                                                       gradient=grad))
        s = 1.6 - 0.6 * ease_out_back(u / 0.35) if u < 0.35 else 1.0 + 0.04 * (u - 0.35)
        a = min(1.0, u / 0.08) * (1.0 - smooth((u - 0.8) / 0.2))
        sp = scaled(spr, s)
        over(out, sp, (self.W - sp.shape[1]) / 2, int(1060 * k) - sp.shape[0] / 2, a)

    def _card(self, out, rnd, t):
        k = self.k
        a = min(1.0, t / 0.12) * (1.0 - smooth((t - 1.1) / 0.3))
        top = 'THE' if rnd == 3 else f'ROUND {rnd + 1}'
        name = 'FINAL' if rnd == 3 else STAGE_NAMES[rnd]
        t1 = self._c(('c1', top), lambda: text_sprite(top, self.f_card1, WHITE, stroke=int(6 * k)))
        t2 = self._c(('c2', name), lambda: text_sprite(name, self.f_card2, YELLOW, stroke=int(8 * k),
                                                        gradient=((255, 236, 120), (255, 170, 20))))
        slide = (1.0 - ease_out_back(t / 0.3)) * 260 * k
        yc = int(700 * k)
        over(out, t1, (self.W - t1.shape[1]) / 2 - slide, yc - t1.shape[0], a)
        over(out, t2, (self.W - t2.shape[1]) / 2 + slide, yc - int(10 * k), a)

    def _banner(self, out, kind, name, extra, t, dur):
        k = self.k
        a = min(1.0, t / 0.08) * (1.0 - smooth((t - (dur - 0.2)) / 0.2))
        label = H.HEADS[name]['label'].upper()
        if kind == 'out':
            head, col, sub = 'ELIMINATED', RED, label
        elif kind == 'saved':
            head, col = 'SAVED!', GOLD
            sub = f'{label} BY {extra:.2f}s' if extra is not None else label
        else:
            head, col, sub = 'JUST MADE IT!', GREEN, label
        w_, h_ = int(900 * k), int(190 * k)
        pan = self._c(('ban', kind), lambda: with_shadow(rounded(w_, h_, int(30 * k), (12, 16, 34, 230),
                                                                  col + (255,), int(6 * k)), 10, (0, 6), 0.5, 20))
        t1 = self._c(('b1', head), lambda: text_sprite(head, self.f_ban, col, shadow=False))
        t2 = self._c(('b2', sub), lambda: text_sprite(sub, self.f_ban2, WHITE, shadow=False))
        ic = self._c(('bic', name, kind), lambda: ring_icon(name, int(150 * k), col, 7, dead=(kind == 'out')))
        s = 1.25 - 0.25 * ease_out_back(t / 0.22)
        x0 = (self.W - pan.shape[1]) / 2
        y0 = int(1150 * k)
        p = scaled(pan, s)
        dx, dy = (pan.shape[1] - p.shape[1]) / 2, (pan.shape[0] - p.shape[0]) / 2
        over(out, p, x0 + dx, y0 + dy, a)
        cy = y0 + pan.shape[0] / 2
        over(out, ic, x0 + int(40 * k), cy - ic.shape[0] / 2, a)
        tx = x0 + int(40 * k) + ic.shape[1] + int(30 * k)
        over(out, t1, tx, cy - t1.shape[0] + int(4 * k), a)
        over(out, t2, tx, cy + int(4 * k), a)

    def _winner(self, out, name, t):
        k = self.k
        a = min(1.0, t / 0.15)
        big = self._c(('wic', name), lambda: ring_icon(name, int(300 * k), GOLD, 10))
        s = 0.3 + 0.7 * ease_out_back(t / 0.45, 1.6)
        b = scaled(big, s)
        cx, cy = self.W / 2, int(600 * k)
        over(out, b, cx - b.shape[1] / 2, cy - b.shape[0] / 2, a)
        cr = self.crown
        cr_s = scaled(cr, 1.3 * s)
        drop = (1.0 - smooth((t - 0.3) / 0.35)) * -160 * k
        over(out, cr_s, cx - cr_s.shape[1] / 2, cy - b.shape[0] / 2 - cr_s.shape[0] * 0.72 + drop,
             a * smooth((t - 0.25) / 0.2))
        t1 = self._c('win1', lambda: text_sprite('WINNER!', self.f_win, GOLD, stroke=int(10 * k),
                                                 gradient=((255, 244, 150), (255, 160, 20))))
        t2 = self._c(('win2', name), lambda: text_sprite(H.HEADS[name]['label'].upper(), self.f_card1, WHITE,
                                                         stroke=int(6 * k)))
        u = smooth((t - 0.35) / 0.3)
        t1s = scaled(t1, 0.8 + 0.2 * ease_out_back((t - 0.35) / 0.35))
        over(out, t1s, cx - t1s.shape[1] / 2, cy + int(175 * k), u)
        over(out, t2, cx - t2.shape[1] / 2, cy + int(175 * k) + t1.shape[0] - int(6 * k), u)


def over_rgba(dst, src, x, y):
    h, w = src.shape[:2]
    Hh, Ww = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(Ww, x + w), min(Hh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    d = dst[y0:y1, x0:x1].astype(np.float32)
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0
    d[..., :3] = d[..., :3] * (1 - a) + s[..., :3] * a
    d[..., 3:4] = np.maximum(d[..., 3:4], s[..., 3:4])
    dst[y0:y1, x0:x1] = d.astype(np.uint8)
