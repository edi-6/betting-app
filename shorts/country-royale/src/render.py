"""Draws a frame of the battle royale at any moment of the simulation: the background, the neon ring, the flag
balls (squashing on impacts, leaning into their motion, eyes that look where they go, blink and panic), the
eliminated balls flying down to the board, bursts in the flags' colours, the board of all 50 flags with each one's
final place, and the HUD. Every element is a function of time (no state carried between frames), so any frame can be
rendered on its own and the render can be split into segments.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image

import balls as BL
import flags as FL
import sim as SIM
import text as TX

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
RC = np.array([540.0, 812.0])        # ring centre on screen
S = 438.0                            # pixels per ring radius
RING_W = 0.038                       # neon tube thickness, ring radii

COLS, ROWS = 10, 5
TILE_W, TILE_H, TILE_GAP = 76, 50, 9
BOARD_Y0 = 1300
BOARD_X0 = (W - (COLS * TILE_W + (COLS - 1) * TILE_GAP)) / 2

OUT_T = 0.9                          # an eliminated ball takes this long to reach its tile on the board

GOLD = (255, 205, 40)
YELLOW = (255, 222, 60)


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def back_out(x, s=1.9):
    x = min(max(x, 0.0), 1.0) - 1
    return 1 + x * x * ((s + 1) * x + s)


def hsv(h, s, v):
    c = cv2.cvtColor(np.uint8([[[int(h * 180) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return c.astype(np.float32) / 255.0


def _hash(*a):
    x = 0x345678
    for v in a:
        x = (x * 1000003) ^ int(v)
        x &= 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0x5bd1e995) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 0xFFFFFFFF


# ----------------------------------------------------------------------------------------------------------------

class Assets:
    def __init__(self):
        cache = os.path.join(HERE, '..', 'cache', 'balls')
        os.makedirs(cache, exist_ok=True)
        self.mips, self.tiles, self.tiles_dim, self.colors = {}, {}, {}, {}
        for code, name, aspect, fn in FL.COUNTRIES:
            p = os.path.join(cache, f'{code}.png')
            if os.path.exists(p):
                spr = np.asarray(Image.open(p), np.float32) / 255.0
            else:
                spr = BL.render_ball(code)
                Image.fromarray((np.clip(spr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)
            self.mips[code] = BL.Mips(spr)
            tex = FL.texture(code, 1024)
            self.tiles[code], self.tiles_dim[code] = self._tile(tex)
            self.colors[code] = self._palette(tex)

    @staticmethod
    def _tile(tex):
        th, tw = tex.shape[:2]
        want = TILE_W / TILE_H
        if tw / th > want:
            cw = int(th * want)
            x0 = (tw - cw) // 2 if tw / th < 1.8 else int((tw - cw) * 0.35)
            crop = tex[:, x0:x0 + cw]
        else:
            ch = int(tw / want)
            crop = tex[(th - ch) // 2:(th - ch) // 2 + ch]
        ss = 4
        img = cv2.resize(crop, (TILE_W * ss, TILE_H * ss), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mask = np.zeros((TILE_H * ss, TILE_W * ss), np.uint8)
        rr = 9 * ss
        cv2.rectangle(mask, (rr, 0), (TILE_W * ss - rr, TILE_H * ss), 255, -1)
        cv2.rectangle(mask, (0, rr), (TILE_W * ss, TILE_H * ss - rr), 255, -1)
        for cx, cy in ((rr, rr), (TILE_W * ss - rr - 1, rr), (rr, TILE_H * ss - rr - 1),
                       (TILE_W * ss - rr - 1, TILE_H * ss - rr - 1)):
            cv2.circle(mask, (cx, cy), rr, 255, -1)
        a = mask.astype(np.float32)[..., None] / 255
        # a light inner border so dark flags read against the background
        edge = cv2.erode(mask, np.ones((7, 7), np.uint8))
        e = ((mask > 0) & (edge == 0)).astype(np.float32)[..., None]
        img = img * (1 - e * 0.5) + e * 0.5
        rgba = np.concatenate([img * a, a], -1)
        rgba = cv2.resize(rgba, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        gray = rgba[..., :3].mean(-1, keepdims=True)
        dim = rgba.copy()
        dim[..., :3] = (gray * 0.45 + rgba[..., :3] * 0.3) * 0.5
        return rgba, dim

    @staticmethod
    def _palette(tex):
        """The flag's main colours, for confetti and bursts (by area)."""
        small = cv2.resize(tex, (48, 32), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        q = (small // 32).astype(int)
        keys, counts = np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], return_counts=True)
        cols = []
        for k in keys[np.argsort(-counts)][:4]:
            sel = (q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]) == k
            cols.append(small[sel].mean(0) / 255.0)
        return cols


# ----------------------------------------------------------------------------------------------------------------

class Story:
    """The run, read back for the frame renderer: positions at any time, who went out when, their places."""

    def __init__(self, d):
        self.d = d
        self.t = d['t']
        self.fps = SIM.FPS
        self.order = d['order']                          # spot -> country index
        self.codes = [FL.COUNTRIES[k][0] for k in self.order]
        self.names = [FL.COUNTRIES[k][1] for k in self.order]
        self.t_out = d['t_out']
        n = len(self.order)
        te = np.where(np.isnan(self.t_out), np.inf, self.t_out)
        rank_order = np.argsort(te)                      # first out first
        self.place = np.empty(n, int)
        for k, i in enumerate(rank_order):
            self.place[i] = n - k                        # first out is 50th, the winner 1st
        self.winner = int(rank_order[-1])
        self.t_win = float(np.sort(te)[-2])
        self.elims = [(float(self.t_out[i]), int(i)) for i in rank_order[:-1]]
        self.out_state = {int(r[0]): r[1:] for r in d['out_state']}
        self.impacts = d['impacts']
        self.imp_by_ball = {}
        for row in self.impacts:
            t, a, b, imp = row[:4]
            self.imp_by_ball.setdefault(int(a), []).append(row)
            if b >= 0:
                r2 = row.copy()
                r2[6:8] *= -1
                self.imp_by_ball.setdefault(int(b), []).append(r2)
        for k in self.imp_by_ball:
            self.imp_by_ball[k] = np.array(self.imp_by_ball[k])
        # board layout: alphabetical by name
        idx = sorted(range(n), key=lambda i: self.names[i])
        self.tile_of = {}
        for slot, i in enumerate(idx):
            r, c = divmod(slot, COLS)
            self.tile_of[i] = (BOARD_X0 + c * (TILE_W + TILE_GAP) + TILE_W / 2,
                               BOARD_Y0 + r * (TILE_H + TILE_GAP) + TILE_H / 2)

    def n_alive(self, ts):
        return int(np.sum(~(self.t_out <= ts)))

    def sample(self, ts):
        """Interpolated state at sim time ts."""
        f = min(max(ts * self.fps, 0.0), len(self.t) - 1.001)
        i = int(f)
        a = f - i
        d = self.d
        pos = d['pos'][i] * (1 - a) + d['pos'][i + 1] * a
        vel = d['vel'][i] * (1 - a) + d['vel'][i + 1] * a
        rad = d['rad'][i] * (1 - a) + d['rad'][i + 1] * a
        ang = d['ring_angle'][i] * (1 - a) + d['ring_angle'][i + 1] * a
        gap = d['gap'][i] * (1 - a) + d['gap'][i + 1] * a
        return pos, vel, rad, ang, gap

    def look(self, ts, i):
        """Smoothed direction of travel (where the eyes look)."""
        acc = np.zeros(2)
        wsum = 0.0
        for k in range(8):
            f = int(min(max((ts - k / 60.0) * self.fps, 0), len(self.t) - 1))
            w = 0.8 ** k
            acc += self.d['vel'][f, i] * w
            wsum += w
        v = acc / wsum
        s = np.linalg.norm(v)
        return v / (s + 1.2)

    def squash(self, ts, i):
        """(amount, nx, ny): the strongest recent impact, ringing down like a spring."""
        rows = self.imp_by_ball.get(i)
        if rows is None:
            return 0.0, 1.0, 0.0
        lo = np.searchsorted(rows[:, 0], ts - 0.25)
        hi = np.searchsorted(rows[:, 0], ts)
        best, bn = 0.0, (1.0, 0.0)
        for row in rows[lo:hi]:
            dt = ts - row[0]
            k = 0.06 if row[2] < 0 else 0.035
            amp = min(0.22, k * row[3]) * math.exp(-dt / 0.055) * math.cos(2 * math.pi * 7.5 * dt)
            if abs(amp) > abs(best):
                best, bn = amp, (row[6], row[7])
        return best, bn[0], bn[1]

# ----------------------------------------------------------------------------------------------------------------

def make_background():
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    top = np.array([0.055, 0.045, 0.13])
    bot = np.array([0.02, 0.02, 0.06])
    bg = top * (1 - y[..., None]) + bot * y[..., None]
    bg = np.broadcast_to(bg, (H, W, 3)).copy()
    yy = (np.arange(H)[:, None] - RC[1]) / S
    xx = (np.arange(W)[None, :] - RC[0]) / S
    d = np.sqrt(xx * xx + yy * yy)
    bg *= (1 - 0.35 * np.clip((np.abs(x) - 0.4) / 0.6, 0, 1) ** 2)[..., None]
    inside = np.clip(1.1 - d, 0, 1)[..., None]
    bg += inside * np.array([0.02, 0.02, 0.05])
    return bg.astype(np.float32), d.astype(np.float32)


class Renderer:
    def __init__(self, story, assets=None):
        self.st = story
        self.A = assets or Assets()
        self.bg, self.bg_d = make_background()
        self.code_of = story.codes

    # -- helpers --------------------------------------------------------------------------------------------------
    def to_screen(self, p, cam):
        c, z = cam['c'], cam['zoom']
        return RC + (np.asarray(p) - c) * S * z + cam.get('shake', (0, 0))

    def ring_color(self, ts):
        n = self.st.n_alive(ts)
        if ts >= self.st.t_win:
            return np.array([1.0, 0.78, 0.2])
        if n <= 2:
            return np.array([1.0, 0.28, 0.25])
        if n <= 10:
            k = (10 - n) / 8
            return hsv(0.86 + 0.08 * k, 0.75, 1.0)
        h = 0.5 + 0.28 * min(1.0, ts / max(self.st.t_win * 0.6, 1))
        return hsv(h, 0.72, 1.0)

    # -- layers ---------------------------------------------------------------------------------------------------
    def wall_hits(self, ts, window):
        """Wall impacts in the last `window` seconds: rows of (t, ball, other, impulse, x, y, nx, ny)."""
        imp = self.st.impacts
        lo = np.searchsorted(imp[:, 0], ts - window)
        hi = np.searchsorted(imp[:, 0], ts)
        rows = imp[lo:hi]
        return rows[rows[:, 2] < 0]

    def draw_ring(self, frame, glow, ts, cam, ring_alpha=1.0):
        pos, vel, rad, ang, gap = self.st.sample(ts)
        col = self.ring_color(ts)
        pulse = 0.0
        for row in self.wall_hits(ts, 0.2):
            pulse += min(1.0, row[3] / 4.0) * math.exp(-(ts - row[0]) / 0.07)
        col = col * (1 + 0.35 * min(1.0, pulse))
        z = cam['zoom']
        c = self.to_screen((0, 0), cam)
        r_mid = (1.0 + RING_W / 2) * S * z
        thick = RING_W * S * z
        sh = 3
        a0 = math.degrees(ang + gap / 2)
        a1 = math.degrees(ang + 2 * math.pi - gap / 2)
        x0, y0 = int(max(0, c[0] - r_mid - thick * 3)), int(max(0, c[1] - r_mid - thick * 3))
        x1, y1 = int(min(W, c[0] + r_mid + thick * 3)), int(min(H, c[1] + r_mid + thick * 3))
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        core = np.zeros_like(m)
        cc = (int(round((c[0] - x0) * 8)), int(round((c[1] - y0) * 8)))
        rr = int(round(r_mid * 8))
        cv2.ellipse(m, cc, (rr, rr), 0, a0, a1, 255, max(1, int(round(thick))), cv2.LINE_AA, sh)
        cv2.ellipse(core, cc, (rr, rr), 0, a0, a1, 255, max(1, int(round(thick * 0.38))), cv2.LINE_AA, sh)
        # round caps at the mouth of the gap
        for a in (ang + gap / 2, ang - gap / 2):
            px = (int(round((c[0] - x0 + r_mid * math.cos(a)) * 8)), int(round((c[1] - y0 + r_mid * math.sin(a)) * 8)))
            cv2.circle(m, px, int(round(thick * 0.5 * 8)), 255, -1, cv2.LINE_AA, sh)
            cv2.circle(core, px, int(round(thick * 0.2 * 8)), 255, -1, cv2.LINE_AA, sh)
        ma = (m.astype(np.float32) / 255.0)[..., None] * ring_alpha
        mc = (core.astype(np.float32) / 255.0)[..., None] * ring_alpha
        reg = frame[y0:y1, x0:x1]
        reg *= 1 - ma
        reg += ma * col * 0.95
        reg += mc * (1 - col) * 0.75
        gx0, gy0 = x0 // 4, y0 // 4
        small = cv2.resize(ma[..., 0], ((x1 - x0) // 4, (y1 - y0) // 4), interpolation=cv2.INTER_AREA)
        g = glow[gy0:gy0 + small.shape[0], gx0:gx0 + small.shape[1]]
        g += small[:g.shape[0], :g.shape[1], None] * col * 1.6

    def draw_ball(self, frame, code, xy, rpx, look=(0, 0), blink=1.0, mood='normal', squash=(0.0, 1.0, 0.0),
                  tilt=0.0, alpha=1.0):
        d = 2 * rpx
        if d < 3:
            return
        mips = self.A.mips[code]
        spr = mips.pick(d)
        m = spr.shape[0]
        patch = spr.copy()
        BL.draw_eyes(patch, look=look, blink=blink, mood=mood)
        k = d / m
        amt, nx, ny = squash
        nrm = np.array([nx, ny])
        nn = np.linalg.norm(nrm)
        nrm = nrm / nn if nn > 1e-6 else np.array([1.0, 0.0])
        tng = np.array([-nrm[1], nrm[0]])
        sq = (1 - amt) * np.outer(nrm, nrm) + (1 + 0.6 * amt) * np.outer(tng, tng)
        ct, stt = math.cos(tilt), math.sin(tilt)
        rot = np.array([[ct, -stt], [stt, ct]])
        A = sq @ rot * k
        half = int(math.ceil(rpx * 1.35)) + 2
        bx0, by0 = int(math.floor(xy[0])) - half, int(math.floor(xy[1])) - half
        size = 2 * half + 1
        t = np.array(xy) - np.array([bx0, by0]) - A @ np.array([m / 2, m / 2])
        M = np.hstack([A, t[:, None]]).astype(np.float32)
        out = cv2.warpAffine(patch, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        TX.over(frame, out, bx0 + size / 2, by0 + size / 2, opacity=alpha)

    def blink(self, i, ts):
        period = 2.2 + 3.0 * _hash(i, 7)
        ph = (ts + _hash(i, 3) * period) % period
        if ph < 0.13:
            return abs(ph - 0.065) / 0.065
        return 1.0

    def draw_balls(self, frame, ts, cam, winner_override=None):
        st = self.st
        pos, vel, rad, ang, gap = st.sample(ts)
        z = cam['zoom']
        alive = ~(st.t_out <= ts)
        for i in np.nonzero(alive)[0]:
            if winner_override is not None and i == st.winner:
                continue
            p = pos[i]
            xy = self.to_screen(p, cam)
            rpx = rad[i] * S * z
            if xy[0] < -rpx * 2 or xy[0] > W + rpx * 2 or xy[1] < -rpx * 2 or xy[1] > H + rpx * 2:
                continue
            look = st.look(ts, i)
            d = math.hypot(p[0], p[1])
            rel = (math.atan2(p[1], p[0]) - ang + math.pi) % (2 * math.pi) - math.pi
            mood = 'normal'
            if abs(rel) < gap / 2 + 0.25 and d > 1.0 - 2.6 * rad[i]:
                mood = 'shock'
            if ts >= st.t_win and i == st.winner:
                mood = 'happy'
            amt, nx, ny = st.squash(ts, i)
            tilt = float(np.clip(-0.05 * vel[i][0], -0.22, 0.22))
            self.draw_ball(frame, self.code_of[i], xy, rpx, look=look, blink=self.blink(i, ts), mood=mood,
                           squash=(amt, nx, ny), tilt=tilt)

    def out_track(self, i, ts, cam):
        """Screen position and radius of an eliminated ball: it leaves the ring with its own velocity and curves
        down into its tile on the board (a cubic Hermite path), shrinking to the tile's size."""
        st = self.st
        t0, x, y, vx, vy, r = st.out_state[i]
        z = cam['zoom']
        u = (ts - t0) / OUT_T
        p0 = self.to_screen((x, y), cam)
        v0 = np.array([vx, vy]) * S * z * OUT_T
        n = np.linalg.norm(v0)
        if n > 420:
            v0 *= 420 / n
        p1 = np.array(st.tile_of[i])
        v1 = np.array([0.0, 160.0])
        uu = min(max(u, 0.0), 1.0)
        h00, h10 = 2 * uu ** 3 - 3 * uu ** 2 + 1, uu ** 3 - 2 * uu ** 2 + uu
        h01, h11 = -2 * uu ** 3 + 3 * uu ** 2, uu ** 3 - uu ** 2
        xy = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
        e = ease(uu)
        rpx = (r * S * z) * (1 - e) + (TILE_H * 0.5) * e
        return xy, rpx, u

    def draw_out_balls(self, frame, ts, cam):
        """Eliminated balls on their way from the ring to the board."""
        st = self.st
        for t0, i in st.elims:
            if ts < t0:
                break
            if ts - t0 > OUT_T:
                continue
            xy, rpx, u = self.out_track(i, ts, cam)
            mood = 'shock' if u < 0.3 else 'dead'
            self.draw_ball(frame, self.code_of[i], xy, rpx, look=(0, 0.6), mood=mood,
                           alpha=1.0 if u < 0.93 else max(0.0, (1 - u) / 0.07))

    def sparks(self, frame, glow, ts, cam):
        n = self.st.n_alive(ts)
        thresh = 2.2 if n > 20 else 1.4
        col = self.ring_color(ts)
        z = cam['zoom']
        for row in self.wall_hits(ts, 0.3):
            t0, b, _, imp, x, y, nx, ny = row
            if imp < thresh:
                continue
            dt = ts - t0
            p = self.to_screen((x, y), cam)
            life = 1 - dt / 0.3
            for k in range(7):
                h1, h2 = _hash(int(t0 * 1000), b, k, 1), _hash(int(t0 * 1000), b, k, 2)
                # sparks fly back along the wall and away from it
                tang = np.array([-ny, nx]) * (1 if h1 > 0.5 else -1)
                d = tang * (0.5 + h2) - np.array([nx, ny]) * (0.6 + h1)
                d /= np.linalg.norm(d) + 1e-9
                sp = (260 + 420 * h2) * z * min(1.6, imp / 3)
                travel = sp * (1 - math.exp(-dt / 0.09)) * 0.09
                a = p + d * travel
                bpt = p + d * max(0.0, travel - 14 * life)
                c = tuple(float(v) for v in np.clip(col * 0.6 + 0.5, 0, 1.4) * life)
                cv2.line(frame, (int(a[0] * 4), int(a[1] * 4)), (int(bpt[0] * 4), int(bpt[1] * 4)), c, 2,
                         cv2.LINE_AA, 2)
                g = (int(a[0] / 4), int(a[1] / 4))
                if 0 <= g[0] < glow.shape[1] and 0 <= g[1] < glow.shape[0]:
                    glow[g[1], g[0]] += col * 1.5 * life

    def trail_color(self, code):
        """The flag's most colourful main colour."""
        best, sat = np.array([0.7, 0.7, 0.8]), 0.0
        for c in self.A.colors[code]:
            c = np.asarray(c)
            sc = c.max() - c.min()
            if sc > sat and c.max() > 0.3:
                best, sat = c, sc
        return best / max(best.max(), 1e-3)

    def trails(self, frame, glow, ts, cam):
        """Neon trails behind the last three balls, in each flag's colour (drawn into the glow, so they add light)."""
        st = self.st
        if st.n_alive(ts) > 3 or ts >= st.t_win:
            return
        f = int(ts * st.fps)
        alive = ~(st.t_out <= ts)
        for i in np.nonzero(alive)[0]:
            col = self.trail_color(self.code_of[i])
            pts = [self.to_screen(st.d['pos'][max(0, f - k), i], cam) / 4 for k in range(0, 18)]
            r = st.d['rad'][f, i] * S * cam['zoom'] / 4
            for k in range(len(pts) - 1):
                a = (1 - k / 17) ** 1.6 * 0.5
                w = max(1, int(round(r * 0.8 * (1 - k / 17))))
                p0, p1 = pts[k], pts[k + 1]
                cv2.line(glow, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)),
                         tuple(float(v) for v in col * a), w, cv2.LINE_AA, 2)

    def bursts(self, frame, glow, ts):
        """A pop of shards in the flag's colours where each ball lands on the board."""
        st = self.st
        for t0, i in st.elims:
            ta = t0 + OUT_T
            dt = ts - ta
            if dt < 0 or dt > 0.9:
                continue
            c = np.array(st.tile_of[i])
            cols = self.A.colors[self.code_of[i]]
            for k in range(16):
                ang = 2 * math.pi * _hash(i, k, 1)
                sp = 220 + 380 * _hash(i, k, 2)
                vx, vy = math.cos(ang) * sp, math.sin(ang) * sp - 180
                drag = math.exp(-2.5 * dt)
                px = c[0] + vx * (1 - drag) / 2.5
                py = c[1] + vy * (1 - drag) / 2.5 + 520 * dt * dt
                life = 1 - dt / 0.9
                col = cols[k % len(cols)]
                sz = 3 + 4 * _hash(i, k, 3)
                cv2.circle(frame, (int(px), int(py)), int(sz * (0.5 + 0.5 * life)), tuple(float(v) for v in col * 1.05),
                           -1, cv2.LINE_AA)
            # a ring that spreads from the tile
            rr = 10 + 90 * ease_out(dt / 0.45)
            a = max(0.0, 1 - dt / 0.45)
            if a > 0:
                x0, y0 = int(max(0, c[0] - rr - 4)), int(max(0, c[1] - rr - 4))
                x1, y1 = int(min(W, c[0] + rr + 5)), int(min(H, c[1] + rr + 5))
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.circle(m, (int(c[0]) - x0, int(c[1]) - y0), int(rr), 255, 3, cv2.LINE_AA)
                frame[y0:y1, x0:x1] += (m.astype(np.float32) / 255 * 0.6 * a)[..., None]

    def draw_board(self, frame, ts, alpha=1.0):
        st = self.st
        for i, (cx, cy) in st.tile_of.items():
            code = self.code_of[i]
            t0 = st.t_out[i]
            ta = t0 + OUT_T if not np.isnan(t0) else np.inf
            gone = ts >= ta
            tile = self.A.tiles_dim[code] if gone else self.A.tiles[code]
            sc = 1.0
            if gone and ts - ta < 0.3:
                sc = 1 + 0.25 * math.sin(math.pi * (ts - ta) / 0.3)
            TX.over(frame, tile, cx, cy, scale=sc, opacity=alpha)
            if gone:
                lab = TX.text(f'#{st.place[i]}', 25, fill=(255, 255, 255), stroke=3)
                TX.over(frame, lab, cx, cy + 1, opacity=alpha)
                if ts - ta < 0.25:
                    fl = 1 - (ts - ta) / 0.25
                    x0, y0 = int(cx - TILE_W / 2), int(cy - TILE_H / 2)
                    frame[y0:y0 + TILE_H, x0:x0 + TILE_W] += fl * 0.6
        if ts >= st.t_win + 0.3:
            cx, cy = st.tile_of[st.winner]
            k = 0.5 + 0.5 * math.sin((ts - st.t_win) * 6)
            x0, y0 = int(cx - TILE_W / 2 - 4), int(cy - TILE_H / 2 - 4)
            cv2.rectangle(frame, (x0, y0), (x0 + TILE_W + 8, y0 + TILE_H + 8),
                          (1.0, 0.8 + 0.1 * k, 0.2), 4, cv2.LINE_AA)
            TX.over(frame, TX.text('#1', 25, fill=GOLD, stroke=3), cx, cy + 1)

    def finish_glow(self, frame, glow):
        g = cv2.GaussianBlur(glow, (0, 0), 7)
        g2 = cv2.GaussianBlur(glow, (0, 0), 22)
        g = cv2.resize(g * 0.55 + g2 * 0.6, (W, H), interpolation=cv2.INTER_LINEAR)
        frame += g

    def frame(self, ts, cam=None, hud=None):
        cam = cam or {'c': np.zeros(2), 'zoom': 1.0}
        frame = self.bg.copy()
        col = self.ring_color(ts)
        frame += (np.clip(1.05 - self.bg_d, 0, 1) ** 2 * 0.07)[..., None] * col
        glow = np.zeros((H // 4, W // 4, 3), np.float32)
        self.draw_ring(frame, glow, ts, cam)
        self.draw_balls(frame, ts, cam)
        self.draw_board(frame, ts)
        self.draw_out_balls(frame, ts, cam)
        self.bursts(frame, glow, ts)
        self.finish_glow(frame, glow)
        if hud is not None:
            hud(frame, ts)
        return frame


def to8(frame):
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
