"""Drawing the living things into the frame (in the renderer's texel space): Steve (a rig of head, body, arms and legs
that swings his pickaxe, falls, lands, cheers and burns), the zombie in the cave, the cave spiders in the mineshaft,
the Warden digging itself out of the Ancient City's floor; and the dropped items, the debris of every block he breaks,
the sculk sensors' vibrations, the shriek, sparks, flames and smoke.
"""
import cv2
import numpy as np

import art as A
import world as WD

TX = 16
LOCAL = 64                      # the character canvas (texels); feet at (32, 52)
FX, FY = 32, 52


def _over(dst, src, x0, y0):
    """Alpha-composite RGBA src (float, alpha 0..255) onto RGBA dst at integer (x0, y0)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    s = src[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4] / 255.0
    d = dst[ya:yb, xa:xb]
    d[..., :3] = d[..., :3] * (1 - a) + s[..., :3] * a
    d[..., 3:4] = np.maximum(d[..., 3:4], s[..., 3:4])


def _part(canvas, spr, pivot, at, angle):
    """Draw sprite rotated by angle (degrees, counter-clockwise on screen) about its pivot, the pivot placed at `at`."""
    M = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), float(angle), 1.0)
    M[0, 2] += at[0] - pivot[0]
    M[1, 2] += at[1] - pivot[1]
    out = cv2.warpAffine(spr, M, (LOCAL, LOCAL), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
    a = out[..., 3:4] / 255.0
    canvas[..., :3] = canvas[..., :3] * (1 - a) + out[..., :3] * a
    canvas[..., 3:4] = np.maximum(canvas[..., 3:4], out[..., 3:4])


def rig(parts, pose, pickaxe=None):
    """A side-on figure facing right on a LOCAL x LOCAL RGBA canvas, feet at (FX, FY).
    pose: arm (front), arm_b (back), leg (front, + = forward), head (tilt, + = looking down), crouch (texels),
    pick (whether he holds the pickaxe)."""
    c = np.zeros((LOCAL, LOCAL, 4), np.float32)
    cr = pose.get('crouch', 0.0)
    hip = (FX, FY - 12 + cr)
    sh = (FX, FY - 24 + cr)
    dark = lambda s: np.concatenate([s[..., :3] * 0.72, s[..., 3:]], -1)
    arm, leg, body, head = parts['arm'], parts['leg'], parts['body'], parts['head']
    _part(c, dark(arm), (2, 1), sh, pose.get('arm_b', 0.0))
    _part(c, dark(leg), (2, 0), hip, -pose.get('leg', 0.0))
    _part(c, body, (2, 12), hip, 0.0)
    _part(c, leg, (2, 0), hip, pose.get('leg', 0.0))
    _part(c, head, (4, 8), sh, -pose.get('head', 0.0))
    ang = pose.get('arm', 0.0)
    if pickaxe is not None and pose.get('pick', True):
        # the pickaxe in his hand: its grip at the end of the arm, its head pointing on from the arm
        th = np.radians(ang)
        hand = (sh[0] + 11 * np.sin(th), sh[1] + 11 * np.cos(th))
        _part(c, pickaxe, (2.5, 11.5), hand, ang - 135.0 + 25.0)
    _part(c, arm, (2, 1), sh, ang)
    return c


class Cast:
    def __init__(self, story, renderer):
        self.s = story
        self.r = renderer
        ch = A.characters()
        self.steve = ch['steve']
        self.zombie = ch['zombie']
        self.spider = ch['spider']
        w = ch['warden']
        self.warden = cv2.resize(w, (int(w.shape[1] * 1.5), int(w.shape[0] * 1.5)), interpolation=cv2.INTER_NEAREST)
        g = (np.abs(self.warden[..., 1] - 210) < 50) & (self.warden[..., 2] > 200)
        self.warden_glow = g
        self.pick = ch['pickaxe']
        self.T, _ = A.block_textures()
        self.icons = {b: A.item_icon(self.T, b) for b in range(len(A.BLOCKS))}
        dia = A.sprite('diamond')
        self.diamond = cv2.resize(dia, (12, 12), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(5)
        self.debris = rng.random((len(story.breaks) + 10, 12, 5))
        self.fire = self._flames()

    @staticmethod
    def _flames():
        rng = np.random.default_rng(3)
        fr = []
        for k in range(8):
            img = np.zeros((20, 16, 4), np.float32)
            for c in range(16):
                h = int(6 + 10 * rng.random() * (1 - abs(c - 8) / 10))
                for r in range(20 - h, 20):
                    f = (r - (20 - h)) / max(h, 1)
                    col = (255, 240, 140) if f > 0.7 else ((255, 170, 40) if f > 0.35 else (230, 80, 20))
                    if rng.random() < 0.85:
                        img[r, c, :3] = col
                        img[r, c, 3] = 230
            fr.append(img)
        return fr

    # -- poses ------------------------------------------------------------------------------------------
    def steve_pose(self, t):
        a = self.s.act_at(t)
        kind, t0, t1, d = a
        u = (t - t0) / max(t1 - t0, 1e-6)
        face = 'right'
        pose = {'arm': 8.0, 'arm_b': -6.0, 'leg': 0.0, 'head': 0.0}
        if kind == 'stand':
            pose['arm'] = 6 + 3 * np.sin(t * 3)
        elif kind == 'mine':
            x, y, f, b = d
            n = max(1, int(round((t1 - t0) / 0.13)))
            ph = (u * n) % 1.0
            sw = 0.5 + 0.5 * np.cos(2 * np.pi * ph)          # 1 raised .. 0 struck
            if f == 'down':
                pose['arm'] = 28 + 120 * sw
                pose['head'] = 28
            else:
                face = f
                pose['arm'] = 55 + 105 * sw
                pose['head'] = 18
            pose['arm_b'] = -10 + 8 * sw
        elif kind == 'turn':
            face = d[0]
        elif kind == 'fall':
            pose['arm'] = 150
            pose['arm_b'] = 165
            pose['leg'] = 12
            pose['head'] = -10
        elif kind == 'land':
            pose['crouch'] = 3 * (1 - u)
            pose['arm'] = 40 * (1 - u)
            pose['head'] = 15
        elif kind == 'torch':
            pose['arm'] = 70 + 20 * np.sin(np.pi * u)
            pose['pick'] = False
        elif kind == 'cheer':
            pose['arm'] = 170
            pose['arm_b'] = 150
            pose['leg'] = 10 * np.sin(u * np.pi * 2)
        elif kind in ('burn', 'dead'):
            pose['arm'] = 140 + 40 * np.sin(t * 17)
            pose['arm_b'] = 150 + 40 * np.sin(t * 13 + 1)
            pose['leg'] = 20 * np.sin(t * 11)
            pose['head'] = -15
        # carry the facing of the last side swing back to the right once he mines down again
        return pose, face

    # -- the frame's entities ---------------------------------------------------------------------------
    def draw(self, img, to_px, rend, t):
        s = self.s
        ev = s.events
        pre = img.copy()
        # the zombie in the cave: shuffles, then comes for the hole and stands at its edge, arms out
        zx = 4.4
        if t >= ev['cave_land'] + 0.4:
            zx = 4.4 - min(2.8, (t - ev['cave_land'] - 0.4) * 1.6)
        walking = ev['cave_land'] + 0.4 <= t < ev['cave_land'] + 0.4 + 2.8 / 1.6
        zy = self._floor(zx)
        zp = {'arm': 90 + 4 * np.sin(t * 6), 'arm_b': 86, 'leg': 22 * np.sin(t * 9) if walking else 0,
              'head': 8 if not walking else 0}
        self._draw_rig(img, to_px, rend, self.zombie, zp, (zx, zy), 'left', pick=False)
        # cave spiders in the mineshaft
        for k, x0 in enumerate((-7.0, -11.5)):
            x = x0
            if 'mine_land' in ev and t >= ev['mine_land']:
                x = min(-1.3 - k * 1.4, x0 + (t - ev['mine_land']) * 3.2)
            self._sprite(img, to_px, rend, self.spider, x, WD.MINE_Y, flip=False,
                         jitter=int((t * 14 + k) % 2), scale=1)
        # the Warden
        self._warden(img, to_px, rend, t)
        # vibrations flying to the sensors
        for (t0, t1, a, b) in s.vibes:
            if t0 <= t < t1:
                u = (t - t0) / (t1 - t0)
                x = a[0] + (b[0] - a[0]) * u
                y = a[1] + (b[1] - a[1]) * u + 0.3 * np.sin(u * np.pi * 3)
                px, py = to_px(x, y)
                cv2.circle(img, (int(px), int(py)), 2, (90, 255, 255), -1)
                cv2.circle(img, (int(px), int(py)), 3, (40, 160, 170), 1)
        if ev['shriek'] <= t < ev['shriek'] + 1.8:
            u = (t - ev['shriek']) / 1.8
            px, py = to_px(11.5, WD.CITY_Y + 1.6)
            for k in range(3):
                v = (u * 3 + k / 3.0) % 1.0
                cv2.circle(img, (int(px), int(py - v * 50)), int(4 + v * 14), (120, 255, 255), 1)
        # Steve
        fx, fy = s.feet_at(t)
        pose, face = self.steve_pose(t)
        tint = None
        if t >= ev['lava']:
            tint = np.array([1.0, 0.55, 0.4])
        hurt = s.hurt_at(t)
        if hurt > 0:
            tint = np.array([1.0, 1 - 0.55 * hurt, 1 - 0.55 * hurt]) * (tint if tint is not None else 1)
        y_off = 0.0
        a = s.act_at(t)
        if a[0] == 'cheer':
            y_off = 0.35 * np.sin(np.pi * (t - a[1]) / (a[2] - a[1]))
        self._draw_rig(img, to_px, rend, self.steve, pose, (fx, fy + y_off), face, pick=True, tint=tint,
                       min_light=0.72)
        if t >= ev['lava']:
            # in the lava: it's in front of him; flames on him
            surf = WD.LAVA_TOP
            _, py_s = to_px(0, surf)
            py_s = int(py_s) + 2
            lo, hi = to_px(-8, surf)
            img[py_s:, :] = pre[py_s:, :]
            fr = self.fire[int(t * 18) % len(self.fire)]
            for k, dx in enumerate((-0.25, 0.05, 0.3)):
                fl = self.fire[(int(t * 18) + k * 3) % len(self.fire)]
                px, py = to_px(fx + dx - 0.5, fy + 1.9 - 0.05 * k)
                self._blit(img, fl, int(px), int(py), alpha=0.85)
        # items flying to him, debris, sparks
        self._items(img, to_px, t, (fx, fy))
        self._debris(img, to_px, t)

    def _floor(self, x):
        y = WD.CAVE_CEIL
        w = self.s.w
        while w.get(int(np.floor(x)), y - 1) == A.B['air'] and y > WD.CAVE_FLOOR - 4:
            y -= 1
        return float(y)

    def _draw_rig(self, img, to_px, rend, parts, pose, feet, face, pick=False, tint=None, min_light=0.62):
        c = rig(parts, pose, self.pick if pick else None)
        if face == 'left':
            c = c[:, ::-1].copy()
        L = np.maximum(rend.light_at(feet[0], feet[1] + 1.0), min_light)
        c[..., :3] *= np.minimum(L, 1.15)
        if tint is not None:
            c[..., :3] *= tint
        px, py = to_px(feet[0], feet[1])
        fx = FX if face == 'right' else LOCAL - FX
        self._blit(img, c, int(round(px - fx)), int(round(py - FY)))

    def _sprite(self, img, to_px, rend, spr, x, y, flip=False, jitter=0, scale=1, min_light=0.6):
        s = spr[:, ::-1] if flip else spr
        L = np.maximum(rend.light_at(x, y + 0.5), min_light)
        s = s.copy()
        s[..., :3] *= np.minimum(L, 1.1)
        px, py = to_px(x, y)
        self._blit(img, s, int(px - s.shape[1] / 2), int(py - s.shape[0]) - jitter)

    def _warden(self, img, to_px, rend, t):
        ev = self.s.events
        if t < ev['warden_rise']:
            return
        w = self.warden.copy()
        h = w.shape[0]
        rise = np.clip((t - ev['warden_rise']) / 1.6, 0, 1)
        x = 8.5
        if t >= ev['warden_walk']:
            x = max(2.7, 8.5 - (t - ev['warden_walk']) * 2.2)
        floor = WD.CITY_Y + 1
        # the glowing ribs beat like a heart
        beat = 0.5 + 0.5 * np.cos(2 * np.pi * t * 1.4)
        roar = ev['warden_roar'] <= t < ev['warden_roar'] + 1.2
        if roar:
            beat = 1.0
        L = np.maximum(rend.light_at(x, floor + 1), 0.95)
        w[..., :3] *= np.minimum(L, 1.2) * 1.25
        g = self.warden_glow
        w[g, :3] = np.array([60, 230, 240]) * (0.7 + 0.8 * beat)
        w = w[:, ::-1]                                   # he faces left, towards the shaft
        vis = int(round(h * rise))
        w = w[:vis]
        px, py = to_px(x, floor)
        jit = int(np.sin(t * 60) * 1.5) if roar else 0
        step = int(abs(np.sin((t - ev['warden_walk']) * 4.4)) * 2) if t >= ev['warden_walk'] else 0
        self._blit(img, w, int(px - w.shape[1] / 2) + jit, int(py - vis) - step)
        if rise < 1:
            rng = np.random.default_rng(int(t * 60))
            for _ in range(10):
                dx, dy = rng.uniform(-1.2, 1.2), rng.uniform(0, 0.6)
                qx, qy = to_px(x + dx, floor + dy)
                img[int(qy):int(qy) + 2, int(qx):int(qx) + 2] = (40, 50, 60)

    def _items(self, img, to_px, t, feet):
        s = self.s
        tx, ty = feet[0], feet[1] + 0.9
        for (tb, x, y, b) in s.breaks:
            a = t - tb
            if a < 0 or a > 0.45:
                continue
            ox, oy = x + 0.5, y + 0.5
            if a < 0.15:
                k = a / 0.15
                px, py = ox, oy + 0.3 * np.sin(np.pi * k)
            else:
                k = (a - 0.15) / 0.3
                k = k * k
                px, py = ox + (tx - ox) * k, oy + (ty - oy) * k
            icon = self.diamond if b in (A.B['deep_diamond'], A.B['diamond_ore']) else self.icons[b]
            if b == A.B['grass']:
                icon = self.icons[A.B['dirt']]
            qx, qy = to_px(px, py)
            self._blit(img, icon, int(qx - icon.shape[1] / 2), int(qy - icon.shape[0] / 2))
            if icon is self.diamond:
                for k2 in range(4):
                    ang = t * 9 + k2 * 1.57
                    sx, sy = qx + 9 * np.cos(ang), qy + 9 * np.sin(ang)
                    img[int(sy):int(sy) + 1, int(sx):int(sx) + 1] = (255, 255, 255)

    def _debris(self, img, to_px, t):
        s = self.s
        T = self.T
        for i, (tb, x, y, b) in enumerate(s.breaks):
            a = t - tb
            if a < 0 or a > 0.5:
                continue
            R = self.debris[i]
            for (u1, u2, u3, u4, u5) in R:
                vx = (u1 - 0.5) * 5
                vy = 2 + u2 * 4
                px = x + 0.2 + 0.6 * u3 + vx * a
                py = y + 0.3 + 0.5 * u4 + vy * a - 18 * a * a
                col = T[b][int(u4 * 15), int(u3 * 15)] * (0.8 + 0.4 * u5)
                qx, qy = to_px(px, py)
                sz = 2 if u5 > 0.5 else 1
                qx, qy = int(qx), int(qy)
                if 0 <= qx < img.shape[1] - 2 and 0 <= qy < img.shape[0] - 2:
                    img[qy:qy + sz, qx:qx + sz] = col
        # little chips on every swing
        for (th, x, y, b) in s.hits:
            a = t - th
            if 0 <= a < 0.18:
                rng = np.random.default_rng(int(th * 1000))
                for _ in range(3):
                    px = x + rng.uniform(0.2, 0.8)
                    py = y + 1.0 + rng.uniform(0, 0.4) * a / 0.18
                    qx, qy = to_px(px, py)
                    col = T[b][int(rng.integers(16)), int(rng.integers(16))]
                    img[int(qy):int(qy) + 1, int(qx):int(qx) + 1] = col

    @staticmethod
    def _blit(img, spr, x0, y0, alpha=1.0):
        h, w = spr.shape[:2]
        H, W = img.shape[:2]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb <= xa or yb <= ya:
            return
        s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
        a = s[..., 3:4] / 255.0 * alpha
        img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + s[..., :3] * a
