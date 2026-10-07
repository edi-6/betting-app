"""What happens, and when: Steve digs straight down from the grass to the diamonds, and one block further.

The digging is scripted block by block from the world itself: mine the block under him (how long depends on the
depth and the story's pace), drop onto whatever is below (a long fall into a cave or the geode costs hearts), place a
torch now and then. Round that: the zombie in the cave, the cave spiders of the mineshaft, the Ancient City's sculk
sensors hearing every swing until the shrieker screams and the Warden digs itself out of the floor, the five
diamonds, the last block, the lava, the death screen. Everything is a function of the video's time t.
"""
import numpy as np

import art as A
import world as WD

B = A.B
G = 30.0                       # gravity (blocks/s^2) for the big falls
INTRO = 1.25                   # the opening: the whole slice, then in to Steve
FPS = 60


def _ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


class Story:
    def __init__(self, world):
        self.w = world
        self.acts = []           # (kind, t0, t1, data)
        self.breaks = []         # (t, x, y, block)
        self.torches = []        # (t, x, y)
        self.damage = []         # (t, hp)
        self.dings = []          # (t, n): diamonds collected
        self.events = {}
        self.captions = []       # (t0, t1, text, style)
        self.hits = []           # (t, x, y, block): every swing that lands
        self._script()

    # -- the script -------------------------------------------------------------------------------------
    def _script(self):
        w = self.w
        t = 0.0
        feet = WD.SURFACE
        fg = w.fg.copy()

        def solid(x, y):
            b = fg[WD.row(y), WD.col(x)]
            return b not in (B['air'], B['lava'], B['water'])

        def pace(y, b):
            if y >= WD.CAVE_CEIL:
                return 0.27
            if y >= 23:
                return 0.15
            if y >= 1:
                return 0.13
            if y >= -12:
                return 0.21
            if y >= -26:
                return 0.19
            if y >= -38:
                return 0.12
            return 0.24                     # by the city: every swing is heard

        def mine(x, y, dur, face):
            nonlocal t
            b = int(fg[WD.row(y), WD.col(x)])
            self.acts.append(('mine', t, t + dur, (x, y, face, b)))
            n = max(1, int(round(dur / 0.13)))
            for k in range(n):
                self.hits.append((t + dur * (k + 0.7) / n, x, y, b))
            t += dur
            fg[WD.row(y), WD.col(x)] = B['air']
            self.breaks.append((t, x, y, b))
            return b

        def fall():
            nonlocal t, feet
            y = feet - 1
            while not solid(0, y) and fg[WD.row(y), WD.col(0)] != B['lava']:
                y -= 1
            to = y + 1
            d = feet - to
            if d <= 0:
                return 0
            dur = 0.045 if d <= 1 else float(np.sqrt(2 * d / G))
            self.acts.append(('fall', t, t + dur, (feet, to)))
            t += dur
            feet = to
            if d > 3:
                hp = d - 3
                self.damage.append((t, hp))
                self.acts.append(('land', t, t + 0.22, (feet,)))
                t += 0.22
            return d

        def torch():
            nonlocal t
            self.acts.append(('torch', t, t + 0.24, (0, feet + 1)))
            t += 0.24
            self.torches.append((t - 0.08, 0, feet + 1))

        # standing on the grass, then straight down
        self.acts.append(('stand', 0.0, INTRO + 0.15, (feet,)))
        t = INTRO + 0.15
        torch_at = {40, 27, 8, -8, -31}
        while feet > WD.DIAMONDS[0][1] + 1:
            y = feet - 1
            if feet in torch_at:
                torch()
                torch_at.discard(feet)
            if solid(0, y):
                b = mine(0, y, pace(y, None), 'down')
                d = fall()
                if y == WD.CAVE_CEIL:
                    self.events['cave_land'] = t
                elif d >= 3 and y > 10:
                    self.events['mine_land'] = t
                elif d >= 3:
                    self.events['geode_land'] = t
                if feet == 0 or (feet == -1 and 'deepslate' not in self.events):
                    self.events.setdefault('deepslate', t)
            else:
                fall()
        # the diamonds
        self.events['diamonds'] = t
        n = 0
        for (x, y, face) in ((0, -54, 'down'), (1, -54, 'right'), (0, -55, 'down'), (-1, -55, 'left'),
                             (1, -55, 'right')):
            if face != 'down':
                self.acts.append(('turn', t, t + 0.08, (face,)))
                t += 0.08
            mine(x, y, 0.42, face)
            n += 1
            self.dings.append((t, n))
            t += 0.12
            if face == 'down':
                fall()
        self.acts.append(('cheer', t, t + 0.5, (feet,)))
        t += 0.5
        mine(0, -56, 0.26, 'down')
        fall()
        # the last block
        self.events['last'] = t
        mine(0, WD.LAST_FLOOR, 1.5, 'down')
        self.events['break_last'] = t
        y = feet - 1
        dur = float(np.sqrt(2 * 3.0 / G))
        self.acts.append(('fall', t, t + dur, (feet, WD.LAVA_TOP - 0.15)))
        t += dur
        feet = WD.LAVA_TOP - 0.15
        self.events['lava'] = t
        self.acts.append(('burn', t, t + 1.7, (feet,)))
        for k in range(5):
            self.damage.append((t + 0.1 + k * 0.32, 2 if k < 4 else 4))
        t += 1.7
        self.events['dead'] = t
        self.acts.append(('dead', t, t + 2.7, (feet,)))
        t += 2.7
        self.T = t
        self.feet_end = feet
        # the city hears him
        sensors = [(4, WD.CITY_Y + 1), (10, WD.CITY_Y + 1), (16, WD.CITY_Y + 1)]
        self.vibes = []                  # (t0, t1, from, to)
        heard = 0
        for (th, x, y, b) in self.hits:
            if -53 <= y <= -38 and heard < 7:
                s = sensors[heard % 3]
                dist = np.hypot(s[0] - x, s[1] - y)
                self.vibes.append((th, th + dist / 9.0, (x + 0.5, y + 0.5), (s[0] + 0.5, s[1] + 0.4)))
                heard += 1
        t_s = self.vibes[3][1] + 0.2 if len(self.vibes) > 3 else self.events['diamonds'] - 3
        self.events['shriek'] = t_s
        self.events['warden_rise'] = t_s + 0.9
        self.events['warden_roar'] = t_s + 2.6
        self.events['warden_walk'] = t_s + 3.4
        self.hp_at = self._hp
        # the camera glides down: his feet's path, smoothed over about a third of a second (no step per block)
        tg = np.arange(0.0, self.T + 1.0, 1.0 / 240)
        fy = np.array([self.feet_at(tt)[1] for tt in tg])
        sig = 0.16 * 240
        k = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
        k /= k.sum()
        pad = len(k) // 2
        fyp = np.concatenate([np.full(pad, fy[0]), fy, np.full(pad, fy[-1])])
        self._cam_t = tg
        self._cam_y = np.convolve(fyp, k, 'valid')
        # captions
        ev = self.events
        cap = self.captions
        cap.append((ev['cave_land'] - 0.75, ev['cave_land'] + 0.2, 'A CAVE!', 'yellow'))
        cap.append((ev['cave_land'], ev['cave_land'] + 0.9, 'OUCH', 'red'))
        cap.append((ev['cave_land'] + 0.9, ev['cave_land'] + 2.0, 'ZOMBIE!!', 'green'))
        if 'mine_land' in ev:
            cap.append((ev['mine_land'] - 0.3, ev['mine_land'] + 1.1, 'ABANDONED MINESHAFT', 'yellow'))
        cap.append((ev['deepslate'], ev['deepslate'] + 1.2, 'Y = 0: DEEPSLATE', 'white'))
        if 'geode_land' in ev:
            cap.append((ev['geode_land'] - 0.7, ev['geode_land'] + 1.0, 'AN AMETHYST GEODE!', 'purple'))
        cap.append((t_s - 1.6, t_s + 0.3, 'SHHH... ANCIENT CITY', 'cyan'))
        cap.append((t_s + 0.3, ev['warden_rise'] + 0.3, 'THE SHRIEKER HEARD HIM', 'cyan'))
        cap.append((ev['warden_rise'] + 0.3, ev['diamonds'] + 0.4, 'THE WARDEN!!!', 'red'))
        cap.append((ev['diamonds'] + 0.3, ev['last'], 'DIAMONDS!!', 'cyan'))
        cap.append((ev['last'] + 0.3, ev['break_last'], 'wait...', 'white'))
        cap.append((ev['lava'], ev['dead'], 'NOOOOO', 'red'))

    # -- state at a time ----------------------------------------------------------------------------------
    def act_at(self, t):
        for a in self.acts:
            if a[1] <= t < a[2]:
                return a
        return self.acts[-1] if t >= self.acts[-1][2] else self.acts[0]

    def feet_at(self, t):
        """Steve's feet (x, y)."""
        y = WD.SURFACE
        for (k, t0, t1, d) in self.acts:
            if k == 'fall':
                if t < t0:
                    break
                if t < t1:
                    f = (t - t0) / (t1 - t0)
                    return 0.5, d[0] + (d[1] - d[0]) * f * f
                y = d[1]
            if k in ('burn', 'dead') and t >= t0:
                sink = 0.6 * _ease((t - t0) / 1.2) if k == 'burn' else 0.6
                y = d[0] - sink
        return 0.5, y

    def fg_at(self, t):
        fg = self.w.fg.copy()
        for (tb, x, y, b) in self.breaks:
            if tb <= t:
                fg[WD.row(y), WD.col(x)] = B['air']
            else:
                break
        return fg

    def lights_at(self, t):
        L = {}
        for k, v in self.w.lights.items():
            kind = 'soul' if self.w.deco.get(k) in ('lantern', 'sensor', 'shrieker') else 'warm'
            L[k] = (v, kind)
        for (tp, x, y) in self.torches:
            if tp <= t:
                L[(x, y)] = (14, 'warm')
        # the shrieker and sensors flare when they fire
        if t >= self.events['shriek']:
            k = np.exp(-(t - self.events['shriek']) / 1.5)
            L[(11, WD.CITY_Y + 1)] = (2 + 10 * k, 'soul')
        return L

    def deco_at(self, t):
        d = dict(self.w.deco)
        for (tp, x, y) in self.torches:
            if tp <= t:
                d[(x, y)] = 'torch'
        return d

    def crack_at(self, t):
        a = self.act_at(t)
        if a[0] == 'mine':
            x, y, face, b = a[3]
            return (x, y), 10 * (t - a[1]) / (a[2] - a[1])
        return None

    def _hp(self, t):
        hp = 20
        for (td, d) in self.damage:
            if td <= t:
                hp -= d
        return max(0, hp)

    def diamonds_at(self, t):
        n = 0
        for (td, k) in self.dings:
            if td <= t:
                n = k
        if t >= self.events['lava'] + 0.4:
            return 0 if t >= self.events['dead'] else n
        return n

    def hurt_at(self, t):
        """0..1: how freshly he was hurt (the red flash)."""
        v = 0.0
        for (td, d) in self.damage:
            if td <= t < td + 0.45:
                v = max(v, 1 - (t - td) / 0.45)
        return v

    def y_display(self, t):
        return int(np.floor(self.feet_at(t)[1] + 1e-6))

    def camera(self, t):
        """(cx, cy, zoom, shake)."""
        x, fy = self.feet_at(t)
        # follow his feet with a little lag, keeping him in the top third so what's below him shows
        ev = self.events
        lag = 0.10
        _, fy_l = self.feet_at(max(0.0, t - lag))
        fy_c = float(np.interp(t, self._cam_t, self._cam_y))
        zoom = 1.0
        off = 1.6
        # the city: pull out to see him and the Warden together
        a = _ease((t - (ev['shriek'] - 1.2)) / 1.0) * (1 - _ease((t - (ev['last'] - 0.2)) / 1.0))
        zoom = zoom * (1 - 0.22 * a)
        off = off - 3.6 * a
        # the last block: closer, the lava in view
        b = _ease((t - ev['last']) / 1.2)
        zoom *= 1 + 0.18 * b
        off = off + 1.4 * b
        cx = 0.5 + 3.0 * a
        cy = fy_c - off
        if t < INTRO:
            # the opening: the whole slice from the sky to the bedrock, zooming in to Steve
            k = _ease(t / INTRO)
            k = k * k
            z0 = 1440 / (96 * 134.0)
            zoom = float(np.exp(np.log(z0) * (1 - k) + np.log(zoom) * k))
            cx = 0.5
            cy = 68 - (960 - 470) / (96 * z0)
            cy = cy * (1 - k) + (fy_c - off) * k
        shake = 0.0
        for (td, d) in self.damage:
            if td <= t < td + 0.35:
                shake = max(shake, 0.12 * d ** 0.5 * (1 - (t - td) / 0.35))
        if ev['warden_roar'] <= t < ev['warden_roar'] + 1.2:
            shake = max(shake, 0.18 * (1 - (t - ev['warden_roar']) / 1.2))
        if t >= ev['lava']:
            shake = max(shake, 0.06 * max(0.0, 1 - (t - ev['lava']) / 1.5))
        return cx, cy, zoom, shake

    def darkness(self, t):
        """The Warden's darkness effect: pulsing, from the shriek for a few seconds."""
        ts = self.events['shriek'] + 0.25
        if t < ts:
            return 0.0
        env = _ease((t - ts) / 0.4) * (1 - _ease((t - (self.events['diamonds'] + 0.2)) / 1.0))
        pulse = 0.5 + 0.5 * np.cos(2 * np.pi * (t - ts) / 1.6)
        return float(env * (0.25 + 0.35 * pulse))
