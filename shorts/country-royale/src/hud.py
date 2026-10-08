"""The HUD: the title, the live counter, the names of the eliminated as they go out, the TOP 10 / TOP 5 / FINAL
banners, the head-to-head for the final, and the winner's celebration (the crown, rays, confetti)."""
import math

import cv2
import numpy as np

import render as RD
import text as TX

WHITE = (255, 255, 255)
YELLOW = (255, 222, 60)
GOLD_GRAD = ((255, 244, 150), (255, 170, 20))
RED = (255, 70, 70)

TITLE_Y = 178
COUNTER_Y = 274


def pop_scale(dt, dur=0.22, amp=0.35):
    if dt < 0:
        return 0.0
    if dt > dur:
        return 1.0
    x = dt / dur
    return 1 + amp * math.sin(math.pi * x) * (1 - x) + 0 * x


class HUD:
    def __init__(self, story, renderer, tl):
        self.st = story
        self.r = renderer
        self.tl = tl
        st = story
        # milestones: the moment the count drops to 10, 5, 3, 2
        te = sorted(t for t, i in st.elims)
        n = len(st.order)
        self.milestones = []
        for k, (label, sub) in {10: ('TOP 10', 'SPEED UP!'), 5: ('TOP 5', None), 3: ('FINAL 3', None),
                                2: ('FINAL 2', '1 VS 1')}.items():
            idx = n - k - 1
            if 0 <= idx < len(te):
                self.milestones.append((te[idx], label, sub))
        self.lift = self._stack_labels()

    def _stack_labels(self):
        """Labels that pop up together near each other are stacked upwards, each keeping its place in the stack
        for as long as it is up."""
        st = self.st
        cam = {'c': np.zeros(2), 'zoom': 1.0}
        lift = {}
        for k, (t0, i) in enumerate(st.elims):
            p, rp, _ = self.r.out_track(i, t0, cam)
            level = 0
            while True:
                clash = False
                for t1, j in st.elims[max(0, k - 8):k]:
                    if t0 - t1 > RD.OUT_T * 0.8:
                        continue
                    q, rq, _ = self.r.out_track(j, t0, cam)
                    dy = (p[1] - rp - 46 * level) - (q[1] - rq - 46 * lift[j])
                    if abs(q[0] - p[0]) < 250 and abs(dy) < 44:
                        clash = True
                        break
                if not clash or level > 4:
                    break
                level += 1
            lift[i] = level
        return lift

    # ------------------------------------------------------------------------------------------------------------
    def title(self, frame, tv, ts, tc):
        st = self.st
        if tc is not None and tc > 0.75:
            name = st.names[st.winner]
            spr = TX.text(f'{name} WINS!', 92 if len(name) < 9 else 76, grad=GOLD_GRAD, stroke=8, shadow=6)
            TX.over(frame, spr, RD.W / 2, TITLE_Y + 8, scale=pop_scale(tc - 0.75, 0.3, 0.45))
            return
        if tv < 2.2:
            a = 1.0 if tv < 1.95 else (2.2 - tv) / 0.25
            spr = TX.hstack(TX.text('FIND YOUR COUNTRY', 70, grad=((255, 240, 120), (255, 196, 30)), stroke=7,
                                    shadow=5), TX.emoji('\U0001F50E', 70), gap=14)
            sc = 1.0 + 0.04 * math.sin(tv * 7.0)
            TX.over(frame, spr, RD.W / 2, TITLE_Y, opacity=a, scale=sc)
        else:
            a = min(1.0, (tv - 2.2) / 0.2)
            spr = TX.hstack(TX.text('LAST ONE INSIDE WINS', 60, stroke=7, shadow=5), TX.emoji('\U0001F3C6', 58),
                            gap=12)
            TX.over(frame, spr, RD.W / 2, TITLE_Y, opacity=a, scale=pop_scale(tv - 2.2, 0.25, 0.35))

    def counter(self, frame, tv, ts, tc):
        st = self.st
        n = st.n_alive(ts)
        if tc is not None and tc > 0.75:
            return
        t2 = st.elims[-2][0]                       # when it came down to two
        if ts >= t2 + 0.9:
            pair = sorted([st.winner, st.elims[-1][1]], key=lambda i: st.names[i])
            a = min(1.0, (ts - t2 - 0.9) / 0.25)
            sprs = []
            for k, i in enumerate(pair):
                tile = self.r.A.tiles[st.codes[i]]
                tile = cv2.resize(tile, (int(tile.shape[1] * 1.15), int(tile.shape[0] * 1.15)),
                                  interpolation=cv2.INTER_LINEAR)
                nm = TX.text(st.names[i], 46, stroke=6, font=TX.NARROW)
                sprs += [tile, nm] if k == 0 else [nm, tile]
                if k == 0:
                    sprs.append(TX.text('VS', 52, fill=RED, stroke=6))
            row = TX.hstack(*sprs, gap=14)
            TX.over(frame, row, RD.W / 2, COUNTER_Y, opacity=a, scale=min(1.0, 980 / row.shape[1]))
            return
        # time of the last change of the count, for the pop
        te = [t for t, i in st.elims if t <= ts]
        dt = ts - te[-1] if te else 10.0
        num = TX.text(str(n), 96, grad=((255, 246, 140), (255, 190, 30)), stroke=8, shadow=6)
        left = TX.text('LEFT', 60, stroke=7, shadow=5)
        row = TX.hstack(num, left, gap=16)
        TX.over(frame, row, RD.W / 2, COUNTER_Y, scale=pop_scale(dt, 0.2, 0.3))

    def names(self, frame, ts, cam):
        """Each eliminated country's name and place rides along above its ball on the way to the board, then
        floats off its tile."""
        st = self.st
        for t0, i in st.elims:
            dt = ts - t0
            if dt < 0:
                break
            if dt > RD.OUT_T * 0.7:
                continue
            xy, rpx, u = self.r.out_track(i, ts, cam)
            y = xy[1] - rpx - 26 - 46 * self.lift.get(i, 0)
            a = 1.0 if u < 0.5 else max(0.0, 1 - (u - 0.5) / 0.2)
            nm = TX.text(st.names[i], 36, stroke=6, font=TX.NARROW, shadow=4)
            rk = TX.text(f'#{st.place[i]}', 30, fill=(255, 92, 92), stroke=5, font=TX.BOLD)
            row = TX.hstack(rk, nm, gap=8)
            x = float(np.clip(xy[0], row.shape[1] / 2 + 12, RD.W - row.shape[1] / 2 - 12))
            TX.over(frame, row, x, y, opacity=a, scale=pop_scale(dt, 0.18, 0.4) * 0.92)

    def banners(self, frame, ts):
        for t0, label, sub in self.milestones:
            dt = ts - t0
            if dt < 0 or dt > 1.35:
                continue
            a = 1.0 if dt < 1.05 else (1.35 - dt) / 0.3
            y = RD.RC[1] - 40
            # a dark band behind the words
            band = np.clip(1 - np.abs(np.arange(RD.H) - y) / 120.0, 0, 1) ** 1.5 * 0.55 * a
            frame *= (1 - band)[:, None, None]
            spr = TX.text(label + '!' if label.startswith('TOP') else label, 128, grad=GOLD_GRAD, stroke=10,
                          shadow=8)
            TX.over(frame, spr, RD.W / 2, y, opacity=a, scale=pop_scale(dt, 0.3, 0.5))
            if sub:
                s2 = TX.text(sub, 62, fill=(255, 255, 255), stroke=7, shadow=5)
                TX.over(frame, s2, RD.W / 2, y + 108, opacity=a * min(1.0, max(0.0, (dt - 0.15) / 0.15)),
                        scale=pop_scale(dt - 0.15, 0.25, 0.4))

    # ------------------------------------------------------------------------------------------------------------
    def celebration_back(self, frame, tc, center, rpx):
        """Light rays turning behind the winner."""
        if tc is None or tc < 0.5:
            return
        a = min(1.0, (tc - 0.5) / 0.6)
        ov = np.zeros((RD.H // 2, RD.W // 2), np.uint8)
        c = (int(center[0] / 2), int(center[1] / 2))
        for k in range(14):
            ang = tc * 0.35 + k * 2 * math.pi / 14
            pts = np.array([c, (c[0] + 1400 * math.cos(ang - 0.09), c[1] + 1400 * math.sin(ang - 0.09)),
                            (c[0] + 1400 * math.cos(ang + 0.09), c[1] + 1400 * math.sin(ang + 0.09))], np.int32)
            cv2.fillConvexPoly(ov, pts, 255, cv2.LINE_AA)
        ov = cv2.GaussianBlur(ov.astype(np.float32) / 255, (0, 0), 6)
        ov = cv2.resize(ov, (RD.W, RD.H), interpolation=cv2.INTER_LINEAR)
        yy, xx = np.mgrid[0:RD.H, 0:RD.W]
        fall = np.clip(1 - np.hypot(xx - center[0], yy - center[1]) / 900.0, 0, 1)
        frame += (ov * fall * 0.22 * a)[..., None] * np.array([1.0, 0.82, 0.35])

    def crown(self, frame, tc, center, rpx):
        if tc is None or tc < 1.25:
            return
        k = RD.back_out((tc - 1.25) / 0.45, 1.6)
        top = center[1] - rpx * 0.92
        y = top - 120 * (1 - k) - rpx * 0.38
        spr = TX.emoji('\U0001F451', int(rpx * 0.95))
        TX.over(frame, spr, center[0] + rpx * 0.06, y, angle=-8 * (1 - k) - 6)

    def confetti(self, frame, tc, palette):
        if tc is None or tc < 0.35:
            return
        dt = tc - 0.35
        cols = [np.array(c) for c in palette] + [np.array([1.0, 0.84, 0.25]), np.array([1.0, 1.0, 1.0])]
        for k in range(220):
            h = RD._hash(k, 11)
            x0 = RD.W * RD._hash(k, 12)
            y0 = -40 - 900 * RD._hash(k, 13)
            vy = 330 + 300 * RD._hash(k, 14)
            y = y0 + vy * dt
            if y < -30 or y > RD.H + 30:
                continue
            x = x0 + 40 * math.sin(dt * (2 + 3 * h) + 6.28 * h)
            ang = dt * (3 + 6 * RD._hash(k, 15)) + h * 6
            w, hh = 13 + 6 * h, 7 + 3 * h
            flip = abs(math.cos(dt * (4 + 5 * h) + h * 3))
            ca, sa = math.cos(ang), math.sin(ang)
            pts = np.array([(-w / 2, -hh / 2 * flip), (w / 2, -hh / 2 * flip), (w / 2, hh / 2 * flip),
                            (-w / 2, hh / 2 * flip)])
            pts = np.stack([x + pts[:, 0] * ca - pts[:, 1] * sa, y + pts[:, 0] * sa + pts[:, 1] * ca], 1)
            col = cols[k % len(cols)] * (0.75 + 0.25 * flip)
            cv2.fillConvexPoly(frame, np.round(pts * 16).astype(np.int32), tuple(float(v) for v in col),
                               cv2.LINE_AA, 4)

    def comment_cta(self, frame, tc):
        if tc is None or tc < 2.4:
            return
        dt = tc - 2.4
        row = TX.hstack(TX.text('COMMENT YOUR COUNTRY', 54, stroke=7, shadow=5), TX.emoji('\U0001F447', 56),
                        gap=10)
        bob = 6 * math.sin(dt * 5)
        TX.over(frame, row, RD.W / 2, RD.BOARD_Y0 - 48 + bob, scale=pop_scale(dt, 0.25, 0.4))
