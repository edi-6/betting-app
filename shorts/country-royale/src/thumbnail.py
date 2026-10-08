"""The 1080x1920 cover: all 50 flag balls packed in the glowing ring, one slipping out through the gap, titled
WHICH COUNTRY WINS? / FIND YOUR COUNTRY. --no-title for the same picture without the text.

python thumbnail.py OUT.jpg [--no-title]
"""
import argparse
import math

import numpy as np
from PIL import Image

import flags as FL
import render as RD
import text as TX
from main import Show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    args = ap.parse_args()
    show = Show()
    R = show.r
    rng = np.random.default_rng(5)
    frame = R.bg.copy()
    col = RD.hsv(0.52, 0.72, 1.0)
    frame += (np.clip(1.05 - R.bg_d, 0, 1) ** 2 * 0.08)[..., None] * col
    glow = np.zeros((RD.H // 4, RD.W // 4, 3), np.float32)
    cam = {'c': np.zeros(2), 'zoom': 1.0}
    gap_ang = math.radians(58)
    # the ring, drawn straight from the renderer with the gap low on the right
    st = show.st
    d = st.d
    k0 = 0
    saved = (d['ring_angle'][k0], d['gap'][k0])
    d['ring_angle'][k0], d['gap'][k0] = gap_ang, math.radians(26)
    d['ring_angle'][k0 + 1], d['gap'][k0 + 1] = gap_ang, math.radians(26)
    R.ring_color = lambda ts: col
    R.draw_ring(frame, glow, 0.0, cam)
    # the balls on a jittered hexagonal grid filling the ring; one on its way out of the gap
    r = 0.105
    dd = 2 * r * 1.06
    spots = []
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = (i + 0.5 * (j % 2)) * dd + rng.uniform(-0.012, 0.012)
            y = j * dd * math.sqrt(3) / 2 + rng.uniform(-0.012, 0.012)
            if math.hypot(x, y) < 1 - r * 1.15:
                spots.append((x, y))
    spots.sort(key=lambda p: math.hypot(p[0], p[1]))
    codes = [c[0] for c in FL.COUNTRIES]
    rng.shuffle(codes)
    out_code = 'BR'
    codes.remove(out_code)
    for (x, y), code in zip(spots[:len(codes)], codes):
        look = (rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.5))
        p = R.to_screen((x, y), cam)
        R.draw_ball(frame, code, p, r * RD.S, look=look, blink=1.0, tilt=rng.uniform(-0.15, 0.15))
    p = R.to_screen((1.06 * math.cos(gap_ang), 1.06 * math.sin(gap_ang)), cam)
    R.draw_ball(frame, out_code, p, r * RD.S, look=(0.6, 0.5), mood='shock', tilt=0.2)
    R.finish_glow(frame, glow)
    if not args.no_title:
        band = np.zeros(RD.H)
        band[60:420] = 1.0
        band[1340:1700] = 1.0
        band = np.convolve(band, np.ones(140) / 140, 'same')
        frame *= (1.0 - 0.5 * band)[:, None, None]
        t1 = TX.text('WHICH COUNTRY', 104, stroke=10, shadow=8)
        t2 = TX.hstack(TX.text('WINS?', 150, grad=((255, 244, 150), (255, 170, 20)), stroke=12, shadow=8),
                       TX.emoji('\U0001F3C6', 140), gap=18)
        TX.over(frame, t1, RD.W / 2, 175, scale=min(1.0, 1000 / t1.shape[1]))
        TX.over(frame, t2, RD.W / 2, 315)
        t3 = TX.hstack(TX.text('FIND YOUR COUNTRY', 72, grad=((255, 240, 120), (255, 196, 30)), stroke=8, shadow=6),
                       TX.emoji('\U0001F50E', 72), gap=12)
        TX.over(frame, t3, RD.W / 2, 1430, scale=min(1.0, 1020 / t3.shape[1]))
        t4 = TX.text('LAST ONE INSIDE WINS', 58, stroke=7, shadow=5)
        TX.over(frame, t4, RD.W / 2, 1540)
    d['ring_angle'][k0], d['gap'][k0] = saved
    Image.fromarray(RD.to8(frame)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
