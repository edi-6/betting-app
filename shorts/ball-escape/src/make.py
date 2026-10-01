"""Can the ball escape all 30 rings? A ball bouncing inside 30 rotating neon rings, each with a gap: slip through a
gap and the ring shatters. Every bounce plays the next note of Beethoven's Fur Elise (public domain); each shattered
ring rings out; the last one ends in a burst and ESCAPED!

python make.py OUT_DIR   (writes OUT_DIR/ball-escape.mp4 and OUT_DIR/thumbnail.jpg; the timing tunes itself so the
                          escape lands between 38 and 50 seconds)
"""
import colorsys
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sift-flight', 'src'))
import audio as AU          # noqa: E402  the synth toolkit from the flight
import pixelfont as PF      # noqa: E402

W, H, FPS, SS = 1080, 1920, 60, 2
N = 30
R0, DR, RW, RB = 62.0, 14.6, 6.0, 11.0          # first ring's radius, spacing, ring width, ball radius (px)
C = np.array([540.0, 1010.0])                    # the rings' centre
G = 900.0                                        # gravity, px/s^2 (down the screen)
SUB = 10
FUR_ELISE = [76, 75, 76, 75, 76, 71, 74, 72, 69, 60, 64, 69, 71, 64, 68, 71, 72, 64, 76, 75, 76, 75, 76, 71, 74, 72,
             69, 60, 64, 69, 71, 64, 72, 71, 69]
PENTA = [69, 72, 74, 76, 79, 81, 84, 86, 88]


def ring_color(k):
    r, g, b = colorsys.hsv_to_rgb((k / N * 0.85 + 0.55) % 1.0, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(seed, growth, t_max=80.0):
    """The ball's path (one sample a frame), the bounces, and when each ring broke. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    th0 = np.arange(N) * 2.39996 + rng.uniform(0, 6.28)
    om = (0.5 + 0.28 * np.sin(1.7 * np.arange(N) + seed)) * np.where(np.arange(N) % 2, -1.0, 1.0)
    p = np.array([0.0, 0.0])
    a = rng.uniform(0, 2 * np.pi)
    v = np.array([np.cos(a), np.sin(a)]) * 560.0
    k = 0
    dt = 1.0 / (FPS * SUB)
    path, bounces, breaks = [], [], []
    t = 0.0
    done_at = None
    while t < t_max:
        for _ in range(SUB):
            v[1] += G * dt
            p = p + v * dt
            t += dt
            if k < N:
                R = R0 + k * DR
                lim = R - RW / 2 - RB
                d = np.hypot(*p)
                if d > lim:
                    gam = np.radians(min(24.0 + growth * t, 75.0))
                    diff = wrap(np.arctan2(p[1], p[0]) - (th0[k] + om[k] * t))
                    if abs(diff) < gam - RB / R:
                        breaks.append(t)
                        k += 1
                        if k == N:
                            done_at = t
                    else:
                        n = p / d
                        vn = v @ n
                        if vn > 0:
                            v = v - 2 * vn * n
                            # a little spin off the moving wall, and the speed kept lively
                            tang = np.array([-n[1], n[0]])
                            v = v + tang * om[k] * R * 0.08
                            s = np.hypot(*v)
                            lo, hi = 560.0 + 6.0 * k, 1300.0
                            v = v * (np.clip(s, lo, hi) / s)
                            bounces.append((t, float(p[0])))
                        p = n * lim
        path.append(p.copy())
        if done_at is not None and t > done_at + 2.6:
            break
    return dict(path=np.array(path), bounces=bounces, breaks=breaks, done=done_at, th0=th0, om=om, growth=growth,
                seed=seed)


def tune():
    best = None
    for growth in (0.6, 0.9, 1.2, 1.6, 2.1, 2.8):
        for seed in range(1, 9):
            s = simulate(seed, growth)
            if s['done'] is None:
                continue
            score = abs(s['done'] - 44.0)
            if best is None or score < best[0]:
                best = (score, s)
            if 38.0 <= s['done'] <= 50.0:
                return s
    return best[1]


def text(img, s, px, y, color=(255, 255, 255), grad=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    h, w = spr.shape[:2]
    x = int((img.shape[1] - w) / 2)
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def render(sim, out_dir):
    path, breaks, done = sim['path'], sim['breaks'], sim['done']
    th0, om, growth = sim['th0'], sim['om'], sim['growth']
    nfr = len(path)
    rng = np.random.default_rng(3)
    # the shards of each ring: spawned when it breaks
    shards = []
    for k, tb in enumerate(breaks):
        R = R0 + k * DR
        ang = rng.uniform(0, 2 * np.pi, 70)
        sp = rng.uniform(80, 420, 70)
        shards.append((tb, k, np.stack([np.cos(ang), np.sin(ang)], 1) * R,
                       np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None] + rng.normal(0, 60, (70, 2))))
    S = SS
    vid = os.path.join(out_dir, 'video.mp4')
    ff = AU_ff()
    proc = subprocess.Popen([ff, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                             str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt',
                             'yuv444p', vid], stdin=subprocess.PIPE)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (np.array([8.0, 8.0, 16.0]) + 26.0 * np.exp(-((xx - C[0]) ** 2 + (yy - C[1]) ** 2) / (2 * 520.0 ** 2))[..., None]
          * np.array([0.5, 0.4, 1.0]))
    thumb = None
    for f in range(nfr):
        t = f / FPS
        k_now = int(np.searchsorted(breaks, t, side='right'))
        sharp = Image.new('RGB', (W * S, H * S), (0, 0, 0))
        glow = Image.new('RGB', (W // 4, H // 4), (0, 0, 0))
        ds, dg = ImageDraw.Draw(sharp), ImageDraw.Draw(glow)
        gam = np.degrees(np.radians(min(24.0 + growth * t, 75.0)))
        for k in range(k_now, N):
            R = R0 + k * DR
            th = np.degrees(th0[k] + om[k] * t)
            a0, a1 = th + gam, th - gam + 360.0
            col = ring_color(k)
            for dr, img, sc, wd in ((ds, sharp, S, RW * S), (dg, glow, 0.25, max(2, int(RW * 0.25 * 2.2)))):
                cx, cy, rr = C[0] * sc, C[1] * sc, R * sc
                dr.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0, a1, fill=col, width=int(round(wd)))
        # the shards of broken rings
        for (tb, k, p0, v0) in shards:
            a = t - tb
            if 0 <= a < 1.3:
                q = C + p0 + v0 * a + np.array([0.0, 0.5 * G * 0.6 * a * a])
                fade = 1.0 - a / 1.3
                col = tuple(int(c * fade) for c in ring_color(k))
                for x, y in q:
                    ds.rectangle([(x - 3) * S, (y - 3) * S, (x + 3) * S, (y + 3) * S], fill=col)
                    dg.rectangle([(x - 3) / 4, (y - 3) / 4, (x + 3) / 4, (y + 3) / 4], fill=col)
        # the ball and its trail (off the screen once it has escaped far enough)
        tc = ring_color(min(k_now, N - 1))
        for j in range(14, -1, -1):
            if f - j < 0:
                continue
            q = C + path[f - j]
            r = RB * (1.0 - j / 18.0)
            fade = (1.0 - j / 15.0) ** 1.5
            col = (255, 255, 255) if j == 0 else tuple(int(c * fade * 0.8) for c in tc)
            ds.ellipse([(q[0] - r) * S, (q[1] - r) * S, (q[0] + r) * S, (q[1] + r) * S], fill=col)
            dg.ellipse([(q[0] - r * 1.6) / 4, (q[1] - r * 1.6) / 4, (q[0] + r * 1.6) / 4, (q[1] + r * 1.6) / 4],
                       fill=col)
        img = np.asarray(sharp.resize((W, H), Image.LANCZOS), np.float32)
        g = np.asarray(glow.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32)
        img = np.clip(bg + img + 1.6 * g, 0, 255)
        # the counter, and the ending
        left = N - k_now
        if done is None or t < done:
            text(img, 'RINGS LEFT', 7, 150)
            text(img, str(left), 22, 230, grad=((255, 255, 255), (170, 210, 255)))
        else:
            u = min(1.0, (t - done) / 0.25)
            text(img, 'ESCAPED!', int(10 + 8 * u), 300, grad=((255, 240, 140), (255, 150, 30)))
        frame = img.astype(np.uint8)
        if f == 0:
            thumb = frame.copy()
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    th = thumb.astype(np.float32)
    text(th, 'CAN IT ESCAPE', 11, 1580, grad=((255, 255, 255), (200, 220, 255)))
    text(th, 'ALL 30 RINGS', 11, 1700, grad=((255, 240, 140), (255, 150, 30)))
    Image.fromarray(th.astype(np.uint8)).save(os.path.join(out_dir, 'thumbnail.jpg'), quality=94)
    return vid, nfr / FPS


def AU_ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def soundtrack(sim, dur, path):
    rng = np.random.default_rng(5)
    mix = AU.Mix(dur + 3.0)
    last = -1.0
    n = 0
    for (t, x) in sim['bounces']:
        if t - last < 0.06 or (sim['done'] and t > sim['done']):
            continue
        last = t
        m = FUR_ELISE[n % len(FUR_ELISE)]
        n += 1
        pan = float(np.clip(x / 500.0, -0.8, 0.8))
        mix.add(AU.pluck(m, 0.35, rng, bright=1.0, decay=0.55), t, 0.55, pan)
        mix.add(AU.bell(m + 12, rng, dur=1.4, bright=0.6), t, 0.16, pan)
    for k, t in enumerate(sim['breaks']):
        m = PENTA[k % len(PENTA)]
        crack = AU.hp(rng.standard_normal(int(0.16 * AU.SR)), 1800, 2) * AU.expenv(int(0.16 * AU.SR), 0.03)
        mix.add(crack, t, 0.30, 0.0)
        mix.add(AU.bell(m + 12, rng, dur=1.8, bright=1.0), t, 0.30, rng.uniform(-0.5, 0.5))
    if sim['done']:
        t = sim['done']
        mix.add(AU.impact(rng, 1.0), t, 0.75, 0.0)
        L, R = AU.supersaw([AU.midi_hz(m) for m in (57, 60, 64, 69, 72, 76)], 2.0, rng, attack=0.01, release=1.2,
                           cutoff=5000)
        mix.add2(L, R, t, 0.5)
        L, R = AU.choir((57, 64, 69, 72), 2.0, rng, attack=0.1, release=1.0)
        mix.add2(L, R, t, 0.35)
    # a soft pad in A minor under it all
    L, R = AU.choir((45, 57, 60, 64), dur, rng, attack=1.5, release=1.0, vowel='o')
    mix.add2(L * 0.25, R * 0.25, 0.0)
    nS = int(dur * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=2.2, wet=0.28)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)
    print(f'[audio] {AU.integrated_lufs(L, R):.1f} LUFS, {n} notes', flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sim = tune()
    print(f"seed {sim['seed']} growth {sim['growth']}: escaped at {sim['done']:.1f}s, {len(sim['bounces'])} bounces, "
          f"rings broke at " + ' '.join(f'{t:.1f}' for t in sim['breaks']), flush=True)
    vid, dur = render(sim, out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(sim, dur, wav)
    final = os.path.join(out, 'ball-escape.mp4')
    subprocess.run([AU_ff(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-r', str(FPS),
                    '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', final], check=True)
    print('done', final, f'{dur:.1f}s', flush=True)


if __name__ == '__main__':
    main()
