"""1000 balls... watch the shape. A Galton board: a thousand balls stream down through 14 rows of pegs, bouncing left
or right at each one, and pile up in 15 bins - into a bell curve. Every peg hit plays a note; at the end the true curve
draws itself over the pile.  python make.py OUT_DIR
"""
import colorsys
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sift-flight', 'src'))
import audio as AU          # noqa: E402
import pixelfont as PF      # noqa: E402

W, H, FPS = 1080, 1920, 60
ROWS, NBALL = 14, 1000
NB = ROWS + 1
DX, DY = 64.0, 52.0
TOP = 360.0
BIN_TOP = TOP + ROWS * DY + 70.0
BIN_W = W / NB
BALL = 9.0
PER_ROW = 6
RELEASE = 0.038
HOP = 0.11
T_REVEAL_PAD, T_TAIL = 0.8, 4.5
PENTA = [0, 2, 4, 7, 9]


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def text(img, s, px, y, color=(255, 255, 255), grad=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    h, w = spr.shape[:2]
    x = (img.shape[1] - w) // 2
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def peg_xy(r, k):
    return W / 2 + (k - r / 2.0) * DX, TOP + r * DY


def plan(seed=7):
    rng = np.random.default_rng(seed)
    balls = []
    counts = np.zeros(NB, int)
    for i in range(NBALL):
        t0 = i * RELEASE
        steps = rng.random(ROWS) < 0.5
        k = np.concatenate([[0], np.cumsum(steps)])
        b = int(k[-1])
        slot = counts[b]
        counts[b] += 1
        col, row = slot % PER_ROW, slot // PER_ROW
        bx = b * BIN_W + (BIN_W - PER_ROW * (BALL * 2 + 1)) / 2 + BALL + col * (BALL * 2 + 1)
        by = H - 60 - BALL - row * (BALL * 2 + 1)
        t_last = t0 + ROWS * HOP
        fall = math.sqrt(max(by - (TOP + ROWS * DY), 1.0) / 1800.0)
        balls.append(dict(t0=t0, k=k, bin=b, land=(bx, by), t_last=t_last, t_land=t_last + fall, fall=fall))
    return balls, counts


def pos(b, t):
    """Where ball b is at time t (None before it starts); it hops from peg to peg, then drops into its bin."""
    a = t - b['t0']
    if a < 0:
        return None
    if a < ROWS * HOP:
        r = int(a // HOP)
        u = a / HOP - r
        x0, y0 = peg_xy(r, b['k'][r])
        x1, y1 = peg_xy(r + 1, b['k'][r + 1])
        return x0 + (x1 - x0) * u, y0 - DY * 0.42 + (y1 - y0) * u - 26.0 * 4 * u * (1 - u)
    x0, y0 = peg_xy(ROWS, b['k'][ROWS])
    u = min(1.0, (t - b['t_last']) / b['fall'])
    bx, by = b['land']
    return x0 + (bx - x0) * u, y0 - DY * 0.42 + (by - y0 + DY * 0.42) * u * u


def render(balls, counts, out):
    t_end = balls[-1]['t_land'] + T_REVEAL_PAD
    nf = int((t_end + T_TAIL) * FPS)
    hue = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(0.92 * i / (NB - 1), 0.72, 1.0)) for i in range(NB)]
    base = Image.new('RGB', (W, H), (12, 12, 22))
    d = ImageDraw.Draw(base)
    for r in range(ROWS + 1):
        for k in range(r + 1):
            x, y = peg_xy(r, k)
            y -= DY * 0.42 - 12
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(150, 150, 180))
    for b in range(NB + 1):
        x = int(b * BIN_W)
        d.rectangle([x - 2, BIN_TOP, x + 2, H - 50], fill=(70, 70, 96))
    d.rectangle([0, H - 52, W, H - 46], fill=(70, 70, 96))
    landed = base.copy()
    ld = ImageDraw.Draw(landed)
    order = sorted(range(len(balls)), key=lambda i: balls[i]['t_land'])
    nxt = 0
    # the true curve, scaled to the pile
    xs = np.linspace(0, W, 400)
    mu, sd = W / 2, math.sqrt(ROWS * 0.25) * BIN_W
    peak = NBALL * BIN_W / (sd * math.sqrt(2 * math.pi))
    rows_h = (BALL * 2 + 1) / PER_ROW
    ys = (H - 60) - peak * np.exp(-0.5 * ((xs - mu) / sd) ** 2) * rows_h
    vid = os.path.join(out, 'video.mp4')
    p = subprocess.Popen([ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                          str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p',
                          vid], stdin=subprocess.PIPE)
    for f in range(nf):
        t = f / FPS
        while nxt < len(order) and balls[order[nxt]]['t_land'] <= t:
            b = balls[order[nxt]]
            x, y = b['land']
            ld.ellipse([x - BALL, y - BALL, x + BALL, y + BALL], fill=hue[b['bin']])
            nxt += 1
        frame = landed.copy()
        fd = ImageDraw.Draw(frame)
        glow = Image.new('RGB', (W // 4, H // 4), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for b in balls:
            if b['t0'] > t:
                break
            if b['t_land'] <= t:
                continue
            q = pos(b, t)
            if q is None:
                continue
            x, y = q
            fd.ellipse([x - BALL, y - BALL, x + BALL, y + BALL], fill=(255, 255, 255))
            gd.ellipse([(x - 16) / 4, (y - 16) / 4, (x + 16) / 4, (y + 16) / 4], fill=hue[b['bin']])
        img = np.asarray(frame, np.float32) + 1.4 * np.asarray(
            glow.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), np.float32)
        n_in = sum(1 for b in balls if b['t0'] <= t)
        if t < t_end:
            text(img, 'WATCH THE SHAPE', 8, 90)
            text(img, f'{n_in} BALLS', 10, 190)
        else:
            u = min(1.0, (t - t_end) / 1.2)
            pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pd = ImageDraw.Draw(pil)
            m = max(2, int(len(xs) * u))
            pts = list(zip(xs[:m], ys[:m]))
            pd.line(pts, fill=(255, 255, 255), width=8, joint='curve')
            img = np.asarray(pil, np.float32).copy()
            text(img, 'THE BELL CURVE', 10, 90, grad=((255, 240, 140), (255, 150, 30)))
            text(img, 'EVERY SINGLE TIME', 6, 200)
        fr = np.clip(img, 0, 255).astype(np.uint8)
        if f == 0:
            Image.fromarray(fr).save(os.path.join(out, 'thumbnail.jpg'), quality=94)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return vid, nf / FPS, t_end


def soundtrack(balls, dur, t_end, path):
    rng = np.random.default_rng(5)
    mix = AU.Mix(dur + 3.0)
    hits = sorted((b['t0'] + r * HOP, int(b['k'][r]) - r / 2.0) for b in balls for r in range(1, ROWS + 1))
    last = -1.0
    for t, off in hits:
        if t - last < 0.035:
            continue
        last = t
        step = int(np.clip(round(off * 1.5) + 10, 0, 24))
        m = 60 + 12 * (step // 5) + PENTA[step % 5] - 12
        mix.add(AU.pluck(m, 0.12, rng, bright=1.0, decay=0.18), t, 0.22, float(np.clip(off / 7.0, -0.8, 0.8)))
    beat, t, b = 60.0 / 120.0, 0.0, 0
    chords = [((60, 64, 67), 36), ((57, 60, 64), 33), ((53, 57, 60), 29), ((55, 59, 62), 31)]
    while t < t_end:
        ch, root = chords[(b // 4) % 4]
        mix.add(AU.kick(b % 2), t, 0.45)
        mix.add(AU.subbass(root, beat * 0.4), t + beat / 2, 0.30)
        if b % 4 == 0:
            L, R = AU.supersaw([AU.midi_hz(m) for m in ch], beat * 4, rng, attack=0.2, release=0.4, cutoff=1800)
            mix.add2(L * 0.25, R * 0.25, t)
        b += 1
        t += beat
    mix.add(AU.riser(rng, 2.5), t_end - 2.5, 0.2)
    mix.add(AU.impact(rng, 0.9), t_end, 0.6)
    L, R = AU.choir((60, 64, 67, 72), T_TAIL, rng, attack=0.3, release=1.0)
    mix.add2(L * 0.45, R * 0.45, t_end)
    for k, m in enumerate((72, 76, 79, 84, 88)):
        mix.add(AU.bell(m, rng, dur=2.0), t_end + 0.25 * k, 0.22, -0.6 + 0.3 * k)
    nS = int(dur * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=1.8, wet=0.22)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    balls, counts = plan()
    print('bins', counts.tolist(), flush=True)
    vid, dur, t_end = render(balls, counts, out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(balls, dur, t_end, wav)
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                    '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', os.path.join(out, 'bell-curve.mp4')], check=True)
    print(f'done {dur:.1f}s (curve at {t_end:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
