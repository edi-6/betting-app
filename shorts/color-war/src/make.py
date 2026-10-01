"""Which color wins? Four colors split a board; each has two balls that bounce through enemy tiles and flip them to
their own color. 45 seconds, then the color holding the most of the board wins. Every flipped tile plays a note (its
column picks the pitch) over a beat; the battle is chosen from 40 seeds for the most lead changes and the closest
finish.

python make.py OUT_DIR   (writes OUT_DIR/color-war.mp4 and OUT_DIR/thumbnail.jpg)
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sift-flight', 'src'))
import audio as AU          # noqa: E402  the synth toolkit from the flight
import pixelfont as PF      # noqa: E402

W, H, FPS = 1080, 1920, 60
NT, TS = 45, 24                       # tiles a side, tile size (px)
BX, BY = 0, 430                       # the board's top-left corner
T_END, T_TAIL = 45.0, 3.5
SUB = 6
RB, SPEED = 10.0, 430.0
TEAMS = [('RED', (240, 66, 56)), ('BLUE', (56, 128, 240)), ('GREEN', (64, 205, 96)), ('YELLOW', (248, 204, 44))]
PENTA = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84]


def simulate(seed):
    rng = np.random.default_rng(seed)
    own = np.zeros((NT, NT), np.int8)
    h = NT // 2
    own[h:, :h], own[:h, h:], own[h:, h:] = 1, 2, 3            # x right, y down: TL red, TR blue, BL green, BR yellow
    balls = []
    for team, (cx, cy) in enumerate(((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))):
        for _ in range(2):
            a = rng.uniform(0, 2 * np.pi)
            balls.append([cx * NT * TS + rng.uniform(-60, 60), cy * NT * TS + rng.uniform(-60, 60),
                          np.cos(a) * SPEED, np.sin(a) * SPEED, team])
    dt = 1.0 / (FPS * SUB)
    frames, pos, flips = [], [], []
    size = NT * TS
    for f in range(int((T_END + T_TAIL) * FPS)):
        t = f / FPS
        if t < T_END:
            for _ in range(SUB):
                for b in balls:
                    for ax in (0, 1):
                        b[ax] += b[2 + ax] * dt
                        s = np.sign(b[2 + ax])
                        px = b[0] + (RB * s if ax == 0 else 0.0)
                        py = b[1] + (RB * s if ax == 1 else 0.0)
                        if not (0 <= px < size and 0 <= py < size):
                            b[2 + ax] *= -1
                            b[ax] = min(max(b[ax], RB), size - RB)
                            continue
                        i, j = int(py // TS), int(px // TS)
                        if own[i, j] != b[4]:
                            own[i, j] = b[4]
                            flips.append((t, b[4], j))
                            b[2 + ax] *= -1
                            b[ax] -= b[2 + ax] * dt * 0 + (-s) * 0.5
                            a = np.arctan2(b[3], b[2]) + rng.uniform(-0.05, 0.05)
                            b[2], b[3] = np.cos(a) * SPEED, np.sin(a) * SPEED
        frames.append(own.copy())
        pos.append([(b[0], b[1], b[4]) for b in balls])
    counts = np.array([[np.sum(fr == k) for k in range(4)] for fr in frames[::30]])
    leaders = counts.argmax(1)
    changes = int(np.sum(leaders[1:] != leaders[:-1]))
    final = counts[min(len(counts) - 1, int(T_END * 2))]
    srt = np.sort(final)[::-1] / NT ** 2
    return dict(seed=seed, frames=frames, pos=pos, flips=flips, changes=changes, margin=float(srt[0] - srt[1]),
                winner=int(final.argmax()), final=final)


def pick():
    best = None
    for seed in range(1, 41):
        s = simulate(seed)
        score = 2.0 * min(s['changes'], 8) - 120.0 * abs(s['margin'] - 0.02)
        if best is None or score > best[0]:
            best = (score, s)
    return best[1]


def blit(img, spr, x, y):
    h, w = spr.shape[:2]
    x, y = int(x), int(y)
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def text(img, s, px, y, color=(255, 255, 255), grad=None, x=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    blit(img, spr, (img.shape[1] - spr.shape[1]) / 2 if x is None else x, y)
    return spr.shape


def render(sim, out):
    tile_col = np.array([c for _, c in TEAMS], np.float32) * 0.62
    ball_col = [tuple(min(255, int(v * 1.15 + 30)) for v in c) for _, c in TEAMS]
    grid = np.ones((NT * TS, NT * TS, 1), np.float32)
    for k in range(NT + 1):
        grid[min(k * TS, NT * TS - 1), :, 0] = 0.78
        grid[:, min(k * TS, NT * TS - 1), 0] = 0.78
    flash = np.zeros((NT, NT), np.float32)
    rng = np.random.default_rng(1)
    conf = np.column_stack([rng.uniform(0, W, 220), rng.uniform(-900, 0, 220), rng.uniform(250, 520, 220),
                            rng.uniform(-60, 60, 220)])
    has_q = '?' in getattr(PF, 'GLYPHS', {}) or True
    q = 'WHICH COLOR WINS?'
    try:
        PF.render(q, px=2)
    except Exception:
        q = 'WHICH COLOR WINS'
    ff = ffmpeg()
    vid = os.path.join(out, 'video.mp4')
    proc = subprocess.Popen([ff, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                             str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt',
                             'yuv444p', vid], stdin=subprocess.PIPE)
    prev = sim['frames'][0]
    thumb = None
    winner = sim['winner']
    for f, own in enumerate(sim['frames']):
        t = f / FPS
        flash *= 0.80
        flash[own != prev] = 1.0
        prev = own
        img = np.empty((H, W, 3), np.float32)
        img[:] = (14, 14, 22)
        board = tile_col[own] + flash[..., None] * 110.0
        board = np.repeat(np.repeat(board, TS, 0), TS, 1) * grid
        img[BY:BY + NT * TS, BX:BX + NT * TS] = board
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        for (x, y, k) in sim['pos'][f]:
            X, Y = BX + x, BY + y
            d.ellipse([X - RB - 4, Y - RB - 4, X + RB + 4, Y + RB + 4], fill=(255, 255, 255))
            d.ellipse([X - RB, Y - RB, X + RB, Y + RB], fill=ball_col[k])
        img = np.asarray(pil, np.float32).copy()
        # the hook, the clock, the standings
        text(img, q, 8, 110)
        left = max(0.0, T_END - t)
        hot = left <= 10.0 and t < T_END
        pulse = 1.0 + (0.15 * max(0.0, np.cos((left % 1.0) * np.pi * 2)) if hot else 0.0)
        text(img, f'0:{int(np.ceil(left)):02d}', int(13 * pulse), 220 - int(10 * (pulse - 1) * 8),
             color=(255, 90, 80) if hot else (255, 255, 255))
        n = np.bincount(own.ravel(), minlength=4) / NT ** 2
        for k, (name, col) in enumerate(TEAMS):
            y = 1560 + k * 82
            w = int(860 * n[k])
            img[y:y + 56, 40:40 + w] = col
            img[y:y + 56, 40 + w:900] = (40, 40, 52)
            text(img, f'{round(100 * n[k])}%', 6, y + 6, x=910)
            text(img, name, 5, y + 10, x=52)
        if t >= T_END:
            u = min(1.0, (t - T_END) / 0.3)
            name, col = TEAMS[winner]
            bright = tuple(min(255, c + 60) for c in col)
            text(img, f'{name} WINS!', int(9 + 7 * u), 860, grad=(bright, col))
            a = t - T_END
            for (cx, cy, vy, vx) in conf:
                yy, xx = int(cy + vy * a), int(cx + vx * a)
                if 0 <= yy < H - 10 and 0 <= xx < W - 10:
                    img[yy:yy + 10, xx:xx + 6] = bright if (xx + yy) % 2 else (255, 255, 255)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        if f == 0:
            thumb = frame.copy()
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    th = thumb.astype(np.float32)
    text(th, 'PICK A COLOR', 12, 880, grad=((255, 255, 255), (200, 220, 255)))
    Image.fromarray(th.astype(np.uint8)).save(os.path.join(out, 'thumbnail.jpg'), quality=94)
    return vid, len(sim['frames']) / FPS


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def soundtrack(sim, dur, path):
    rng = np.random.default_rng(5)
    mix = AU.Mix(dur + 3.0)
    last = [-1.0] * 4
    for (t, k, col) in sim['flips']:
        if t - last[k] < 0.06:
            continue
        last[k] = t
        m = PENTA[int(col * len(PENTA) / NT)] + 12 * (k % 2)
        mix.add(AU.pluck(m, 0.18, rng, bright=1.0, decay=0.25), t, 0.32, (k % 2) * 0.8 - 0.4)
    beat = 0.5
    chords = [((60, 64, 67), 36), ((57, 60, 64), 33), ((53, 57, 60), 29), ((55, 59, 62), 31)]
    b = 0
    t = 0.0
    while t < T_END - 1e-6:
        ch, root = chords[(b // 4) % 4]
        if b % 4 == 0:
            L, R = AU.supersaw([AU.midi_hz(m) for m in ch], 2.0, rng, attack=0.05, release=0.3, cutoff=2400)
            mix.add2(L * 0.35, R * 0.35, t)
        mix.add(AU.kick(b % 2), t, 0.75)
        mix.add(AU.subbass(root, beat * 0.45), t + beat / 2, 0.45)
        mix.add(AU.hat(b % 3, True), t + beat / 2, 0.12, 0.3)
        if b % 2 == 1 and t > 8.0:
            mix.add(AU.clap(b), t, 0.32)
        b += 1
        t += beat
    mix.add(AU.snare_roll(rng, 3.0, 4, 32, 120.0), T_END - 3.0, 0.32)
    mix.add(AU.riser(rng, 3.0), T_END - 3.0, 0.22)
    mix.add(AU.impact(rng, 1.1), T_END, 0.8)
    L, R = AU.supersaw([AU.midi_hz(m) for m in (48, 60, 64, 67, 72, 76)], 2.6, rng, attack=0.01, release=1.0,
                       cutoff=5200)
    mix.add2(L * 0.55, R * 0.55, T_END)
    L, R = AU.choir((60, 64, 67, 72), 2.6, rng, attack=0.1, release=1.0)
    mix.add2(L * 0.35, R * 0.35, T_END)
    nS = int(dur * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=1.4, wet=0.18)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)
    print(f'[audio] {AU.integrated_lufs(L, R):.1f} LUFS', flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sim = pick()
    print(f"seed {sim['seed']}: {sim['changes']} lead changes, {TEAMS[sim['winner']][0]} wins by "
          f"{100 * sim['margin']:.1f}%, final {sim['final'].tolist()}", flush=True)
    vid, dur = render(sim, out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(sim, dur, wav)
    final = os.path.join(out, 'color-war.mp4')
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-r', str(FPS),
                    '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', final], check=True)
    print('done', final, f'{dur:.1f}s', flush=True)


if __name__ == '__main__':
    main()
