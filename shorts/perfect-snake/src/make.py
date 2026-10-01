"""Can the snake fill the whole board? A snake plays a perfect game of Snake: it follows a Hamiltonian cycle of the
board (so it can never trap itself), taking safe shortcuts towards the apple while it's short, until it fills every
square. It speeds up as it grows; every apple plays the next note of a climbing scale.  python make.py OUT_DIR
"""
import colorsys
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sift-flight', 'src'))
import audio as AU          # noqa: E402
import pixelfont as PF      # noqa: E402

W, H, FPS = 1080, 1920, 60
COLS, ROWS, CELL = 22, 34, 44
X0, Y0 = (W - COLS * CELL) // 2, 380
N = COLS * ROWS
TARGET_S, TAIL_S = 43.0, 3.5
PENTA = [0, 2, 4, 7, 9]


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def text(img, s, px, y, color=(255, 255, 255), grad=None, x=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    h, w = spr.shape[:2]
    x = (img.shape[1] - w) // 2 if x is None else x
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def play(seed=4):
    """The whole game: per step the head's cell, the length, the apple's cell; and the steps where it ate."""
    cyc = []
    for y in range(ROWS):
        xs = range(1, COLS) if y % 2 == 0 else range(COLS - 1, 0, -1)
        cyc += [(x, y) for x in xs]
    cyc += [(0, y) for y in range(ROWS - 1, -1, -1)]
    order = {c: i for i, c in enumerate(cyc)}
    rng = np.random.default_rng(seed)
    occ = np.zeros((COLS, ROWS), bool)
    body = [cyc[0]]
    occ[cyc[0]] = True

    def dist(a, b):
        return (order[b] - order[a]) % N

    def new_apple():
        free = np.flatnonzero(~occ.ravel())
        if not len(free):
            return None
        k = int(rng.choice(free))
        return (k // ROWS, k % ROWS)

    apple = new_apple()
    heads, lens, apples, eats = [body[-1]], [1], [apple], []
    while apple is not None:
        h, tail = body[-1], body[0]
        nxt = cyc[(order[h] + 1) % N]
        if len(body) < N * 0.55:
            dA, dT = dist(h, apple), dist(h, tail)
            best = dist(h, nxt)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (h[0] + d[0], h[1] + d[1])
                if not (0 <= n[0] < COLS and 0 <= n[1] < ROWS) or occ[n]:
                    continue
                dn = dist(h, n)
                if best < dn <= dA and dn < dT - 4:
                    best, nxt = dn, n
        if occ[nxt] and nxt != tail:
            break                                        # never happens with the cycle; stop rather than crash
        if nxt == apple:
            body.append(nxt)
            occ[nxt] = True
            eats.append(len(heads))
            apple = new_apple()
        else:
            occ[body[0]] = False
            body.pop(0)
            body.append(nxt)
            occ[nxt] = True
        heads.append(nxt)
        lens.append(len(body))
        apples.append(apple)
    return heads, lens, apples, eats


def schedule(lens):
    """Steps per frame growing with the snake, scaled so the game takes about TARGET_S."""
    L = np.asarray(lens, float) / N
    lo, hi = 0.0, 5000.0
    for _ in range(60):
        A = (lo + hi) / 2
        frames = np.sum(1.0 / (1.0 + A * L ** 2.2))
        if frames / FPS > TARGET_S:
            lo = A
        else:
            hi = A
    rate = 1.0 + A * L ** 2.2
    pos = np.cumsum(1.0 / rate)                         # the frame each step lands on
    return pos


def render(heads, lens, apples, pos, out):
    nf = int(pos[-1]) + 1 + int(TAIL_S * FPS)
    step_at = np.searchsorted(pos, np.arange(nf), side='right') - 1
    step_at = np.clip(step_at, 0, len(heads) - 1)
    H_ = np.array(heads)
    gap = np.ones((ROWS * CELL, COLS * CELL, 1), np.float32)
    for k in range(ROWS + 1):
        gap[max(0, k * CELL - 2):k * CELL + 2] = 0.0
    for k in range(COLS + 1):
        gap[:, max(0, k * CELL - 2):k * CELL + 2] = 0.0
    hue = np.array([colorsys.hsv_to_rgb(0.92 * i / N, 0.7, 1.0) for i in range(N)], np.float32) * 255
    rng = np.random.default_rng(2)
    conf = np.column_stack([rng.uniform(0, W, 240), rng.uniform(-1000, 0, 240), rng.uniform(260, 560, 240),
                            rng.uniform(-70, 70, 240), rng.uniform(0, 1, 240)])
    vid = os.path.join(out, 'video.mp4')
    p = subprocess.Popen([ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                          str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p',
                          vid], stdin=subprocess.PIPE)
    t_done = pos[-1] / FPS
    for f in range(nf):
        s = int(step_at[f])
        L = lens[s]
        seg = H_[max(0, s - L + 1):s + 1]
        grid = np.zeros((COLS, ROWS, 3), np.float32)
        grid[:] = (26, 26, 38)
        grid[seg[:, 0], seg[:, 1]] = hue[np.linspace(0, L - 1, len(seg)).astype(int)]
        hx, hy = seg[-1]
        grid[hx, hy] = (255, 255, 255)
        ap = apples[s]
        if ap is not None:
            grid[ap] = (235, 40, 50)
        board = np.repeat(np.repeat(grid.transpose(1, 0, 2), CELL, 0), CELL, 1) * gap + (1 - gap) * 14.0
        img = np.empty((H, W, 3), np.float32)
        img[:] = (12, 12, 20)
        img[Y0:Y0 + ROWS * CELL, X0:X0 + COLS * CELL] = board
        t = f / FPS
        pct = int(100 * L / N)
        if t < t_done:
            text(img, 'CAN IT FILL THE BOARD?', 7, 70)
            text(img, f'LENGTH {L}', 11, 150)
            img[290:330, 90:990] = (40, 40, 56)
            img[290:330, 90:90 + int(900 * L / N)] = (90, 230, 120)
            text(img, f'{pct}%', 5, 296, x=1000 - 10 * 5 * len(f'{pct}%') // 2 - 60)
        else:
            u = min(1.0, (t - t_done) / 0.3)
            text(img, '100%', int(14 + 8 * u), 90, grad=((255, 240, 140), (255, 150, 30)))
            text(img, 'PERFECT GAME!', 9, 250)
            a = t - t_done
            for (cx, cy, vy, vx, hh) in conf:
                yy, xx = int(cy + vy * a), int(cx + vx * a)
                if 0 <= yy < H - 12 and 0 <= xx < W - 8:
                    img[yy:yy + 12, xx:xx + 7] = np.array(colorsys.hsv_to_rgb(hh, 0.6, 1.0)) * 255
        fr = np.clip(img, 0, 255).astype(np.uint8)
        if f == 0:
            Image.fromarray(fr).save(os.path.join(out, 'thumbnail.jpg'), quality=94)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return vid, nf / FPS, t_done


def soundtrack(eats, pos, dur, t_done, path):
    rng = np.random.default_rng(5)
    mix = AU.Mix(dur + 3.0)
    last = -1.0
    for k, s in enumerate(eats):
        t = pos[s] / FPS
        if t - last < 0.07 or t > t_done:
            continue
        last = t
        frac = k / max(1, len(eats) - 1)
        step = int(frac * 24)
        m = 55 + 12 * (step // 5) + PENTA[step % 5]
        mix.add(AU.pluck(m, 0.16, rng, bright=1.0, decay=0.22), t, 0.30, float(np.sin(k) * 0.4))
    beat, t, b = 60.0 / 128.0, 0.0, 0
    chords = [((57, 60, 64), 33), ((53, 57, 60), 29), ((48, 52, 55), 36), ((55, 59, 62), 31)]
    while t < t_done:
        ch, root = chords[(b // 4) % 4]
        mix.add(AU.kick(b % 2), t, 0.55)
        mix.add(AU.hat(b % 3, True), t + beat / 2, 0.10, 0.3)
        mix.add(AU.subbass(root, beat * 0.4), t + beat / 2, 0.32)
        if b % 4 == 0:
            L, R = AU.supersaw([AU.midi_hz(m) for m in ch], beat * 4, rng, attack=0.05, release=0.3, cutoff=2200)
            mix.add2(L * 0.25, R * 0.25, t)
        b += 1
        t += beat
    mix.add(AU.riser(rng, 3.0), t_done - 3.0, 0.22)
    mix.add(AU.impact(rng, 1.0), t_done, 0.75)
    L, R = AU.supersaw([AU.midi_hz(m) for m in (45, 57, 60, 64, 69, 72)], 3.0, rng, attack=0.01, release=1.2,
                       cutoff=5200)
    mix.add2(L * 0.5, R * 0.5, t_done)
    L, R = AU.choir((57, 64, 69, 72), 3.0, rng, attack=0.1, release=1.0)
    mix.add2(L * 0.35, R * 0.35, t_done)
    mix.add(AU.crash(1, 3.0), t_done, 0.3)
    nS = int(dur * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=1.5, wet=0.18)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    heads, lens, apples, eats = play()
    pos = schedule(lens)
    print(f'{len(heads)} steps, {len(eats)} apples, final length {lens[-1]} of {N}', flush=True)
    vid, dur, t_done = render(heads, lens, apples, pos, out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(eats, pos, dur, t_done, wav)
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                    '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', os.path.join(out, 'perfect-snake.mp4')], check=True)
    print(f'done {dur:.1f}s (board full at {t_done:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
