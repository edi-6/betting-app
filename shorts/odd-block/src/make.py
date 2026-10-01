"""Find the odd block: five grids of Minecraft blocks, one different block hidden in each, five seconds to find it,
then the reveal. The pairs are picked by how alike their textures are, so the levels get really hard.
python make.py OUT_DIR
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sift-flight', 'src'))
import audio as AU          # noqa: E402
import pixelfont as PF      # noqa: E402
import textures as TX       # noqa: E402

W, H, FPS = 1080, 1920, 60
SEARCH, REVEAL, OUTRO = 5.0, 1.8, 2.6
GRIDS = [(5, 8), (6, 10), (7, 11), (8, 13), (9, 14)]
PICK = [0.45, 0.20, 0.08, 0.03, 0.0]                 # where in the sorted list of pairs (most alike first)
LEVEL_COL = [(120, 230, 120), (120, 200, 255), (255, 220, 90), (255, 150, 60), (255, 80, 80)]
BAN = ('water', 'lava', 'ichor', 'portal', 'fire', 'glass', 'flow', 'still', 'moon', 'sun')


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def text(img, s, px, y, color=(255, 255, 255), grad=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    h, w = spr.shape[:2]
    x = (img.shape[1] - w) // 2
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def pairs():
    T, _ = TX.make_all()
    names = [n for n, im in T.items() if im.shape[:2] == (16, 16) and im.shape[2] == 4 and (im[..., 3] == 255).all()
             and not any(b in n for b in BAN)]
    im = {n: T[n][..., :3].astype(np.float32) for n in names}
    feats = {n: (im[n].reshape(-1, 3).mean(0), im[n].std()) for n in names}
    ps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (np.abs(feats[a][0] - feats[b][0]).sum() + 0.5 * abs(feats[a][1] - feats[b][1]) +
                 0.3 * np.abs(im[a] - im[b]).mean())
            if d > 7.0:
                ps.append((d, a, b))
    ps.sort()
    used, out = set(), []
    for q in PICK:
        k = int(q * (len(ps) - 1))
        while ps[k][1] in used or ps[k][2] in used:
            k += 1
        d, a, b = ps[k]
        used |= {a, b}
        out.append((a, b, d))
    return T, out


def build_levels(T, chosen, rng):
    levels = []
    for (a, b, d), (cols, rows) in zip(chosen, GRIDS):
        k = max(1, int(min(1000 / cols, 1340 / rows) // 16))
        cell, gap = 16 * k, max(4, k)
        gw, gh = cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap
        x0, y0 = (W - gw) // 2, 400 + (1340 - gh) // 2
        if rng.random() < 0.5:
            a, b = b, a                                 # which of the pair is the crowd and which the odd one
        oc, orow = int(rng.integers(0, cols)), int(rng.integers(0, rows))
        big = {n: np.kron(T[n][..., :3].astype(np.float32), np.ones((k, k, 1), np.float32)) for n in (a, b)}
        grid = np.zeros((H, W, 3), np.float32)
        grid[:] = (22, 20, 30)
        for c in range(cols):
            for r in range(rows):
                x, y = x0 + c * (cell + gap), y0 + r * (cell + gap)
                grid[y:y + cell, x:x + cell] = big[b] if (c, r) == (oc, orow) else big[a]
        centre = (x0 + oc * (cell + gap) + cell / 2, y0 + orow * (cell + gap) + cell / 2)
        levels.append(dict(img=grid, centre=centre, cell=cell, crowd=a, odd=b, d=d,
                           box=(x0 + oc * (cell + gap), y0 + orow * (cell + gap))))
    return levels


def frames(levels, out):
    per = SEARCH + REVEAL
    total = per * len(levels) + OUTRO
    vid = os.path.join(out, 'video.mp4')
    p = subprocess.Popen([ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                          str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p',
                          vid], stdin=subprocess.PIPE)
    for f in range(int(total * FPS)):
        t = f / FPS
        li = min(int(t // per), len(levels) - 1)
        lv = levels[li]
        a = t - li * per
        if t >= per * len(levels):
            img = levels[-1]['img'] * 0.25
            text(img, 'HOW MANY', 12, 760)
            text(img, 'DID YOU GET?', 12, 880, grad=((255, 240, 140), (255, 150, 30)))
            text(img, 'COMMENT YOUR SCORE', 6, 1060)
        else:
            img = lv['img'].copy()
            text(img, 'FIND THE ODD BLOCK', 8, 70)
            text(img, f'LEVEL {li + 1} OF 5', 6, 160, color=LEVEL_COL[li])
            if a < SEARCH:
                n = int(np.ceil(SEARCH - a))
                text(img, str(n), 14, 235, color=(255, 90, 80) if n <= 2 else (255, 255, 255))
                text(img, 'CAN YOU FIND ALL 5?', 6, 1800)
            else:
                u = a - SEARCH
                x, y = lv['box']
                c = lv['cell']
                keep = img[y:y + c, x:x + c].copy()
                img *= 0.32
                img[y:y + c, x:x + c] = keep
                pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                d = ImageDraw.Draw(pil)
                cx, cy = lv['centre']
                rr = c * (0.85 + 1.2 * max(0.0, 1.0 - u / 0.25))
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 70, 60), width=max(6, c // 14))
                img = np.asarray(pil, np.float32).copy()
                text(img, 'FOUND IT?', 12, 230, grad=((255, 240, 140), (255, 150, 30)))
        fr = np.clip(img, 0, 255).astype(np.uint8)
        if f == 0:
            Image.fromarray(fr).save(os.path.join(out, 'thumbnail.jpg'), quality=94)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return vid, total


def soundtrack(total, path, n_levels):
    rng = np.random.default_rng(3)
    mix = AU.Mix(total + 3.0)
    beat = 0.5
    chords = [((60, 64, 67), 36), ((57, 60, 64), 33), ((53, 57, 60), 29), ((55, 59, 62), 31)]
    b, t = 0, 0.0
    while t < total - OUTRO:
        ch, root = chords[(b // 4) % 4]
        mix.add(AU.kick(b % 2), t, 0.55)
        mix.add(AU.hat(b % 3, True), t + beat / 2, 0.10, 0.3)
        mix.add(AU.subbass(root, beat * 0.4), t + beat / 2, 0.35)
        m = ch[b % 3] + 12
        mix.add(AU.pluck(m, 0.2, rng, bright=0.8, decay=0.2), t, 0.18, -0.3)
        b += 1
        t += beat
    per = SEARCH + REVEAL
    for li in range(n_levels):
        t0 = li * per
        for s in range(int(SEARCH)):
            hot = s >= SEARCH - 2
            tick = AU.modal(0.1, 2400.0 if hot else 1800.0, (1.0, 2.7), (1.0, 0.4), (0.02, 0.01), rng)
            mix.add(tick, t0 + s, 0.35 if hot else 0.22)
        tr = t0 + SEARCH
        mix.add(AU.bell(88, rng, dur=1.4), tr, 0.35, -0.2)
        mix.add(AU.bell(93, rng, dur=1.4), tr + 0.08, 0.30, 0.2)
        mix.add(AU.crash(li, 1.6), tr, 0.18)
    te = per * n_levels
    L, R = AU.supersaw([AU.midi_hz(m) for m in (48, 60, 64, 67, 72)], OUTRO, rng, attack=0.01, release=1.0,
                       cutoff=4800)
    mix.add2(L * 0.5, R * 0.5, te)
    mix.add(AU.impact(rng, 0.8), te, 0.5)
    nS = int(total * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=1.5, wet=0.18)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(11)
    T, chosen = pairs()
    for i, (a, b, d) in enumerate(chosen):
        print(f'level {i + 1}: {a} vs {b} (difference {d:.1f})', flush=True)
    levels = build_levels(T, chosen, rng)
    vid, total = frames(levels, out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(total, wav, len(levels))
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                    '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', os.path.join(out, 'odd-block.mp4')], check=True)
    print(f'done {total:.1f}s', flush=True)


if __name__ == '__main__':
    main()
