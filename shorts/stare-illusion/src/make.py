"""Stare at the red dot for 20 seconds... then look at your hand. A slowly turning, expanding spiral with a countdown;
then a still of the golden Sift valley, which seems to breathe and shrink (the motion aftereffect, the 'waterfall
illusion'), and the prompts to look at your hand.  python make.py OUT_DIR
"""
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
T_STARE, T_END = 20.0, 28.5
CX, CY = W / 2, H / 2 + 60
PHOTO = os.path.join(HERE, '..', '..', 'sift-flight', 'release', 'thumbnail_clean.jpg')


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def text(img, s, px, y, color=(255, 255, 255), grad=None):
    spr = PF.render(s, px=px, color=color, grad=grad, outline=1)
    h, w = spr.shape[:2]
    x = (img.shape[1] - w) // 2
    a = spr[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a


def frames(out):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - CX, yy - CY
    r = np.hypot(dx, dy) + 1.0
    th = np.arctan2(dy, dx)
    lr = np.log(r)
    calm = np.clip((r - 34.0) / 70.0, 0, 1)
    vign = 1.0 - 0.35 * np.clip((r - 520.0) / 700.0, 0, 1)
    dot = r < 16.0
    ring = (r >= 16.0) & (r < 21.0)
    photo = np.asarray(Image.open(PHOTO).convert('RGB').resize((W, H), Image.LANCZOS), np.float32)
    vid = os.path.join(out, 'video.mp4')
    p = subprocess.Popen([ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
                          str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p',
                          vid], stdin=subprocess.PIPE)
    for f in range(int(T_END * FPS)):
        t = f / FPS
        if t < T_STARE:
            ease = min(1.0, t / 1.5)
            s = np.tanh(2.6 * np.sin(6.0 * th + 9.0 * lr - 6.5 * t * ease - 3.0 * ease * ease))
            g = (128.0 + 82.0 * s * calm) * vign
            img = np.repeat(g[..., None], 3, 2) * np.array([1.0, 0.97, 1.04], np.float32)
            img[ring] = 255.0
            img[dot] = (235, 28, 40)
            img[60:420] *= 0.55
            text(img, 'STARE AT THE RED DOT', 7, 110)
            text(img, 'FOR 20 SECONDS', 7, 190)
            n = int(np.ceil(T_STARE - t))
            text(img, str(n), 20, 270, color=(255, 90, 80) if n <= 5 else (255, 255, 255))
        else:
            a = t - T_STARE
            img = photo.copy()
            if a < 3.5:
                img[ring] = 255.0
                img[dot] = (235, 28, 40)
                img[60:330] *= 0.55
                text(img, 'KEEP LOOKING', 8, 110)
                text(img, 'AT THE DOT', 8, 200)
            elif a < 6.0:
                img[60:330] *= 0.55
                text(img, 'NOW LOOK AT', 8, 110)
                text(img, 'YOUR HAND', 8, 200, grad=((255, 240, 140), (255, 150, 30)))
            else:
                img[60:330] *= 0.55
                text(img, 'DID IT MOVE?', 10, 140, grad=((255, 240, 140), (255, 150, 30)))
        fr = np.clip(img, 0, 255).astype(np.uint8)
        if f == 0:
            Image.fromarray(fr).save(os.path.join(out, 'thumbnail.jpg'), quality=94)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return vid


def soundtrack(path):
    rng = np.random.default_rng(7)
    mix = AU.Mix(T_END + 3.0)
    L, R = AU.choir((45, 52, 57, 64), T_STARE, rng, attack=4.0, release=1.5, vowel='o')
    mix.add2(L * 0.55, R * 0.55, 0.0)
    n = int(T_STARE * AU.SR)
    tt = np.arange(n) / AU.SR
    hum = np.sin(2 * np.pi * 55.0 * tt) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * tt)) * np.clip(tt / 3.0, 0, 1)
    mix.add(hum * 0.25, 0.0)
    for k in range(int(T_STARE)):
        tick = AU.modal(0.12, 1800.0, (1.0, 2.7, 4.1), (1.0, 0.5, 0.3), (0.02, 0.012, 0.008), rng)
        mix.add(tick, float(k), 0.16 + 0.02 * k)
    mix.add(AU.riser(rng, 4.0, 200, 8000), T_STARE - 4.0, 0.30)
    mix.add(AU.reverse_crash(rng, 1.0), T_STARE - 1.0, 0.35)
    mix.add(AU.impact(rng, 0.9), T_STARE, 0.55)
    L, R = AU.choir((57, 64, 69, 73, 76), T_END - T_STARE, rng, attack=0.3, release=1.5, vowel='a')
    mix.add2(L * 0.45, R * 0.45, T_STARE)
    for k, m in enumerate((81, 85, 88, 93)):
        mix.add(AU.bell(m, rng, dur=2.5), T_STARE + 0.15 * k, 0.22, -0.6 + 0.4 * k)
    mix.add(AU.bell(88, rng, dur=2.0), T_STARE + 6.0, 0.3)
    nS = int(T_END * AU.SR)
    L, R = AU.reverb(mix.L[:nS], mix.R[:nS], seed=4, decay=2.6, wet=0.3)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    vid = frames(out)
    wav = os.path.join(out, 'audio.wav')
    soundtrack(wav)
    common = ['-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p',
              '-profile:v', 'high', '-x264-params', f'keyint={FPS // 2}:min-keyint={FPS // 2}:scenecut=0',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-ar',
              '48000', '-movflags', '+faststart', '-shortest']
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav] + common +
                   ['-crf', '20', '-maxrate', '16M', '-bufsize', '32M', '-b:a', '320k',
                    os.path.join(out, 'stare-illusion.mp4')], check=True)
    subprocess.run([ffmpeg(), '-y', '-v', 'error', '-i', vid, '-i', wav, '-vf', 'scale=720:1280:flags=lanczos'] + common +
                   ['-crf', '23', '-maxrate', '6M', '-bufsize', '12M', '-b:a', '192k',
                    os.path.join(out, 'stare-illusion-720p.mp4')], check=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
