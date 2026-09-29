"""Thumbnails (1280 x 720) from frames of the rendered film (the 3D pictures, without the interface), cropped,
graded a touch harder, with a few words in the game's font.

    python thumbnail.py            # -> release/thumb_a.jpg, thumb_b.jpg, thumb_c.jpg
"""
import os
import subprocess

import numpy as np
from PIL import Image

import film as FM
import pixelfont as PF

OUT = os.path.join(FM.ROOT, 'release')
# name, shot, time, crop (x, y, width) of the 1920 x 1080 frame, lines [(text, colour)], text at top / bottom
THUMBS = [
    ('thumb_a', 'f5_shadows', 10.4, (240, 60, 1440), [('2 SHADOWS.', (240, 240, 240)), ('1 PLAYER.', (235, 40, 40))],
     'bottom'),
    ('thumb_b', 's8_turn', 10.9, (300, 40, 1320), [("DON'T LOOK", (240, 240, 240)), ('AT HIM', (235, 40, 40))],
     'top'),
    ('thumb_c', 'u6b_streets', 3.0, (0, 0, 1920), [('24 COPIES', (240, 240, 240)), ('OF MY HOUSE', (235, 40, 40))],
     'top'),
]


def frame(shot, t):
    path = os.path.join(FM.frames_dir(False), shot + '.mp4')
    cmd = [FM.ffmpeg_exe(), '-v', 'error', '-ss', f'{t:.3f}', '-i', path, '-frames:v', '1', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-']
    b = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(FM.H, FM.W, 3)


def grade(img):
    f = img.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * 1.12 + 0.5 + 0.02, 0, 1) ** 0.95            # a little more contrast, lifted mids
    h, w = f.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    v = 1.0 - 0.35 * np.clip(np.sqrt(x * x * 0.8 + y * y) - 0.45, 0, 1) ** 1.4
    return (f * v[..., None] * 255).astype(np.uint8)


def over(dst, spr, x, y):
    h, w = spr.shape[:2]
    x, y = int(x), int(y)
    a = spr[..., 3:4].astype(np.float32) / 255.0
    region = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (region * (1 - a) + spr[..., :3] * a).astype(np.uint8)


def make(name, shot, t, crop, lines, where):
    img = frame(shot, t)
    x, y, w = crop
    h = w * 9 // 16
    img = np.asarray(Image.fromarray(img[y:y + h, x:x + w]).resize((1280, 720), Image.LANCZOS))
    img = grade(img)
    sprites = [PF.render(txt, px=11, color=col, outline=2, outline_col=(8, 6, 10)) for (txt, col) in lines]
    total = sum(s.shape[0] for s in sprites) + 14 * (len(sprites) - 1)
    yy = 40 if where == 'top' else 720 - 44 - total
    # a soft dark band behind the words
    band = np.zeros(720, np.float32)
    band[max(0, yy - 30):min(720, yy + total + 30)] = 1.0
    band = np.convolve(band, np.ones(60) / 60, 'same')
    img = (img.astype(np.float32) * (1.0 - 0.45 * band)[:, None, None]).astype(np.uint8)
    for s in sprites:
        over(img, s, (1280 - s.shape[1]) // 2, yy)
        yy += s.shape[0] + 14
    path = os.path.join(OUT, name + '.jpg')
    Image.fromarray(img).save(path, quality=92)
    print(path)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for spec in THUMBS:
        make(*spec)
