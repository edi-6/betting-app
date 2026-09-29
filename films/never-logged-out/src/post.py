"""The film look, applied to finished frames (after the interface is drawn): film grain, a tape (VHS) layer that
only creeps in when something is wrong, frame tearing, the frozen game, fades.

All effects are deterministic in (frame number, seed) so re-renders match.
"""
import numpy as np

H, W = 1080, 1920


def _rng(frame, salt=0):
    return np.random.default_rng((frame * 7919 + salt * 104729) & 0x7FFFFFFF)


def grain(img, amount, frame, size=2):
    """Monochrome grain, strongest in the mid tones (like film), a little coarser than a pixel."""
    if amount <= 0:
        return img
    rng = _rng(frame, 1)
    h, w = img.shape[:2]
    n = rng.standard_normal((h // size + 1, w // size + 1)).astype(np.float32)
    n = np.repeat(np.repeat(n, size, 0), size, 1)[:h, :w]
    f = img.astype(np.float32)
    lum = f.mean(2, keepdims=True) / 255.0
    k = amount * 255.0 * (0.35 + 1.6 * lum * (1.0 - lum))
    return np.clip(f + n[..., None] * k, 0, 255).astype(np.uint8)


def vhs(img, amount, frame, t=0.0):
    """Tape damage: colour channels drift apart, the chroma smears, scan lines, a band of tracking noise, a soft
    jitter of whole lines."""
    if amount <= 0:
        return img
    rng = _rng(frame, 2)
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    # line jitter
    jit = (rng.standard_normal(h) * 1.2 * amount).astype(int)
    if np.abs(jit).max() > 0:
        idx = (np.arange(w)[None, :] - jit[:, None]) % w
        f = np.take_along_axis(f, idx[..., None].repeat(3, 2), 1)
    # channel drift + smear
    s = int(round(2 + 5 * amount))
    r = np.roll(f[..., 0], s, axis=1)
    b = np.roll(f[..., 2], -s, axis=1)
    f[..., 0] = f[..., 0] * (1 - 0.7 * amount) + r * 0.7 * amount
    f[..., 2] = f[..., 2] * (1 - 0.7 * amount) + b * 0.7 * amount
    lum = f.mean(2, keepdims=True)
    f = f * (1 - 0.35 * amount) + lum * 0.35 * amount
    # scan lines
    f[::2] *= 1.0 - 0.10 * amount
    # a band of tracking noise drifting up the frame
    by = int((h - (t * 90.0) % (h + 200)) + 100)
    bh = int(12 + 30 * amount)
    y0, y1 = max(by, 0), min(by + bh, h)
    if y1 > y0:
        noise = rng.random((y1 - y0, w)).astype(np.float32)
        f[y0:y1] = f[y0:y1] * (1 - 0.6 * amount) + (noise[..., None] * 255.0) * 0.6 * amount
        sh = int(rng.integers(8, 40) * amount)
        f[y0:y1] = np.roll(f[y0:y1], sh, axis=1)
    return np.clip(f, 0, 255).astype(np.uint8)


def tear(img, amount, frame):
    """Frame tearing: a few horizontal slices slip sideways (one of them from a moment earlier would be better,
    but a sideways slip reads the same at 24 fps)."""
    if amount <= 0:
        return img
    rng = _rng(frame, 3)
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(1 + 4 * amount)):
        y = int(rng.integers(0, h - 20))
        hh = int(rng.integers(8, int(20 + 140 * amount)))
        dx = int(rng.normal(0, 60 * amount))
        out[y:y + hh] = np.roll(img[y:y + hh], dx, axis=1)
    return out


def chroma(img, px):
    """Plain lateral colour split (for a jolt)."""
    if px <= 0:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], px, axis=1)
    out[..., 2] = np.roll(img[..., 2], -px, axis=1)
    return out


def fade(img, a, col=(0, 0, 0)):
    """a = 1: the image; a = 0: solid col."""
    if a >= 1:
        return img
    return (img.astype(np.float32) * a + np.array(col, np.float32) * (1 - a)).astype(np.uint8)


def dim(img, k):
    return (img.astype(np.float32) * k).astype(np.uint8)


def freeze_dim(img, a):
    """The frozen game: the frame goes slightly milky, like a window that stopped responding."""
    if a <= 0:
        return img
    f = img.astype(np.float32)
    return np.clip(f * (1 - 0.35 * a) + 255.0 * 0.22 * a, 0, 255).astype(np.uint8)


def vignette(img, amount):
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :] * (w / h)
    r = np.sqrt(x * x + y * y) / np.sqrt(1 + (w / h) ** 2)
    k = 1.0 - amount * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6
    return (img.astype(np.float32) * k[..., None]).astype(np.uint8)


def apply(img, frame, t, fx):
    """fx: dict(grain, vhs, tear, chroma, fade, freeze, vignette)."""
    out = img
    if fx.get('vignette', 0) > 0:
        out = vignette(out, fx['vignette'])
    if fx.get('freeze', 0) > 0:
        out = freeze_dim(out, fx['freeze'])
    if fx.get('tear', 0) > 0:
        out = tear(out, fx['tear'], frame)
    if fx.get('vhs', 0) > 0:
        out = vhs(out, fx['vhs'], frame, t)
    if fx.get('chroma', 0) > 0:
        out = chroma(out, int(fx['chroma']))
    out = grain(out, fx.get('grain', 0.035), frame)
    if fx.get('fade', 1.0) < 1.0:
        out = fade(out, fx['fade'])
    return out
