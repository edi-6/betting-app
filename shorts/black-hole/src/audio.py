"""Procedural sound design driven by the cue sheet of the render.

Everything is synthesised here (no samples). The black hole: a hum that starts as a thin whine and grows into a
sub-bass roar as it grows, air rushing in, blocks being torn out of the ground, the sizzle of the hot disk,
TNT going off in it; the giants groaning as they lean into the pull, torn off the ground, stretched (a rubbery
creak) and swallowed with a deep gulp; the creeper's hiss and blast, the Warden's sonic boom being swallowed;
the collapse (everything sucked in, then dead silence) and the explosion, with the blocks raining back down.
The score: a cathedral organ in the spirit of space films, one chord per SIZE (A minor, F, D minor, then the
dominant E while the Warden goes and the hole collapses, resolving to A major on the explosion) over a ticking
clock. Stereo panning from the 3D event positions; slow motion drops the pitch and adds reverb; loudness
normalised (BS.1770 style) to -14 LUFS with a peak limiter.
"""
import json
import wave

import numpy as np
from scipy import signal

SR = 48000


# ---------------------------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------------------------
def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output='sos')


def bp(x, lo, hi, order=2):
    hi = min(hi, SR * 0.45)
    return signal.sosfilt(_sos('bandpass', [lo, hi], order), x)


def lp(x, f, order=2):
    return signal.sosfilt(_sos('lowpass', min(f, SR * 0.45), order), x)


def hp(x, f, order=2):
    return signal.sosfilt(_sos('highpass', f, order), x)


def expenv(n, tau, attack=0.0):
    t = np.arange(n) / SR
    e = np.exp(-t / tau)
    if attack > 0:
        e *= np.clip(t / attack, 0, 1)
    return e


def norm(x, peak=1.0):
    m = np.abs(x).max() if len(x) else 0.0
    return x * (peak / m) if m > 0 else x


def shaped_noise(dur, center, width_oct, env, rng, nfft=1024):
    """Noise with a time-varying band-pass (log-gaussian) spectral shape. center/width/env: callables of t."""
    n = int(dur * SR)
    x = rng.standard_normal(n + nfft)
    f, t, Z = signal.stft(x, fs=SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    t = t[:Z.shape[1]]
    c = np.maximum(center(t), 30.0)
    w = np.maximum(width_oct(t), 0.1)
    lf = np.log2(np.maximum(f, 1.0))[:, None]
    g = np.exp(-0.5 * ((lf - np.log2(c)[None, :]) / w[None, :]) ** 2)
    Z = Z * g * env(t)[None, :]
    _, y = signal.istft(Z, fs=SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    return y[:n]


def sine_sweep(dur, f0, f1, tau_f, rng=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / tau_f)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def resample(x, ratio):
    """Pitch/speed change by ratio (>1 = higher)."""
    n = int(len(x) / ratio)
    if n < 2:
        return x
    return np.interp(np.arange(n) * ratio, np.arange(len(x)), x)


def grains(dur, n_grains, lo, hi, tau_spread, rng, glen=(0.002, 0.007), decay=None):
    n = int(dur * SR)
    out = np.zeros(n)
    for _ in range(n_grains):
        pos = int(min(n - 1, rng.exponential(tau_spread) * SR))
        gl = int(rng.uniform(*glen) * SR)
        g = rng.standard_normal(gl) * np.hanning(gl)
        g = bp(g, lo * rng.uniform(0.8, 1.25), hi * rng.uniform(0.8, 1.2))
        amp = rng.uniform(0.3, 1.0)
        if decay:
            amp *= np.exp(-pos / SR / decay)
        e = min(n, pos + gl)
        out[pos:e] += g[:e - pos] * amp
    return out


def modal(dur, f0, ratios, amps, taus, rng, detune=0.004):
    """Sum of decaying inharmonic partials (struck metal)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a, tau in zip(ratios, amps, taus):
        f = f0 * r * (1 + rng.uniform(-detune, detune))
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    return x


def reverb_ir(rng, dur=1.6, tau=0.45, lp_hz=5000):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / tau)
    ir = lp(ir, lp_hz, 1)
    ir[: int(0.012 * SR)] = 0.0                  # pre-delay
    return ir / np.sqrt(np.sum(ir ** 2))


def compress(L, R, thresh_db=-22.0, ratio=3.0, attack=0.006, release=0.22):
    """Feed-forward RMS compressor (stereo linked), evaluated at 1 kHz and interpolated."""
    hop = SR // 1000
    m = 0.5 * (L * L + R * R)
    win = int(0.03 * SR)
    rms = np.sqrt(np.convolve(m, np.ones(win) / win, mode='same') + 1e-12)[::hop]
    lev = 20 * np.log10(rms)
    target = -np.maximum(lev - thresh_db, 0.0) * (1.0 - 1.0 / ratio)
    aa = np.exp(-1.0 / (attack * 1000))
    ar = np.exp(-1.0 / (release * 1000))
    g = np.zeros_like(target)
    cur = 0.0
    for i, tv in enumerate(target):
        cur = aa * cur + (1 - aa) * tv if tv < cur else ar * cur + (1 - ar) * tv
        g[i] = cur
    gl = 10 ** (np.interp(np.arange(len(L)), np.arange(len(g)) * hop, g) / 20.0)
    return L * gl, R * gl


# ---------------------------------------------------------------------------------------------
# sounds reused from the earlier videos
# ---------------------------------------------------------------------------------------------
def make_crunch(rng, lo=500, hi=2600, dur=0.25):
    x = grains(dur, int(rng.integers(20, 40)), lo, hi, 0.05, rng, glen=(0.002, 0.009))
    x += 0.4 * bp(rng.standard_normal(len(x)), 150, 600) * expenv(len(x), 0.05)
    return norm(x, 0.8)


def make_tick(rng):
    dur = 0.06
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 1400, 6500) * expenv(n, rng.uniform(0.003, 0.008))
    x += 0.5 * np.sin(2 * np.pi * rng.uniform(300, 520) * np.arange(n) / SR) * expenv(n, 0.012)
    return norm(x, 0.7)


def make_riser(rng, dur):
    x = shaped_noise(dur, lambda t: 300 + 5000 * (t / dur) ** 2, lambda t: 0.6 + 0 * t,
                     lambda t: np.clip(t / dur, 0, 1) ** 2.5, rng)
    n = len(x)
    tt = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(55 + 55 * (tt / dur) ** 2) / SR) * (tt / dur) ** 2
    return norm(norm(x) + 0.35 * tone, 0.7)


def make_hit(rng):
    """Short cinematic 'trailer hit': tight low punch + bright transient."""
    dur = 0.9
    n = int(dur * SR)
    punch = sine_sweep(dur, 140, 52, 0.035) * expenv(n, 0.16, 0.001)
    snap = bp(rng.standard_normal(n), 900, 9000) * expenv(n, 0.02)
    body = bp(rng.standard_normal(n), 120, 700) * expenv(n, 0.12)
    x = norm(punch) + 0.45 * norm(snap) + 0.5 * norm(body)
    return norm(hp(np.tanh(1.4 * x), 35.0, 2), 0.9)


def make_slowmo_down(rng):
    """Time slowing down: a deep downward 'vwoom' with the air sucked out."""
    dur = 1.2
    n = int(dur * SR)
    sub = sine_sweep(dur, 170, 34, 0.25) * expenv(n, 0.5, 0.02)
    air = shaped_noise(dur, lambda t: 3000 * np.exp(-t / 0.25) + 150, lambda t: 0.9 + 0 * t,
                       lambda t: np.clip(t / 0.05, 0, 1) * np.exp(-t / 0.35), rng)
    return norm(norm(sub) + 0.6 * norm(air), 0.85)


def make_slowmo_up(rng):
    dur = 0.5
    n = int(dur * SR)
    t = np.arange(n) / SR
    air = shaped_noise(dur, lambda tt: 300 + 3500 * (tt / dur) ** 2, lambda tt: 0.8 + 0 * tt,
                       lambda tt: (tt / dur) ** 2 * np.clip((dur - tt) / 0.03, 0, 1), rng)
    tone = np.sin(2 * np.pi * np.cumsum(50 + 110 * (t / dur) ** 2) / SR) * (t / dur) ** 2
    return norm(norm(air) + 0.35 * tone, 0.7)


IRON = (1.0, 2.32, 3.87, 5.12, 7.35, 9.8)


def make_clang(rng):
    """An anvil knocked loose: a hard iron clunk with a short inharmonic ring."""
    dur = 1.0
    n = int(dur * SR)
    f0 = rng.uniform(330, 470)
    ring = modal(dur, f0, IRON, (1.0, 0.7, 0.55, 0.4, 0.3, 0.2), (0.20, 0.14, 0.10, 0.07, 0.05, 0.04), rng)
    knock = bp(rng.standard_normal(n), 250, 1800) * expenv(n, 0.012, 0.0004)
    body = sine_sweep(dur, 190, 90, 0.02) * expenv(n, 0.05, 0.0008)
    x = 0.8 * norm(ring) + 0.9 * norm(knock) + 0.7 * norm(body)
    return norm(np.tanh(1.3 * x), 0.9)


def make_hail(rng, dur, maker, density, bank_n=10):
    """A continuous hail of one kind of hit to sit under mass events."""
    n = int(dur * SR)
    bank = [maker(rng) for _ in range(bank_n)]
    x = np.zeros(n)
    for _ in range(int(dur * density)):
        s = int(rng.uniform(0, max(1, n - 1)))
        b = resample(bank[rng.integers(bank_n)], rng.uniform(0.8, 1.2))
        e = min(n, s + len(b))
        x[s:e] += b[:e - s] * rng.uniform(0.3, 1.0)
    return norm(x, 0.8)


def make_growl(rng, dur=0.9, f0=52.0):
    """A low, rattling growl: a buzzing pulse train through moving formants."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 + 10 * np.sin(2 * np.pi * 3.1 * t) + rng.uniform(-3, 3)
    ph = np.cumsum(f) / SR
    buzz = (np.mod(ph, 1.0) < 0.14).astype(float) - 0.14
    buzz += 0.5 * rng.standard_normal(n) * (np.mod(ph, 1.0) < 0.3)
    out = np.zeros(n)
    for lo, hi, g in ((180, 420, 1.0), (520, 900, 0.6), (1200, 2200, 0.25)):
        out += g * bp(buzz, lo, hi, 2)
    env = np.clip(t / 0.06, 0, 1) * np.clip((dur - t) / 0.35, 0, 1)
    return norm(out * env, 0.8)


def make_boom_charge(rng, dur=0.8):
    """Sonic boom charging: an inhaled rising drone with a crackle of energy."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    drone = np.sin(2 * np.pi * np.cumsum(70 + 240 * u ** 2) / SR) + 0.4 * np.sin(2 * np.pi * np.cumsum(140 + 480 * u ** 2) / SR)
    air = shaped_noise(dur, lambda tt: 400 + 3000 * (tt / dur) ** 2, lambda tt: 0.8 + 0 * tt,
                       lambda tt: (tt / dur) ** 2, rng)
    crackle = grains(dur, 60, 2000, 7000, dur * 0.6, rng, glen=(0.001, 0.003))
    x = norm(drone * u ** 1.5) + 0.7 * norm(air) + 0.25 * norm(crackle * u)
    return norm(x, 0.8)


def make_sonic_boom(rng):
    """The Warden's sonic boom: an enormous low WHUMP, a bright crack, a resonant pitch dive and an airy wash."""
    dur = 2.4
    n = int(dur * SR)
    t = np.arange(n) / SR
    whump = sine_sweep(dur, 120, 32, 0.12) * expenv(n, 0.45, 0.002)
    crack = bp(rng.standard_normal(n), 1200, 12000) * expenv(n, 0.008)
    dive = np.sin(2 * np.pi * np.cumsum(90 + 700 * np.exp(-t / 0.18)) / SR) * expenv(n, 0.5, 0.004)
    wash = shaped_noise(dur, lambda tt: 300 + 3500 * np.exp(-tt / 0.3), lambda tt: 1.1 + 0 * tt,
                        lambda tt: np.exp(-tt / 0.7) * np.clip(tt / 0.01, 0, 1), rng)
    ringy = modal(dur, 220.0, (1.0, 1.5, 2.02, 3.1), (1.0, 0.6, 0.4, 0.2), (0.6, 0.4, 0.3, 0.2), rng, 0.0)
    x = 1.0 * norm(whump) + 0.5 * norm(crack) + 0.6 * norm(dive) + 0.7 * norm(wash) + 0.25 * norm(ringy)
    return norm(hp(np.tanh(1.4 * x), 28.0, 2), 0.98)


def make_evening(rng, dur):
    """A calm afternoon: soft wind with the odd far-off bird."""
    n = int(dur * SR)
    x = lp(rng.standard_normal(n), 700, 2)
    x = hp(x, 60, 2)
    mod = lp(rng.standard_normal(n), 0.3, 1)
    mod = 0.6 + 0.4 * (mod - mod.min()) / (np.ptp(mod) + 1e-9)
    x = norm(x * mod, 0.3)
    t = np.arange(n) / SR
    for _ in range(int(dur * 0.9)):
        s = rng.uniform(0, dur - 0.5)
        f0 = rng.uniform(2600, 4200)
        for k in range(int(rng.integers(2, 4))):
            st = s + k * rng.uniform(0.09, 0.14)
            m = (t >= st) & (t < st + 0.07)
            u = (t[m] - st) / 0.07
            x[m] += 0.05 * np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.25 * np.sin(np.pi * u))) / SR) * np.sin(np.pi * u)
    return x


def make_squelch(rng, dur=0.4):
    x = shaped_noise(dur, lambda t: 1100 * np.exp(-t / 0.2) + 300, lambda t: 1.0 + 0 * t,
                     lambda t: np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.1), rng)
    n = len(x)
    t = np.arange(n) / SR
    x *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * rng.uniform(25, 40) * t + rng.uniform(0, 6)))
    x = lp(x, 3000, 2)
    return norm(x + 0.4 * grains(dur, 30, 400, 2200, 0.08, rng), 0.7)


def make_rip(rng):
    """An arm torn off: tearing crunch over a heavy wet thump."""
    dur = 0.9
    n = int(dur * SR)
    tear = grains(dur, 220, 300, 2600, 0.12, rng, glen=(0.002, 0.012))
    thump = sine_sweep(dur, 120, 45, 0.05) * expenv(n, 0.18, 0.002)
    wet = make_squelch(rng, dur) * 0.6
    crack = bp(rng.standard_normal(n), 1000, 7000) * expenv(n, 0.01)
    x = norm(tear) + 0.9 * norm(thump) + 0.5 * norm(wet) + 0.5 * norm(crack)
    return norm(np.tanh(1.5 * x), 0.9)


def make_explosion(rng, size=1.0, distant=False):
    """A blast built to read on phone speakers: sharp crack, punchy mid body, roaring noise burst that sweeps
    down, crackling tail, and only a short sub thump."""
    dur = 2.0 + 0.6 * size
    n = int(dur * SR)
    t = np.arange(n) / SR
    crack = bp(rng.standard_normal(n), 1500, 12000) * expenv(n, 0.005)
    thump = sine_sweep(dur, rng.uniform(110, 135), rng.uniform(48, 58), 0.05) * expenv(n, 0.13 + 0.04 * size, 0.002)
    body = bp(rng.standard_normal(n), 160, 900, 2) * expenv(n, 0.16 + 0.08 * size, 0.003)
    roar = shaped_noise(dur, lambda tt: 450 + 3800 * np.exp(-tt / (0.09 + 0.05 * size)),
                        lambda tt: 1.2 + 0 * tt, lambda tt: np.exp(-tt / (0.30 + 0.16 * size)) *
                        np.clip(tt / 0.004, 0, 1), rng)
    crackle = np.zeros(n)
    for _ in range(int(170 * size)):
        p = int(min(n - 400, rng.exponential(0.3 + 0.18 * size) * SR))
        L = rng.integers(40, 260)
        crackle[p:p + L] += rng.standard_normal(L) * np.hanning(L) * rng.uniform(0.3, 1.0) * np.exp(-p / SR / 0.7)
    crackle = bp(crackle, 900, 7000)
    rumble = bp(rng.standard_normal(n), 45, 220, 2) * expenv(n, 0.32 + 0.12 * size, 0.02)
    x = 0.55 * norm(crack) + 0.7 * norm(thump) + 0.75 * norm(body) + 1.0 * norm(roar) + 0.4 * norm(crackle) \
        + 0.3 * norm(rumble)
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    x = hp(x, 38.0, 2)
    if distant:
        x = lp(x, 1800, 2) + 0.2 * x
        x = np.concatenate([np.zeros(int(0.03 * SR)), x])[:n]
    return norm(x * np.clip((dur - t) / 0.3, 0, 1), 0.95)


def make_whoosh(rng, dur, heavy=0.0):
    """Rising band-passed noise rushing in; denser and lower when heavy."""
    def center(t):
        u = np.clip(t / dur, 0, 1)
        return 380 + (1500 + 900 * (1 - heavy)) * u ** 1.6

    def env(t):
        u = np.clip(t / dur, 0, 1)
        return (u ** 2.2) * (1.0 - np.clip((u - 0.97) / 0.03, 0, 1))
    x = shaped_noise(dur, center, lambda t: 0.9 + 0.4 * heavy + 0 * t, env, rng)
    if heavy > 0:
        n = len(x)
        tt = np.arange(n) / SR
        x = x * (1 + 0.35 * heavy * np.sin(2 * np.pi * (23 + 9 * rng.random()) * tt + rng.uniform(0, 6)))
        rumble = lp(rng.standard_normal(n), 140, 4) * np.clip(tt / dur, 0, 1) ** 2.5
        x = norm(x) + 0.35 * heavy * norm(rumble)
    return norm(x, 0.8)


def _saws(f0, n, rng, ratios=((1.0, 1.0),), detune=(-0.006, 0.0, 0.007)):
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a in ratios:
        for d in detune:
            ph = np.cumsum(np.full(n, f0 * r * (1 + d))) / SR + rng.uniform()
            x += a * (2 * np.mod(ph, 1.0) - 1)
    return x, t


def make_braam(rng, dur=1.8, f0=55.0):
    """Trailer 'braam': a brassy detuned saw stack whose filter snaps open and slowly closes, over a punch."""
    n = int(dur * SR)
    x, t = _saws(f0, n, rng, ((1.0, 1.0), (1.5, 0.45), (2.0, 0.6), (3.0, 0.18)))
    bright = lp(x, 2600, 2) * np.exp(-t / 0.11)
    dark = lp(x, 420, 2) * np.exp(-t / 0.8)
    env = np.clip(t / 0.012, 0, 1) * np.clip((dur - t) / 0.35, 0, 1)
    punch = sine_sweep(dur, 115, 42, 0.05) * expenv(n, 0.22, 0.002)
    y = (0.55 * norm(bright) + norm(dark)) * env + 0.8 * norm(punch)
    return norm(hp(np.tanh(1.3 * y), 30.0, 2), 0.9)


def make_taiko(rng):
    dur = 1.1
    n = int(dur * SR)
    body = sine_sweep(dur, rng.uniform(95, 110), rng.uniform(46, 52), 0.03) * expenv(n, 0.24, 0.001)
    skin = bp(rng.standard_normal(n), 150, 900) * expenv(n, 0.035, 0.0005)
    slap = bp(rng.standard_normal(n), 1500, 5000) * expenv(n, 0.006)
    return norm(hp(np.tanh(1.4 * (norm(body) + 0.5 * norm(skin) + 0.2 * norm(slap))), 30.0, 2), 0.9)


def make_rewind(rng, dur=0.45):
    """Out of the cold open: a fluttering tape-rewind whoosh with a rising zip."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = shaped_noise(dur, lambda tt: 600 + 6000 * (tt / dur) ** 1.5, lambda tt: 0.5 + 0 * tt,
                     lambda tt: (tt / dur) ** 1.5, rng)
    x = x[:n] * (1 + 0.5 * np.sin(2 * np.pi * (18 + 40 * t / dur) * t))
    zp = np.sin(2 * np.pi * np.cumsum(200 + 3000 * (t / dur) ** 2) / SR) * (t / dur) ** 2
    return norm(norm(x) + 0.3 * zp, 0.8)


def make_hiss(rng, dur):
    """The creeper's hiss: a soft, breathy 'ssss' that swells."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 3500, 11000, 2) + 0.5 * bp(rng.standard_normal(n), 1800, 3500, 2)
    x *= 0.85 + 0.15 * np.sin(2 * np.pi * 6.5 * t)
    return norm(x, 0.7)


# ---------------------------------------------------------------------------------------------
# the black hole
# ---------------------------------------------------------------------------------------------
def hole_hum(Renv, rng):
    """The hole's voice from its radius over time (per sample): a thin, eerie whine while it is tiny, growing
    into a throbbing sub roar with a whirring, swirling body as it gets huge."""
    n = len(Renv)
    t = np.arange(n) / SR
    s = np.clip(Renv / 9.0, 0.0, 1.0)
    on = (Renv > 1e-3).astype(float)
    on = lp(on, 8.0, 1)
    a = np.sqrt(s) * on
    f = 64.0 - 22.0 * np.sqrt(s)
    ph = 2 * np.pi * np.cumsum(f) / SR
    wob = 1.0 + 0.25 * np.sin(2 * np.pi * (0.9 + 1.2 * s) * t) * np.sin(2 * np.pi * 0.23 * t + 1.0)
    sub = (np.sin(ph) + 0.55 * np.sin(2 * ph + 0.3) + 0.32 * np.sin(3 * ph + 1.1) + 0.2 * np.sin(4 * ph + 2.0))
    sub = np.tanh(1.6 * sub) * a * wob
    whirr = bp(rng.standard_normal(n), 180, 900, 2) * (0.55 + 0.45 * np.sin(2 * np.pi * np.cumsum(0.8 + 2.2 * (1 - s)) / SR))
    whirr = whirr * a
    fw = 1500.0 / (1.0 + 1.6 * Renv) + 180.0
    whine = np.sin(2 * np.pi * np.cumsum(fw * (1 + 0.012 * np.sin(2 * np.pi * 5.2 * t))) / SR)
    whine = whine * on * np.clip(1.0 - s * 1.6, 0.0, 1.0) * 0.5
    return norm(sub, 1.0), norm(whirr, 1.0), whine


def make_air(rng, dur):
    """Air rushing into the hole: a wide, swirling noise bed."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 250, 2600, 2)
    sw = 0.65 + 0.35 * np.sin(2 * np.pi * 0.55 * t) * np.sin(2 * np.pi * 0.17 * t + 0.8)
    gust = lp(rng.standard_normal(n), 1.5, 1)
    gust = 0.7 + 0.3 * (gust - gust.min()) / (np.ptp(gust) + 1e-9)
    return norm(x * sw * gust, 0.8)


def make_spawn(rng):
    """It pops into existence: air sucked backwards into a deep 'thoom', with a glassy shimmer. Returns
    (sound, pre-roll seconds)."""
    pre = 0.55
    suck = shaped_noise(pre, lambda t: 5000 * np.exp(-(pre - t) / 0.2) + 300, lambda t: 0.7 + 0 * t,
                        lambda t: (t / pre) ** 3, rng)
    dur = 2.4
    n = int(dur * SR)
    t = np.arange(n) / SR
    thoom = sine_sweep(dur, 160, 38, 0.12) * expenv(n, 0.6, 0.003)
    shimmer = modal(dur, 1250.0, (1.0, 1.41, 2.03, 2.76, 3.9), (1.0, 0.7, 0.5, 0.3, 0.2),
                    (0.9, 0.7, 0.5, 0.35, 0.25), rng, 0.01)
    shimmer *= 1.0 + 0.3 * np.sin(2 * np.pi * 7.0 * t)
    pop = bp(rng.standard_normal(n), 800, 5000) * expenv(n, 0.01)
    body = norm(thoom) + 0.28 * norm(shimmer) + 0.35 * norm(pop)
    s = int(pre * SR)
    x = np.zeros(s + n)
    x[:len(suck)] += 0.6 * norm(suck)
    x[s:] += body
    return norm(hp(np.tanh(1.2 * x), 30.0, 2), 0.9), pre


def make_whomp(rng, depth=1.0):
    """The hole grows a size: a sucked-in pre-swell, a huge detuned sub drop and a gritty body. Returns
    (sound, pre-roll seconds)."""
    pre = 0.35
    swell = shaped_noise(pre, lambda t: 300 + 3000 * (t / pre) ** 2, lambda t: 0.8 + 0 * t,
                         lambda t: (t / pre) ** 2.5, rng)
    dur = 2.6
    n = int(dur * SR)
    f1 = 36.0 - 6.0 * depth
    drop = sine_sweep(dur, 190 - 30 * depth, f1, 0.18) + 0.5 * sine_sweep(dur, 380 - 60 * depth, 2 * f1, 0.15)
    drop *= expenv(n, 0.7 + 0.4 * depth, 0.004)
    grit = bp(rng.standard_normal(n), 90, 700, 2) * expenv(n, 0.35 + 0.2 * depth, 0.003)
    crack = bp(rng.standard_normal(n), 1500, 9000) * expenv(n, 0.012)
    tail = shaped_noise(dur, lambda tt: 200 + 1500 * np.exp(-tt / 0.4), lambda tt: 1.0 + 0 * tt,
                        lambda tt: np.exp(-tt / (0.8 + 0.5 * depth)) * np.clip(tt / 0.02, 0, 1), rng)
    body = norm(drop) + 0.55 * norm(grit) + 0.3 * norm(crack) + 0.4 * norm(tail)
    s = int(pre * SR)
    x = np.zeros(s + n)
    x[:len(swell)] += 0.5 * norm(swell)
    x[s:] += body
    return norm(hp(np.tanh(1.5 * x), 28.0, 2), 0.95), pre


def make_block_break(rng):
    """A block torn out of the ground: a short gritty crunch (dirt, grass, stone)."""
    dur = 0.18
    n = int(dur * SR)
    lo, hi = (220, 2200) if rng.random() < 0.6 else (500, 4200)
    x = grains(dur, int(rng.integers(14, 30)), lo, hi, 0.03, rng, glen=(0.002, 0.008))
    x += 0.5 * bp(rng.standard_normal(n), 120, 500) * expenv(n, 0.03, 0.001)
    return norm(x, 0.8)


def make_sizzle(rng, dur):
    """The hot disk: dense crackling like fat in a pan, and a faint hiss."""
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(int(dur * 900)):
        p = int(rng.uniform(0, n - 200))
        L = int(rng.integers(10, 90))
        x[p:p + L] += rng.standard_normal(L) * np.hanning(L) * rng.uniform(0.2, 1.0)
    x = hp(x, 2200, 2)
    hiss = bp(rng.standard_normal(n), 4000, 12000, 2) * 0.25
    return norm(x + hiss, 0.7)


def make_groan(rng, dur, f0=70.0):
    """A giant straining against the pull: slow groaning creaks and a trembling low body."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    k = max(2, int(dur * 1.6))
    for _ in range(k):
        s = rng.uniform(0.0, max(0.05, dur - 0.5))
        L = rng.uniform(0.35, 0.9)
        m = (t >= s) & (t < s + L)
        u = (t[m] - s) / L
        f = rng.uniform(0.8, 1.3) * f0 * (1 + 0.35 * u)
        ph = np.cumsum(f) / SR
        saw = (np.mod(ph, 1.0) - 0.5) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(15, 28) * t[m]))
        out[m] += saw * np.sin(np.pi * u) ** 2
    creak = bp(out, 110, 1600, 2)
    tremble = lp(rng.standard_normal(n), 120, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 9.0 * t))
    env = np.clip(t / 0.3, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)
    return norm((norm(creak) + 0.5 * norm(tremble)) * env * (0.5 + 0.5 * t / dur), 0.8)


def make_ground_rip(rng):
    """Torn off the ground: a heavy tear of roots and dirt, a deep thump and a whoosh upwards."""
    dur = 1.6
    n = int(dur * SR)
    tear = grains(dur, 420, 180, 2400, 0.22, rng, glen=(0.002, 0.014))
    thump = sine_sweep(dur, 100, 36, 0.06) * expenv(n, 0.3, 0.002)
    body = bp(rng.standard_normal(n), 70, 450, 2) * expenv(n, 0.25, 0.003)
    up = shaped_noise(dur, lambda tt: 250 + 2500 * np.clip(tt / 0.8, 0, 1) ** 1.5, lambda tt: 0.9 + 0 * tt,
                      lambda tt: np.clip(tt / 0.15, 0, 1) * np.exp(-tt / 0.6), rng)
    x = norm(tear) + 0.9 * norm(thump) + 0.6 * norm(body) + 0.45 * norm(up)
    return norm(hp(np.tanh(1.5 * x), 30.0, 2), 0.9)


def make_stretch(rng, dur):
    """Spaghettification: a rubbery, creaking stretch whose pitch sags lower and lower, with fibres snapping."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    f = 190.0 * (1 - 0.7 * u ** 1.3) + 25.0 * np.sin(2 * np.pi * 3.0 * t) * u
    ph = np.cumsum(f) / SR
    saw = 2 * np.mod(ph, 1.0) - 1
    fm = 0.5 + 0.5 * np.sin(2 * np.pi * (11 + 6 * u) * t)
    x = bp(saw, 150, 1200, 2) * (0.5 + 0.5 * fm)
    snaps = grains(dur, int(dur * 40), 900, 5000, dur * 0.6, rng, glen=(0.001, 0.004))
    env = np.clip(t / 0.15, 0, 1) * np.clip((dur - t) / 0.08, 0, 1) * (0.4 + 0.6 * u)
    return norm((norm(x) + 0.35 * norm(snaps)) * env, 0.8)


def make_gulp(rng, size=1.0):
    """Swallowed: a rushing inhale that cuts off into a deep, wet 'gulp' and a sub drop."""
    pre = 0.4
    inhale = shaped_noise(pre, lambda t: 400 + 3500 * (t / pre) ** 2, lambda t: 0.7 + 0 * t,
                          lambda t: (t / pre) ** 2.5, rng)
    dur = 1.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    gl = np.sin(2 * np.pi * np.cumsum(260 * np.exp(-t / 0.05) + 55) / SR) * expenv(n, 0.14, 0.002)
    form = bp(gl, 150, 900, 2)
    sub = sine_sweep(dur, 90, 28, 0.1) * expenv(n, 0.5 + 0.2 * size, 0.004)
    x = np.zeros(int(pre * SR) + n)
    x[:len(inhale)] += 0.5 * norm(inhale)
    s = int(pre * SR)
    x[s:] += norm(form) * 0.8 + norm(sub)
    return norm(hp(np.tanh(1.4 * x), 26.0, 2), 0.9), pre


def make_implosion(rng, dur):
    """The collapse: everything sucked in, rising and tightening into a point, then cut dead."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    whine = np.sin(2 * np.pi * np.cumsum(90 + 900 * u ** 2.5) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(180 + 1900 * u ** 2.5) / SR)
    rush = shaped_noise(dur, lambda tt: 200 + 6000 * (tt / dur) ** 2.2, lambda tt: 0.8 + 0 * tt,
                        lambda tt: (tt / dur) ** 1.5, rng)
    sub = np.sin(2 * np.pi * np.cumsum(40 + 60 * u ** 2) / SR)
    x = norm(whine * u ** 1.6) * 0.6 + norm(rush) + 0.7 * norm(sub * (0.3 + 0.7 * u))
    x *= np.clip((dur - t) / 0.004, 0, 1)
    return norm(x, 0.9)


def make_final_boom(rng):
    """The explosion at the end: a crack, a gigantic sub drop, a long roaring body and debris crackle."""
    dur = 5.0
    n = int(dur * SR)
    big = make_explosion(rng, 4.0)
    big = np.concatenate([big, np.zeros(max(0, n - len(big)))])[:n]
    drop = sine_sweep(dur, 150, 26, 0.35) * expenv(n, 1.4, 0.003)
    roar = shaped_noise(dur, lambda tt: 300 + 2500 * np.exp(-tt / 0.6), lambda tt: 1.2 + 0 * tt,
                        lambda tt: np.exp(-tt / 1.3) * np.clip(tt / 0.01, 0, 1), rng)
    debris = grains(dur, 700, 500, 4500, 1.0, rng, glen=(0.002, 0.012), decay=1.6)
    x = norm(big) + 0.9 * norm(drop) + 0.6 * norm(roar) + 0.3 * norm(debris)
    return norm(hp(np.tanh(1.3 * x), 26.0, 2), 0.99)


def make_block_land(rng):
    """A block (or a chunk of a giant) landing back on the ground."""
    dur = 0.3
    n = int(dur * SR)
    thud = sine_sweep(dur, rng.uniform(130, 190), 70, 0.02) * expenv(n, 0.05, 0.001)
    grit = grains(dur, int(rng.integers(8, 18)), 300, 2500, 0.03, rng, glen=(0.002, 0.008))
    return norm(norm(thud) + 0.6 * norm(grit), 0.8)


# ---------------------------------------------------------------------------------------------
# the score: a cathedral organ and a ticking clock
# ---------------------------------------------------------------------------------------------
NOTE = {'A1': 55.0, 'C#2': 69.3, 'D2': 73.42, 'E2': 82.41, 'F2': 87.31, 'G#2': 103.83, 'A2': 110.0,
        'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G#3': 207.65, 'A3': 220.0,
        'B2': 123.47, 'B3': 246.94, 'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'A4': 440.0, 'E1': 41.2, 'F1': 43.65, 'D1': 36.71}
CHORDS = {
    'pedal': ['A2', 'E3'],
    'Am': ['A2', 'C3', 'E3', 'A3', 'C4'],
    'F': ['F2', 'A2', 'C3', 'F3', 'A3', 'C4'],
    'Dm': ['D2', 'A2', 'D3', 'F3', 'A3', 'D4'],
    'E': ['E2', 'B2', 'E3', 'G#3', 'B3', 'E4'],
    'A': ['A2', 'E3', 'A3', 'C#4', 'E4', 'A4'],
}
STOPS = ((1.0, 1.0), (2.0, 0.7), (3.0, 0.36), (4.0, 0.4), (6.0, 0.16), (8.0, 0.12))


def organ_voice(freqs, n, rng, bright=1.0):
    """Sustained organ tone (additive pipes with a slight chorus and a slow tremulant) for a set of notes."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    trem = 1.0 + 0.025 * np.sin(2 * np.pi * 5.3 * t + rng.uniform(0, 6))
    for f in freqs:
        for k, (r, a) in enumerate(STOPS):
            if f * r > 5000:
                continue
            a = a * (bright if r >= 3 else 1.0)
            for d in (-0.0012, 0.0011):
                x += a * np.sin(2 * np.pi * f * r * (1 + d) * t + rng.uniform(0, 6))
    return x * trem / max(1, len(freqs))


def build_score(sections, n, rng):
    """sections: list of (t0, t1, chord, level, bright). Each chord swells in and hands over to the next."""
    out = np.zeros(n)
    for t0, t1, chord, level, bright in sections:
        s0 = max(0, int(t0 * SR))
        s1 = min(n, int(t1 * SR) + int(0.6 * SR))
        m = s1 - s0
        if m <= 0:
            continue
        v = organ_voice([NOTE[k] for k in CHORDS[chord]], m, rng, bright)
        tt = np.arange(m) / SR
        env = np.clip(tt / 0.35, 0, 1) * np.clip(((t1 - t0) + 0.6 - tt) / 0.6, 0, 1)
        out[s0:s1] += v * env * level
    return out


def make_clock(rng, times, gains):
    """The ticking clock under the score: a dry woodblock tick per beat."""
    ticks = [make_tick(rng) for _ in range(6)]
    n = int((max(times) + 1.0) * SR) if len(times) else SR
    x = np.zeros(n)
    for tm, g in zip(times, gains):
        s = int(tm * SR)
        b = ticks[rng.integers(len(ticks))]
        e = min(n, s + len(b))
        x[s:e] += b[:e - s] * g
    return x


# ---------------------------------------------------------------------------------------------
# mixing
# ---------------------------------------------------------------------------------------------
class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, x, t, gain=1.0, pan=0.0):
        s = int(t * SR)
        if s >= self.n or s + len(x) <= 0:
            return
        if s < 0:
            x = x[-s:]
            s = 0
        e = min(self.n, s + len(x))
        pan = float(np.clip(pan, -1, 1))
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        self.L[s:e] += x[:e - s] * gain * gl
        self.R[s:e] += x[:e - s] * gain * gr


def k_weight(x):
    # ITU-R BS.1770 pre-filter (high shelf) + RLB high-pass, coefficients for 48 kHz
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x))


def integrated_lufs(L, R):
    kl, kr = k_weight(L), k_weight(R)
    blk = int(0.4 * SR)
    hop = int(0.1 * SR)
    ms = []
    for s in range(0, len(kl) - blk, hop):
        ms.append(np.mean(kl[s:s + blk] ** 2) + np.mean(kr[s:s + blk] ** 2))
    ms = np.array(ms)
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    if len(g) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = g[(-0.691 + 10 * np.log10(g + 1e-12)) > rel]
    return -0.691 + 10 * np.log10(g2.mean())


def limiter(L, R, ceiling=0.89, release=0.08):
    peak = np.maximum(np.abs(L), np.abs(R))
    la = int(0.002 * SR)
    g = np.ones_like(peak)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * la + 1)
    a = np.exp(-1.0 / (release * SR))
    cur = 1.0
    for i in range(len(need)):
        cur = need[i] if need[i] < cur else cur * a + need[i] * (1 - a)
        g[i] = cur
    return L * g, R * g


def _pan_gain(pos, cam, tgt):
    if pos is None or cam is None:
        return 0.0, 1.0
    cam = np.asarray(cam, float)
    tgt = np.asarray(tgt, float)
    fwd = tgt - cam
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.array([fwd[1], -fwd[0], 0.0])
    d = np.asarray(pos, float) - cam
    dist = np.linalg.norm(d)
    x = np.dot(d, right) / (dist + 1e-9)
    pan = np.clip(x * 1.4, -0.85, 0.85)
    gain = 1.0 / (1.0 + max(0.0, dist - 30.0) / 50.0)
    return pan, gain


def _env_track(values, fps, n, smooth_hz):
    """Per-frame values -> smoothed per-sample envelope of length n."""
    v = np.interp(np.arange(n) / SR * fps, np.arange(len(values)), values)
    return np.clip(lp(np.concatenate([v, np.zeros(SR)])[:n], smooth_hz, 1), 0.0, None)


# ---------------------------------------------------------------------------------------------
# build from cues
# ---------------------------------------------------------------------------------------------
def build(cues, fps, out_path, seed=7):
    rng = np.random.default_rng(seed)
    nfr = len(cues)
    dur = nfr / fps
    mix = Mix(dur + 5.0)                       # effects
    verb = Mix(dur + 5.0)                      # reverb send (slow motion, big moments)
    music = Mix(dur + 5.0)                     # the score, ducked under the effects
    n_all = int((dur + 5.0) * SR)

    scale = np.array([c.get('scale', 1.0) for c in cues], float)
    segs = [c['seg'] for c in cues]
    shots = [c['shot'] for c in cues]
    Rf = np.array([c.get('R', 0.0) for c in cues], float)
    stage = np.array([c.get('stage', -1) for c in cues], int)
    main = np.array([s == 'main' for s in segs])
    n_cold = int((~main).sum())

    def ev_n(name):
        return np.array([c['events'].get(name, [0])[0] for c in cues], float)

    def env(values, hz=5.0):
        return _env_track(np.asarray(values, float), fps, n_all, hz)

    boom_f = next((i for i in range(nfr) if main[i] and 'explode' in cues[i]['events']), None)
    col_f = next((i for i in range(nfr) if main[i] and 'collapse' in cues[i]['events']), None)
    silent = np.zeros(nfr)                        # the dead air between the collapse and the explosion
    if boom_f is not None:
        gone = next((i for i in range(col_f or 0, boom_f) if Rf[i] <= 0.0), boom_f)
        silent[gone:boom_f] = 1.0
    quiet = 1.0 - env(silent, 40.0).clip(0, 1)

    # ---- ambience: a calm day that the hole slowly swallows
    amb = np.clip(1.0 - Rf / 3.0, 0.15, 1.0)
    amb[silent > 0] = 0.0
    if boom_f is not None:
        amb[boom_f:] = np.clip(np.linspace(0.0, 1.0, nfr - boom_f) * 3.0, 0, 0.6)
    ev_amb = make_evening(rng, dur + 5.0)
    mix.add(ev_amb * env(amb, 2.0), 0.0, 0.5, pan=-0.3)
    mix.add(make_evening(rng, dur + 5.0) * env(amb, 2.0), 0.0, 0.5, pan=0.3)

    # ---- the hole: hum, whirr, whine, rushing air (slow motion pitches it down a little)
    Renv = env(Rf, 12.0)
    sub, whirr, whine = hole_hum(Renv, rng)
    near = np.array([np.clip(35.0 / (c.get('hole_dist', 60.0) + 5.0), 0.5, 1.4) for c in cues])
    ne = env(near, 3.0)
    mix.add(hp(sub, 45.0, 2) * ne * quiet, 0.0, 0.32)
    mix.add(whirr * ne * quiet, 0.0, 0.32)
    mix.add(whine * ne * quiet, 0.0, 0.16)
    peel = ev_n('peel')
    air_lev = np.clip(0.25 * np.sqrt(Rf / 9.0) + 0.6 * np.log1p(peel) / np.log1p(400), 0, 1) * (Rf > 0)
    mix.add(make_air(rng, dur + 5.0) * env(air_lev, 4.0) * quiet, 0.0, 0.4)
    mix.add(make_air(rng, dur + 5.0) * env(air_lev, 4.0) * quiet, 0.0, 0.3, pan=0.5)

    # ---- blocks torn out of the ground, the disk sizzling as they heat up and vanish
    rt = np.clip((scale - 0.5) / 0.4, 0.2, 1.0)
    e = np.clip(np.log1p(peel) / np.log1p(300), 0, 1) * rt
    if e.max() > 0.01:
        mix.add(make_hail(rng, dur + 5.0, make_block_break, 70.0) * env(e, 8.0), 0.0, 0.42)
    eat = ev_n('eat') + ev_n('eat_small') * 0.1 + ev_n('eat_prop')
    nb = np.array([c.get('n_blocks', 0) + 0.2 * c.get('n_debris', 0) for c in cues], float)
    sz = np.clip(0.5 * np.log1p(eat) / np.log1p(200) + 0.5 * np.log1p(nb) / np.log1p(8000), 0, 1) * (Rf > 0)
    mix.add(make_sizzle(rng, dur + 5.0) * env(sz, 5.0) * quiet, 0.0, 0.2)

    # ---- after the explosion: everything raining back down
    land = ev_n('land')
    e = np.clip(np.log1p(land) / np.log1p(250), 0, 1) * rt
    if e.max() > 0.01:
        mix.add(make_hail(rng, dur + 5.0, make_block_land, 90.0) * env(e, 10.0), 0.0, 0.5)

    # ---- the cold open: a riser under it, a rewind out of it, a hit on the first frame of the story
    if n_cold:
        mix.add(hp(make_riser(rng, n_cold / fps), 180.0, 2), 0.0, 0.3)
        rw = make_rewind(rng)
        mix.add(rw, n_cold / fps - len(rw) / SR, 0.5)
        mix.add(make_hit(rng), n_cold / fps, 0.5)

    # ---- the creeper's hiss, from when it starts to flash until it goes off
    for seg in ('cold', 'main'):
        idx = [i for i, c in enumerate(cues) if segs[i] == seg and any(
            b['name'] == 'creeper' and b.get('white', 0.0) > 0.0 for b in c.get('bodies', []))]
        if idx:
            a, b = idx[0], idx[-1] + 1
            h = make_hiss(rng, (b - a) / fps + 0.1)
            u = np.linspace(0, 1, len(h))
            mix.add(h * (0.35 + 0.65 * u ** 1.5) * np.clip((len(h) - np.arange(len(h))) / (0.05 * SR), 0, 1),
                    a / fps, 0.45)

    # ---- the giants: groaning while they lean, stretched while they fly in
    for seg in ('cold', 'main'):
        for name, f0 in (('creeper', 85.0), ('zombie', 70.0), ('steve', 78.0), ('warden', 52.0)):
            lean = [i for i, c in enumerate(cues) if segs[i] == seg and any(
                b['name'] == name and b['state'] == 'lean' for b in c.get('bodies', []))]
            if len(lean) > 5:
                a, b = lean[0], lean[-1] + 1
                g = make_groan(rng, (b - a) / fps + 0.2, f0)
                mix.add(g, a / fps, 0.4)
                if name == 'warden':
                    mix.add(make_growl(rng, min(2.0, (b - a) / fps), 44.0), a / fps + 0.1, 0.35)
            st = [i for i, c in enumerate(cues) if segs[i] == seg and any(
                b['name'] == name and b['state'] == 'fly' and b['stretch'] > 1.25 for b in c.get('bodies', []))]
            if len(st) > 3:
                a, b = st[0], st[-1] + 1
                pf = 0.55 + 0.45 * float(np.mean(scale[a:b]))
                s = resample(make_stretch(rng, (b - a) / fps * pf + 0.1), pf)
                mix.add(s, a / fps, 0.4)
                verb.add(s, a / fps, 0.25)
            fl = [i for i, c in enumerate(cues) if segs[i] == seg and any(
                b['name'] == name and b['state'] == 'fly' for b in c.get('bodies', []))]
            if len(fl) > 5:
                a, b = fl[0], fl[-1] + 1
                w = make_whoosh(rng, (b - a) / fps + 0.05, 0.8)
                mix.add(w, a / fps, 0.35)

    # ---- slow motion: pitch drop in, rush out
    slow = (scale < 0.55) & main
    s_down, s_up = make_slowmo_down(rng), make_slowmo_up(rng)
    for f in range(1, nfr):
        if slow[f] and not slow[f - 1]:
            mix.add(s_down, f / fps, 0.45)
        if slow[f - 1] and not slow[f]:
            mix.add(s_up, f / fps - 0.25, 0.3)

    # ---- per-frame events
    booms = [make_explosion(rng, 1.0) for _ in range(5)]
    clangs = [make_clang(rng) for _ in range(5)]
    last_tnt = -99
    for i, c in enumerate(cues):
        t = i / fps
        ev = c['events']
        cam = c.get('cam')
        tgt = c.get('tgt', [0.0, 0.0, 15.0])
        pf = 0.55 + 0.45 * scale[i]
        wet = float(np.clip((0.9 - scale[i]) / 0.5, 0.0, 1.0))

        def at(name):
            e_ = ev.get(name)
            return _pan_gain(e_[1] if e_ else None, cam, tgt)

        if 'spawn' in ev:
            s, pre = make_spawn(rng)
            mix.add(resample(s, pf), t - pre, 0.85)
            verb.add(s, t - pre, 0.5)
        if 'stage' in ev and 'spawn' not in ev:
            depth = min(1.0, stage[i] / 3.0)
            s, pre = make_whomp(rng, depth)
            mix.add(resample(s, pf), t - pre, 0.8 + 0.2 * depth)
            verb.add(s, t - pre, 0.4)
        if 'tnt_boom' in ev and i - last_tnt > 2:
            pan, g = at('tnt_boom')
            s = resample(booms[rng.integers(len(booms))], pf * rng.uniform(0.9, 1.1))
            mix.add(s, t, 0.6 * max(g, 0.5), pan)
            verb.add(s, t, 0.3 * wet)
            last_tnt = i
        if 'prop_lift' in ev and rng.random() < 0.5:
            pan, g = at('prop_lift')
            mix.add(resample(clangs[rng.integers(len(clangs))], pf * rng.uniform(0.7, 0.9)), t, 0.2 * g, pan)
        if 'lift' in ev:
            pan, g = at('lift')
            s = resample(make_ground_rip(rng), pf)
            mix.add(s, t, 0.9 * max(g, 0.6), pan)
            verb.add(s, t, 0.3)
        if 'sever' in ev:
            pan, g = at('sever')
            s = resample(make_rip(rng), pf)
            mix.add(s, t, 0.8 * max(g, 0.6), pan)
            verb.add(s, t, 0.35)
        if 'creeper_boom' in ev:
            pan, g = at('creeper_boom')
            b = resample(make_explosion(rng, 2.6), pf)
            mix.add(b, t, 1.0 * max(g, 0.7), pan)
            verb.add(b, t, 0.5)
            # ... and the blast is sucked back in
            sk = make_whoosh(rng, 1.1, 0.6)
            mix.add(resample(sk, pf), t + 0.35 / pf, 0.45, pan)
        if 'eaten' in ev and 'creeper_boom' not in ev:
            pan, g = at('eaten')
            s, pre = make_gulp(rng, 1.0)
            mix.add(resample(s, pf), t - pre / pf, 0.9, pan)
            verb.add(s, t - pre / pf, 0.4)
        if 'boom' in ev:
            ch = resample(make_boom_charge(rng), pf)
            mix.add(ch, t - len(ch) / SR, 0.55)
            b = resample(make_sonic_boom(rng), pf)
            mix.add(b, t, 0.95)
            verb.add(b, t, 0.45)
            s, pre = make_gulp(rng, 0.6)
            mix.add(resample(s, pf), t + 0.6 / max(pf, 0.3) - pre, 0.6)
        if 'collapse' in ev and main[i] and i == col_f:
            gone = next((k for k in range(i, nfr) if Rf[k] <= 0.0), i + int(fps))
            d = max(0.3, (gone - i) / fps)
            s = make_implosion(rng, d)
            mix.add(s, t, 0.9)
        if 'explode' in ev:
            b = make_final_boom(rng)
            mix.add(b, t, 1.25)
            verb.add(b, t, 0.6)
            music.add(make_braam(rng, 3.0, 36.7), t, 0.8)
            music.add(make_taiko(rng), t, 0.7)
        hud = c.get('hud', {})
        if 'banner' in hud:
            music.add(make_taiko(rng), t + 0.02, 0.45)
        if 'stamp' in hud:
            music.add(make_braam(rng, 1.6, 41.2), t, 0.45)

    # ---- the score: organ chords by SIZE, the dominant through the end, A major on the explosion
    def vt(f):
        return f / fps

    first = {}
    for i in range(nfr):
        if main[i] and stage[i] >= 0 and stage[i] not in first:
            first[stage[i]] = i
    w_boom = next((i for i in range(nfr) if main[i] and 'boom' in cues[i]['events']), None)
    end_main = nfr
    secs = []
    if n_cold:
        secs.append((0.0, vt(n_cold), 'E', 0.55, 0.7))
    order = [(0, 'pedal', 0.45, 0.5), (1, 'Am', 0.6, 0.6), (2, 'F', 0.75, 0.8), (3, 'Dm', 0.9, 1.0)]
    for k, (st_, chord, lev, br) in enumerate(order):
        if st_ not in first:
            continue
        t0 = vt(first[st_])
        nxt = [first[s2] for s2, *_ in order[k + 1:] if s2 in first]
        t1 = vt(nxt[0]) if nxt else vt(w_boom or boom_f or end_main)
        if st_ == 3 and w_boom is not None:
            t1 = vt(w_boom)
        secs.append((t0, t1, chord, lev, br))
    if w_boom is not None:
        gone = next((k for k in range(col_f or w_boom, nfr) if Rf[k] <= 0.0 and k > (col_f or 0)), boom_f or nfr)
        secs.append((vt(w_boom), vt(gone), 'E', 1.0, 1.1))
    if boom_f is not None:
        secs.append((vt(boom_f), vt(nfr) + 1.0, 'A', 1.1, 1.2))
    score = build_score(secs, n_all, rng)
    # the collapse swells the organ up into the silence
    if col_f is not None and boom_f is not None:
        sw = np.ones(nfr)
        for i in range(col_f, boom_f):
            sw[i] = 1.0 + 0.8 * (i - col_f) / max(1, boom_f - col_f)
        score = score * env(sw, 20.0) * quiet
    # the end: fade the chord down under the reborn hole, a low note stays
    end_f = next((i for i in range(nfr) if main[i] and shots[i] == 'reborn'), None)
    if end_f is not None:
        fade = np.ones(nfr)
        fade[end_f:] = np.linspace(1.0, 0.35, nfr - end_f)
        score = score * env(fade, 4.0)
    ir_big = reverb_ir(rng, 3.2, 1.1, 3800)
    wet_score = signal.fftconvolve(score, ir_big)[:n_all]
    sc = lp(score, 5200, 2) * 0.6 + wet_score * 0.9
    music.add(norm(sc, 0.8), 0.0, 0.42)
    # the clock: a tick every beat while the hole is there, faster as it grows
    ticks, gains = [], []
    tm = vt(first.get(0, nfr))
    stop = vt(col_f) if col_f is not None else dur
    while tm < stop:
        f = int(tm * fps)
        sg = stage[min(f, nfr - 1)]
        ticks.append(tm)
        gains.append(0.35 + 0.15 * max(sg, 0))
        tm += 0.6 if sg < 2 else 0.45
    if ticks:
        music.add(make_clock(rng, ticks, gains), 0.0, 0.2)

    n = int(dur * SR)
    ir = reverb_ir(rng)
    wl = signal.fftconvolve(verb.L[:n], ir)[:n]
    wr = signal.fftconvolve(verb.R[:n], ir)[:n]
    L, R = mix.L[:n] + wl, mix.R[:n] + wr
    fx_env = np.sqrt(lp(0.5 * (L * L + R * R), 6.0, 1).clip(0))
    ref = np.percentile(fx_env, 95) + 1e-9
    duck = np.clip(1.0 - 0.55 * fx_env / ref, 0.4, 1.0)
    L = L + music.L[:n] * duck
    R = R + music.R[:n] * duck
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    # phones can't play the deep sub anyway: a low shelf under 90 Hz and a little presence
    L = L - 0.5 * lp(L, 90.0, 2) + 0.25 * bp(L, 1800.0, 4500.0, 1)
    R = R - 0.5 * lp(R, 90.0, 2) + 0.25 * bp(R, 1800.0, 4500.0, 1)
    lufs = integrated_lufs(L, R)
    g0 = 10 ** ((-18.0 - lufs) / 20.0)
    L, R = compress(L * g0, R * g0, thresh_db=-24.0, ratio=3.0)
    gain = 10 ** ((-14.0 - integrated_lufs(L, R)) / 20.0)
    L, R = L * gain, R * gain
    L, R = limiter(L, R, ceiling=10 ** (-1.2 / 20))
    lufs2 = integrated_lufs(L, R)
    f = int(0.01 * SR)
    for ch_ in (L, R):
        ch_[:f] *= np.linspace(0, 1, f)
        ch_[-f:] *= np.linspace(1, 0, f)
    data = np.stack([L, R], -1)
    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f'audio: {dur:.2f}s  loudness before {lufs:.1f} LUFS -> after {lufs2:.1f} LUFS, '
          f'peak {20 * np.log10(np.abs(data).max() + 1e-9):.1f} dBFS', flush=True)
    return out_path


def loudness_report(path, win=1.0):
    """Short-term loudness (LUFS) per window of a 16-bit stereo WAV (for checking the mix balance)."""
    with wave.open(path) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768.0
    L, R = k_weight(d[:, 0]), k_weight(d[:, 1])
    n = int(win * SR)
    out = []
    for s in range(0, len(L) - n + 1, n):
        ms = np.mean(L[s:s + n] ** 2) + np.mean(R[s:s + n] ** 2)
        out.append(-0.691 + 10 * np.log10(ms + 1e-12))
    return np.array(out)


if __name__ == '__main__':
    import sys
    d = json.load(open(sys.argv[1]))
    build(d['frames'], d['fps'], sys.argv[2])
