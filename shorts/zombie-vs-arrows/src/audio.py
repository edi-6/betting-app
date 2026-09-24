"""Procedural sound design driven by the simulation cue sheet.

Everything is synthesised here (no samples): bowstring twangs and distant volley releases, arrows whistling
past the lens, the roar of a sky full of arrows, flesh thunks, bone cracks and wet spray, ground thuds and the
patter of a hail of arrows, arrows clattering down, severed parts landing, a slow-motion pitch drop with a
heartbeat, wind rush on the arrow-cam, UI pops/stamps and a light wind bed. Mixed in stereo with panning from
the 3D event positions, slow-motion hits get a reverb send, then the mix is loudness-normalised (BS.1770 style)
to -14 LUFS with a peak limiter.
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
    m = np.abs(x).max()
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


# ---------------------------------------------------------------------------------------------
# sound banks
# ---------------------------------------------------------------------------------------------
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


def make_pop(rng):
    dur = 0.28
    n = int(dur * SR)
    sw = shaped_noise(dur, lambda t: 900 + 5000 * (t / dur) ** 1.5, lambda t: 0.7 + 0 * t,
                      lambda t: np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2, rng)
    click = np.sin(2 * np.pi * np.cumsum(1400 - 900 * np.linspace(0, 1, n)) / SR) * expenv(n, 0.018)
    return norm(0.6 * norm(sw) + 0.5 * click, 0.6)


def make_x_sound(rng):
    """Short, soft 'nope' buzzer: two descending square-ish notes, low-passed."""
    out = []
    for f, d in ((392.0, 0.12), (294.0, 0.2)):
        n = int(d * SR)
        t = np.arange(n) / SR
        w = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + 0.5 * np.sin(2 * np.pi * f * t)
        w = lp(w, 2200) * np.minimum(1, t / 0.005) * np.minimum(1, (d - t) / 0.03)
        out.append(w)
        out.append(np.zeros(int(0.02 * SR)))
    x = np.concatenate(out)
    thump = sine_sweep(len(x) / SR, 120, 60, 0.05) * expenv(len(x), 0.08)
    return norm(0.8 * norm(x) + 0.5 * thump, 0.6)


def make_check_sound(rng):
    """Bright two-note chime (major sixth up) with bell-like partials."""
    dur = 1.4
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for start, f in ((0.0, 1318.5), (0.11, 2217.5)):
        s = int(start * SR)
        tt = t[:n - s]
        v = np.zeros(n - s)
        for ratio, a, tau in ((1.0, 1.0, 0.55), (2.0, 0.25, 0.25), (3.01, 0.12, 0.12), (4.2, 0.06, 0.08)):
            v += a * np.sin(2 * np.pi * f * ratio * tt) * np.exp(-tt / tau)
        v *= np.minimum(1, tt / 0.002)
        x[s:] += v
    sparkle = bp(rng.standard_normal(n), 5000, 12000) * expenv(n, 0.05) * 0.05
    return norm(x + sparkle, 0.55)


def make_hurt(rng):
    """Punchy 'hit taken' cue: low body thump + short mid knock (no voice)."""
    dur = 0.3
    n = int(dur * SR)
    thump = sine_sweep(dur, 190, 75, 0.03) * expenv(n, 0.07, 0.001)
    knock = bp(rng.standard_normal(n), 350, 1400) * expenv(n, 0.025)
    x = norm(thump) + 0.45 * norm(knock)
    return norm(np.tanh(1.5 * x), 0.7)

# ---------------------------------------------------------------------------------------------
# arrows
# ---------------------------------------------------------------------------------------------
def pluck(f0, dur, rng, decay=0.996, bright=0.5):
    """Karplus-Strong plucked string."""
    n = int(dur * SR)
    L = max(2, int(SR / f0))
    buf = rng.uniform(-1, 1, L)
    buf = lp(buf, 1500 + 6000 * bright, 1)
    out = np.zeros(n)
    idx = 0
    for i in range(n):
        v = buf[idx]
        nxt = buf[(idx + 1) % L]
        buf[idx] = decay * 0.5 * (v + nxt)
        out[i] = v
        idx = (idx + 1) % L
    return out


def make_twang(rng):
    """Bow release: a short bowstring twang under a quick air 'fwip'."""
    dur = 0.45
    n = int(dur * SR)
    string = pluck(rng.uniform(105, 135), dur, rng, decay=0.993, bright=0.35) * expenv(n, 0.09)
    swish = shaped_noise(dur, lambda t: 5200 * np.exp(-t / 0.05) + 900, lambda t: 0.8 + 0 * t,
                         lambda t: np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.045), rng)
    knock = bp(rng.standard_normal(n), 200, 900) * expenv(n, 0.012)
    return norm(0.8 * norm(string) + 0.75 * norm(swish) + 0.3 * norm(knock), 0.8)


def make_volley(rng, n_bows=40):
    """Hundreds of bows released far away: a soft, smeared chorus of twangs and a rising air rush."""
    dur = 1.1
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(n_bows):
        tw = make_twang(rng)
        s = int(rng.exponential(0.1) * SR)
        e = min(n, s + len(tw))
        if s < n:
            x[s:e] += tw[:e - s] * rng.uniform(0.3, 1.0)
    x = lp(x, 2600, 2)
    rush = shaped_noise(dur, lambda t: 700 + 1600 * np.clip(t / 0.4, 0, 1), lambda t: 1.0 + 0 * t,
                        lambda t: np.clip(t / 0.08, 0, 1) * np.exp(-t / 0.5), rng)
    return norm(norm(x) + 0.5 * norm(rush), 0.8)


def make_flyby(rng, dur=0.42):
    """An arrow whistling past the lens: a band of noise sweeping down through the pass (Doppler)."""
    c = dur * 0.45
    x = shaped_noise(dur, lambda t: 1100 + 2600 / (1 + np.exp((t - c) / 0.03)), lambda t: 0.55 + 0 * t,
                     lambda t: np.exp(-((t - c) / 0.07) ** 2), rng)
    n = len(x)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(900 + 700 / (1 + np.exp((t - c) / 0.03))) / SR) \
        * np.exp(-((t - c) / 0.05) ** 2)
    return norm(norm(x) + 0.12 * tone, 0.7)


def make_thunk(rng):
    """Arrow burying itself in flesh: woody knock + meaty low thump + short wet squelch + the head's tick."""
    dur = 0.35
    n = int(dur * SR)
    knock = bp(rng.standard_normal(n), rng.uniform(500, 800), rng.uniform(2200, 3200)) * expenv(n, 0.012, 0.0005)
    thump = sine_sweep(dur, rng.uniform(170, 210), rng.uniform(70, 90), 0.02) * expenv(n, 0.06, 0.001)
    squelch = shaped_noise(dur, lambda t: 900 * np.exp(-t / 0.08) + 350, lambda t: 0.7 + 0 * t,
                           lambda t: np.clip(t / 0.006, 0, 1) * np.exp(-t / 0.05), rng)
    tick = bp(rng.standard_normal(n), 3500, 9000) * expenv(n, 0.002)
    x = 0.9 * norm(knock) + 1.0 * norm(thump) + 0.45 * norm(squelch) + 0.3 * norm(tick)
    return norm(np.tanh(1.6 * x), 0.85)


def make_quiver(rng):
    """The 'dooiinng' of an arrow shaft vibrating after it sticks (13 Hz wobble, like the animation)."""
    dur = 0.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = rng.uniform(150, 190) * (1 + 0.04 * np.sin(2 * np.pi * 13 * t))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.35 * np.sin(2 * np.pi * np.cumsum(2.7 * f) / SR)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 13 * t + 1.0)
    return norm(tone * am * expenv(n, 0.14, 0.004), 0.5)


def make_thud(rng):
    """Arrow sinking into the ground: dull low thud and a little dirt."""
    dur = 0.3
    n = int(dur * SR)
    thud = sine_sweep(dur, rng.uniform(120, 150), rng.uniform(50, 65), 0.02) * expenv(n, 0.045, 0.001)
    body = lp(rng.standard_normal(n), 700, 2) * expenv(n, 0.03)
    dirt = grains(dur, int(rng.integers(8, 16)), 700, 3200, 0.03, rng, glen=(0.002, 0.006))
    return norm(np.tanh(1.3 * (norm(thud) + 0.6 * norm(body) + 0.35 * norm(dirt))), 0.8)


def make_bonecrack(rng):
    dur = 0.25
    n = int(dur * SR)
    snap = bp(rng.standard_normal(n), 1200, 9000) * expenv(n, 0.004)
    res = bp(rng.standard_normal(n), 600, 1100, 3) * expenv(n, 0.03)
    splinter = grains(dur, 14, 1500, 6000, 0.02, rng, glen=(0.001, 0.004))
    return norm(norm(snap) + 0.6 * norm(res) + 0.5 * norm(splinter), 0.8)


def make_squelch(rng, dur=0.4):
    """Wet spray of flesh knocked out of the wound."""
    x = shaped_noise(dur, lambda t: 1100 * np.exp(-t / 0.2) + 300, lambda t: 1.0 + 0 * t,
                     lambda t: np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.1), rng)
    n = len(x)
    t = np.arange(n) / SR
    x *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * rng.uniform(25, 40) * t + rng.uniform(0, 6)))
    x = lp(x, 3000, 2)
    return norm(x + 0.4 * grains(dur, 30, 400, 2200, 0.08, rng), 0.7)


def make_clatter(rng):
    """Loose arrow landing flat: two or three light wooden taps."""
    dur = 0.3
    n = int(dur * SR)
    x = np.zeros(n)
    s = 0
    for k in range(int(rng.integers(2, 4))):
        L = int(0.03 * SR)
        tap = bp(rng.standard_normal(L), 900, 3500) * expenv(L, 0.004) * (0.7 ** k)
        tap += 0.4 * np.sin(2 * np.pi * rng.uniform(600, 900) * np.arange(L) / SR) * expenv(L, 0.008) * (0.7 ** k)
        e = min(n, s + L)
        x[s:e] += tap[:e - s]
        s += int(rng.uniform(0.03, 0.08) * SR)
        if s >= n:
            break
    return norm(x, 0.6)


def make_rip(rng):
    """A limb or the head shot off: tearing crunch over a heavy wet thump."""
    dur = 0.9
    n = int(dur * SR)
    tear = grains(dur, 220, 300, 2600, 0.12, rng, glen=(0.002, 0.012))
    thump = sine_sweep(dur, 120, 45, 0.05) * expenv(n, 0.18, 0.002)
    wet = make_squelch(rng, dur) * 0.8
    crack = bp(rng.standard_normal(n), 1000, 7000) * expenv(n, 0.01)
    x = norm(tear) + 0.9 * norm(thump) + 0.6 * norm(wet) + 0.5 * norm(crack)
    return norm(np.tanh(1.5 * x), 0.9)


def make_land(rng):
    """A giant severed part hitting the ground: heavy thud with a crunch of debris."""
    dur = 1.4
    n = int(dur * SR)
    thud = sine_sweep(dur, 95, 38, 0.06) * expenv(n, 0.22, 0.002)
    body = bp(rng.standard_normal(n), 80, 500, 2) * expenv(n, 0.15, 0.003)
    debris = grains(dur, 160, 400, 3000, 0.2, rng, glen=(0.002, 0.01))
    x = norm(thud) + 0.7 * norm(body) + 0.45 * norm(debris)
    return norm(hp(np.tanh(1.4 * x), 32.0, 2), 0.9)


def make_rush(rng, dur):
    """Wind roaring past the arrow-cam."""
    n = int(dur * SR)
    low = lp(rng.standard_normal(n), 900, 2)
    high = bp(rng.standard_normal(n), 2000, 7000, 2) * 0.35
    t = np.arange(n) / SR
    buffet = 0.8 + 0.2 * np.sin(2 * np.pi * 7.3 * t) * np.sin(2 * np.pi * 2.1 * t + 1.0)
    return norm((low + high) * buffet, 0.8)


def make_swarm_layers(rng, dur):
    """The sound of thousands of arrows in the air: a bright fluttering hiss and a deep whooshing body."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    hiss = bp(rng.standard_normal(n), 2200, 8000, 2)
    flutter = 0.7 + 0.3 * lp(rng.standard_normal(n), 30, 1) / 0.05
    hiss *= np.clip(flutter, 0.2, 1.4)
    body = bp(rng.standard_normal(n), 220, 1100, 2) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.6 * t))
    return norm(hiss, 0.8), norm(body, 0.8)


def make_hail(rng, dur, maker, density, bank_n=10):
    """A continuous hail of one kind of hit (thunks or thuds) to sit under mass impacts."""
    n = int(dur * SR)
    bank = [maker(rng) for _ in range(bank_n)]
    x = np.zeros(n)
    k = int(dur * density)
    for _ in range(k):
        s = int(rng.uniform(0, max(1, n - 1)))
        b = resample(bank[rng.integers(bank_n)], rng.uniform(0.8, 1.25))
        e = min(n, s + len(b))
        x[s:e] += b[:e - s] * rng.uniform(0.3, 1.0)
    return norm(x, 0.8)


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


def make_heartbeat(rng):
    """Lub-dub."""
    dur = 0.55
    n = int(dur * SR)
    x = np.zeros(n)
    for start, f, a in ((0.0, 62.0, 1.0), (0.2, 55.0, 0.7)):
        s = int(start * SR)
        m = n - s
        x[s:] += a * sine_sweep(m / SR, f * 1.6, f, 0.015) * expenv(m, 0.06, 0.004)
    return norm(lp(x, 400, 2), 0.9)


def reverb_ir(rng, dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / 0.45)
    ir = lp(ir, 5000, 1)
    ir[: int(0.012 * SR)] = 0.0                  # pre-delay
    return ir / np.sqrt(np.sum(ir ** 2))


def make_riser(rng, dur):
    x = shaped_noise(dur, lambda t: 300 + 5000 * (t / dur) ** 2, lambda t: 0.6 + 0 * t,
                     lambda t: np.clip(t / dur, 0, 1) ** 2.5, rng)
    n = len(x)
    tt = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(55 + 55 * (tt / dur) ** 2) / SR) * (tt / dur) ** 2
    return norm(norm(x) + 0.35 * tone, 0.7)


def make_wind(rng, dur):
    n = int(dur * SR)
    x = lp(rng.standard_normal(n), 900, 2)
    x = hp(x, 60, 2)
    mod = lp(rng.standard_normal(n), 0.4, 1)
    mod = 0.6 + 0.4 * (mod - mod.min()) / (np.ptp(mod) + 1e-9)
    return norm(x * mod, 0.3)


def make_hit(rng):
    """Short cinematic 'trailer hit' used on the hook cuts: tight low punch + bright transient."""
    dur = 0.9
    n = int(dur * SR)
    punch = sine_sweep(dur, 140, 52, 0.035) * expenv(n, 0.16, 0.001)
    snap = bp(rng.standard_normal(n), 900, 9000) * expenv(n, 0.02)
    body = bp(rng.standard_normal(n), 120, 700) * expenv(n, 0.12)
    x = norm(punch) + 0.45 * norm(snap) + 0.5 * norm(body)
    return norm(hp(np.tanh(1.4 * x), 35.0, 2), 0.9)


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
    la = int(0.002 * SR)      # 2 ms look-ahead
    g = np.ones_like(peak)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    # smooth gain: instant attack (with look-ahead via min filter), exponential release
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
    gain = 1.0 / (1.0 + max(0.0, dist - 25.0) / 40.0)
    return pan, gain

# ---------------------------------------------------------------------------------------------
# build from cues
# ---------------------------------------------------------------------------------------------
def _env_track(values, fps, n, smooth_hz):
    """Per-frame values -> smoothed per-sample envelope of length n."""
    v = np.interp(np.arange(n) / SR * fps, np.arange(len(values)), values)
    return np.clip(lp(np.concatenate([v, np.zeros(SR)])[:n], smooth_hz, 1), 0.0, None)


def build(cues, fps, out_path, seed=7):
    rng = np.random.default_rng(seed)
    nfr = len(cues)
    dur = nfr / fps
    mix = Mix(dur + 4.0)
    verb = Mix(dur + 4.0)                      # reverb send for slow-motion hits

    thunks = [make_thunk(rng) for _ in range(12)]
    thuds = [make_thud(rng) for _ in range(10)]
    cracks = [make_bonecrack(rng) for _ in range(6)]
    squelches = [make_squelch(rng) for _ in range(8)]
    clatters = [make_clatter(rng) for _ in range(8)]
    ticks = [make_tick(rng) for _ in range(24)]
    gravels = [make_crunch(rng, 250, 1400, 0.35) for _ in range(10)]
    twangs = [make_twang(rng) for _ in range(6)]
    volleys = [make_volley(rng) for _ in range(3)]
    flybys = [make_flyby(rng) for _ in range(6)]
    quivers = [make_quiver(rng) for _ in range(4)]
    rips = [make_rip(rng) for _ in range(2)]
    lands = [make_land(rng) for _ in range(2)]
    pop = make_pop(rng)
    xs = make_x_sound(rng)
    ck = make_check_sound(rng)
    hurt = make_hurt(rng)
    hit = make_hit(rng)
    slow_down, slow_up = make_slowmo_down(rng), make_slowmo_up(rng)
    beat = make_heartbeat(rng)
    mix.add(make_wind(rng, dur + 2.0), 0.0, 0.11, pan=-0.6)
    mix.add(make_wind(rng, dur + 2.0), 0.0, 0.11, pan=0.6)

    scale = np.array([c.get('scale', 1.0) for c in cues], float)
    riding = np.array([bool(c.get('riding')) for c in cues])
    near = np.array([c.get('swarm', 0) for c in cues], float)
    total = np.array([c['events'].get('flying', [0])[0] for c in cues], float)
    impacts = np.array([c['events'].get('impact', [0])[0] for c in cues], float)
    grounds = np.array([c['events'].get('ground_hit', [0])[0] + c['events'].get('pile_hit', [0])[0]
                        for c in cues], float)
    n_all = int((dur + 4.0) * SR)

    # the swarm in the air: deep whoosh plus a bright hiss that dulls in slow motion
    lev = 0.55 * np.log1p(near) / np.log1p(3000) + 0.5 * np.log1p(total) / np.log1p(10000)
    hiss, body = make_swarm_layers(rng, dur + 4.0)
    e_body = _env_track(lev, fps, n_all, 5.0)
    e_hiss = _env_track(lev * (0.3 + 0.7 * scale), fps, n_all, 5.0)
    mix.add(body * e_body, 0.0, 0.55)
    mix.add(hiss * e_hiss, 0.0, 0.32)

    # a hail of hits under mass impacts (real-time only; slow motion gets individual pitched-down hits)
    rt = np.clip((scale - 0.6) / 0.3, 0.0, 1.0)
    e_meat = _env_track(np.clip(np.log1p(np.maximum(impacts - 2, 0)) / np.log1p(150), 0, 1) * rt, fps, n_all, 8.0)
    e_dirt = _env_track(np.clip(np.log1p(np.maximum(grounds - 2, 0)) / np.log1p(250), 0, 1) * rt, fps, n_all, 8.0)
    if e_meat.max() > 0.01:
        mix.add(make_hail(rng, dur + 4.0, make_thunk, 70.0) * e_meat, 0.0, 0.6)
    if e_dirt.max() > 0.01:
        mix.add(make_hail(rng, dur + 4.0, make_thud, 90.0) * e_dirt, 0.0, 0.5)

    # arrow-cam wind rush
    i = 0
    while i < nfr:
        if riding[i]:
            j = i
            while j < nfr and riding[j] and (j == i or not cues[j].get('cut')):
                j += 1
            L = (j - i) / fps
            r = make_rush(rng, L + 0.3)
            m = len(r)
            t = np.arange(m) / SR
            env = np.clip(t / 0.12, 0, 1) * np.clip((L + 0.25 - t) / 0.25, 0, 1)
            mix.add(r * env, i / fps, 0.5)
            i = j
        else:
            i += 1

    # slow motion (inside the rounds): pitch drop on the way in with a riser building up to it, a rush on the
    # way out, heartbeat while it lasts
    in_round = np.array([c['seg'] != 'hook' for c in cues])
    slow = (scale < 0.5) & in_round
    for f in range(1, nfr):
        if slow[f] and not slow[f - 1]:
            mix.add(slow_down, f / fps, 0.6)
            start = f
            while start > 0 and cues[start - 1]['seg'] == cues[f]['seg'] and not cues[start].get('cut'):
                start -= 1
            if f - start > int(0.8 * fps):
                L = (f - start) / fps
                mix.add(hp(make_riser(rng, L), 150.0, 2), start / fps, 0.5)
            k = f + int(0.35 * fps)
            while k < nfr and scale[k] < 0.4:
                mix.add(beat, k / fps, 0.7)
                k += int(0.8 * fps)
        if in_round[f] and scale[f] < 0.95 and scale[f] > scale[f - 1] + 0.004 \
                and scale[f - 1] <= scale[max(0, f - 2)] + 1e-6:
            mix.add(slow_up, f / fps, 0.45)

    # hook: riser under the montage, a hit on every cut and on the cut into round 1
    hook_end = next((i for i, c in enumerate(cues) if c['seg'] != 'hook'), 0)
    if hook_end > 0:
        mix.add(hp(make_riser(rng, hook_end / fps), 180.0, 2), 0.0, 0.45)
        for f in range(hook_end):
            if cues[f].get('cut'):
                mix.add(hit, f / fps, 0.55 if f == 0 else 0.42)
        mix.add(hit, hook_end / fps, 0.5)

    # fly-bys: an arrow passing close to a camera whistles past
    dmin = np.array([c.get('swarm_dmin', 999.0) for c in cues], float)
    last_fly = -99
    for f in range(1, nfr - 1):
        if dmin[f] < 7.0 and dmin[f] <= dmin[f - 1] and dmin[f] <= dmin[f + 1] and f - last_fly > 3:
            c = cues[f]
            pan, _ = _pan_gain(c.get('swarm_pos'), c.get('cam'), c.get('tgt'))
            pf = 0.55 + 0.45 * scale[f]
            s = resample(flybys[rng.integers(len(flybys))], pf * rng.uniform(0.9, 1.1))
            g = float(np.clip((7.0 - dmin[f]) / 5.0, 0.25, 1.0)) * (0.3 if riding[f] else 0.5)
            mix.add(s, f / fps - 0.19 / pf, g, pan)
            last_fly = f

    last_hurt = -99
    last_volley = -99
    for i, c in enumerate(cues):
        t = i / fps
        ev = c['events']
        cam = c.get('cam')
        tgt = c.get('tgt', [0.0, 0.0, 14.0])
        pf = 0.55 + 0.45 * scale[i]                # slow motion drops the pitch of everything that happens
        wet = float(np.clip((0.9 - scale[i]) / 0.5, 0.0, 1.0))
        if c.get('label'):
            mix.add(pop, t, 0.5)
        if c.get('stamp') == 'x':
            mix.add(xs, t + 0.02, 0.55)
        if c.get('stamp') == 'check':
            mix.add(ck, t + 0.02, 0.6)
        if 'hp' in c and c['hp'] < c['hp_prev'] and (i - last_hurt) > fps * 0.45:
            mix.add(resample(hurt, pf), t, 0.45)
            last_hurt = i

        def put(bank, count, cap, gain, pos, spread=1.0 / fps, pitch=(0.85, 1.2), pan_spread=0.5, send=0.0):
            if count <= 0:
                return
            pan, g = _pan_gain(pos, cam, tgt)
            cc = min(float(count), float(cap))
            k = int(np.floor(cc)) + int(rng.random() < (cc - np.floor(cc)))
            if k <= 0:
                return
            total_g = gain * g * (0.45 + 0.55 * min(1.0, np.log1p(count) / np.log1p(cap * 4)))
            gg = total_g / np.sqrt(k)
            for _ in range(k):
                s = resample(bank[rng.integers(len(bank))], rng.uniform(*pitch) * pf)
                dt = rng.uniform(0, spread)
                p = pan + rng.uniform(-pan_spread, pan_spread)
                a = gg * rng.uniform(0.7, 1.0)
                mix.add(s, t + dt, a, p)
                if send > 0:
                    verb.add(s, t + dt, a * send, p)

        if 'launch' in ev:
            n, pos = ev['launch']
            d = np.linalg.norm(np.asarray(pos) - np.asarray(cam)) if pos is not None and cam is not None else 99.0
            if d < 45.0:
                put(twangs, n, 3, 0.55, pos, pitch=(0.9, 1.1), pan_spread=0.2)
            elif i - last_volley > 0.35 * fps:
                pan, g = _pan_gain(pos, cam, tgt)
                mix.add(volleys[rng.integers(len(volleys))], t, 0.5 * g * min(1.0, 0.4 + np.log10(max(n, 1)) / 3),
                        pan)
                last_volley = i
        if 'impact' in ev:
            n, pos = ev['impact']
            put(thunks, n, 4, 0.9, pos, pan_spread=0.35, send=0.6 * wet)
        if 'stick' in ev and ev['stick'][0] <= 3:
            n, pos = ev['stick']
            put(quivers, n, 2, 0.3, pos, pan_spread=0.1)
        if 'carve' in ev:
            n, pos = ev['carve']
            put(squelches, n / 60.0, 2, 0.35, pos, send=0.4 * wet)
        if 'bone' in ev:
            n, _ = ev['bone']
            put(cracks, n / 20.0, 2, 0.45, None, send=0.4 * wet)
        if 'ground_hit' in ev:
            n, pos = ev['ground_hit']
            put(thuds, n, 3, 0.5, pos, pan_spread=0.6)
        if 'pile_hit' in ev:
            n, pos = ev['pile_hit']
            put(thunks, n, 2, 0.3, pos, pitch=(0.65, 0.85), pan_spread=0.6)
        if 'arrow_land' in ev:
            n, pos = ev['arrow_land']
            put(clatters, n, 2, 0.25, pos, pan_spread=0.6)
        if 'debris_ground' in ev:
            n, pos = ev['debris_ground']
            put(ticks, n / 6.0, 8, 0.22, pos, pitch=(0.8, 1.4), pan_spread=0.6)
        if 'wound' in ev or 'crumble' in ev:
            n = ev.get('wound', [0])[0] + ev.get('crumble', [0])[0]
            put(gravels, n / 150.0, 2, 0.18, None, pitch=(0.7, 1.0))
        if 'sever' in ev:
            pan, g = _pan_gain(ev['sever'][1], cam, tgt)
            s = resample(rips[rng.integers(len(rips))], pf * rng.uniform(0.9, 1.05))
            mix.add(s, t, 0.8 * g, pan)
            verb.add(s, t, 0.3 * g, pan)
        if 'chunk_land' in ev:
            pan, g = _pan_gain(ev['chunk_land'][1], cam, tgt)
            mix.add(resample(lands[rng.integers(len(lands))], pf * rng.uniform(0.9, 1.1)), t, 0.8 * g, pan)
        if 'detach' in ev and ev['detach'][0] > 400:
            put(cracks, 1, 1, 0.4, ev['detach'][1])

    # slow-motion reverb
    n = int(dur * SR)
    ir = reverb_ir(rng)
    wl = signal.fftconvolve(verb.L[:n], ir)[:n]
    wr = signal.fftconvolve(verb.R[:n], ir)[:n]
    L, R = mix.L[:n] + wl, mix.R[:n] + wr
    L, R = hp(L, 35.0, 2), hp(R, 35.0, 2)
    lufs = integrated_lufs(L, R)
    g0 = 10 ** ((-18.0 - lufs) / 20.0)
    L, R = compress(L * g0, R * g0, thresh_db=-24.0, ratio=3.0)
    gain = 10 ** ((-14.0 - integrated_lufs(L, R)) / 20.0)
    L, R = L * gain, R * gain
    L, R = limiter(L, R, ceiling=10 ** (-1.2 / 20))
    lufs2 = integrated_lufs(L, R)
    f = int(0.01 * SR)
    for ch in (L, R):
        ch[:f] *= np.linspace(0, 1, f)
        ch[-f:] *= np.linspace(1, 0, f)
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
