"""Procedural sound design driven by the simulation cue sheet.

Everything is synthesised here (no samples): anvils (heavy iron clunks with an inharmonic ring, brighter when
iron hits iron, dull and crunchy when they crush flesh, thuddy on the ground), falling whistles and the roar of
a sky full of anvils, clattering avalanches down the pile, the Warden's heartbeat (it races as the danger grows
and stops when he dies), his growl when hurt, the charge and blast of his sonic boom, heavy pieces landing,
a slow-motion pitch drop with reverb, anvil-cam wind, UI pops/stamps and an evening wind bed. Mixed in stereo
with panning from the 3D event positions, then loudness-normalised (BS.1770 style) to -14 LUFS with a peak
limiter.
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


def make_rush(rng, dur):
    """Wind roaring past the arrow-cam."""
    n = int(dur * SR)
    low = lp(rng.standard_normal(n), 900, 2)
    high = bp(rng.standard_normal(n), 2000, 7000, 2) * 0.35
    t = np.arange(n) / SR
    buffet = 0.8 + 0.2 * np.sin(2 * np.pi * 7.3 * t) * np.sin(2 * np.pi * 2.1 * t + 1.0)
    return norm((low + high) * buffet, 0.8)


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


# ---------------------------------------------------------------------------------------------
# anvils and the Warden
# ---------------------------------------------------------------------------------------------
def modal(dur, f0, ratios, amps, taus, rng, detune=0.004):
    """Sum of decaying inharmonic partials (struck metal)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a, tau in zip(ratios, amps, taus):
        f = f0 * r * (1 + rng.uniform(-detune, detune))
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    return x


IRON = (1.0, 2.32, 3.87, 5.12, 7.35, 9.8)          # a chunky block of iron: dense, uneven partials


def make_clang(rng, bright=1.0):
    """Anvil hitting anvil: a hard iron CLUNK with a short, bright, inharmonic ring."""
    dur = 1.1
    n = int(dur * SR)
    f0 = rng.uniform(330, 470)
    ring = modal(dur, f0, IRON, (1.0, 0.7, 0.55, 0.4, 0.3, 0.2),
                 (0.20, 0.14, 0.10, 0.07, 0.05, 0.04), rng)
    hi = modal(dur, f0 * rng.uniform(4.1, 4.6), (1.0, 1.51, 2.13), (0.5, 0.35, 0.2), (0.06, 0.05, 0.03), rng)
    knock = bp(rng.standard_normal(n), 250, 1800) * expenv(n, 0.012, 0.0004)
    body = sine_sweep(dur, 190, 90, 0.02) * expenv(n, 0.05, 0.0008)
    click = bp(rng.standard_normal(n), 3000, 12000) * expenv(n, 0.0025)
    x = 0.8 * norm(ring) + 0.35 * bright * norm(hi) + 0.9 * norm(knock) + 0.7 * norm(body) + 0.4 * bright * norm(click)
    return norm(np.tanh(1.3 * x), 0.9)


def make_crush(rng):
    """Anvil landing on the Warden: a massive dull thump, a wet crunch, the iron's short dead ring."""
    dur = 1.0
    n = int(dur * SR)
    thump = sine_sweep(dur, rng.uniform(95, 120), rng.uniform(38, 48), 0.04) * expenv(n, 0.16, 0.001)
    body = bp(rng.standard_normal(n), 90, 600, 2) * expenv(n, 0.09, 0.002)
    crunch = grains(dur, 70, 300, 2500, 0.05, rng, glen=(0.002, 0.01))
    wet = make_squelch(rng, 0.5)
    ring = modal(dur, rng.uniform(300, 420), IRON[:4], (1.0, 0.6, 0.4, 0.3), (0.07, 0.05, 0.04, 0.03), rng)
    x = 1.0 * norm(thump) + 0.7 * norm(body) + 0.55 * norm(crunch) + 0.35 * norm(np.pad(wet, (0, n - len(wet))))\
        + 0.35 * norm(ring)
    return norm(np.tanh(1.5 * x), 0.95)


def make_ground_thud(rng):
    """Anvil burying itself in the dirt: heavy thud, dirt spray and a muffled clank."""
    dur = 0.8
    n = int(dur * SR)
    thud = sine_sweep(dur, rng.uniform(110, 135), rng.uniform(42, 52), 0.03) * expenv(n, 0.1, 0.001)
    dirt = grains(dur, 40, 500, 3500, 0.06, rng, glen=(0.002, 0.008))
    clank = modal(dur, rng.uniform(280, 380), IRON[:3], (1.0, 0.5, 0.3), (0.05, 0.04, 0.03), rng)
    x = norm(thud) + 0.5 * norm(dirt) + 0.35 * norm(clank)
    return norm(np.tanh(1.3 * x), 0.85)


def make_rattle(rng):
    """Anvil tumbling down the pile: three or four quick, smaller clanks."""
    dur = 0.6
    n = int(dur * SR)
    x = np.zeros(n)
    s = 0
    for k in range(int(rng.integers(3, 5))):
        c = make_clang(rng, bright=0.6)
        c = resample(c, rng.uniform(1.05, 1.35))[: int(0.25 * SR)]
        e = min(n, s + len(c))
        x[s:e] += c[:e - s] * (0.75 ** k)
        s += int(rng.uniform(0.06, 0.13) * SR)
        if s >= n:
            break
    return norm(x, 0.7)


def make_whistle(rng, dur=0.7):
    """A heavy lump of iron falling past: a low whoosh that drops in pitch (Doppler)."""
    c = dur * 0.5
    x = shaped_noise(dur, lambda t: 380 + 900 / (1 + np.exp((t - c) / 0.05)), lambda t: 0.7 + 0 * t,
                     lambda t: np.exp(-((t - c) / 0.13) ** 2), rng)
    return norm(x, 0.7)


def make_storm(rng, dur):
    """Thousands of anvils in the air: a deep roaring rush with a fluttering whistle on top."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    roar = bp(rng.standard_normal(n), 60, 420, 2)
    roar *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.45 * t)
    whistle = bp(rng.standard_normal(n), 700, 2600, 2)
    whistle *= np.clip(0.6 + 0.4 * lp(rng.standard_normal(n), 20, 1) / 0.05, 0.1, 1.4)
    return norm(roar, 0.8), norm(whistle, 0.8)


def make_hail(rng, dur, maker, density, bank_n=10):
    """A continuous hail of one kind of hit to sit under mass landings."""
    n = int(dur * SR)
    bank = [maker(rng) for _ in range(bank_n)]
    x = np.zeros(n)
    k = int(dur * density)
    for _ in range(k):
        s = int(rng.uniform(0, max(1, n - 1)))
        b = resample(bank[rng.integers(bank_n)], rng.uniform(0.8, 1.2))
        e = min(n, s + len(b))
        x[s:e] += b[:e - s] * rng.uniform(0.3, 1.0)
    return norm(x, 0.8)


def make_warden_beat(rng):
    """The Warden's heartbeat: a deep, muffled double thump with a little sub."""
    dur = 0.6
    n = int(dur * SR)
    x = np.zeros(n)
    for start, f, a in ((0.0, 48.0, 1.0), (0.19, 42.0, 0.75)):
        s = int(start * SR)
        m = n - s
        thump = sine_sweep(m / SR, f * 1.9, f, 0.02) * expenv(m, 0.075, 0.003)
        knock = lp(rng.standard_normal(m), 300, 2) * expenv(m, 0.02, 0.001)
        x[s:] += a * (norm(thump) + 0.35 * norm(knock))
    return norm(lp(x, 500, 2), 0.95)


def make_growl(rng, dur=0.9):
    """A low, rattling growl when he gets hurt: a buzzing pulse train through moving formants."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 52 + 10 * np.sin(2 * np.pi * 3.1 * t) + rng.uniform(-3, 3)
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
    """The blast: an enormous low WHUMP, a bright crack, a resonant pitch dive and a long airy wash."""
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


def make_heavy_land(rng):
    """A big piece of the Warden (or a slab of anvils) hitting the ground."""
    dur = 1.4
    n = int(dur * SR)
    thud = sine_sweep(dur, 90, 34, 0.06) * expenv(n, 0.25, 0.002)
    body = bp(rng.standard_normal(n), 70, 450, 2) * expenv(n, 0.18, 0.003)
    debris = grains(dur, 160, 400, 3000, 0.2, rng, glen=(0.002, 0.01))
    x = norm(thud) + 0.7 * norm(body) + 0.45 * norm(debris)
    return norm(hp(np.tanh(1.4 * x), 30.0, 2), 0.9)


def make_creak(rng, dur=2.5):
    """The pile settling: slow, groaning iron creaks."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(3):
        s = rng.uniform(0.0, dur * 0.6)
        L = rng.uniform(0.4, 0.9)
        m = (t >= s) & (t < s + L)
        u = (t[m] - s) / L
        f = rng.uniform(90, 160) * (1 + 0.3 * u)
        ph = np.cumsum(f) / SR
        saw = (np.mod(ph, 1.0) - 0.5) * (0.6 + 0.4 * np.sin(2 * np.pi * 23 * t[m]))
        out[m] += saw * np.sin(np.pi * u) ** 2
    return norm(bp(out, 150, 1800, 2), 0.6)


def make_evening(rng, dur):
    """Evening air: soft wind with the odd far-off bird chirp."""
    n = int(dur * SR)
    x = lp(rng.standard_normal(n), 700, 2)
    x = hp(x, 60, 2)
    mod = lp(rng.standard_normal(n), 0.3, 1)
    mod = 0.6 + 0.4 * (mod - mod.min()) / (np.ptp(mod) + 1e-9)
    x = norm(x * mod, 0.3)
    t = np.arange(n) / SR
    for _ in range(int(dur / 3.5)):
        s = rng.uniform(0, dur - 0.5)
        for k in range(int(rng.integers(2, 4))):
            s0 = s + k * rng.uniform(0.09, 0.14)
            m = (t >= s0) & (t < s0 + 0.07)
            u = (t[m] - s0) / 0.07
            f = rng.uniform(2800, 3600) * (1 + 0.25 * u)
            x[m] += 0.015 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u)
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
    verb = Mix(dur + 4.0)                      # reverb send (slow motion, sonic boom)

    clangs = [make_clang(rng) for _ in range(12)]
    crushes = [make_crush(rng) for _ in range(8)]
    thuds = [make_ground_thud(rng) for _ in range(8)]
    rattles = [make_rattle(rng) for _ in range(6)]
    whistles = [make_whistle(rng) for _ in range(5)]
    cracks = [make_bonecrack(rng) for _ in range(4)]
    gravels = [make_crunch(rng, 250, 1400, 0.35) for _ in range(8)]
    ticks = [make_tick(rng) for _ in range(16)]
    heavy = [make_heavy_land(rng) for _ in range(2)]
    growls = [make_growl(rng) for _ in range(3)]
    beat = make_warden_beat(rng)
    pop = make_pop(rng)
    xs = make_x_sound(rng)
    ck = make_check_sound(rng)
    hit = make_hit(rng)
    slow_down, slow_up = make_slowmo_down(rng), make_slowmo_up(rng)
    charge, sboom = make_boom_charge(rng), make_sonic_boom(rng)
    mix.add(make_evening(rng, dur + 2.0), 0.0, 0.3, pan=-0.5)
    mix.add(make_evening(rng, dur + 2.0), 0.0, 0.3, pan=0.5)

    scale = np.array([c.get('scale', 1.0) for c in cues], float)
    riding = np.array([bool(c.get('riding')) for c in cues])
    near = np.array([c.get('swarm', 0) for c in cues], float)
    total = np.array([c['events'].get('falling', [0])[0] for c in cues], float)
    lands = np.array([c['events'].get('land_anvil', [0])[0] + c['events'].get('land_ground', [0])[0]
                      for c in cues], float)
    flesh = np.array([c['events'].get('land_flesh', [0])[0] for c in cues], float)
    n_all = int((dur + 4.0) * SR)

    # the storm in the air
    lev = 0.5 * np.log1p(near) / np.log1p(2000) + 0.6 * np.log1p(total) / np.log1p(10000)
    roar, whistle = make_storm(rng, dur + 4.0)
    mix.add(roar * _env_track(lev, fps, n_all, 4.0), 0.0, 0.7)
    mix.add(whistle * _env_track(lev * (0.3 + 0.7 * scale), fps, n_all, 4.0), 0.0, 0.25)

    # hail beds under mass landings (real time only; slow motion gets individual pitched-down hits)
    rt = np.clip((scale - 0.6) / 0.3, 0.0, 1.0)
    e_metal = _env_track(np.clip(np.log1p(np.maximum(lands - 2, 0)) / np.log1p(120), 0, 1) * rt, fps, n_all, 8.0)
    e_meat = _env_track(np.clip(np.log1p(np.maximum(flesh - 1, 0)) / np.log1p(60), 0, 1) * rt, fps, n_all, 8.0)
    if e_metal.max() > 0.01:
        mix.add(make_hail(rng, dur + 4.0, make_clang, 60.0) * e_metal, 0.0, 0.55)
    if e_meat.max() > 0.01:
        mix.add(make_hail(rng, dur + 4.0, make_crush, 40.0) * e_meat, 0.0, 0.5)

    # anvil-cam wind rush
    i = 0
    while i < nfr:
        if riding[i]:
            j = i
            while j < nfr and riding[j] and (j == i or not cues[j].get('cut')):
                j += 1
            L = (j - i) / fps
            r = make_rush(rng, L + 0.3)
            t = np.arange(len(r)) / SR
            mix.add(r * np.clip(t / 0.12, 0, 1) * np.clip((L + 0.25 - t) / 0.25, 0, 1), i / fps, 0.45)
            i = j
        else:
            i += 1

    # slow motion (inside the rounds): pitch drop in, rush out
    in_round = np.array([c['seg'] != 'hook' for c in cues])
    slow = (scale < 0.5) & in_round
    for f in range(1, nfr):
        if slow[f] and not slow[f - 1]:
            mix.add(slow_down, f / fps, 0.55)
        if in_round[f] and scale[f] < 0.95 and scale[f] > scale[f - 1] + 0.004 \
                and scale[f - 1] <= scale[max(0, f - 2)] + 1e-6:
            mix.add(slow_up, f / fps, 0.4)

    # hook: riser under the montage, a hit on every cut and on the cut into round 1
    hook_end = next((i for i, c in enumerate(cues) if c['seg'] != 'hook'), 0)
    if hook_end > 0:
        mix.add(hp(make_riser(rng, hook_end / fps), 180.0, 2), 0.0, 0.4)
        for f in range(hook_end):
            if cues[f].get('cut'):
                mix.add(hit, f / fps, 0.5 if f == 0 else 0.4)
        mix.add(hit, hook_end / fps, 0.5)

    # whistles: anvils dropping past the camera
    dmin = np.array([c.get('swarm_dmin', 999.0) for c in cues], float)
    last = -99
    for f in range(1, nfr - 1):
        if dmin[f] < 8.0 and dmin[f] <= dmin[f - 1] and dmin[f] <= dmin[f + 1] and f - last > 4:
            c = cues[f]
            pan, _ = _pan_gain(c.get('swarm_pos'), c.get('cam'), c.get('tgt'))
            pf = 0.55 + 0.45 * scale[f]
            s = resample(whistles[rng.integers(len(whistles))], pf * rng.uniform(0.9, 1.1))
            mix.add(s, f / fps - 0.35 / pf, float(np.clip((8.0 - dmin[f]) / 6.0, 0.25, 1.0)) * 0.5, pan)
            last = f

    last_hurt = -99
    boom_frames = []
    for i, c in enumerate(cues):
        t = i / fps
        ev = c['events']
        cam = c.get('cam')
        tgt = c.get('tgt', [0.0, 0.0, 15.0])
        pf = 0.55 + 0.45 * scale[i]                # slow motion drops the pitch of everything that happens
        wet = float(np.clip((0.9 - scale[i]) / 0.5, 0.0, 1.0))
        if c.get('label'):
            mix.add(pop, t, 0.5)
        if c.get('stamp') == 'x':
            mix.add(xs, t + 0.02, 0.55)
        if c.get('stamp') == 'check':
            mix.add(ck, t + 0.02, 0.6)
        if c.get('beat'):
            quiet = 1.0 - 0.5 * min(1.0, lev[i] * 1.4)           # the heart sits under the chaos
            loud = 0.55 if c['seg'] in ('r5', 'hook') else 0.32    # it takes over in the finale
            mix.add(resample(beat, pf), t, loud * quiet)
        if 'hp' in c and c['hp'] < c['hp_prev'] and (i - last_hurt) > fps * 1.2:
            mix.add(resample(growls[rng.integers(len(growls))], pf * rng.uniform(0.92, 1.05)), t + 0.05, 0.4)
            last_hurt = i

        def put(bank, count, cap, gain, pos, spread=1.0 / fps, pitch=(0.88, 1.15), pan_spread=0.5, send=0.0):
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

        if 'land_flesh' in ev:
            n, pos = ev['land_flesh']
            put(crushes, n, 3, 0.95, pos, pan_spread=0.3, send=0.6 * wet)
        if 'land_anvil' in ev:
            n, pos = ev['land_anvil']
            put(clangs, n, 4, 0.7, pos, pan_spread=0.4, send=0.5 * wet)
        if 'land_ground' in ev:
            n, pos = ev['land_ground']
            put(thuds, n, 3, 0.55, pos, pan_spread=0.6)
        if 'slide' in ev:
            n, pos = ev['slide']
            put(rattles, n / 3.0, 2, 0.35, pos, pan_spread=0.5)
        if 'crush' in ev and ev['crush'][0] > 150:
            put(cracks, ev['crush'][0] / 400.0, 2, 0.35, ev['crush'][1], send=0.4 * wet)
        if 'debris_ground' in ev:
            n, pos = ev['debris_ground']
            put(ticks, n / 8.0, 6, 0.18, pos, pitch=(0.8, 1.3), pan_spread=0.6)
        if 'crumble' in ev or 'sink' in ev:
            n = ev.get('crumble', [0])[0] + ev.get('sink', [0])[0]
            put(gravels, n / 200.0, 2, 0.2, None, pitch=(0.7, 1.0))
        if 'chunk_land' in ev or 'stack_drop' in ev:
            pos = (ev.get('chunk_land') or [0, None])[1]
            pan, g = _pan_gain(pos, cam, tgt)
            mix.add(resample(heavy[rng.integers(len(heavy))], pf), t, 0.7 * g, pan)
        if 'boom' in ev:
            boom_frames.append(i)

    # sonic boom: charge-up before, blast on the frame it fires (both follow the slow-motion schedule)
    for f in boom_frames:
        t = f / fps
        pf = 0.55 + 0.45 * scale[f]
        ch = resample(charge, pf)
        mix.add(ch, t - len(ch) / SR, 0.55)
        b = resample(sboom, pf)
        mix.add(b, t, 1.0)
        verb.add(b, t, 0.45)

    # settling iron after the finale
    last_seg = cues[-1]['seg']
    f_end = next((i for i in range(nfr - 1, 0, -1) if cues[i]['events'].get('land_anvil') or
                  cues[i]['events'].get('land_ground')), None)
    if f_end is not None and cues[f_end]['seg'] == last_seg:
        mix.add(make_creak(rng, 2.5), f_end / fps + 0.3, 0.25)

    n = int(dur * SR)
    ir = reverb_ir(rng)
    wl = signal.fftconvolve(verb.L[:n], ir)[:n]
    wr = signal.fftconvolve(verb.R[:n], ir)[:n]
    L, R = mix.L[:n] + wl, mix.R[:n] + wr
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
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
