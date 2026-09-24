"""Procedural sound design driven by the cue sheet of the render.

Everything is synthesised here (no samples). The battle: bow releases and a thousand arrows whistling in, flesh
thunks and bone cracks, the creeper's hiss and its blast, iron anvils clanging, crushing and rattling down the
piles, the fuse hiss of ten thousand TNT and their explosions (layered by distance, with a rolling rumble under
mass detonations), the Warden's heartbeat, growl and sonic boom, severed limbs and a toppling giant, debris.
The edit: a riser and a rewind out of the cold open, trailer hits on the fighter introductions and round
banners, an elimination stinger, a dark drone that grows round by round and a victory chord for the winner.
Mixed in stereo with panning from the 3D event positions, slow motion drops the pitch and adds reverb; then the
mix is loudness-normalised (BS.1770 style) to -14 LUFS with a peak limiter.
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

# ---------------------------------------------------------------------------------------------
# arrows (from the Zombie video)
# ---------------------------------------------------------------------------------------------


def make_hurt(rng):
    """Punchy 'hit taken' cue: low body thump + short mid knock (no voice)."""
    dur = 0.3
    n = int(dur * SR)
    thump = sine_sweep(dur, 190, 75, 0.03) * expenv(n, 0.07, 0.001)
    knock = bp(rng.standard_normal(n), 350, 1400) * expenv(n, 0.025)
    x = norm(thump) + 0.45 * norm(knock)
    return norm(np.tanh(1.5 * x), 0.7)


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


def make_swarm_layers(rng, dur):
    """The sound of thousands of arrows in the air: a bright fluttering hiss and a deep whooshing body."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    hiss = bp(rng.standard_normal(n), 2200, 8000, 2)
    flutter = 0.7 + 0.3 * lp(rng.standard_normal(n), 30, 1) / 0.05
    hiss *= np.clip(flutter, 0.2, 1.4)
    body = bp(rng.standard_normal(n), 220, 1100, 2) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.6 * t))
    return norm(hiss, 0.8), norm(body, 0.8)


# ---------------------------------------------------------------------------------------------
# TNT and the creeper (from the Creeper video)
# ---------------------------------------------------------------------------------------------


def make_boom(rng, dur=2.2):
    n = int(dur * SR)
    sub = sine_sweep(dur, 92, 44, 0.2) * expenv(n, 0.38, 0.004)
    rumble = lp(rng.standard_normal(n), 200, 4) * expenv(n, 0.42, 0.01)
    crack = bp(rng.standard_normal(n), 900, 6000) * expenv(n, 0.02)
    debris = grains(dur, 260, 400, 3500, 0.35, rng, glen=(0.002, 0.012))
    x = 1.0 * norm(sub) + 0.7 * norm(rumble) + 0.5 * norm(crack) + 0.5 * norm(debris)
    x = np.tanh(x * 1.3) / np.tanh(1.3)
    return norm(x, 0.95)


def make_fuse(rng, dur):
    """Primed TNT hiss: bright band-limited noise with a slow flutter and tiny sparks."""
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 2400, 9500, 2)
    t = np.arange(n) / SR
    flutter = 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(5, 8) * t + rng.uniform(0, 6)) \
        * np.sin(2 * np.pi * rng.uniform(0.7, 1.3) * t)
    x *= flutter
    sparks = np.zeros(n)
    for _ in range(int(dur * 38)):
        p = rng.integers(0, max(1, n - 200))
        L = rng.integers(20, 120)
        sparks[p:p + L] += rng.standard_normal(L) * np.hanning(L) * rng.uniform(0.5, 1.5)
    x = norm(x) + 0.35 * norm(hp(sparks, 3000, 2))
    fade = int(0.03 * SR)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return norm(x, 0.6)


def make_explosion(rng, size=1.0, distant=False):
    """TNT blast built to read on phone speakers: sharp crack, punchy mid body, roaring noise burst that sweeps
    down, crackling tail, and only a short sub thump (long sub rumbles just turn to mud on small speakers)."""
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


def make_rumble_bed(rng, dur):
    """Rolling thunder under mass detonations: low-mid roar (60-600 Hz) with some crackle on top."""
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 60, 600, 2) + 0.5 * bp(rng.standard_normal(n), 600, 2500, 2)
    return norm(x, 0.8)


def make_spray(rng, dur=0.35, lo=300, hi=2500):
    """Wet/gritty spray of blasted flesh or gunpowder."""
    x = shaped_noise(dur, lambda tt: 0.5 * (lo + hi) * np.exp(-tt / 0.25), lambda tt: 1.1 + 0 * tt,
                     lambda tt: np.exp(-tt / 0.09) * np.clip(tt / 0.003, 0, 1), rng)
    x += 0.5 * grains(dur, 40, lo, hi, 0.06, rng)
    return norm(x, 0.7)


def make_whoosh(rng, dur, heavy=0.0):
    """A formation rushing in: rising band-passed noise, denser/lower for big formations."""
    def center(t):
        u = np.clip(t / dur, 0, 1)
        return 380 + (1500 + 900 * (1 - heavy)) * u ** 1.6

    def width(t):
        return 0.9 + 0.4 * heavy + 0 * t

    def env(t):
        u = np.clip(t / dur, 0, 1)
        return (u ** 2.2) * (1.0 - np.clip((u - 0.97) / 0.03, 0, 1))
    x = shaped_noise(dur, center, width, env, rng)
    if heavy > 0:
        n = len(x)
        tt = np.arange(n) / SR
        flutter = 1 + 0.35 * heavy * np.sin(2 * np.pi * (23 + 9 * rng.random()) * tt + rng.uniform(0, 6))
        x = x * flutter
        rumble = lp(rng.standard_normal(n), 140, 4) * np.clip(tt / dur, 0, 1) ** 2.5
        x = norm(x) + 0.35 * heavy * norm(rumble)
    return norm(x, 0.8)


# ---------------------------------------------------------------------------------------------
# the edit: trailer hits, stingers, the drone and the victory chord
# ---------------------------------------------------------------------------------------------
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
    """A big low drum hit."""
    dur = 1.1
    n = int(dur * SR)
    body = sine_sweep(dur, rng.uniform(95, 110), rng.uniform(46, 52), 0.03) * expenv(n, 0.24, 0.001)
    skin = bp(rng.standard_normal(n), 150, 900) * expenv(n, 0.035, 0.0005)
    slap = bp(rng.standard_normal(n), 1500, 5000) * expenv(n, 0.006)
    return norm(hp(np.tanh(1.4 * (norm(body) + 0.5 * norm(skin) + 0.2 * norm(slap))), 30.0, 2), 0.9)


def make_slam(rng):
    """Fighter-intro slam: an air whoosh into a tight punchy hit with a metallic snap. Returns (sound, pre-roll)."""
    pre = 0.22
    w = shaped_noise(pre, lambda t: 500 + 4500 * (t / pre) ** 2, lambda t: 0.7 + 0 * t, lambda t: (t / pre) ** 2.2, rng)
    hit = make_hit(rng)
    ring = modal(0.6, rng.uniform(600, 800), IRON[:4], (1.0, 0.5, 0.3, 0.2), (0.12, 0.08, 0.05, 0.04), rng)
    s = int(pre * SR)
    x = np.zeros(s + len(hit))
    x[:len(w)] += 0.6 * norm(w)
    x[s:s + len(hit)] += hit
    x[s:s + len(ring)] += 0.22 * norm(ring)
    return norm(x, 0.9), pre


def make_elim(rng):
    """Elimination stinger: a very low braam, a power-down synth dive and a crash."""
    dur = 2.2
    n = int(dur * SR)
    t = np.arange(n) / SR
    br = make_braam(rng, dur, 41.2)
    f = 330 * np.exp(-t / 0.3) + 60
    down = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * expenv(n, 0.4, 0.004)
    down = lp(down, 1600, 2)
    crash = bp(rng.standard_normal(n), 2500, 12000) * expenv(n, 0.3, 0.002)
    return norm(norm(br) + 0.3 * norm(down) + 0.2 * norm(crash), 0.95)


def make_rewind(rng, dur=0.45):
    """Out of the cold open: a fluttering tape-rewind whoosh with a rising zip."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = shaped_noise(dur, lambda tt: 600 + 6000 * (tt / dur) ** 1.5, lambda tt: 0.5 + 0 * tt,
                     lambda tt: (tt / dur) ** 1.5, rng)
    x = x[:n] * (1 + 0.5 * np.sin(2 * np.pi * (18 + 40 * t / dur) * t))
    zp = np.sin(2 * np.pi * np.cumsum(200 + 3000 * (t / dur) ** 2) / SR) * (t / dur) ** 2
    return norm(norm(x) + 0.3 * zp, 0.8)


def make_swell(rng, dur=0.8):
    """Reverse-cymbal swell leading into a hit."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 2500, 13000) * (t / dur) ** 3
    return norm(x, 0.7)


def make_victory(rng, dur=3.4):
    """The winner: a bright D major brass chord swelling over timpani, a cymbal and chimes."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, a in ((146.83, 1.0), (185.0, 0.8), (220.0, 0.85), (293.66, 0.7), (369.99, 0.45), (440.0, 0.35)):
        s, _ = _saws(f, n, rng, ((1.0, 1.0), (2.0, 0.25)), detune=(-0.004, 0.004))
        x += a * s
    brass = lp(x, 900, 2) + 0.5 * lp(x, 3000, 2) * (1 - np.exp(-t / 0.4)) * np.exp(-t / 1.6)
    env = np.clip(t / 0.05, 0, 1) * (0.6 + 0.4 * np.exp(-t / 0.3)) * np.clip((dur - t) / 1.4, 0, 1)
    timp = np.zeros(n)
    for k, st in enumerate((0.0, 0.12, 0.2, 0.26)):
        s = int(st * SR)
        m = n - s
        timp[s:] += (0.6 if k else 1.0) * sine_sweep(m / SR, 110, 73.4, 0.04) * expenv(m, 0.35, 0.002)
    cym = bp(rng.standard_normal(n), 3000, 14000) * expenv(n, 1.1, 0.003)
    chime = np.zeros(n)
    for st, f in ((0.35, 1174.66), (0.47, 1479.98), (0.59, 1760.0), (0.71, 2349.3)):
        s = int(st * SR)
        tt = t[:n - s]
        v = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt / tau)
                for r, a, tau in ((1.0, 1.0, 0.6), (2.0, 0.25, 0.25), (3.01, 0.1, 0.12)))
        chime[s:] += v * np.minimum(1, tt / 0.002)
    y = norm(brass * env) + 0.8 * norm(timp) + 0.18 * norm(cym) + 0.22 * norm(chime)
    return norm(hp(np.tanh(1.2 * y), 35.0, 2), 0.9)


def make_drone(rng, dur, f0=73.42):
    """Dark tension drone (root, fifth, octave and sub octave), in a dark and a brighter layer."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a in ((0.5, 0.5), (1.0, 1.0), (1.5, 0.5), (2.0, 0.4)):
        for d in (-0.004, 0.003):
            f = f0 * r * (1 + d) * (1 + 0.002 * np.sin(2 * np.pi * 0.13 * t + rng.uniform(0, 6)))
            x += a * (2 * np.mod(np.cumsum(f) / SR + rng.uniform(), 1.0) - 1)
    mod = 0.8 + 0.2 * np.sin(2 * np.pi * 0.21 * t)
    return norm(lp(x, 260, 2) * mod, 0.8), norm(bp(x, 300, 1400, 2) * mod, 0.8)


def make_hiss(rng, dur):
    """The creeper's hiss: a soft, breathy 'ssss' that swells."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 3500, 11000, 2) + 0.5 * bp(rng.standard_normal(n), 1800, 3500, 2)
    x *= 0.85 + 0.15 * np.sin(2 * np.pi * 6.5 * t)
    return norm(x, 0.7)


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


def _local_minima(d, thresh, gap):
    """Frames where d has a local minimum below thresh, at least `gap` frames apart."""
    out = []
    last = -10 ** 6
    for f in range(1, len(d) - 1):
        if d[f] < thresh and d[f] <= d[f - 1] and d[f] <= d[f + 1] and f - last > gap:
            out.append(f)
            last = f
    return out


def build(cues, fps, out_path, seed=7):
    rng = np.random.default_rng(seed)
    nfr = len(cues)
    dur = nfr / fps
    mix = Mix(dur + 4.0)                       # effects
    verb = Mix(dur + 4.0)                      # reverb send (slow motion, big blasts)
    music = Mix(dur + 4.0)                     # the score, ducked under the effects
    n_all = int((dur + 4.0) * SR)              # length of the beds

    thunks = [make_thunk(rng) for _ in range(10)]
    quivers = [make_quiver(rng) for _ in range(4)]
    thuds = [make_thud(rng) for _ in range(8)]
    clatters = [make_clatter(rng) for _ in range(6)]
    flybys = [make_flyby(rng) for _ in range(6)]
    cracks = [make_bonecrack(rng) for _ in range(5)]
    clangs = [make_clang(rng) for _ in range(12)]
    crushes = [make_crush(rng) for _ in range(8)]
    gthuds = [make_ground_thud(rng) for _ in range(8)]
    rattles = [make_rattle(rng) for _ in range(6)]
    whistles = [make_whistle(rng) for _ in range(5)]
    booms = [make_explosion(rng, 1.0) for _ in range(8)]
    big_booms = [make_explosion(rng, 1.8) for _ in range(4)]
    far_booms = [make_explosion(rng, 1.2, distant=True) for _ in range(6)]
    sprays = [make_spray(rng) for _ in range(8)]
    powders = [make_spray(rng, 0.45, 1500, 7000) for _ in range(6)]
    gravels = [make_crunch(rng, 250, 1400, 0.35) for _ in range(10)]
    ticks = [make_tick(rng) for _ in range(20)]
    heavy = [make_heavy_land(rng) for _ in range(3)]
    rips = [make_rip(rng) for _ in range(2)]
    growls = [make_growl(rng) for _ in range(3)]
    taikos = [make_taiko(rng) for _ in range(3)]
    beat = make_warden_beat(rng)
    charge, sboom = make_boom_charge(rng), make_sonic_boom(rng)
    slow_down, slow_up = make_slowmo_down(rng), make_slowmo_up(rng)
    hit = make_hit(rng)

    scale = np.array([c.get('scale', 1.0) for c in cues], float)
    segs = [c['seg'] for c in cues]
    shots = [c['shot'] for c in cues]

    def ev_n(name):
        return np.array([c['events'].get(name, [0])[0] for c in cues], float)

    def env(values, hz=5.0):
        return _env_track(np.asarray(values, float), fps, n_all, hz)

    # ---- ambience
    mix.add(make_evening(rng, dur + 2.0), 0.0, 0.2, pan=-0.5)
    mix.add(make_evening(rng, dur + 2.0), 0.0, 0.2, pan=0.5)

    # ---- beds: a sky full of arrows, of anvils, the fuse hiss of the TNT, rolling thunder
    fly = ev_n('flying')
    near_a = np.array([c.get('arrows', [0])[0] for c in cues], float)
    if fly.max() > 0:
        hiss_a, body_a = make_swarm_layers(rng, dur + 4.0)
        lev = 0.55 * np.log1p(near_a) / np.log1p(300) + 0.45 * np.log1p(fly) / np.log1p(1000)
        mix.add(body_a * env(lev), 0.0, 0.5)
        mix.add(hiss_a * env(lev * (0.3 + 0.7 * scale)), 0.0, 0.3)
    fall = ev_n('falling')
    near_n = np.array([c.get('anvils', [0])[0] for c in cues], float)
    if fall.max() > 0:
        roar, whistle = make_storm(rng, dur + 4.0)
        lev = 0.5 * np.log1p(near_n) / np.log1p(300) + 0.5 * np.log1p(fall) / np.log1p(1000)
        mix.add(roar * env(lev, 4.0), 0.0, 0.55)
        mix.add(whistle * env(lev * (0.3 + 0.7 * scale), 4.0), 0.0, 0.22)
    lit = ev_n('tnt_hanging') + ev_n('tnt_falling')
    if lit.max() > 0:
        fz = make_fuse(rng, dur + 4.0)
        lev = np.clip(np.log1p(lit) / np.log1p(10000), 0, 1) * (lit > 0)
        mix.add(fz * env(lev, 6.0), 0.0, 0.34)
        mix.add(make_fuse(rng, dur + 4.0) * env(lev * np.clip(np.array([c.get('tnt', [0])[0] for c in cues]) / 40.0,
                                                                  0, 1), 6.0), 0.0, 0.3, pan=0.3)
    rate = ev_n('explode')
    if rate.max() > 6:
        bed = make_rumble_bed(rng, dur + 4.0)
        e = np.clip(np.log1p(rate) / np.log1p(400), 0, 1) ** 1.3
        mix.add(bed * lp(env(e, 30.0), 3.0, 1) * 1.8, 0.0, 0.24)
    # mass landings: hail beds (real time only; slow motion gets individual pitched-down hits)
    rt = np.clip((scale - 0.6) / 0.3, 0.0, 1.0)
    lands = ev_n('land_anvil') + ev_n('land_ground')
    flesh = ev_n('land_flesh')
    impacts = ev_n('impact')
    for arr, maker, cap, g in ((lands, make_clang, 120, 0.45), (flesh, make_crush, 60, 0.45),
                               (impacts, make_thunk, 80, 0.4)):
        e = np.clip(np.log1p(np.maximum(arr - 2, 0)) / np.log1p(cap), 0, 1) * rt
        if e.max() > 0.01:
            mix.add(make_hail(rng, dur + 4.0, maker, 50.0) * env(e, 8.0), 0.0, g)

    # ---- the cold open: a riser under it, a rewind out of it, a hit on the first frame of the fight
    n_cold = sum(1 for s in segs if s == 'cold')
    if n_cold:
        mix.add(hp(make_riser(rng, n_cold / fps), 180.0, 2), 0.0, 0.35)
        rw = make_rewind(rng)
        mix.add(rw, n_cold / fps - len(rw) / SR, 0.55)
        mix.add(hit, n_cold / fps, 0.55)
        music.add(taikos[0], n_cold / fps, 0.6)

    # ---- the creeper's hiss, from when it starts to flash until it goes off
    hs = [i for i, c in enumerate(cues) if c.get('hissing')]
    if hs:
        a, b = hs[0], hs[-1] + 1
        h = make_hiss(rng, (b - a) / fps + 0.1)
        u = np.linspace(0, 1, len(h))
        mix.add(h * (0.35 + 0.65 * u ** 1.5) * np.clip((len(h) - np.arange(len(h))) / (0.05 * SR), 0, 1), a / fps, 0.5)

    # ---- the volley: bows released far away
    for i in range(nfr):
        if cues[i]['events'].get('launch') and not any(cues[k]['events'].get('launch') for k in range(max(0, i - 3), i)):
            mix.add(make_volley(rng, 60), i / fps, 0.55)

    # ---- fly-bys: arrows past the lens, anvils whistling down past it, fuses hissing past
    for key, bank, thresh, g in (('arrows', flybys, 6.0, 0.5), ('anvils', whistles, 8.0, 0.45)):
        d = np.array([c.get(key, [0, 999.0])[1] for c in cues], float)
        for f in _local_minima(d, thresh, 4):
            c = cues[f]
            pan, _ = _pan_gain(c.get(key)[2], c.get('cam'), c.get('tgt'))
            pf = 0.55 + 0.45 * scale[f]
            s = resample(bank[rng.integers(len(bank))], pf * rng.uniform(0.9, 1.1))
            mix.add(s, f / fps - 0.2 / pf, float(np.clip((thresh - d[f]) / (thresh - 1.0), 0.3, 1.0)) * g, pan)

    # ---- slow motion (in the fight): pitch drop in, rush out
    main = np.array([s == 'main' for s in segs])
    slow = (scale < 0.5) & main
    for f in range(1, nfr):
        if slow[f] and not slow[f - 1]:
            mix.add(slow_down, f / fps, 0.5)
        if main[f] and scale[f] < 0.95 and scale[f] > scale[f - 1] + 0.004 and scale[f - 1] <= scale[max(0, f - 2)] + 1e-6:
            mix.add(slow_up, f / fps, 0.35)

    # ---- per-frame events
    last_growl = -99
    last_big = -99
    for i, c in enumerate(cues):
        t = i / fps
        ev = c['events']
        cam = c.get('cam')
        tgt = c.get('tgt', [0.0, 0.0, 15.0])
        pf = 0.55 + 0.45 * scale[i]
        wet = float(np.clip((0.9 - scale[i]) / 0.5, 0.0, 1.0))

        def put(bank, count, cap, gain, pos, spread=1.0 / fps, pitch=(0.88, 1.15), pan_spread=0.5, send=0.0):
            if count <= 0:
                return
            pan, g = _pan_gain(pos, cam, tgt)
            cc = min(float(count), float(cap))
            k = int(np.floor(cc)) + int(rng.random() < (cc - np.floor(cc)))
            if k <= 0:
                return
            gg = gain * g * (0.45 + 0.55 * min(1.0, np.log1p(count) / np.log1p(cap * 4))) / np.sqrt(k)
            for _ in range(k):
                s = resample(bank[rng.integers(len(bank))], rng.uniform(*pitch) * pf)
                dt = rng.uniform(0, spread)
                p = pan + rng.uniform(-pan_spread, pan_spread)
                a = gg * rng.uniform(0.7, 1.0)
                mix.add(s, t + dt, a, p)
                if send > 0:
                    verb.add(s, t + dt, a * send, p)

        # arrows
        if 'impact' in ev:
            put(thunks, ev['impact'][0], 4, 0.8, ev['impact'][1], pan_spread=0.4, send=0.4 * wet)
            if rng.random() < 0.3:
                put(quivers, 1, 1, 0.25, ev['impact'][1])
        if 'bone' in ev:
            put(cracks, ev['bone'][0] / 40.0, 2, 0.4, None, send=0.3 * wet)
        if 'ground_hit' in ev:
            put(thuds, ev['ground_hit'][0], 3, 0.45, ev['ground_hit'][1], pan_spread=0.6)
        if 'arrow_land' in ev:
            put(clatters, ev['arrow_land'][0] / 4.0, 3, 0.25, ev['arrow_land'][1], pan_spread=0.6)
        # the creeper goes off
        if 'creeper_boom' in ev:
            pos = ev['creeper_boom'][1]
            pan, g = _pan_gain(pos, cam, tgt)
            b = resample(make_explosion(rng, 2.6), pf)
            mix.add(b, t, 1.0 * max(g, 0.7), pan)
            verb.add(b, t, 0.5)
            mix.add(resample(make_boom(rng, 2.6), pf), t, 0.8, pan)
        if 'powder' in ev:
            put(powders, ev['powder'][0] / 300.0, 2, 0.25, None, pitch=(0.9, 1.3))
        # anvils
        if 'land_flesh' in ev:
            put(crushes, ev['land_flesh'][0], 3, 0.9, ev['land_flesh'][1], pan_spread=0.3, send=0.5 * wet)
        if 'land_anvil' in ev:
            put(clangs, ev['land_anvil'][0], 4, 0.65, ev['land_anvil'][1], pan_spread=0.4, send=0.4 * wet)
        if 'land_ground' in ev:
            put(gthuds, ev['land_ground'][0], 3, 0.5, ev['land_ground'][1], pan_spread=0.6)
        if 'slide' in ev:
            put(rattles, ev['slide'][0] / 3.0, 2, 0.3, ev['slide'][1], pan_spread=0.5)
        if 'anvil_thrown' in ev or 'anvil_knock' in ev:
            n = ev.get('anvil_thrown', [0])[0] + ev.get('anvil_knock', [0])[0]
            put(clangs, n / 4.0, 3, 0.4, (ev.get('anvil_thrown') or ev.get('anvil_knock'))[1], pitch=(1.0, 1.3))
        if 'crush' in ev and ev['crush'][0] > 150:
            put(cracks, ev['crush'][0] / 400.0, 2, 0.35, ev['crush'][1], send=0.4 * wet)
        if 'stack_drop' in ev:
            put(rattles, 1, 1, 0.35, None)
        # limbs, a toppling giant, big pieces
        if 'sever' in ev:
            pan, g = _pan_gain(ev['sever'][1], cam, tgt)
            s = resample(rips[rng.integers(len(rips))], pf)
            mix.add(s, t, 0.8 * g, pan)
            verb.add(s, t, 0.35)
        if 'chunk_land' in ev and ev['chunk_land'][0] > 2000:
            pan, g = _pan_gain(ev['chunk_land'][1], cam, tgt)
            s = resample(heavy[rng.integers(len(heavy))], pf * (0.8 if ev['chunk_land'][0] > 20000 else 1.0))
            mix.add(s, t, (1.0 if ev['chunk_land'][0] > 20000 else 0.7) * max(g, 0.6), pan)
            verb.add(s, t, 0.3)
        # TNT
        if 'explode' in ev:
            n, pos = ev['explode']
            pan, g = _pan_gain(pos, cam, tgt)
            if n <= 3:
                for _ in range(n):
                    s = resample(booms[rng.integers(len(booms))], rng.uniform(0.9, 1.1) * pf)
                    mix.add(s, t + rng.uniform(0, 1.0 / fps), 0.8 * g, pan + rng.uniform(-0.2, 0.2))
                    verb.add(s, t, 0.3 * wet)
            else:
                if i - last_big >= 3:
                    s = resample(big_booms[rng.integers(len(big_booms))], rng.uniform(0.85, 1.05) * pf)
                    mix.add(s, t, 0.75 * g, pan + rng.uniform(-0.3, 0.3))
                    verb.add(s, t, 0.3 * wet)
                    last_big = i
                put(far_booms, n / 6.0, 3, 0.45, pos, pitch=(0.8, 1.15), pan_spread=0.7)
        if 'blast' in ev:
            put(sprays, ev['blast'][0] / 400.0, 3, 0.3, ev['blast'][1], pan_spread=0.6)
        if 'ground' in ev:
            put(gravels, ev['ground'][0] / 10.0, 4, 0.35, ev['ground'][1], pitch=(0.7, 1.0), pan_spread=0.6)
        # debris
        if 'debris_ground' in ev:
            put(ticks, ev['debris_ground'][0] / 8.0, 6, 0.18, ev['debris_ground'][1], pitch=(0.8, 1.3), pan_spread=0.6)
        if 'crumble' in ev:
            put(gravels, ev['crumble'][0] / 200.0, 2, 0.18, None, pitch=(0.7, 1.0))
        # the Warden
        if c.get('beat'):
            dang = c.get('danger', 1.0 if segs[i] == 'cold' else 0.3)
            quiet = 1.0 - 0.45 * min(1.0, np.log1p(rate[i]) / np.log1p(200))
            mix.add(resample(beat, pf), t, (0.18 + 0.4 * dang) * quiet)
        if 'boom' in ev:
            ch = resample(charge, pf)
            mix.add(ch, t - len(ch) / SR, 0.55)
            b = resample(sboom, pf)
            mix.add(b, t, 1.0)
            verb.add(b, t, 0.45)
        if 'hp' in c and c['hp'][3] < c['hp_prev'][3] and (i - last_growl) > fps * 1.2:
            mix.add(resample(growls[rng.integers(len(growls))], pf * rng.uniform(0.92, 1.05)), t + 0.05, 0.35)
            last_growl = i
        # the edit
        hud = c.get('hud', {})
        if 'intro' in hud:
            s, pre = make_slam(rng)
            mix.add(s, t - pre, 0.6)
            music.add(taikos[rng.integers(len(taikos))], t, 0.55)
        if 'banner' in hud:
            w = make_whoosh(rng, 0.35, 0.5)
            mix.add(hp(w, 200.0, 2), t - 0.33, 0.45)
            music.add(make_braam(rng, 1.8, 55.0), t, 0.75)
            music.add(taikos[rng.integers(len(taikos))], t, 0.5)
        if 'elim' in hud:
            music.add(make_elim(rng), t, 0.85)
        if 'winner' in hud:
            sw = make_swell(rng, 0.7)
            music.add(sw, t - len(sw) / SR, 0.4)
            music.add(make_victory(rng), t, 0.9)

    # ---- the score: a dark drone that grows round by round and makes way for the victory chord
    inten = np.zeros(nfr)
    for i, s in enumerate(shots):
        if segs[i] == 'cold':
            v = 0.75
        elif s.startswith('L_'):
            v = 0.35
        elif s.startswith('R1'):
            v = 0.45
        elif s.startswith('R2'):
            v = 0.6
        elif s.startswith('R3'):
            v = 0.85
        else:
            v = 0.0
        inten[i] = v
    dark, bright = make_drone(rng, dur + 4.0)
    music.add(dark * env(inten, 1.2), 0.0, 0.3)
    music.add(bright * env(inten ** 2, 1.2), 0.0, 0.14)

    n = int(dur * SR)
    ir = reverb_ir(rng)
    wl = signal.fftconvolve(verb.L[:n], ir)[:n]
    wr = signal.fftconvolve(verb.R[:n], ir)[:n]
    L, R = mix.L[:n] + wl, mix.R[:n] + wr
    # duck the score under the effects
    fx_env = np.sqrt(lp(0.5 * (L * L + R * R), 6.0, 1).clip(0))
    ref = np.percentile(fx_env, 95) + 1e-9
    duck = np.clip(1.0 - 0.6 * fx_env / ref, 0.35, 1.0)
    L = L + music.L[:n] * duck
    R = R + music.R[:n] * duck
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
