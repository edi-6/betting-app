"""Procedural sound design and music for the hydraulic press, driven by the cue sheet of the render.

Everything is synthesised here (no samples). The music is drift phonk at 130 BPM in C# minor: distorted 808s, a
clap on 2 and 4, metallic 808 hats and the pitched 808 cowbell riff, on a beat grid that every crush lands on. A
muffled version under the cold open that stops like a tape when the video rewinds; a pickup into the first crush;
a breakdown when the bedrock goes on (a heartbeat, a sub drone climbing a semitone, the riff opening up) and the
drop when the press blows up. The effects: the hotbar click and each block's place sound, the pump's whine and the
oil's hiss as the ram moves, the clunk of contact, metal groaning under load, each material giving way its own way
(dirt, glass, melon, slime's boing, TNT's fuse and blast, splintering wood and item pops, diamond, obsidian), sparks
grinding, a hydraulic line bursting, the alarm, the press exploding and its pieces raining down, the
'Challenge Complete!' fanfare. Slow motion drops the pitch and muffles the music. Loudness normalised to -14 LUFS
with a peak limiter, balanced for phone speakers.
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


def make_squelch(rng, dur=0.4):
    x = shaped_noise(dur, lambda t: 1100 * np.exp(-t / 0.2) + 300, lambda t: 1.0 + 0 * t,
                     lambda t: np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.1), rng)
    n = len(x)
    t = np.arange(n) / SR
    x *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * rng.uniform(25, 40) * t + rng.uniform(0, 6)))
    x = lp(x, 3000, 2)
    return norm(x + 0.4 * grains(dur, 30, 400, 2200, 0.08, rng), 0.7)


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


def make_block_land(rng):
    """A block (or a chunk of a giant) landing back on the ground."""
    dur = 0.3
    n = int(dur * SR)
    thud = sine_sweep(dur, rng.uniform(130, 190), 70, 0.02) * expenv(n, 0.05, 0.001)
    grit = grains(dur, int(rng.integers(8, 18)), 300, 2500, 0.03, rng, glen=(0.002, 0.008))
    return norm(norm(thud) + 0.6 * norm(grit), 0.8)


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


def _env_track(values, fps, n, smooth_hz):
    """Per-frame values -> smoothed per-sample envelope of length n."""
    v = np.interp(np.arange(n) / SR * fps, np.arange(len(values)), values)
    return np.clip(lp(np.concatenate([v, np.zeros(SR)])[:n], smooth_hz, 1), 0.0, None)


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

# ---------------------------------------------------------------------------------------------
# the press and the blocks
# ---------------------------------------------------------------------------------------------
def make_click(rng):
    """The hotbar: a short, dry UI click."""
    dur = 0.05
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 1850 * t) * expenv(n, 0.006)
    x += 0.6 * bp(rng.standard_normal(n), 2500, 9000) * expenv(n, 0.002)
    x += 0.4 * np.sin(2 * np.pi * 620 * t) * expenv(n, 0.01)
    return norm(x, 0.7)


def make_place(rng, kind):
    """A block going down on the platen, per material (like the game's place sounds)."""
    dur = 0.28
    n = int(dur * SR)
    t = np.arange(n) / SR
    thud = sine_sweep(dur, 170, 80, 0.02) * expenv(n, 0.045, 0.001)
    if kind in ('grass', 'tnt', 'melon'):
        body = grains(dur, 24, 250, 2200, 0.025, rng, glen=(0.002, 0.008))
    elif kind == 'glass':
        body = modal(dur, 2300.0, (1.0, 1.52, 2.31, 3.1), (1.0, 0.6, 0.5, 0.3), (0.06, 0.05, 0.04, 0.03), rng, 0.02)
    elif kind == 'slime':
        body = make_squelch(rng, dur) * 0.8
    elif kind == 'chest':
        body = bp(rng.standard_normal(n), 250, 1400) * expenv(n, 0.03) + 0.5 * modal(
            dur, 190.0, (1.0, 2.7, 5.1), (1.0, 0.5, 0.3), (0.05, 0.03, 0.02), rng)
    elif kind == 'diamond':
        body = modal(dur, 1400.0, (1.0, 2.0, 3.01, 4.2), (1.0, 0.5, 0.35, 0.2), (0.14, 0.1, 0.08, 0.05), rng, 0.0)
    else:                                                  # obsidian, bedrock: heavy stone
        body = grains(dur, 30, 150, 1600, 0.03, rng, glen=(0.002, 0.01))
        thud = sine_sweep(dur, 120, 45, 0.03) * expenv(n, 0.09, 0.001)
    x = norm(thud) + 0.7 * norm(body)
    return norm(x * np.clip((dur - t) / 0.02, 0, 1), 0.8)


def make_hydraulic(rng, vz, fps, n):
    """The ram moving: the pump's whine (pitch following the speed) and the oil's hiss, per sample from the
    ram's vertical speed per frame (negative going down)."""
    v = _env_track(np.abs(vz), fps, n, 12.0)
    down = _env_track(np.clip(-np.asarray(vz), 0, None), fps, n, 12.0)
    a = np.clip(v / 18.0, 0, 1) ** 0.7
    t = np.arange(n) / SR
    f = 150.0 + 11.0 * np.minimum(v, 40.0) - 30.0 * np.clip(down / 18.0, 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / SR
    whine = np.sin(ph) + 0.5 * np.sin(2 * ph + 0.3) + 0.3 * np.sin(3 * ph) + 0.2 * np.sign(np.sin(4 * ph))
    whine = bp(whine, 120, 2600, 2) * (1 + 0.15 * np.sin(2 * np.pi * 31 * t))
    hiss = bp(rng.standard_normal(n), 1200, 7000, 2)
    return norm(whine) * a * 0.55 + norm(hiss) * a * 0.35


def make_contact(rng):
    """The ram meets the block: a heavy metal clunk."""
    dur = 0.7
    n = int(dur * SR)
    ring = modal(dur, 140.0, IRON, (1.0, 0.6, 0.5, 0.35, 0.25, 0.2), (0.25, 0.18, 0.12, 0.08, 0.05, 0.04), rng)
    knock = bp(rng.standard_normal(n), 150, 1500) * expenv(n, 0.02, 0.0005)
    body = sine_sweep(dur, 110, 48, 0.03) * expenv(n, 0.12, 0.001)
    return norm(np.tanh(1.4 * (0.6 * norm(ring) + 0.8 * norm(knock) + norm(body))), 0.9)


def make_strain(rng, dur, hardness):
    """Metal under load: a groan that climbs, creaks and ticks (hardness 0..1 makes it heavier)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / max(dur, 1e-3)
    g = make_groan(rng, dur, f0=58.0 + 30.0 * hardness)
    ticks = grains(dur, int(dur * (25 + 90 * hardness)), 1500, 6000, dur * 0.7, rng, glen=(0.0008, 0.003))
    ticks *= u ** 1.5
    x = norm(g) + 0.5 * norm(ticks)
    return norm(x * np.clip(t / 0.08, 0, 1), 0.8)


def make_grind(rng, dur):
    """Metal grinding on stone with sparks flying: a harsh, fluttering screech."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    am = 0.6 + 0.4 * lp(rng.standard_normal(n), 25.0, 1) / 0.05
    am = np.clip(am, 0.1, 1.6)
    x = bp(rng.standard_normal(n), 2400, 7800, 2) * am
    f = 3100 + 400 * np.sin(2 * np.pi * 5.3 * t) + 200 * rng.standard_normal(n) * 0.02
    scr = np.sin(2 * np.pi * np.cumsum(f) / SR)
    crack = grains(dur, int(dur * 220), 2500, 9000, dur * 0.8, rng, glen=(0.0005, 0.002))
    return norm(norm(x) + 0.08 * norm(scr) + 0.6 * norm(crack), 0.8)


def make_shatter(rng, size=1.0):
    """Glass: a bright crash and a shower of tinkling shards."""
    dur = 1.4
    n = int(dur * SR)
    crash = bp(rng.standard_normal(n), 2500, 12000) * expenv(n, 0.05, 0.0005)
    body = bp(rng.standard_normal(n), 600, 3000) * expenv(n, 0.03, 0.0005)
    shards = np.zeros(n)
    for _ in range(int(70 * size)):
        s = int(min(n - 1, rng.exponential(0.18) * SR))
        f0 = rng.uniform(2200, 7500)
        L = int(rng.uniform(0.03, 0.18) * SR)
        e = min(n, s + L)
        tt = np.arange(e - s) / SR
        shards[s:e] += np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / rng.uniform(0.01, 0.05)) * rng.uniform(0.2, 1.0)
    x = 0.9 * norm(crash) + 0.5 * norm(body) + 0.8 * norm(shards)
    return norm(x, 0.9)


def make_splat(rng):
    """The melon: a wet, pulpy burst."""
    dur = 0.9
    n = int(dur * SR)
    thump = sine_sweep(dur, 150, 60, 0.03) * expenv(n, 0.07, 0.001)
    wet = make_squelch(rng, 0.5)
    wet = np.concatenate([wet, np.zeros(n - len(wet))])
    crunch = grains(dur, 80, 400, 3000, 0.05, rng, glen=(0.002, 0.01))
    drips = np.zeros(n)
    for _ in range(14):
        s = int(rng.uniform(0.08, 0.8) * SR)
        L = int(0.03 * SR)
        e = min(n, s + L)
        tt = np.arange(e - s) / SR
        drips[s:e] += np.sin(2 * np.pi * np.cumsum(rng.uniform(500, 900) * (1 + 3 * tt)) / SR) * np.exp(-tt / 0.01)
    x = norm(thump) + 0.9 * norm(wet) + 0.6 * norm(crunch) + 0.2 * norm(drips)
    return norm(np.tanh(1.3 * x), 0.9)


def make_boing(rng):
    """Slime throwing the ram back up: a springy, wobbling BOING."""
    dur = 0.8
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 150 + 260 * (1 - np.exp(-t / 0.04)) * (1 + 0.25 * np.sin(2 * np.pi * 11 * t) * np.exp(-t / 0.3))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    x *= expenv(n, 0.3, 0.004)
    sq = make_squelch(rng, 0.35)
    x[:len(sq)] += 0.5 * norm(sq)
    return norm(x, 0.85)


def make_wood_burst(rng):
    """The chest: planks cracking and splintering apart."""
    dur = 0.8
    n = int(dur * SR)
    crack = bp(rng.standard_normal(n), 900, 6000) * expenv(n, 0.012, 0.0003)
    splinter = grains(dur, 160, 700, 4200, 0.09, rng, glen=(0.002, 0.012))
    knock = modal(dur, 170.0, (1.0, 2.6, 4.9), (1.0, 0.5, 0.3), (0.06, 0.04, 0.03), rng) + \
        modal(dur, 240.0, (1.0, 2.4, 4.3), (1.0, 0.5, 0.3), (0.05, 0.035, 0.02), rng)
    thump = sine_sweep(dur, 140, 55, 0.03) * expenv(n, 0.08, 0.001)
    x = 0.8 * norm(crack) + norm(splinter) + 0.6 * norm(knock) + 0.8 * norm(thump)
    return norm(np.tanh(1.3 * x), 0.9)


def make_pop(rng):
    """The game's item pop: a short blip that bends upwards."""
    dur = 0.09
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(380, 620)
    f = f0 * (1 + 1.6 * (1 - np.exp(-t / 0.02)))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
    return norm(x, 0.6)


def make_crystal(rng):
    """Diamond giving way: a glassy crystalline shatter over a hard crunch."""
    dur = 1.5
    n = int(dur * SR)
    sh = make_shatter(rng, 1.3)
    sh = np.concatenate([sh, np.zeros(max(0, n - len(sh)))])[:n]
    chime = np.zeros(n)
    for f0 in rng.uniform(1800, 5200, 10):
        s = int(rng.exponential(0.12) * SR)
        if s >= n:
            continue
        L = n - s
        tt = np.arange(L) / SR
        chime[s:] += np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / rng.uniform(0.15, 0.5)) * rng.uniform(0.3, 1.0)
    crunch = make_crunch(rng, 300, 2500, dur)
    thump = sine_sweep(dur, 160, 55, 0.03) * expenv(n, 0.1, 0.001)
    x = norm(sh) + 0.4 * norm(chime) + 0.6 * norm(crunch) + 0.7 * norm(thump)
    return norm(np.tanh(1.2 * x), 0.92)


def make_stone_break(rng, heavy=1.0):
    """Stone (grass block's dirt at heavy 0, obsidian at 1): a gritty crunch and a thump, with rubble."""
    dur = 1.0 + 0.4 * heavy
    n = int(dur * SR)
    crunch = grains(dur, int(120 + 200 * heavy), 180 + 150 * heavy, 2200 + 1800 * heavy, 0.06 + 0.05 * heavy, rng,
                    glen=(0.002, 0.012))
    thump = sine_sweep(dur, 150, 42, 0.04) * expenv(n, 0.1 + 0.08 * heavy, 0.001)
    crack = bp(rng.standard_normal(n), 1200, 9000) * expenv(n, 0.006 + 0.004 * heavy)
    rubble = grains(dur, int(60 + 120 * heavy), 300, 3000, 0.35, rng, glen=(0.002, 0.01), decay=0.5)
    x = norm(crunch) + 0.9 * norm(thump) + 0.5 * norm(crack) + 0.4 * norm(rubble)
    return norm(np.tanh(1.4 * x), 0.92)


def make_magic(rng, dur=1.4):
    """Obsidian's purple portal shimmer: a detuned glassy swirl."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f0 in (880.0, 1318.5, 1760.0, 2093.0):
        x += np.sin(2 * np.pi * f0 * (1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)) * t + rng.uniform(0, 6))
    x *= np.exp(-t / 0.5) * np.clip(t / 0.02, 0, 1)
    air = shaped_noise(dur, lambda tt: 3000 + 2000 * np.sin(tt * 9), lambda tt: 0.5 + 0 * tt,
                       lambda tt: np.exp(-tt / 0.4), rng)
    return norm(norm(x) + 0.4 * norm(air), 0.6)


def make_hose_burst(rng):
    """A hydraulic line giving way: a sharp pop and a violent spray of oil."""
    dur = 1.0
    n = int(dur * SR)
    pop = bp(rng.standard_normal(n), 500, 6000) * expenv(n, 0.01, 0.0003)
    thump = sine_sweep(dur, 180, 70, 0.02) * expenv(n, 0.06, 0.001)
    spray = shaped_noise(dur, lambda tt: 2500 + 1500 * np.exp(-tt / 0.1), lambda tt: 1.0 + 0 * tt,
                         lambda tt: np.clip(tt / 0.01, 0, 1) * (0.6 + 0.4 * np.exp(-tt / 0.2)), rng)
    return norm(0.9 * norm(pop) + 0.8 * norm(thump) + norm(spray), 0.9)


def make_spray(rng, dur):
    """Oil spraying out under pressure: a sputtering hiss."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 1800, 9000, 2)
    sput = 0.7 + 0.3 * np.sign(np.sin(2 * np.pi * 17 * t + 3 * np.sin(2 * np.pi * 2.3 * t)))
    return norm(x * sput, 0.6)


def make_alarm(dur, period):
    """An industrial alarm: a square tone sweeping up once per light pulse."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = np.mod(t / period, 1.0)
    f = 620 + 420 * u
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sign(np.sin(ph)) * 0.6 + 0.4 * np.sin(2 * ph)
    x = bp(x, 400, 3500, 2) * (0.35 + 0.65 * np.sin(np.pi * u) ** 0.5)
    return norm(x, 0.6)


def make_shrapnel(rng, dur=2.4):
    """Pieces of the press raining down: clanks and rattles of steel."""
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(26):
        s = int(min(n - 1, (0.25 + rng.exponential(0.5)) * SR))
        c = make_clang(rng) * rng.uniform(0.2, 0.8)
        c = resample(c, rng.uniform(0.7, 1.8))
        e = min(n, s + len(c))
        x[s:e] += c[:e - s]
    return norm(x, 0.8)


def make_fanfare(rng):
    """'Challenge Complete!': a bright rising arpeggio into a shimmering chord."""
    dur = 2.2
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    notes = (523.25, 659.25, 783.99, 1046.5)
    for k, f in enumerate(notes):
        s = int(k * 0.09 * SR)
        L = n - s
        tt = np.arange(L) / SR
        tone = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt) + 0.2 * np.sin(2 * np.pi * 3 * f * tt)
        x[s:] += tone * np.exp(-tt / (0.25 if k < 3 else 0.9)) * np.clip(tt / 0.004, 0, 1)
    shimmer = np.zeros(n)
    for f in (1567.98, 2093.0, 2637.0):
        shimmer += np.sin(2 * np.pi * f * t * (1 + 0.002 * np.sin(2 * np.pi * 6 * t)))
    shimmer *= np.clip((t - 0.27) / 0.05, 0, 1) * np.exp(-np.clip(t - 0.27, 0, None) / 0.7)
    return norm(norm(x) + 0.3 * norm(shimmer), 0.75)


def make_title_hit(rng):
    """Under 'UNBREAKABLE': a dark, heavy braam."""
    return make_braam(rng, 2.2, 34.65 * 2)


# ---------------------------------------------------------------------------------------------
# the beat: drift phonk at 130 BPM in C# minor (808s, claps, metallic hats, the cowbell riff)
# ---------------------------------------------------------------------------------------------
def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def kick(rng):
    dur = 0.45
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 48 + 160 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.2, 0.001)
    click = bp(rng.standard_normal(n), 2000, 9000) * expenv(n, 0.003)
    return norm(np.tanh(2.2 * (body + 0.25 * click)), 0.95)


def bass808(f0, dur, glide_from=None, drive=2.4):
    """A long distorted 808: pitch drop on the attack, an optional glide from the previous note."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.full(n, f0)
    if glide_from is not None:
        f = f0 + (glide_from - f0) * np.exp(-t / 0.05)
    f = f * (1 + 1.2 * np.exp(-t / 0.015))
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.clip(t / 0.003, 0, 1) * np.exp(-t / max(0.25, dur * 0.8)) * np.clip((dur - t) / 0.03, 0, 1)
    x = np.tanh(drive * np.sin(ph) * env) / np.tanh(drive)
    return x + 0.12 * np.tanh(3 * np.sin(2 * ph)) * env


def clap(rng):
    dur = 0.4
    n = int(dur * SR)
    x = np.zeros(n)
    for k, d in enumerate((0.0, 0.009, 0.019, 0.03)):
        s = int(d * SR)
        L = int(0.01 * SR) if k < 3 else n - s
        tt = np.arange(L) / SR
        x[s:s + L] += rng.standard_normal(L) * np.exp(-tt / (0.003 if k < 3 else 0.1))
    x = bp(x, 800, 5200, 2)
    return norm(x, 0.8)


HAT_F = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def hat(rng, open_=False):
    dur = 0.35 if open_ else 0.06
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(np.sign(np.sin(2 * np.pi * f * 1.3 * t + rng.uniform(0, 6))) for f in HAT_F)
    x = hp(bp(x, 6000, 16000, 2), 7000, 2) + 0.3 * hp(rng.standard_normal(n), 8000, 2)
    return norm(x * expenv(n, 0.12 if open_ else 0.018, 0.0005), 0.6)


def cowbell(f, rng, dur=0.42):
    """The 808 cowbell (two square waves, ~1:1.48) tuned to a note, with a click and a ringing tail."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * f * t)) + np.sign(np.sin(2 * np.pi * f * 1.48 * t + 0.4))
    x = bp(x, f * 0.8, f * 5.0, 2)
    env = 0.65 * np.exp(-t / 0.02) + 0.35 * np.exp(-t / 0.16)
    return norm(np.tanh(1.6 * x * env), 0.7)


# the riff: 32 sixteenths (two bars), semitones above C#5, in a 3-3-2 bounce
RIFF = {0: 0, 3: 0, 6: 3, 8: 0, 10: 7, 12: 5, 14: 3,
        16: 0, 19: 0, 22: 3, 24: 0, 26: -2, 28: -4, 30: -5}
RIFF_B = {0: 12, 3: 12, 6: 10, 8: 7, 10: 12, 12: 15, 14: 12,
          16: 12, 19: 12, 22: 10, 24: 7, 26: 5, 28: 3, 30: 2}
# the 808 line: step -> (midi note, length in steps)
BASS = {0: (37, 10), 10: (40, 6), 16: (37, 6), 22: (35, 4), 26: (33, 6)}
KICKS = (0, 10, 16, 22, 26)


class Beat:
    """Places drum hits and notes on the video's beat grid (beat 0 = the first block giving way)."""

    def __init__(self, g0, beat, n, rng):
        self.g0, self.beat, self.n = g0, beat, n
        self.rng = rng
        self.L = np.zeros(n)
        self.R = np.zeros(n)
        self.bank_hat = [hat(rng) for _ in range(6)]
        self.bank_ohat = [hat(rng, True) for _ in range(2)]
        self.bank_clap = [clap(rng) for _ in range(3)]
        self.kick = kick(rng)
        self.cow = {}

    def t(self, b):
        return self.g0 + b * self.beat

    def put(self, x, b, gain=1.0, pan=0.0):
        s = int(self.t(b) * SR)
        if s >= self.n or s + len(x) <= 0:
            return
        if s < 0:
            x = x[-s:]
            s = 0
        e = min(self.n, s + len(x))
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        self.L[s:e] += x[:e - s] * gain * gl
        self.R[s:e] += x[:e - s] * gain * gr

    def cowbell(self, semi, b, gain, pan=0.0):
        f = midi_hz(73 + semi)
        if semi not in self.cow:
            self.cow[semi] = cowbell(f, self.rng)
        self.put(self.cow[semi], b, gain, pan)
        self.put(self.cow[semi], b + 0.75, gain * 0.22, -pan)      # a dotted-eighth echo

    def bar2(self, b0, level=1.0, cow=True, hats16=False, riff=RIFF, kick_on=True, bass_on=True, clap_on=True):
        """Two bars of the groove starting at beat b0."""
        rng = self.rng
        for s in range(32):
            b = b0 + s / 4.0
            if kick_on and s in KICKS:
                self.put(self.kick, b, 0.95 * level)
            if bass_on and s in BASS:
                m, L = BASS[s]
                prev = [BASS[k][0] for k in sorted(BASS) if k < s]
                glide = midi_hz(prev[-1]) if prev and s in (10, 26) else None
                self.put(bass808(midi_hz(m), L / 4.0 * self.beat, glide), b, 0.8 * level)
            if clap_on and s % 8 == 4:
                self.put(self.bank_clap[rng.integers(3)], b, 0.7 * level)
            if s % 2 == 0 or hats16 or (s >= 28 and s % 1 == 0):
                self.put(self.bank_hat[rng.integers(6)], b, (0.32 if s % 4 == 0 else 0.22) * level,
                         pan=rng.uniform(-0.3, 0.3))
            if s == 14:
                self.put(self.bank_ohat[rng.integers(2)], b, 0.25 * level)
            if cow and s in riff:
                self.cowbell(riff[s], b, 0.55 * level, pan=0.15 if s % 2 else -0.15)

    def fill(self, b0, beats=2.0, level=1.0):
        """A clap roll into a drop."""
        k = int(beats * 4)
        for s in range(k):
            u = s / max(1, k - 1)
            step = 0.25 if u < 0.5 else 0.125
            self.put(self.bank_clap[s % 3], b0 + s * 0.25, (0.25 + 0.5 * u) * level)
            if u >= 0.5:
                self.put(self.bank_clap[(s + 1) % 3], b0 + s * 0.25 + step, (0.25 + 0.5 * u) * level * 0.8)


# ---------------------------------------------------------------------------------------------
# build from cues
# ---------------------------------------------------------------------------------------------
HARD = {'grass': 0.1, 'glass': 0.25, 'melon': 0.1, 'slime': 0.0, 'tnt': 0.2, 'chest': 0.3, 'diamond': 0.75,
        'obsidian': 0.9, 'bedrock': 1.0}


def tape_stop(x, s0, dur):
    """Slow a buffer to a halt from sample s0 over dur seconds (pitch falls with the speed); silence after."""
    n = int(dur * SR)
    rate = np.linspace(1.0, 0.0, n) ** 1.3
    pos = s0 + np.cumsum(rate)
    out = x.copy()
    seg = np.interp(pos, np.arange(len(x)), x) * np.linspace(1.0, 0.0, n) ** 0.5
    e = min(len(x), s0 + n)
    out[s0:e] = seg[:e - s0]
    out[e:] = 0.0
    return out


def build(meta, out_path, seed=7):
    rng = np.random.default_rng(seed)
    cues = meta['frames']
    fps = meta['fps']
    g0, beat = meta['grid']
    nf = len(cues)
    dur = nf / fps
    n = int(dur * SR) + SR
    mix = Mix(dur)

    def tv(i):
        return i / fps

    seg = [c.get('seg') for c in cues]
    rew = [i for i in range(nf) if seg[i] == 'rewind']
    t_rew = tv(rew[0]) if rew else None
    t_rew_end = tv(rew[-1] + 1) if rew else None
    scale = np.array([c.get('scale', 1.0) for c in cues], float)
    slow = (scale > 0) & (scale < 0.95)

    def put(x, t, g, pan=0.0, sc=1.0):
        if 0 < sc < 0.95:
            x = resample(x, max(0.45, np.sqrt(sc)))
        mix.add(x, t, g, pan)

    # ---- one-shot sounds from the story
    if seg and seg[0] == 'cold':
        put(make_hit(rng), 0.0, 0.9)
        put(make_contact(rng), 0.0, 0.6)
    prev_vz = 0.0
    prime_t = None
    hoses_t = None
    press_t = None
    strain = []                                   # (t0, t1, kind)
    duck_hits = []                                # (t, depth, tau)
    t_place_bedrock = None
    for i, c in enumerate(cues):
        ev = c.get('events', {}) or {}
        t = tv(i)
        sc = scale[i]
        kind = c.get('kind')
        if 'place' in ev:
            k = ev['place'].get('kind')
            if seg[i] == 'main':
                put(make_click(rng), t, 0.55)
            put(make_place(rng, k), t + 0.06, 0.55)
            if k == 'bedrock' and seg[i] == 'main':
                t_place_bedrock = t
        vz = c.get('ram_vz', 0.0) or 0.0
        if prev_vz < -12.0 and vz > -6.0 and c.get('state') == 'whole':
            put(make_contact(rng), t, 0.7, sc=sc)
        prev_vz = vz
        # strain: the ram pressing on a whole block
        pressing = c.get('state') == 'whole' and (c.get('ram_z', 99.0) or 99.0) <= 10.05 and kind is not None
        if pressing and (not strain or strain[-1][1] is not None):
            strain.append([t, None, kind])
        if not pressing and strain and strain[-1][1] is None:
            strain[-1][1] = t
        if 'bounce' in ev:
            put(make_boing(rng), t, 0.85, sc=sc)
        if 'prime' in ev:
            prime_t = t
        if 'break' in ev:
            k = ev['break'].get('kind')
            if k == 'grass':
                put(make_stone_break(rng, 0.0), t, 0.9, sc=sc)
            elif k == 'glass':
                put(make_shatter(rng), t, 0.95, sc=sc)
            elif k == 'melon':
                put(make_splat(rng), t, 0.95, sc=sc)
            elif k == 'slime':
                put(make_splat(rng), t, 0.8, sc=sc)
                put(make_squelch(rng, 0.6), t, 0.7, sc=sc)
            elif k == 'chest':
                put(make_wood_burst(rng), t, 0.95, sc=sc)
            elif k == 'diamond':
                put(make_crystal(rng), t, 1.0, sc=sc)
            elif k == 'obsidian':
                put(make_stone_break(rng, 1.0), t, 1.0, sc=sc)
                put(make_magic(rng), t + 0.05, 0.45, sc=sc)
            if k != 'tnt':
                put(make_hit(rng), t, 0.5, sc=sc)
                duck_hits.append((t, 0.35, 0.25))
        if 'explode' in ev:
            if prime_t is not None:
                put(make_hiss(rng, max(0.1, t - prime_t)), prime_t, 0.35)
                prime_t = None
            put(make_explosion(rng, 1.6), t, 1.0, sc=sc)
            duck_hits.append((t, 0.6, 0.5))
        if 'loot' in ev:
            for k in range(int(ev['loot'].get('n', 8))):
                put(make_pop(rng), t + 0.12 + rng.exponential(0.25), 0.4, pan=rng.uniform(-0.5, 0.5))
        if 'land' in ev and ev['land'].get('speed', 0) > 6.0:
            m = ev['land'].get('n', 1)
            for _ in range(min(3, 1 + m // 15)):
                put(resample(make_block_land(rng), rng.uniform(0.9, 1.6)), t + rng.uniform(0, 1.0 / fps),
                    min(0.3, 0.06 + 0.01 * m), pan=rng.uniform(-0.6, 0.6), sc=sc)
        if 'hoses' in ev:
            hoses_t = t
            put(make_hose_burst(rng), t, 0.8, sc=sc)
        if 'press_break' in ev:
            press_t = t
            put(make_explosion(rng, 3.2), t, 1.0, sc=sc)
            put(make_shrapnel(rng), t + 0.2, 0.55, sc=sc)
            put(make_title_hit(rng), t, 0.55, sc=sc)
            duck_hits.append((t, 0.7, 1.2))
        h = c.get('hud', {}) or {}
        if h.get('title') == 'UNBREAKABLE':
            put(make_braam(rng, 2.4, 69.3), t, 0.55)
        if h.get('toast'):
            put(make_fanfare(rng), t, 0.5)
        # the end of a segment: sounds that last until it
        if i + 1 == nf or seg[i + 1] != seg[i]:
            if hoses_t is not None:
                end = press_t if press_t is not None else t
                put(make_spray(rng, end - hoses_t + 0.6), hoses_t + 0.05, 0.4)
                put(make_alarm(end - hoses_t + 0.5, 1.0 / 2.2), hoses_t, 0.42)
                hoses_t = None
            if strain and strain[-1][1] is None:
                strain[-1][1] = t
            press_t = None
    for (t0, t1, k) in strain:
        d = max(0.15, t1 - t0)
        if k == 'slime':
            put(make_squelch(rng, min(0.9, d)), t0, 0.6)
            continue
        put(make_strain(rng, d + 0.1, HARD.get(k, 0.3)), t0, 1.05 if k == 'bedrock' else 0.35 + 0.4 * HARD.get(k, 0.3))
    # slow motion in and out
    for i in range(1, nf):
        if slow[i] and not slow[i - 1] and seg[i] == 'main':
            put(make_slowmo_down(rng), tv(i), 0.5)
        if slow[i - 1] and not slow[i] and seg[i] == 'main':
            put(make_slowmo_up(rng), tv(i) - 0.45, 0.4)
    if t_rew is not None:
        put(make_rewind(rng, t_rew_end - t_rew), t_rew, 0.7)

    # ---- continuous layers: the ram's hydraulics, the grinding sparks
    vz = np.array([c.get('ram_vz', 0.0) or 0.0 for c in cues], float)
    vz[[i for i in range(nf) if seg[i] == 'rewind']] = 0.0
    hyd = make_hydraulic(rng, vz, fps, n)
    sp = np.array([c.get('n_sparks', 0) or 0 for c in cues], float)
    genv = _env_track(np.clip(sp / 150.0, 0, 1) ** 0.6, fps, n, 8.0)
    grind = make_grind(rng, n / SR)[:n] * genv
    mix.L[:n] += 0.5 * hyd + 0.62 * grind
    mix.R[:n] += 0.5 * hyd + 0.62 * grind

    # ---- the music
    music_L = np.zeros(n)
    music_R = np.zeros(n)
    # cold open: a dark drone, a filtered riff and a heartbeat, stopped like a tape when it rewinds
    if t_rew is not None:
        cb = Beat(0.0, beat, n, rng)
        nb = int(t_rew_end / beat) + 2
        for b in range(0, nb, 8):
            cb.bar2(b, 0.8, cow=True, kick_on=False, bass_on=False, clap_on=False)
        for b in range(nb):
            cb.put(cb.kick, b, 0.7)
        tt = np.arange(n) / SR
        drone = (np.tanh(2.0 * np.sin(2 * np.pi * 34.65 * tt)) + 0.5 * np.sin(2 * np.pi * 69.3 * tt)
                 + 0.25 * np.sin(2 * np.pi * 73.4 * tt))
        drone *= np.clip(tt / 0.3, 0, 1)
        cl = lp(cb.L, 900, 2) + 0.35 * drone
        cr = lp(cb.R, 900, 2) + 0.35 * drone
        s0 = int(t_rew * SR)
        cl = tape_stop(cl, s0, t_rew_end - t_rew)
        cr = tape_stop(cr, s0, t_rew_end - t_rew)
        music_L += cl
        music_R += cr
    # the main beat
    B = Beat(g0, beat, n, rng)
    t_end = dur
    b_end = (t_end - g0) / beat
    b_bd = np.ceil((t_place_bedrock - g0) / beat) if t_place_bedrock is not None else b_end
    press_frames = [i for i in range(nf) if 'press_break' in (cues[i].get('events') or {}) and seg[i] == 'main']
    b_drop = round((tv(press_frames[0]) - g0) / beat * 2) / 2 if press_frames else b_end
    B.fill(-2.0, 2.0, 0.8)
    B.put(make_riser(rng, 2 * beat), -2.0, 0.35)
    b = 0.0
    while b < b_bd:
        B.bar2(b, 1.0, cow=True, hats16=b >= 24, riff=RIFF)
        b += 8
    main_mask = np.ones(n)
    s_bd = int(B.t(b_bd) * SR)
    s_drop = int(B.t(b_drop) * SR)
    fade = int(0.03 * SR)
    main_mask[s_bd:] = 0.0
    main_mask[max(0, s_bd - fade):s_bd] = np.linspace(1, 0, min(fade, s_bd))
    gl, gr = B.L * main_mask, B.R * main_mask
    # the breakdown: a sub drone that climbs a semitone, a heartbeat, the riff muffled and opening up
    D = Beat(g0, beat, n, rng)
    k = b_bd
    while k < b_drop - 0.5:
        D.put(D.kick, k, 0.55 + 0.35 * (k - b_bd) / max(1.0, b_drop - b_bd))
        k += 1
    b = b_bd
    while b < b_drop:
        D.bar2(b, 0.8, cow=True, kick_on=False, bass_on=False, clap_on=False, riff=RIFF)
        b += 8
    tt = np.arange(n) / SR
    u = np.clip((tt - B.t(b_bd)) / max(0.1, B.t(b_drop) - B.t(b_bd)), 0, 1)
    f = 34.65 * 2 ** (u / 12.0)
    sub = np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR)) + 0.4 * np.sin(2 * np.pi * np.cumsum(2 * f) / SR)
    win = ((tt >= B.t(b_bd)) & (tt < B.t(b_drop) - 0.5 * beat)).astype(float)
    win = lp(win, 20.0, 1)
    lo = lp(D.L, 500, 2)
    hi = D.L
    dL = (lo * (1 - u ** 2) + hi * u ** 2) * win + 0.4 * sub * win * (0.4 + 0.6 * u)
    lo = lp(D.R, 500, 2)
    hi = D.R
    dR = (lo * (1 - u ** 2) + hi * u ** 2) * win + 0.4 * sub * win * (0.4 + 0.6 * u)
    riser_t = B.t(b_drop - 4)
    riser = make_riser(rng, 4 * beat - 0.5 * beat)
    # the drop: the press blows up and the beat comes back, harder
    E = Beat(g0, beat, n, rng)
    b = b_drop
    while b < b_end + 1:
        E.bar2(b, 1.05, cow=True, hats16=True, riff=RIFF_B)
        b += 8
    crash = bp(rng.standard_normal(int(2.0 * SR)), 3000, 14000) * expenv(int(2.0 * SR), 0.5, 0.001)
    E.put(norm(crash, 0.6), b_drop, 0.5)
    drop_mask = np.zeros(n)
    drop_mask[s_drop:] = 1.0
    music_L += gl + dL + E.L * drop_mask
    music_R += gr + dR + E.R * drop_mask
    mix_riser = np.zeros(n)
    s = int(riser_t * SR)
    e = min(n, s + len(riser))
    mix_riser[s:e] = riser[:e - s]
    music_L += 0.3 * mix_riser
    music_R += 0.3 * mix_riser
    # muffled under slow motion, ducked under the big hits
    m_slow = _env_track(slow.astype(float), fps, n, 6.0)
    mlL, mlR = lp(music_L, 420, 2), lp(music_R, 420, 2)
    music_L = music_L * (1 - m_slow) + mlL * m_slow * 1.3
    music_R = music_R * (1 - m_slow) + mlR * m_slow * 1.3
    duck = np.ones(n)
    for (t0, depth, tau) in duck_hits:
        s = int(t0 * SR)
        if s >= n:
            continue
        L = min(n - s, int(tau * 5 * SR))
        duck[s:s + L] = np.minimum(duck[s:s + L], 1.0 - depth * np.exp(-np.arange(L) / SR / tau))
    L = mix.L[:n] + 0.46 * music_L * duck
    R = mix.R[:n] + 0.46 * music_R * duck
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    # phone speakers: tame the deep sub, add a little presence
    L = L - 0.45 * lp(L, 80.0, 2) + 0.25 * bp(L, 1800.0, 4500.0, 1)
    R = R - 0.45 * lp(R, 80.0, 2) + 0.25 * bp(R, 1800.0, 4500.0, 1)
    L, R = L[:int(dur * SR)], R[:int(dur * SR)]
    lufs = integrated_lufs(L, R)
    g = 10 ** ((-18.0 - lufs) / 20.0)
    L, R = compress(L * g, R * g, thresh_db=-24.0, ratio=3.0)
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


if __name__ == '__main__':
    import sys
    build(json.load(open(sys.argv[1])), sys.argv[2])
