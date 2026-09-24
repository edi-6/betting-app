"""Procedural music and sound design for the mob marble race, driven by the cue sheet of the render.

Everything is synthesised here (no samples). The music is a 144 BPM chiptune dance track in A minor: square-wave
arpeggios, a detuned saw pad, an off-beat bass and a four-on-the-floor beat pumping the rest. At 144 BPM the
countdown's numbers land on every second beat and the beat drops on GO; each round builds the arrangement up
(rolling arpeggios, sixteenth hats, then a lead); slow motion muffles it; the final strips it to a snare roll and
a riser, cuts it dead for the decisive moment (a heartbeat and a swell) and answers with a brass fanfare and a
victory groove in C major. The effects: glassy ticks off the end-rod pegs, marbles clacking together, knocks on
wood and stone, clinks on ice, boings off slime, the pack rolling, pistons, the TNT's fuse and blast, trapdoors
dropping open, lava splashing, sizzling and bubbling, a blip for every marble counted through, the broadcast
stings (round cards, ELIMINATED, SAVED, JUST IN), fireworks and confetti. Sounds of the race play slower and
deeper in slow motion. Loudness normalised to -14 LUFS with a peak limiter, balanced for phone speakers.
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
# sound effects
# ---------------------------------------------------------------------------------------------
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


def make_hiss(rng, dur):
    """The creeper's hiss: a soft, breathy 'ssss' that swells."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 3500, 11000, 2) + 0.5 * bp(rng.standard_normal(n), 1800, 3500, 2)
    x *= 0.85 + 0.15 * np.sin(2 * np.pi * 6.5 * t)
    return norm(x, 0.7)


def make_pop(rng):
    """The game's item pop: a short blip that bends upwards."""
    dur = 0.09
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(380, 620)
    f = f0 * (1 + 1.6 * (1 - np.exp(-t / 0.02)))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
    return norm(x, 0.6)


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


def make_peg(rng):
    """A marble off an end rod: a bright, glassy tick."""
    dur = 0.12
    n = int(dur * SR)
    f0 = rng.uniform(2300, 3400)
    x = modal(dur, f0, (1.0, 2.6, 4.1), (1.0, 0.4, 0.2), (0.03, 0.015, 0.01), rng)
    x += 0.5 * bp(rng.standard_normal(n), 3000, 9000) * expenv(n, 0.003)
    return norm(x, 0.7)


def make_clack(rng):
    """Marble on marble: a hard, short clack."""
    dur = 0.08
    n = int(dur * SR)
    x = modal(dur, rng.uniform(2800, 3800), (1.0, 1.9, 3.3), (1.0, 0.5, 0.3), (0.012, 0.008, 0.005), rng)
    x += 0.8 * bp(rng.standard_normal(n), 1500, 8000) * expenv(n, 0.002)
    return norm(x, 0.8)


def make_knock(rng, lo=180.0, hi=420.0, snap=0.6):
    """A marble landing on wood (or, higher and drier, on stone): a hollow knock."""
    dur = 0.18
    n = int(dur * SR)
    f0 = rng.uniform(lo, hi)
    x = modal(dur, f0, (1.0, 2.3, 3.9), (1.0, 0.5, 0.25), (0.05, 0.03, 0.02), rng)
    x += snap * bp(rng.standard_normal(n), 300, 3000) * expenv(n, 0.006)
    return norm(x, 0.8)


def make_clink(rng):
    """On ice: a glassy clink."""
    dur = 0.2
    x = modal(dur, rng.uniform(1600, 2400), (1.0, 2.76, 5.4), (1.0, 0.6, 0.3), (0.08, 0.05, 0.03), rng, 0.0)
    return norm(x, 0.6)


def make_ping(rng):
    """On gold: a small metallic ping."""
    dur = 0.3
    n = int(dur * SR)
    x = modal(dur, rng.uniform(900, 1300), IRON, (1.0, 0.6, 0.45, 0.3, 0.2, 0.1), (0.12, 0.09, 0.07, 0.05, 0.04,
                                                                                    0.03), rng)
    x += 0.5 * bp(rng.standard_normal(n), 2000, 8000) * expenv(n, 0.002)
    return norm(x, 0.6)


def make_boing(rng):
    """Off a slime block: a short springy boing."""
    dur = 0.35
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 180 + 200 * (1 - np.exp(-t / 0.03)) * (1 + 0.2 * np.sin(2 * np.pi * 14 * t) * np.exp(-t / 0.15))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.12, 0.003)
    sq = make_squelch(rng, 0.2)
    x[:len(sq)] += 0.4 * norm(sq)
    return norm(x, 0.8)


def make_trapdoor(rng, low=1.0):
    """A wooden trapdoor dropping open: a short creak and a double clonk."""
    dur = 0.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    # the creak: stick-slip pulses at a wandering rate ringing a wooden body
    rate = 70 + 40 * np.sin(2 * np.pi * 3.0 * t) + 30 * t / dur
    ph = np.cumsum(rate) / SR
    pulses = (np.diff(np.floor(ph), prepend=0.0) > 0).astype(float) * rng.uniform(0.5, 1.0, n)
    body = modal(0.05, 620 * low, (1.0, 2.1, 3.4), (1.0, 0.6, 0.4), (0.012, 0.008, 0.006), rng)
    creak = np.convolve(pulses, body)[:n] * np.clip(t / 0.01, 0, 1) * np.exp(-t / 0.09)
    clonk = np.zeros(n)
    for d, g in ((0.1, 1.0), (0.16, 0.45)):
        s = int(d * SR)
        kn = make_knock(rng, 130 * low, 190 * low, snap=0.9)
        clonk[s:s + len(kn)] += g * kn[:n - s]
    return norm(0.45 * norm(creak) + norm(clonk), 0.9)


def make_clank(rng):
    """The golden trapdoor dropping open: a metallic clank."""
    return norm(resample(make_clang(rng), 0.8), 0.8)


def make_piston(rng):
    """A piston: a pneumatic push and a wooden thunk."""
    dur = 0.4
    n = int(dur * SR)
    whoosh = shaped_noise(dur, lambda t: 1800 + 600 * np.exp(-t / 0.05), lambda t: 0.8 + 0 * t,
                          lambda t: np.clip(t / 0.01, 0, 1) * np.exp(-t / 0.07), rng)[:n]
    thunk = make_knock(rng, 120, 170)
    x = 0.6 * norm(whoosh)
    x[:len(thunk)] += norm(thunk)
    return norm(x, 0.8)


def make_sizzle(rng, dur=1.6):
    """Into the lava: a thick splash, then sizzling and crackling."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    splash = bp(rng.standard_normal(n), 400, 3000) * expenv(n, 0.06, 0.002)
    fizz = bp(rng.standard_normal(n), 2500, 11000) * np.exp(-t / 0.6) * (0.5 + 0.5 * np.clip(t / 0.08, 0, 1))
    crackle = np.zeros(n)
    for _ in range(90):
        p = int(min(n - 300, rng.exponential(0.45) * SR))
        L = int(rng.integers(30, 180))
        crackle[p:p + L] += rng.standard_normal(L) * np.hanning(L) * rng.uniform(0.3, 1.0)
    crackle = bp(crackle, 800, 6000)
    thump = sine_sweep(dur, 160, 60, 0.05) * expenv(n, 0.1, 0.002)
    x = 0.8 * norm(splash) + 0.5 * norm(fizz) + 0.5 * norm(crackle) + 0.6 * norm(thump)
    return norm(x * np.clip((dur - t) / 0.2, 0, 1), 0.9)


def make_lava_pop(rng):
    """Lava's bubbling: a small low pop that bends upwards."""
    dur = 0.12
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = rng.uniform(110, 170) * (1 + 3.0 * (1 - np.exp(-t / 0.025)))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.035, 0.002)
    x += 0.25 * bp(rng.standard_normal(n), 300, 1500) * expenv(n, 0.01)
    return norm(x, 0.6)


def make_sparkle(rng, dur=0.7):
    """Gold sparkles: a flurry of tiny high pings."""
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(14):
        s = int(rng.uniform(0, dur * 0.6) * SR)
        p = modal(0.2, rng.uniform(2800, 5200), (1.0, 2.4), (1.0, 0.3), (0.05, 0.03), rng)
        e = min(n, s + len(p))
        x[s:e] += p[:e - s] * rng.uniform(0.3, 1.0)
    return norm(x, 0.5)


def make_beep(freq, dur=0.22):
    """The countdown's beep."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * freq * t)) * 0.6 + 0.4 * np.sin(2 * np.pi * freq * t)
    return lp(x, 6000, 2) * np.clip(t / 0.004, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)


def make_blip(rng, m):
    """A marble through a trapdoor, counted: a two-note chiptune blip (pitched up with each one)."""
    x = []
    for mm, d in ((m, 0.05), (m + 5, 0.14)):
        n = int(d * SR)
        t = np.arange(n) / SR
        f = 440.0 * 2 ** ((mm - 69) / 12.0)
        sq = np.where(np.mod(f * t, 1.0) < 0.5, 1.0, -1.0)
        x.append(lp(sq, 7000, 2) * np.clip((d - t) / 0.01, 0, 1) * (np.exp(-t / 0.08) if d > 0.1 else 1.0))
    return norm(np.concatenate(x), 0.5)


def make_out_sting(rng):
    """ELIMINATED: a heavy 'dun-DUNN' falling onto a low hit."""
    dur = 1.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    s1 = int(0.12 * SR)
    f = np.where(t < 0.12, 329.6, 220.0 * np.where(t > 0.55, 2 ** (-2 * (t - 0.55) / 12.0 / 0.45), 1.0))
    ph = np.cumsum(f) / SR
    x = np.sign(np.sin(2 * np.pi * ph)) * 0.45 + np.sin(2 * np.pi * ph) + 0.5 * np.sin(np.pi * ph)
    x = lp(x, 1600, 2)
    env = np.where(t < 0.12, np.clip(t / 0.004, 0, 1) * np.clip((0.12 - t) / 0.01, 0.3, 1),
                   np.clip((t - 0.12) / 0.005, 0, 1) * np.exp(-np.clip(t - 0.12, 0, None) / 0.45))
    x = x * env
    hit = make_hit(rng)
    x[s1:s1 + len(hit)] += 0.7 * hit[:n - s1]
    return norm(x * np.clip((dur - t) / 0.05, 0, 1), 0.85)


def make_save_sting(rng):
    """SAVED: a bright rising whoosh into a double ding."""
    dur = 1.1
    n = int(dur * SR)
    t = np.arange(n) / SR
    up = shaped_noise(0.35, lambda tt: 800 + 6000 * (tt / 0.35) ** 2, lambda tt: 0.6 + 0 * tt,
                      lambda tt: (tt / 0.35) ** 2, rng)
    x = np.zeros(n)
    x[:len(up)] += 0.5 * norm(up)
    for st, f in ((0.3, 1318.5), (0.42, 1975.5)):
        s = int(st * SR)
        tt = np.arange(n - s) / SR
        x[s:] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / 0.35)
    return norm(x * np.clip((dur - t) / 0.05, 0, 1), 0.8)


def make_justin_sting(rng):
    """JUST IN: a quick upward slide into a ding."""
    dur = 0.8
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.where(t < 0.1, 700 + 7000 * t, 1760.0)
    ph = np.cumsum(f) / SR
    x = np.sin(2 * np.pi * ph) * np.where(t < 0.1, 0.5 * t / 0.1, np.exp(-(t - 0.1) / 0.3))
    s = int(0.1 * SR)
    tt = np.arange(n - s) / SR
    x[s:] += 0.5 * np.sin(2 * np.pi * 2637.0 * tt) * np.exp(-tt / 0.2)
    x[s:] += 0.3 * bp(rng.standard_normal(n - s), 3000, 9000) * np.exp(-tt / 0.01)
    return norm(x * np.clip((dur - t) / 0.05, 0, 1), 0.7)


def make_crash(rng, dur=2.2):
    n = int(dur * SR)
    x = hp(rng.standard_normal(n), 3500, 2)
    x += 0.5 * modal(dur, 3150, (1.0, 1.41, 1.93, 2.66, 3.3), (1.0, 0.8, 0.7, 0.5, 0.4), (0.9, 0.7, 0.6, 0.5, 0.4),
                     rng, 0.02)
    return norm(x * expenv(n, 0.7, 0.001), 0.6)


def make_whoosh_hit(rng):
    """A round card: a riser into an impact with a crash."""
    pre = 0.5
    r = make_riser(rng, pre)
    dur = 1.6
    n = int(dur * SR)
    x = np.zeros(int(pre * SR) + n)
    x[:len(r)] += 0.6 * norm(r)
    s = int(pre * SR)
    h = make_hit(rng)
    x[s:s + len(h)] += h[:len(x) - s]
    x[s:] += 0.35 * make_crash(rng, dur)[:len(x) - s]
    return norm(x, 0.9), pre


def make_firework(rng):
    """A launch whistle, a bang and crackle."""
    pre = 0.45
    t = np.arange(int(pre * SR)) / SR
    whistle = np.sin(2 * np.pi * np.cumsum(900 + 1400 * t / pre) / SR) * (t / pre) * 0.5
    bang = make_explosion(rng, 0.4)
    crackle = np.zeros(int(1.5 * SR))
    for _ in range(160):
        p = int(min(len(crackle) - 200, rng.exponential(0.35) * SR))
        L = int(rng.integers(20, 90))
        crackle[p:p + L] += rng.standard_normal(L) * np.hanning(L) * rng.uniform(0.2, 1.0)
    crackle = hp(crackle, 2000, 2)
    x = np.zeros(len(whistle) + max(len(bang), len(crackle)))
    x[:len(whistle)] += whistle
    s = len(whistle)
    x[s:s + len(bang)] += 0.7 * norm(bang)
    x[s:s + len(crackle)] += 0.45 * norm(crackle)
    return norm(x, 0.85), pre


def make_rumble(rng, speed_env, n):
    """Marbles rolling: a band of low rumble that follows how fast they go."""
    base = bp(rng.standard_normal(n), 70, 420, 2)
    grit = bp(rng.standard_normal(n), 900, 3200, 2)
    t = np.arange(n) / SR
    am = 0.75 + 0.25 * np.sin(2 * np.pi * (6.0 + 5.0 * speed_env) * t)
    return (norm(base) + 0.25 * norm(grit)) * am * speed_env


# ---------------------------------------------------------------------------------------------
# the track: 144 BPM, A minor, chiptune arpeggios over a four-on-the-floor dance beat
# ---------------------------------------------------------------------------------------------
BPM = 144.0
BEAT = 60.0 / BPM
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]     # Am F C G
ROOTS = [45, 41, 48, 43]
# the lead for the third round: eighth notes over each bar of Am F C G (None = rest)
LEAD = [(81, None, 84, None, 88, 86, 84, None), (81, None, 84, None, 86, 84, 81, None),
        (79, None, 84, None, 88, 86, 84, None), (83, 84, 86, None, 83, None, 79, None)]
HAT_F = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def square(f, n, duty=0.25, phase=0.0):
    t = np.arange(n) / SR
    return np.where(np.mod(f * t + phase, 1.0) < duty, 1.0, -1.0)


def arp_note(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = square(f, n, 0.25) * 0.7 + square(f * 2, n, 0.5) * 0.15
    return lp(x, 5200, 2) * np.clip(t / 0.003, 0, 1) * np.exp(-t / 0.09)


def lead_note(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 6.0 * t) * np.clip((t - 0.08) / 0.1, 0, 1)
    ph = np.cumsum(f * vib) / SR
    x = np.where(np.mod(ph, 1.0) < 0.5, 1.0, -1.0) * 0.8 + 0.3 * np.sin(2 * np.pi * ph)
    env = np.clip(t / 0.004, 0, 1) * (0.6 + 0.4 * np.exp(-t / 0.08)) * np.clip((dur - t) / 0.02, 0, 1)
    return lp(x, 5000, 2) * env


def brass_note(f, dur, rng):
    """A synth brass note: detuned saws and a square an octave down, the filter opening on the attack."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    x = np.zeros(n)
    for d in (-0.006, 0.0, 0.006):
        x += 2 * np.mod(np.cumsum(f * (1 + d) * vib) / SR + rng.uniform(), 1.0) - 1
    x += 0.5 * np.sign(np.sin(2 * np.pi * np.cumsum(f / 2 * vib) / SR))
    w = np.exp(-t / 0.12)
    y = lp(x, 1400, 2) * (1 - w) + lp(x, 4200, 2) * w
    env = np.clip(t / 0.012, 0, 1) * np.clip((dur - t) / 0.06, 0, 1) * (0.8 + 0.2 * np.exp(-t / 0.1))
    return y * env / 3.0


def saw_chord(freqs, dur, rng, detune=(-0.008, 0.0, 0.009), cutoff=2400):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in freqs:
        for d in detune:
            x += 2 * np.mod(f * (1 + d) * t + rng.uniform(), 1.0) - 1
    env = np.clip(t / 0.02, 0, 1) * np.clip((dur - t) / 0.08, 0, 1)
    return lp(x, cutoff, 2) * env / (len(freqs) * len(detune))


def bass_note(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (2 * np.mod(f * t, 1.0) - 1) * 0.6 + np.sin(2 * np.pi * f * t) * 0.8
    w = np.exp(-t / 0.06)                        # the filter closing: bright pluck into a round body
    y = lp(x, 350, 2) * (1 - w) + lp(x, 1900, 2) * w
    return np.tanh(1.6 * y) * np.clip(t / 0.004, 0, 1) * np.clip((dur - t) / 0.02, 0, 1)


def kick(rng):
    dur = 0.45
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 48 + 160 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.2, 0.001)
    click = bp(rng.standard_normal(n), 2000, 9000) * expenv(n, 0.003)
    return norm(np.tanh(2.2 * (body + 0.25 * click)), 0.95)


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


def snare(rng):
    dur = 0.25
    n = int(dur * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(185 + 60 * np.exp(-t / 0.01)) / SR) * expenv(n, 0.05, 0.001)
    nz = bp(rng.standard_normal(n), 1200, 9000) * expenv(n, 0.09, 0.001)
    return norm(0.6 * norm(tone) + norm(nz), 0.8)


def hat(rng, open_=False):
    dur = 0.35 if open_ else 0.06
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(np.sign(np.sin(2 * np.pi * f * 1.3 * t + rng.uniform(0, 6))) for f in HAT_F)
    x = hp(bp(x, 6000, 16000, 2), 7000, 2) + 0.3 * hp(rng.standard_normal(n), 8000, 2)
    return norm(x * expenv(n, 0.12 if open_ else 0.018, 0.0005), 0.6)


class Track:
    """The music on a beat grid (beat 0 = GO): tonal parts pumped by the kick, and the drums."""

    def __init__(self, n, g0, rng):
        self.n, self.g0 = n, g0
        self.rng = rng
        self.L, self.R = np.zeros(n), np.zeros(n)
        self.DL, self.DR = np.zeros(n), np.zeros(n)
        self.duck = np.ones(n)
        self.kick = kick(rng)
        self.snare = snare(rng)
        self.claps = [clap(rng) for _ in range(3)]
        self.hats = [hat(rng) for _ in range(6)]
        self.ohat = hat(rng, True)
        self.cache = {}

    def t(self, b):
        return self.g0 + b * BEAT

    def _add(self, L, R, x, t, g, pan):
        s = int(round(t * SR))
        if s >= self.n or s + len(x) <= 0:
            return
        if s < 0:
            x = x[-s:]
            s = 0
        e = min(self.n, s + len(x))
        L[s:e] += x[:e - s] * g * np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        R[s:e] += x[:e - s] * g * np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)

    def tone(self, x, b, g=1.0, pan=0.0):
        self._add(self.L, self.R, x, self.t(b), g, pan)

    def drum(self, x, b, g=1.0, pan=0.0):
        self._add(self.DL, self.DR, x, self.t(b), g, pan)

    def pump(self, b, depth=0.6):
        """Sidechain: the tonal parts dip under each kick."""
        s = int(round(self.t(b) * SR))
        if s < 0 or s >= self.n:
            return
        e = min(self.n, s + int(0.25 * SR))
        env = 1.0 - depth * np.exp(-np.arange(e - s) / SR / 0.07)
        self.duck[s:e] = np.minimum(self.duck[s:e], env)

    def note(self, kind, m, dur):
        key = (kind, m, round(dur, 4))
        if key not in self.cache:
            f = midi_hz(m)
            if kind == 'arp':
                self.cache[key] = arp_note(f, dur)
            elif kind == 'lead':
                self.cache[key] = lead_note(f, dur)
            elif kind == 'bass':
                self.cache[key] = bass_note(f, dur)
            else:
                self.cache[key] = brass_note(f, dur, self.rng)
        return self.cache[key]

    def bar(self, b0, chord, root, level=1.0, beats=4, kick_on=True, clap_on=True, hats=1, arp=1, bass=1,
            pad=True, lead=None):
        """A bar of the groove from beat b0. hats: 0 none, 1 off-beat open + eighths, 2 with sixteenths;
        arp: 0 none, 1 up, 2 up-and-down; bass: 0 none, 1 off-beat eighths, 2 with sixteenth pickups."""
        rng = self.rng
        pattern = [0, 1, 2, 3] if arp == 1 else [0, 1, 2, 3, 2, 1]
        notes = list(chord) + [chord[0] + 12]
        for beat in range(beats):
            b = b0 + beat
            if kick_on:
                self.drum(self.kick, b, 0.9 * level)
                self.pump(b)
            if clap_on and beat % 2 == 1:
                self.drum(self.claps[rng.integers(3)], b, 0.5 * level)
            if hats:
                self.drum(self.ohat, b + 0.5, 0.15 * level, pan=0.25)
                self.drum(self.hats[rng.integers(6)], b, 0.16 * level, pan=-0.2)
                if hats == 2:
                    for h in (0.25, 0.75):
                        self.drum(self.hats[rng.integers(6)], b + h, 0.1 * level, pan=-0.3)
            if bass:
                for off in ((0.5,) if bass == 1 else (0.5, 0.75)):
                    self.tone(self.note('bass', root, BEAT * (0.45 if off == 0.5 else 0.22)), b + off,
                              0.5 * level)
            if arp:
                for s16 in range(4):
                    k = (beat * 4 + s16) % len(pattern)
                    m = notes[pattern[k]] + 12
                    self.tone(self.note('arp', m, BEAT / 4 * 0.95), b + s16 / 4.0, 0.2 * level,
                              pan=0.35 * np.sin(s16 * 1.3 + b0))
        if pad:
            self.tone(saw_chord([midi_hz(m) for m in chord], beats * BEAT, rng), b0, 0.3 * level)
        if lead is not None:
            for k, m in enumerate(lead):
                if m is None or k / 2.0 >= beats:
                    continue
                x = self.note('lead', m, BEAT / 2 * 0.9)
                self.tone(x, b0 + k / 2.0, 0.2 * level, pan=-0.1)
                self.tone(x, b0 + k / 2.0 + 0.75, 0.06 * level, pan=0.5)          # a dotted-eighth echo

    def roll(self, b0, b1, level=1.0):
        """A snare roll that speeds up (eighths, sixteenths, then thirty-seconds) and swells."""
        b = b0
        while b < b1 - 1e-6:
            u = (b - b0) / max(1e-6, b1 - b0)
            step = 0.5 if u < 0.4 else (0.25 if u < 0.8 else 0.125)
            self.drum(self.snare, b, (0.2 + 0.45 * u) * level, pan=0.1)
            b += step


# ---------------------------------------------------------------------------------------------
# build from the cue sheet
# ---------------------------------------------------------------------------------------------
HITS = {'peg': (0.5, 14.0), 'marble': (0.5, 14.0), 'wood': (0.65, 16.0), 'stone': (0.62, 18.0),
        'ice': (0.48, 20.0), 'slime': (0.62, 30.0), 'metal': (0.5, 15.0)}      # gain, impulse for full level
COUNT = (81, 84, 86, 88, 91, 93)          # the counter blips climb the A minor pentatonic
FANFARE = [(0.0, (67,), 1 / 3), (1 / 3, (72,), 1 / 3), (2 / 3, (76,), 1 / 3), (1.0, (79, 76, 72), 2.0),
           (3.0, (76,), 0.5), (3.5, (79,), 0.5), (4.0, (84, 79, 76), 3.5)]      # beats from its start, midi, beats
C_MAJOR, F_MAJOR, G_MAJOR = (48, 52, 55), (53, 57, 60), (55, 59, 62)
MUSIC = 0.32                              # the music's level under the effects


def _write(path, L, R):
    pcm = np.clip(np.stack([L, R], -1) * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return pcm


def build(meta, out_path, seed=7, stems=False):
    """Write the soundtrack for the cue sheet `meta` to out_path (48 kHz 16-bit stereo WAV); with stems, also the
    effects and the music on their own, before mastering (for checking the balance)."""
    rng = np.random.default_rng(seed)
    cues = meta['frames']
    fps = meta['fps']
    nf = len(cues)
    dur = nf / fps
    n = int(dur * SR) + SR
    mix = Mix(dur)
    go = int(meta.get('go', 0))
    t_go = go / fps
    win = meta.get('win')
    t_win = win / fps if win is not None else dur + 10.0
    speed = np.array([c['dt'] * fps for c in cues])
    speed[:go] = 1.0                                  # the countdown's slow clock isn't slow motion
    tgt = np.array([c['tgt'] for c in cues])
    cam_d = np.linalg.norm(np.array([c['cam'] for c in cues]) - tgt, axis=1)

    def tv(i):
        return i / fps

    def put(x, t, g, pan=0.0, sc=1.0):
        """Sounds of the race play slower and deeper in slow motion."""
        if sc < 0.95:
            x = resample(x, max(0.5, np.sqrt(sc)))
        mix.add(x, t, g, pan)

    def pan_of(i, x):
        return float(np.clip((x - tgt[i][0]) / 9.0, -0.8, 0.8))

    def near(i, x, z, r=16.0):
        """1 in the middle of the shot, fading to 0 away from it."""
        return float(np.clip(1.2 - np.hypot(x - tgt[i][0], z - tgt[i][2]) / r, 0.0, 1.0))

    def x_of(c, name):
        return c['pos'][c['names'].index(name)][0]

    # ---- the race: every knock, clack and boing, the rolling, the machinery, the lava
    banks = {'peg': [make_peg(rng) for _ in range(10)], 'marble': [make_clack(rng) for _ in range(8)],
             'wood': [make_knock(rng) for _ in range(8)], 'stone': [make_knock(rng, 380, 760, 0.9) for _ in range(8)],
             'ice': [make_clink(rng) for _ in range(8)], 'slime': [make_boing(rng) for _ in range(6)],
             'metal': [make_ping(rng) for _ in range(6)]}
    pistons = meta.get('pistons', [])
    lit = {}
    counted = {}
    duck_hits = []                                    # (t, depth, tau): the music dips under the big moments
    stall = [make_knock(rng, 300, 480, 0.8) for _ in range(4)]
    for i, c in enumerate(cues):
        t = tv(i)
        sc = speed[i]
        att = (16.0 / max(16.0, cam_d[i])) ** 0.7
        for kind, (cnt, imp, x) in c['hits'].items():
            gain, full = HITS.get(kind, HITS['stone'])
            u = float(np.clip((imp - 0.6) / full, 0.0, 1.0)) ** 0.6
            if u < 0.04:
                continue
            bank = banks.get(kind, banks['stone'])
            for j in range(min(int(cnt), 3)):
                s = resample(bank[rng.integers(len(bank))], rng.uniform(0.93, 1.08))
                put(s, t + rng.uniform(0, 1.0 / fps), gain * (0.25 + 0.75 * u) * att * (1.0 if j == 0 else 0.55),
                    pan_of(i, x) + rng.uniform(-0.1, 0.1), sc)
        opened = {e[1] for e in c['events'] if e[0] == 'open'}
        for e in c['events']:
            k = e[0]
            if k == 'go':                              # the stalls' floors drop
                for j in range(8):
                    put(stall[j % 4], t + rng.uniform(0, 0.04), 0.2, (j - 3.5) / 5.0)
            elif k == 'piston' and e[1] < len(pistons):
                px, pz = pistons[e[1]]
                g = near(i, px, pz)
                if g > 0.02:
                    put(make_piston(rng), t, 0.45 * g, pan_of(i, px), sc)
            elif k == 'lit':
                lit[e[1]] = t
            elif k == 'boom':
                x = e[2][0]
                t0 = lit.pop(e[1], None)
                if t0 is not None:
                    put(make_hiss(rng, t - t0 + 0.05), t0, 0.3, pan_of(i, x))
                put(make_explosion(rng, 1.3), t, 1.0, 0.5 * pan_of(i, x), sc)
                duck_hits.append((t, 0.6, 0.6))
            elif k == 'through':
                cnt = counted.get(e[1], 0)
                counted[e[1]] = cnt + 1
                if e[1] not in opened:
                    put(make_blip(rng, COUNT[min(cnt, len(COUNT) - 1)]), t, 0.2, pan_of(i, x_of(c, e[2])))
            elif k == 'open':
                if i == win:
                    put(make_clank(rng), t, 0.55, 0.0, sc)
                else:
                    put(make_trapdoor(rng), t, 0.95, 0.0, sc)
                    duck_hits.append((t, 0.3, 0.4))
            elif k == 'saved':
                put(make_sparkle(rng), t, 0.4, pan_of(i, x_of(c, e[2])), sc)
            elif k == 'lava':
                put(make_sizzle(rng), t, 0.85, pan_of(i, e[2][0]), sc)
            elif k == 'release':                       # the pen's floor splits and drops
                for j in range(2):
                    put(make_trapdoor(rng, 1.25), t + 0.025 * j, 0.45, (-0.4, 0.4)[j])
        # ---- the broadcast: round cards and banners
        h = c['hud']
        if h.get('round', 0) > 0:
            x, pre = make_whoosh_hit(rng)
            mix.add(x, t - pre, 0.55)
            duck_hits.append((t, 0.35, 0.3))
        ban = h.get('banner')
        if ban:
            if ban[0] == 'out':
                late = t >= t_win                       # the winner's moment keeps the spotlight
                put(make_out_sting(rng), t, 0.3 if late else 0.55)
                if not late:
                    duck_hits.append((t, 0.4, 0.35))
            elif ban[0] == 'saved':
                put(make_save_sting(rng), t, 0.55)
                duck_hits.append((t, 0.3, 0.3))
            elif ban[0] == 'justin':
                put(make_justin_sting(rng), t, 0.5)
                duck_hits.append((t, 0.25, 0.25))
    # the pack rolling
    spd = np.array([c['speed'] for c in cues])
    rolling = np.clip(np.clip(spd / 12.0, 0, 1).sum(1) / 3.0, 0, 1) * np.sqrt(np.clip(speed, 0.1, 1.0))
    rolling[:go] = 0.0
    rum = make_rumble(rng, _env_track(rolling, fps, n, 4.0), n)
    # lava bubbling near the camera
    pools = meta.get('lava', [])
    lava = np.array([max([near(i, (p[0] + p[2]) / 2.0, p[3], 14.0) for p in pools], default=0.0) for i in range(nf)])
    pops = [make_lava_pop(rng) for _ in range(8)]
    for i in range(nf):
        for _ in range(rng.poisson(lava[i] * 5.0 / fps)):
            put(pops[rng.integers(8)], tv(i) + rng.uniform(0, 1.0 / fps), 0.22 * lava[i], rng.uniform(-0.6, 0.6),
                speed[i])
    bed = norm(bp(rng.standard_normal(n), 40, 260, 2)) * _env_track(lava, fps, n, 2.0)
    mix.L[:n] += 0.2 * rum + 0.08 * bed
    mix.R[:n] += 0.2 * rum + 0.08 * bed
    # slow motion in and out (the final's is part of the music's big moment)
    slow = speed < 0.9
    for i in range(1, nf):
        if slow[i] and not slow[i - 1] and tv(i) < t_win - 2.0:
            put(make_slowmo_down(rng), tv(i), 0.45)
        if slow[i - 1] and not slow[i] and tv(i) < t_win:
            put(make_slowmo_up(rng), tv(i) - 0.45, 0.35)
    # the countdown: three beeps on every second beat, a high one on GO
    for k in range(3):
        put(make_beep(880.0), t_go - (6 - 2 * k) * BEAT, 0.42)
    put(make_beep(1760.0, 0.5), t_go, 0.45)
    put(make_hit(rng), max(0.0, t_go - 6 * BEAT), 0.3)
    # the win
    if win is not None:
        put(make_hit(rng), t_win, 0.8)
        put(make_fanfare(rng), t_win + 0.05, 0.3)
        for _ in range(10):
            put(make_pop(rng), t_win + 0.05 + rng.exponential(0.2), 0.22, rng.uniform(-0.6, 0.6))
        for j, k in enumerate((6, 18, 30, 44)):
            x, pre = make_firework(rng)
            tf = t_win + k / fps
            if tf - pre < t_win + 0.05:                # no launch whistle before the winning moment
                x, pre = x[int(pre * SR):], 0.0
            put(x, tf - pre, 0.42, (-0.45, 0.45)[j % 2])

    # ---- the music, on the beat grid from GO
    T = Track(n, t_go, rng)
    X = Mix(dur)                                      # the parts off the grid's bars (risers, swells, the fanfare)
    rounds = sorted(f for f, r in meta.get('rounds', []) if r > 0)

    def bar_near(f):
        return int(round((tv(f) - t_go) / BEAT / 4.0)) * 4

    b_end = int(np.ceil((dur - t_go) / BEAT))
    bB = bar_near(rounds[0]) if len(rounds) > 0 else b_end
    bC = bar_near(rounds[1]) if len(rounds) > 1 else b_end
    bF = bar_near(rounds[2]) if len(rounds) > 2 else b_end
    b_win = (t_win - t_go) / BEAT
    slow_final = [i for i in range(nf) if slow[i] and not slow[i - 1] and tv(i) > T.t(bF)]
    b_cut = int(np.floor((tv(slow_final[0]) - t_go) / BEAT)) if slow_final else int(np.floor(b_win))
    b_cut = int(np.clip(b_cut, bF + 2, max(bF + 2, np.floor(b_win))))
    t_cut = T.t(b_cut)
    # intro (the countdown, six beats): F then G, the arp, pad and a pulsing bass opening up, a kick on every
    # beat and on every half beat at the end, a snare roll and a riser into the drop on GO
    I = Track(n, t_go, rng)
    I.bar(-6, CHORDS[1], ROOTS[1], 1.6, beats=2, kick_on=False, clap_on=False, hats=0, bass=0)
    I.bar(-4, CHORDS[3], ROOTS[3], 1.6, beats=4, kick_on=False, clap_on=False, hats=0, bass=0)
    for k in range(12):
        root = ROOTS[1] if k < 4 else ROOTS[3]
        I.tone(I.note('bass', root, BEAT * 0.4), -6 + k * 0.5, 0.8)
    s0, s1 = int(T.t(-6) * SR), int(t_go * SR)
    ramp = np.zeros(n)
    ramp[s0:s1] = np.linspace(0.0, 1.0, s1 - s0) ** 2
    ramp[s1:] = 1.0
    IL = lp(I.L, 380, 2) * (1 - ramp) + I.L * ramp
    IR = lp(I.R, 380, 2) * (1 - ramp) + I.R * ramp
    for k in range(8):
        T.drum(T.kick, -6 + k * (1.0 if k < 4 else 0.5), 0.6 + 0.04 * k)
    T.roll(-2, 0, 1.0)
    X.add(make_riser(rng, 6 * BEAT), T.t(-6), 0.5)
    X.add(make_crash(rng), t_go, 0.45)
    # round 1 (A), round 2 (B: rolling arps, sixteenth hats, busier bass), round 3 (C: and the lead)
    for b0 in range(0, bF, 4):
        k = (b0 // 4) % 4
        if b0 < bB:
            T.bar(b0, CHORDS[k], ROOTS[k], 1.0, hats=1, arp=1, bass=1)
        elif b0 < bC:
            T.bar(b0, CHORDS[k], ROOTS[k], 1.0, hats=2, arp=2, bass=2)
        else:
            T.bar(b0, CHORDS[k], ROOTS[k], 1.0, hats=2, arp=2, bass=2, lead=LEAD[k])
    for b in (bB, bC):
        if 2 <= b < bF:
            T.roll(b - 2, b, 0.7)
            X.add(make_crash(rng), T.t(b), 0.35)
    # the final: a build on F and G (kick on every beat, the roll speeding up, a riser), cut dead as time slows
    k = (bF // 4) % 4
    T.bar(bF, CHORDS[k], ROOTS[k], 1.0, beats=min(4, b_cut - bF), clap_on=False, hats=2, arp=1, bass=1)
    if b_cut > bF + 4:
        T.bar(bF + 4, G_MAJOR, 43, 1.05, beats=b_cut - bF - 4, clap_on=False, hats=2, arp=1, bass=1)
    T.roll(bF, b_cut, 1.0)
    X.add(make_riser(rng, t_cut - T.t(bF)), T.t(bF), 0.35)
    # the moment: silence but a heartbeat and a dominant swell rising into the win
    sw = max(0.2, t_win - t_cut)
    tt = np.arange(int((sw + 0.05) * SR)) / SR
    swell = saw_chord([midi_hz(m) for m in (43, 55, 59, 62, 65)], sw + 0.05, rng, cutoff=1800) * \
        (tt / (sw + 0.05)) ** 2
    X.add(norm(swell, 0.6), t_cut, 0.55)
    X.add(make_riser(rng, sw), t_cut, 0.3)
    X.add(make_slowmo_down(rng), t_cut, 0.35)
    heart = lp(T.kick, 300, 2)
    X.add(heart, t_cut, 0.6)
    X.add(heart, t_cut + 0.2, 0.4)
    if win is not None:
        # the win: a crash and a C major chord blooming, the brass fanfare on the next beat, then the victory
        # groove in C major into a last stab
        X.add(make_crash(rng, 3.0), t_win, 0.5)
        b_fan = int(np.ceil(b_win + 0.25))
        b_vic = b_fan + 4
        pad = saw_chord([midi_hz(m) for m in (48, 55, 60, 64, 67)], T.t(b_vic) - t_win + 0.1, rng, cutoff=3200)
        X.add(pad * np.clip(np.arange(len(pad)) / SR / 0.3, 0, 1), t_win, 0.6)
        for (bo, ms, bl) in FANFARE:
            for j, m in enumerate(ms):
                x = T.note('brass', m, bl * BEAT)
                X.add(x, T.t(b_fan + bo), 0.5 if j == 0 else 0.3, (0.0, -0.3, 0.3)[j])
        b_stab = b_vic + 8
        if T.t(b_stab) + 0.3 > dur:
            b_stab = b_vic + 4
        prog = [(b_vic, C_MAJOR, 48, 4), (b_vic + 4, F_MAJOR, 41, 2), (b_vic + 6, G_MAJOR, 43, 2)]
        for (b0, ch, root, nb) in prog:
            if b0 < b_stab:
                T.bar(b0, ch, root, 1.0, beats=min(nb, b_stab - b0), hats=2, arp=1, bass=1)
        X.add(make_crash(rng), T.t(b_vic), 0.35)
        for j, m in enumerate((60, 64, 67, 72)):
            X.add(T.note('brass', m, 0.9 * BEAT), T.t(b_stab), 0.35, (-0.3, -0.1, 0.1, 0.3)[j])
        T.drum(T.kick, b_stab, 1.0)
        X.add(make_crash(rng), T.t(b_stab), 0.45)
    # the grid's parts stop dead at the cut and stay out until the win
    gate = np.ones(n)
    sc0, sc1 = int(t_cut * SR), int(t_win * SR)
    f = int(0.015 * SR)
    gate[sc0:sc1] = 0.0
    gate[max(0, sc0 - f):sc0] = np.linspace(1, 0, min(f, sc0))
    mL = (T.L * T.duck + T.DL + IL) * gate + X.L[:n]
    mR = (T.R * T.duck + T.DR + IR) * gate + X.R[:n]
    # muffled under slow motion (not the final's: that's the music's own moment)
    sm = np.array([1.0 if (slow[i] and tv(i) < t_cut - 0.3) else 0.0 for i in range(nf)])
    m = _env_track(sm, fps, n, 6.0)
    mL = mL * (1 - m) + lp(mL, 480, 2) * m * 1.25
    mR = mR * (1 - m) + lp(mR, 480, 2) * m * 1.25
    duck = np.ones(n)
    for (t0, depth, tau) in duck_hits:
        s = int(t0 * SR)
        if s >= n:
            continue
        L = min(n - s, int(tau * 5 * SR))
        duck[s:s + L] = np.minimum(duck[s:s + L], 1.0 - depth * np.exp(-np.arange(L) / SR / tau))

    # ---- mix and master
    L = mix.L[:n] + MUSIC * mL * duck
    R = mix.R[:n] + MUSIC * mR * duck
    if stems:
        _write(out_path[:-4] + '_sfx.wav', 0.25 * mix.L[:n], 0.25 * mix.R[:n])
        _write(out_path[:-4] + '_music.wav', 0.25 * MUSIC * mL * duck, 0.25 * MUSIC * mR * duck)
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    # phone speakers: tame the deep sub, add a little presence
    L = L - 0.45 * lp(L, 80.0, 2) + 0.25 * bp(L, 1800.0, 4500.0, 1)
    R = R - 0.45 * lp(R, 80.0, 2) + 0.25 * bp(R, 1800.0, 4500.0, 1)
    L, R = L[:int(dur * SR)], R[:int(dur * SR)]
    g = 10 ** ((-18.0 - integrated_lufs(L, R)) / 20.0)
    L, R = compress(L * g, R * g, thresh_db=-24.0, ratio=3.0)
    g = 10 ** ((-14.0 - integrated_lufs(L, R)) / 20.0)
    L, R = limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    lufs = integrated_lufs(L, R)
    f0, f1 = int(0.01 * SR), int(0.3 * SR)
    for ch in (L, R):
        ch[:f0] *= np.linspace(0, 1, f0)
        ch[-f1:] *= np.linspace(1, 0, f1) ** 0.5
    pcm = _write(out_path, L, R)
    print(f'[audio] {dur:.1f}s, {lufs:.1f} LUFS integrated, peak {20 * np.log10(np.abs(pcm).max() / 32768):.1f} dBFS',
          flush=True)
    return out_path


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fh:
        build(json.load(fh), sys.argv[2], stems='--stems' in sys.argv)
