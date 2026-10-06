"""Procedural music and sound design for part 2, driven by the cue sheet of the render.

Everything is synthesised here (no samples). The music: part 1's soft felt piano in D major for the run, the
strings coming in as the dominoes grow; a held breath as the giant tips, and a slam; a driving race theme (piano
octaves, strings, a low drum) held in suspense through the slow-motion finish, and the chord as gold wins; the
piano again over the bridge, cut off where the run stops. Then nothing but the wind, the rain coming and a heartbeat;
thunder; and under the falling field the same piano gone dark, D minor, building with the strings to a wide D minor
chord at 100,000. A low drone, a whoosh as you turn, a dissonant sting when he's there, a glitch, silence.
The effects: every one of the 100,000 dominoes clicks as it's knocked and again as it comes to rest (thousands of
them become the rattle of the field), each growth domino's knock bigger and deeper than the last, the giant's slam,
red's 'wah wah', the punch, the river by the bridge, birds, wind, rain and thunder. Loudness normalised to -14 LUFS
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


def make_crash(rng, dur=2.2):
    n = int(dur * SR)
    x = hp(rng.standard_normal(n), 3500, 2)
    x += 0.5 * modal(dur, 3150, (1.0, 1.41, 1.93, 2.66, 3.3), (1.0, 0.8, 0.7, 0.5, 0.4), (0.9, 0.7, 0.6, 0.5, 0.4),
                     rng, 0.02)
    return norm(x * expenv(n, 0.7, 0.001), 0.6)


def _write(path, L, R):
    pcm = np.clip(np.stack([L, R], -1) * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return pcm


# ---------------------------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------------------------
def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


_NOTE_CACHE = {}


def piano(m, dur, vel, seed=0):
    """A soft felt piano note: inharmonic partials from a pair of slightly detuned strings, a quick first decay into
    a long ring, a muted hammer thump, and the damper after `dur` (with a little pedal left on)."""
    key = (m, round(dur, 2), round(vel, 2), seed % 4)
    if key in _NOTE_CACHE:
        return _NOTE_CACHE[key]
    rng = np.random.default_rng(1000 * m + seed % 4)
    f0 = midi_hz(m)
    ring = 2.4
    n = int((dur + ring) * SR)
    t = np.arange(n) / SR
    B = 2.5e-4 * (f0 / 261.6) ** 0.6
    tau0 = float(np.clip(3.4 * (261.6 / f0) ** 0.5, 0.9, 6.0))
    x = np.zeros(n)
    for k in range(1, 12):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1.0 / k) ** (1.55 - 0.55 * vel)
        tau = tau0 / (1 + 0.45 * (k - 1))
        env = 0.62 * np.exp(-t / (tau * 0.22)) + 0.38 * np.exp(-t / tau)
        for d in (-1.0, 1.0):
            x += 0.5 * amp * np.sin(2 * np.pi * fk * (1 + d * 0.0005) * t + rng.uniform(0, 6.28)) * env
    x *= np.clip(t / 0.005, 0, 1)
    thump = lp(rng.standard_normal(n), 900, 2) * np.exp(-t / 0.018) * 0.05 * vel
    x = lp(x + thump, 2200 + 3800 * vel, 2)
    rel = np.where(t < dur, 1.0, 0.22 + 0.78 * np.exp(-np.clip(t - dur, 0, None) / 0.14))
    x = x * rel * np.clip((n / SR - t) / 0.3, 0, 1)
    out = (x / (np.abs(x).max() + 1e-9) * vel).astype(np.float32)
    if len(_NOTE_CACHE) > 600:
        _NOTE_CACHE.clear()
    _NOTE_CACHE[key] = out
    return out


def strings(freqs, dur, rng, attack=0.5, release=0.9, bright=2200.0):
    """A string-section chord: detuned saw voices per note, vibrato, soft attack and release, warm filter."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in freqs:
        for d in (-0.011, -0.004, 0.0, 0.005, 0.012):
            vib = 1 + 0.0035 * np.sin(2 * np.pi * rng.uniform(4.6, 5.6) * t + rng.uniform(0, 6.28))
            ph = np.cumsum(f * (1 + d) * vib) / SR + rng.uniform()
            x += 2 * np.mod(ph, 1.0) - 1
    x = lp(x, bright, 2)
    x = lp(x, bright * 1.6, 2)
    env = np.clip(t / attack, 0, 1) ** 1.5 * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / release, 0, 1))
    return x * env / (5 * len(freqs))


def bell(m, rng, dur=2.2, bright=1.0):
    """A celesta / glass bell."""
    f0 = midi_hz(m)
    x = modal(dur, f0, (1.0, 2.0, 3.0, 4.2, 5.4), (1.0, 0.45 * bright, 0.25 * bright, 0.12, 0.06),
              (1.1, 0.6, 0.35, 0.2, 0.12), rng, 0.001)
    n = len(x)
    x *= np.clip(np.arange(n) / SR / 0.002, 0, 1)
    return norm(x, 0.8)


def reverb(L, R, seed=3, decay=2.0, wet=0.3, pre=0.018):
    """Convolution with a generated stereo hall: exponentially decaying, darkening noise."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    n = int(decay * 1.6 * SR)
    t = np.arange(n) / SR
    out = []
    for ch, x in enumerate((L, R)):
        ir = rng.standard_normal(n) * np.exp(-t / (decay / 6.9))
        ir = lp(ir, 5200, 2) * 0.7 + lp(ir, 1400, 2) * 0.3
        ir[:int(pre * SR)] = 0.0
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        y = fftconvolve(x, ir)[:len(x)]
        out.append(x * (1 - wet) + y * wet * 1.4)
    return out[0], out[1]


# ---------------------------------------------------------------------------------------------
# sound effects
# ---------------------------------------------------------------------------------------------
def make_click(rng, low=1.0):
    """A domino knocking into the next: a hard little tick over a short woody body."""
    dur = 0.07
    n = int(dur * SR)
    f0 = rng.uniform(2300, 3700) * low
    x = modal(dur, f0, (1.0, 1.61, 2.83), (1.0, 0.5, 0.28), (0.009, 0.006, 0.004), rng)
    x += 0.7 * modal(dur, rng.uniform(760, 1150) * low, (1.0, 2.1), (1.0, 0.4), (0.016, 0.01), rng)
    x += 0.5 * hp(rng.standard_normal(n), 2500, 2) * expenv(n, 0.0012)
    return norm(x * np.clip(np.arange(n) / SR / 0.0004, 0, 1), 0.8)


def make_settle(rng):
    """A domino coming to rest on the one in front: softer and lower."""
    return norm(resample(make_click(rng, 0.62), 1.0) * 0.8, 0.55)


def make_punch(rng):
    """The game's hit: a whoosh of the swing and a dull thud."""
    dur = 0.35
    n = int(dur * SR)
    t = np.arange(n) / SR
    sw = shaped_noise(0.2, lambda tt: 500 + 3000 * (tt / 0.2), lambda tt: 0.7 + 0 * tt,
                      lambda tt: np.sin(np.pi * np.clip(tt / 0.2, 0, 1)) ** 2, rng)
    x = np.zeros(n)
    x[:len(sw)] += 0.35 * norm(sw)
    s = int(0.14 * SR)
    th = sine_sweep((n - s + 2) / SR, 190, 70, 0.03)[:n - s] * expenv(n - s, 0.06, 0.001)
    th += 0.5 * lp(rng.standard_normal(n - s), 800, 2) * expenv(n - s, 0.03)
    x[s:] += norm(th)
    return norm(x * np.clip((dur - t) / 0.02, 0, 1), 0.8)


def make_step(rng):
    """A footstep on grass: a soft crunch."""
    dur = 0.12
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(int(rng.integers(8, 16))):
        p = int(rng.uniform(0, 0.06) * SR)
        L = int(rng.uniform(0.002, 0.008) * SR)
        g = rng.standard_normal(L) * np.hanning(L)
        x[p:p + L] += g[:n - p] * rng.uniform(0.3, 1.0)
    x = bp(x, 500, 4500, 2) + 0.4 * lp(rng.standard_normal(n), 300, 2) * expenv(n, 0.02)
    return norm(x * expenv(n, 0.05, 0.003), 0.6)


def make_levelup(rng):
    """The counter reaching 10,000: a bright rising chime."""
    x = np.zeros(int(2.6 * SR))
    for k, m in enumerate((86, 90, 93, 98)):
        b = bell(m, rng, 2.2)
        s = int(k * 0.075 * SR)
        x[s:s + len(b)] += b[:len(x) - s] * (0.8 if k < 3 else 1.0)
    return norm(x, 0.8)


def make_thunder(rng, dur=4.0):
    """A close lightning strike: a hard crack, then rolling thunder."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    crack = hp(rng.standard_normal(n), 900, 2) * expenv(n, 0.045, 0.0008)
    crack += 0.8 * bp(rng.standard_normal(n), 150, 2500, 2) * expenv(n, 0.12, 0.002)
    # rolls: a few swells of low noise
    env = np.zeros(n)
    for c, w, a in ((0.12, 0.25, 1.0), (0.55, 0.5, 0.7), (1.3, 0.8, 0.55), (2.2, 1.0, 0.35)):
        env += a * np.exp(-0.5 * ((t - c) / w) ** 2)
    rumble = lp(rng.standard_normal(n), 220, 2) + 0.5 * bp(rng.standard_normal(n), 60, 600, 2)
    rumble *= env * (0.7 + 0.3 * lp(rng.standard_normal(n), 8, 1) * 8)
    x = 0.9 * norm(crack) + 1.1 * norm(rumble)
    return norm(np.tanh(1.3 * x) * np.clip((dur - t) / 0.4, 0, 1), 0.95)


def make_chirp(rng):
    """A bird: a few quick whistles."""
    x = []
    for _ in range(int(rng.integers(2, 5))):
        d = rng.uniform(0.05, 0.12)
        n = int(d * SR)
        t = np.arange(n) / SR
        f = rng.uniform(2600, 4200) * (1 + rng.uniform(-0.35, 0.35) * t / d)
        f += 300 * np.sin(2 * np.pi * rng.uniform(25, 60) * t)
        x.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2)
        x.append(np.zeros(int(rng.uniform(0.03, 0.09) * SR)))
    return norm(np.concatenate(x), 0.5)


def make_drone(rng, dur):
    """Something is wrong: a low beating drone, a minor second, a whisper of noise."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, a in ((36.7, 1.0), (38.9, 0.8), (73.4, 0.5), (77.8, 0.4)):
        x += a * np.tanh(1.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)))
    x = lp(x, 400, 2)
    wh = bp(rng.standard_normal(n), 1800, 5200, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    x = norm(x) + 0.25 * norm(wh)
    return x * np.clip(t / (dur * 0.6), 0, 1) ** 2


def make_swell(rng, dur):
    """A reverse cymbal: noise that swells into the strike."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 3000, 2) + 0.5 * bp(rng.standard_normal(n), 800, 3000, 2)
    return norm(x * (t / dur) ** 3, 0.7)


# ---------------------------------------------------------------------------------------------
# part 2's sounds
# ---------------------------------------------------------------------------------------------
def make_knock(rng, size):
    """A growth domino knocking into the next: bigger and deeper with its size (a woody knock, and for the big ones a
    thump you feel)."""
    s = max(float(size), 1.0)
    dur = min(0.3 + 0.1 * s, 2.6)
    f0 = 1500.0 / s ** 0.8
    k = s ** 0.5
    x = modal(dur, f0, (1.0, 1.58, 2.71, 3.9), (1.0, 0.5, 0.24, 0.1), (0.03 * k, 0.02 * k, 0.012 * k, 0.008 * k), rng)
    n = len(x)
    x += 0.6 * lp(rng.standard_normal(n), 1800 / k, 2) * expenv(n, 0.01 * k, 0.0005)
    x = norm(x)
    if s > 2.5:
        sub = sine_sweep(dur, 110.0 / s ** 0.3, 32.0, 0.12 * k) * expenv(n, 0.09 * k, 0.002)
        x = x + min(1.2, 0.25 * s ** 0.5) * norm(sub)
    return norm(x * np.clip(np.arange(n) / SR / 0.0005, 0, 1), 0.9)


def make_slam(rng):
    """The giant coming down: a hit you feel in your chest, a crack, the ground rumbling and dust settling."""
    dur = 5.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    sub = sine_sweep(dur, 72.0, 24.0, 0.35) * expenv(n, 1.0, 0.002)
    body = lp(rng.standard_normal(n), 260, 2) * expenv(n, 0.4, 0.001)
    crack = bp(rng.standard_normal(n), 500, 5000, 2) * expenv(n, 0.05, 0.0005)
    rumble = lp(rng.standard_normal(n), 110, 2) * expenv(n, 1.6, 0.08)
    debris = hp(rng.standard_normal(n), 2500, 2) * expenv(n, 1.2, 0.2) * \
        (0.5 + 0.5 * np.abs(lp(rng.standard_normal(n), 20, 1)) * 25)
    x = 1.1 * norm(sub) + 0.8 * norm(body) + 0.55 * norm(crack) + 0.6 * norm(rumble) + 0.12 * norm(debris)
    return norm(np.tanh(1.5 * x) * np.clip((dur - t) / 0.5, 0, 1), 0.98)


def make_womp(rng):
    """Red falls short: a muted trombone's 'wah wah'."""
    out = np.zeros(int(1.3 * SR))
    for (t0, d, f0, f1) in ((0.0, 0.34, 233.1, 220.0), (0.40, 0.75, 207.7, 164.8)):
        n = int(d * SR)
        tt = np.arange(n) / SR
        f = f0 + (f1 - f0) * (tt / d) ** 1.6 + 2.5 * np.sin(2 * np.pi * 5.5 * tt) * (tt / d)
        saw = 2 * np.mod(np.cumsum(f) / SR, 1.0) - 1
        env = np.clip(tt / 0.035, 0, 1) * np.clip((d - tt) / 0.09, 0, 1)
        y = lp(saw, 900, 2) * (0.35 + 0.65 * env) + 0.6 * lp(saw, 2600, 2) * env ** 2      # the mute opening
        s = int(t0 * SR)
        out[s:s + n] += y * env
    return norm(out, 0.7)


def make_tom(rng):
    """A low drum for the race."""
    dur = 0.35
    n = int(dur * SR)
    x = sine_sweep(dur, 150.0, 68.0, 0.05) * expenv(n, 0.13, 0.002)
    x += 0.3 * lp(rng.standard_normal(n), 1200, 2) * expenv(n, 0.015)
    return norm(x, 0.8)


def make_whoosh(rng, dur=0.5):
    x = shaped_noise(dur, lambda tt: 300 + 4000 * np.sin(np.pi * np.clip(tt / dur, 0, 1)), lambda tt: 0.8 + 0 * tt,
                     lambda tt: np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2, rng)
    return norm(x, 0.8)


def make_sting(rng):
    """He's right there: a dissonant hit (a cluster in the strings, a slam of noise, the floor dropping out)."""
    dur = 2.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    cl = strings([midi_hz(m) for m in (38, 39, 45, 50, 51, 56, 62, 63, 68, 74)], 1.4, rng, attack=0.012,
                 release=1.0, bright=5200)
    cl = np.pad(cl, (0, max(0, n - len(cl))))[:n]
    hit = bp(rng.standard_normal(n), 120, 7000, 2) * expenv(n, 0.09, 0.001)
    sub = sine_sweep(dur, 90.0, 22.0, 0.4) * expenv(n, 0.9, 0.002)
    shriek = np.sin(2 * np.pi * np.cumsum(1760 + 120 * np.sin(2 * np.pi * 7 * t)) / SR) * expenv(n, 0.5, 0.01)
    x = 1.0 * norm(cl) + 0.8 * norm(hit) + 0.9 * norm(sub) + 0.12 * shriek
    return norm(np.tanh(1.6 * x), 0.98)


def make_glitch(rng, dur=0.45):
    """The picture tearing: bit-crushed noise and a stuttering buzz, then nothing."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    k = 0
    while k < n:
        L = int(rng.uniform(0.012, 0.05) * SR)
        f = float(rng.choice([110.0, 220.0, 440.0, 880.0, 1760.0])) * rng.uniform(0.9, 1.1)
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(L) / SR)) * rng.uniform(0.3, 1.0)
        if rng.random() < 0.4:
            seg = rng.standard_normal(L)
        x[k:k + L] = seg[:n - k]
        k += L
    x = np.round(x * 6) / 6
    return norm(x * np.clip((dur - t) / 0.02, 0, 1), 0.7)


def make_creak(rng, dur=2.4):
    """A giant wooden thing tipping: a low groan that slides up as it goes over, and a few cracks."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    g = shaped_noise(dur, lambda tt: 120 + 180 * (tt / dur) ** 1.5 + 25 * np.sin(2 * np.pi * 0.9 * tt),
                     lambda tt: 0.18 + 0 * tt, lambda tt: np.clip(tt / 0.4, 0, 1) * np.clip((dur - tt) / 0.3, 0, 1),
                     rng)
    x = norm(g)
    for _ in range(5):
        p = int(rng.uniform(0.1, dur - 0.2) * SR)
        m = int(0.03 * SR)
        x[p:p + m] += 0.5 * bp(rng.standard_normal(m), 800, 4000, 2) * expenv(m, 0.006)
    return norm(x * (0.6 + 0.4 * t / dur), 0.8)


def make_heartbeat(rng):
    dur = 0.5
    n = int(dur * SR)
    x = np.zeros(n)
    for (s, a) in ((0.0, 1.0), (0.17, 0.7)):
        m = n - int(s * SR)
        x[int(s * SR):] += a * sine_sweep(m / SR, 75.0, 42.0, 0.05) * expenv(m, 0.09, 0.004)
    return norm(lp(x, 200, 2), 0.9)


# ---------------------------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------------------------
RUN_CHORDS = [((50, 57, 61, 66), (62, 69, 73, 78)),           # Dmaj7: left hand, arpeggio tones
              ((47, 54, 57, 62), (59, 66, 69, 74)),           # Bm7
              ((43, 50, 54, 59), (62, 67, 71, 74)),           # Gmaj7
              ((45, 52, 57, 61), (64, 69, 73, 76)),           # A
              ((40, 47, 50, 55), (59, 64, 67, 71))]           # Em7
RUN_MELODY = [[], [(0, 78, 2), (2, 76, 1), (3, 74, 1)], [(0, 71, 2), (2, 74, 1), (3, 78, 1)],
              [(0, 76, 3), (3, 73, 1)], [(0, 79, 2), (2, 78, 1), (3, 76, 1)]]
MUSIC = 0.5                                                   # the music under the effects


def _bar(mus, rng, tb, bar, lh, rh, mel=(), vel=1.0, cut=None):
    beat = bar / 4.0
    mus.add(piano(lh[0], bar, 0.55 * vel), tb, 0.8, -0.2)
    mus.add(piano(lh[1], bar * 0.5, 0.42 * vel), tb + 2 * beat, 0.6, -0.15)
    for k, m in enumerate([rh[0], rh[1], rh[2], rh[3], rh[3], rh[2], rh[1], rh[2]]):
        tn = tb + k * beat / 2
        if cut is not None and tn >= cut:
            break                                  # the run stops: the arpeggio doesn't finish
        mus.add(piano(m, beat * 0.6, (0.34 + 0.08 * (k % 4 == 0)) * vel, k), tn, 0.5, 0.2 + 0.1 * np.sin(k))
    for (bt, m, ln) in mel:
        if cut is None or tb + bt * beat < cut:
            mus.add(piano(m, ln * beat * 0.95, 0.62 * vel), tb + bt * beat, 0.62, 0.05)


def score_open(mus, rng, t0, t1, t_land):
    """The run and the growth: four bars of the felt piano in D, the strings coming in as the dominoes grow. Then
    the giant tips: a held A in the strings, rising, and a reversed swell into the slam."""
    bar = (t1 - t0) / 4.0
    for b in range(4):
        lh, rh = RUN_CHORDS[b]
        tb = t0 + b * bar
        _bar(mus, rng, tb, bar, lh, rh, RUN_MELODY[b])
        if b >= 1:
            mus.add(strings([midi_hz(m) for m in lh[1:]], bar * 1.02, rng, attack=0.6 if b == 1 else 0.3), tb,
                    0.4 + 0.12 * b, 0.0)
    d = max(0.5, t_land - t1)
    mus.add(strings([midi_hz(m) for m in (45, 57, 64, 69, 73, 76)], d, rng, attack=d * 0.85, release=0.05,
                    bright=3800), t1, 1.4, 0.0)
    rise = np.sin(2 * np.pi * np.cumsum(np.linspace(55.0, 110.0, int(d * SR))) / SR)
    mus.add(0.3 * rise * np.linspace(0, 1, len(rise)) ** 2, t1, 1.0, 0.0)
    mus.add(make_swell(rng, min(1.4, d)), t_land - min(1.4, d), 0.9, 0.0)


def score_race(mus, rng, t0, t_fin, t_merge):
    """The race: driving, in D, a bar a chord (D G A Bm), piano octaves in eighths, strings, a low drum on the
    beat; held in suspense through the slow-motion finish, and the chord as gold wins."""
    chords = [(38, 50, 54, 57), (43, 50, 55, 59), (45, 52, 57, 61), (47, 54, 59, 62)]
    span = max(0.5, t_fin - t0)
    nb = max(2, int(round(span / 1.3)))
    bar = span / nb
    beat = bar / 4
    for b in range(nb):
        ch = chords[b % 4]
        tb = t0 + b * bar
        for k in range(8):
            mus.add(piano(ch[0] + 12 + 12 * (k % 2), beat * 0.45, 0.45 + 0.1 * (k % 4 == 0), k), tb + k * beat / 2,
                    0.55, -0.15)
        mus.add(strings([midi_hz(m) for m in ch[1:]] + [midi_hz(ch[1] + 12)], bar, rng, attack=0.08, release=0.3,
                        bright=2600), tb, 0.6, 0.0)
        for k in range(4):
            mus.add(make_tom(rng), tb + k * beat, 0.45 if k % 2 == 0 else 0.3, 0.0)
    d = max(0.3, t_merge - t_fin)
    mus.add(strings([midi_hz(m) for m in (45, 50, 52, 57, 62, 64)], d + 0.15, rng, attack=0.25, release=0.25,
                    bright=3000), t_fin, 0.75, 0.0)
    for k, m in enumerate((38, 50, 54, 57, 62, 66, 69, 74)):
        mus.add(piano(m, 0.9, 0.7, k), t_merge + 0.01 * k, 0.6, (k - 4) * 0.08)
    mus.add(strings([midi_hz(m) for m in (50, 54, 57, 62, 66)], 0.9, rng, attack=0.02, release=0.6, bright=3200),
            t_merge, 0.8, 0.0)


def score_tail(mus, rng, t0, t1):
    """Over the bridge and on to the field: the felt piano again, cut off where the run stops."""
    chords = [RUN_CHORDS[0], RUN_CHORDS[1], RUN_CHORDS[2], RUN_CHORDS[4]]
    bar = max(1.3, (t1 - t0) / 3.5)
    tb, b = t0, 0
    while tb < t1:
        lh, rh = chords[b % 4]
        _bar(mus, rng, tb, bar, lh, rh, RUN_MELODY[(b % 4) + 1] if b < 3 else (), vel=0.9, cut=t1)
        tb += bar
        b += 1


def score_field(mus, rng, t0, t_done):
    """The field falling: dark now, D minor, i VI iv V (Dm Bb Gm A) twice, building: low strings, a piano ostinato,
    a high string line the second time round, and at 100,000 a wide D minor chord."""
    chords = [((38, 50, 53, 57), (62, 65, 69, 74)), ((34, 46, 50, 53), (58, 62, 65, 70)),
              ((31, 43, 46, 50), (55, 58, 62, 67)), ((33, 45, 49, 52), (57, 61, 64, 69))]
    n = 8
    seg = (t_done - t0) / n
    for c in range(n):
        lh, rh = chords[c % 4]
        tb = t0 + c * seg
        grow = 0.55 + 0.45 * c / (n - 1)
        mus.add(piano(lh[0], seg * 1.05, 0.6 * grow), tb, 0.85, -0.2)
        mus.add(strings([midi_hz(m) for m in lh], seg * 1.02, rng, attack=0.3, release=0.4, bright=1400 + 1200 * grow),
                tb, 0.75 * grow, 0.0)
        for k in range(8):
            m = rh[k % 4] + (12 if c >= 4 and k % 2 else 0)
            mus.add(piano(m, seg / 8 * 0.8, 0.28 + 0.2 * grow, k), tb + k * seg / 8, 0.42 * grow, 0.25 * np.sin(k + c))
        if c >= 4:
            top = (74, 70, 67, 69)[c % 4] + 12
            mus.add(strings([midi_hz(top)], seg * 0.98, rng, attack=0.2, release=0.3, bright=4000), tb, 0.4, 0.1)
    for k, m in enumerate((26, 38, 45, 50, 53, 57, 62, 65, 69)):
        mus.add(piano(m, 2.0, 0.75, k), t_done + 0.012 * k, 0.6, (k - 4) * 0.08)
    mus.add(strings([midi_hz(m) for m in (38, 45, 50, 53, 57, 62)], 1.8, rng, attack=0.05, release=1.2, bright=2400),
            t_done, 0.9, 0.0)


# ---------------------------------------------------------------------------------------------
def build(meta, out_path, seed=11, stems=False):
    """Write the soundtrack for the cue sheet `meta` to out_path (48 kHz 16-bit stereo WAV)."""
    rng = np.random.default_rng(seed)
    cues = meta['frames']
    fps = meta['fps']
    nf = len(cues)
    dur = nf / fps
    n = int(dur * SR) + SR
    sfx, amb, mus = Mix(dur), Mix(dur), Mix(dur)
    times = np.array([c['t'] for c in cues])

    def video_t(t_sim):
        return float(np.interp(t_sim, times, np.arange(nf) / fps))

    ev = meta['events']
    tv = {k: video_t(v) for k, v in ev.items() if not isinstance(v, list)}
    first = {}
    for i, c in enumerate(cues):
        for e in c['events']:
            first.setdefault(e, i / fps)
    for k in ('punch', 'land', 'red_end', 'merge', 'stop', 'done', 'turn', 'face', 'black'):
        tv.setdefault(k, first.get(k, 0.0))
        if k in first:
            tv[k] = first[k]
    strikes = [i / fps for i, c in enumerate(cues) if 'strike' in c['events']]

    # ---- the dominoes: every one clicks as it's hit and again as it comes to rest; the growth knocks
    clicks = [make_click(rng) for _ in range(28)]
    settles = [make_settle(rng) for _ in range(16)]
    for i, c in enumerate(cues):
        t = i / fps
        cam = np.array(c['cam'])
        tgt = np.array(c['tgt'])
        dist = np.linalg.norm(cam - tgt)
        att = float(np.clip(9.0 / max(dist, 1.0), 0.18, 1.3)) ** 0.8
        slow = c['dt'] * fps
        for kind, cnt, gain in (('run', c['run'], 1.1), ('field', c['field'], 0.34), ('rest', c['rest'], 0.3)):
            if cnt <= 0:
                continue
            m = min(cnt, 90)
            g = gain * att * (1.0 if cnt <= m else np.sqrt(cnt / m))
            if kind != 'run' and cnt > 12:
                g = gain * max(att, 0.55) * (1.0 + cnt / 30.0) ** -0.35 * (1.0 if cnt <= m else np.sqrt(cnt / m))
            for _ in range(m):
                bank = settles if kind == 'rest' else clicks
                x = bank[rng.integers(len(bank))]
                if slow < 0.9:
                    x = resample(x, max(0.55, np.sqrt(max(slow, 0.05))))
                pan = rng.uniform(-0.35, 0.35) if kind == 'run' else rng.uniform(-0.8, 0.8)
                sfx.add(x, t + rng.uniform(0, 1.0 / fps), g * rng.uniform(0.7, 1.1), pan)
        for s in c.get('grow', []):
            sfx.add(make_knock(rng, s), t, min(1.25, 0.45 + 0.13 * s ** 0.8), 0.0)

    # ---- the story's sounds
    sfx.add(make_punch(rng), tv['punch'] - 0.14, 0.85, 0.3)
    sfx.add(make_slam(rng), tv['land'], 1.15, 0.0)
    t_tip = video_t(ev['giant'])
    sfx.add(make_creak(rng, max(0.8, tv['land'] - t_tip)), t_tip + 0.1, 0.55, 0.0)
    sfx.add(make_womp(rng), tv['red_end'], 0.5, -0.3)
    sfx.add(resample(make_click(rng, 0.8), 0.7), tv['stop'], 0.7, 0.0)
    sfx.add(make_thunder(rng, 4.0) * 0.5, first.get('rumble', tv['stop'] + 0.8), 0.45, 0.4)
    for k, ts in enumerate(strikes):
        th = make_thunder(rng, max(1.0, min(5.0, dur - ts + 0.5)))
        sfx.add(th if k == 0 else lp(th, 1800, 2), ts + (0.0 if k == 0 else 0.08 * k), 1.05 if k == 0 else 0.55,
                [0.0, -0.4, 0.4, -0.7, 0.7][k % 5])
    if strikes:
        sfx.add(make_swell(rng, 0.8), strikes[0] - 0.8, 0.45, 0.0)
    sfx.add(make_levelup(rng), tv['done'], 0.5, 0.0)
    sfx.add(make_whoosh(rng, 0.5), tv['turn'] - 0.05, 0.7, 0.4)
    sfx.add(make_sting(rng), first.get('face', tv['turn'] + 0.4), 1.1, 0.0)
    sfx.add(make_glitch(rng, max(0.2, dur - tv['black'])), tv['black'], 0.7, 0.0)
    # a heartbeat in the silence, and again before you turn
    hb = make_heartbeat(rng)
    for a, b, bpm in ((tv['stop'] + 0.5, (strikes[0] if strikes else tv['stop'] + 3) - 0.2, 78),
                      (tv['done'] + 0.5, tv['turn'], 96)):
        tb = a
        while tb < b:
            sfx.add(hb, tb, 0.75, 0.0)
            tb += 60.0 / bpm
    amb.add(make_drone(rng, max(0.5, tv['turn'] - tv['done'] + 0.6)), tv['done'], 0.7, 0.0)

    # ---- ambience: wind and birds on a sunny day (gone in the storm), the river by the bridge, the rain
    tt = np.arange(n) / SR
    dark = _env_track(np.array([c['dark'] for c in cues]), fps, n, 3.0)
    rain = _env_track(np.array([c['rain'] for c in cues]), fps, n, 3.0)
    wind = bp(rng.standard_normal(n), 180, 1100, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * tt) *
                                                      np.sin(2 * np.pi * 0.047 * tt + 1.0))
    gust = (0.05 + 0.10 * dark) * (0.7 + 0.3 * np.abs(lp(rng.standard_normal(n), 0.8, 1)) * 40)
    amb.L[:n] += gust * norm(wind)
    amb.R[:n] += gust * norm(np.roll(wind, 2400))
    tb = 0.3
    while tb < tv['stop']:
        amb.add(make_chirp(rng), tb, 0.07 * rng.uniform(0.5, 1.0), rng.uniform(-0.9, 0.9))
        tb += rng.uniform(1.2, 3.2)
    river = np.array([np.clip(1.0 - abs(np.array(c['tgt'])[1] - (-26.0)) / 14.0, 0, 1) for c in cues])
    renv = _env_track(river, fps, n, 2.0)
    water = bp(rng.standard_normal(n), 300, 2400, 2) * (0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 6, 1)) * 12)
    amb.L[:n] += 0.09 * norm(water) * renv
    amb.R[:n] += 0.09 * norm(np.roll(water, 1800)) * renv
    rn = bp(rng.standard_normal(n), 700, 9000, 2)
    drops = hp(rng.standard_normal(n) * (rng.random(n) < 0.002), 2000, 2)
    amb.L[:n] += 0.16 * rain * (norm(rn) + 0.5 * norm(drops))
    amb.R[:n] += 0.16 * rain * (norm(np.roll(rn, 3100)) + 0.5 * norm(np.roll(drops, 777)))

    # ---- the music
    t_land = tv['land']
    score_open(mus, rng, 0.02, video_t(ev['giant']), t_land)
    t_fin = video_t(ev['red_end'] - 0.3)
    score_race(mus, rng, t_land + 0.05, t_fin, tv['merge'])
    score_tail(mus, rng, tv['merge'] + 0.45, tv['stop'])
    if strikes:
        score_field(mus, rng, strikes[0] + 0.5, tv['done'])
    mL, mR = reverb(mus.L[:n], mus.R[:n], decay=2.4, wet=0.32)
    # the music stops dead with the run, comes back with the lightning, and dies as you turn
    g = np.ones(n)
    s0 = int(tv['stop'] * SR)
    s1 = int((strikes[0] if strikes else tv['stop'] + 3.0) * SR)
    g[s0:s1] = np.clip(1.0 - (np.arange(s1 - s0) / SR) / 0.25, 0, 1)
    g *= np.clip((tv['turn'] + 0.3 - tt) / 0.9, 0.0, 1.0)
    mL, mR = mL * g, mR * g
    sL, sR = reverb(sfx.L[:n], sfx.R[:n], seed=5, decay=1.2, wet=0.12)

    # ---- mix and master
    L = sL + MUSIC * mL + amb.L[:n]
    R = sR + MUSIC * mR + amb.R[:n]
    if stems:
        _write(out_path[:-4] + '_sfx.wav', 0.3 * sL, 0.3 * sR)
        _write(out_path[:-4] + '_music.wav', 0.3 * MUSIC * mL, 0.3 * MUSIC * mR)
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    L = L - 0.4 * lp(L, 80.0, 2) + 0.2 * bp(L, 1800.0, 4500.0, 1)
    R = R - 0.4 * lp(R, 80.0, 2) + 0.2 * bp(R, 1800.0, 4500.0, 1)
    L, R = L[:int(dur * SR)], R[:int(dur * SR)]
    # after the glitch: silence
    cut = int(min(dur, tv['black'] + 0.42) * SR)
    L[cut:] = 0.0
    R[cut:] = 0.0
    g = 10 ** ((-18.0 - integrated_lufs(L, R)) / 20.0)
    L, R = compress(L * g, R * g, thresh_db=-24.0, ratio=2.5)
    g = 10 ** ((-14.0 - integrated_lufs(L, R)) / 20.0)
    L, R = limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    lufs = integrated_lufs(L, R)
    f0 = int(0.01 * SR)
    for ch in (L, R):
        ch[:f0] *= np.linspace(0, 1, f0)
    pcm = _write(out_path, L, R)
    print(f'[audio] {dur:.1f}s, {lufs:.1f} LUFS integrated, peak {20 * np.log10(np.abs(pcm).max() / 32768):.1f} dBFS',
          flush=True)
    return out_path


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fh:
        build(json.load(fh), sys.argv[2], stems='--stems' in sys.argv)
