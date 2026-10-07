"""The sound toolkit (from the earlier videos): DSP helpers, a felt piano, strings, bells, a hall reverb, a
compressor, a limiter and BS.1770 loudness, and a few effects. Everything is synthesised (no samples). The
zoom's soundtrack itself is put together in zaudio.py.
"""
import wave

import numpy as np
from scipy import signal

SR = 48000
_NOTE_CACHE = {}


def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output='sos')


def bp(x, lo, hi, order=2):
    hi = min(hi, SR * 0.45)
    return signal.sosfilt(_sos('bandpass', [lo, hi], order), x)


def lp(x, f, order=2):
    return signal.sosfilt(_sos('lowpass', min(f, SR * 0.45), order), x)


def hp(x, f, order=2):
    return signal.sosfilt(_sos('highpass', f, order), x)


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


def _write(path, L, R):
    pcm = np.clip(np.stack([L, R], -1) * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return pcm


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


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


def make_levelup(rng):
    """The counter reaching 10,000: a bright rising chime."""
    x = np.zeros(int(2.6 * SR))
    for k, m in enumerate((86, 90, 93, 98)):
        b = bell(m, rng, 2.2)
        s = int(k * 0.075 * SR)
        x[s:s + len(b)] += b[:len(x) - s] * (0.8 if k < 3 else 1.0)
    return norm(x, 0.8)


def make_swell(rng, dur):
    """A reverse cymbal: noise that swells into the strike."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 3000, 2) + 0.5 * bp(rng.standard_normal(n), 800, 3000, 2)
    return norm(x * (t / dur) ** 3, 0.7)


def make_whoosh(rng, dur=0.5):
    x = shaped_noise(dur, lambda tt: 300 + 4000 * np.sin(np.pi * np.clip(tt / dur, 0, 1)), lambda tt: 0.8 + 0 * tt,
                     lambda tt: np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2, rng)
    return norm(x, 0.8)
