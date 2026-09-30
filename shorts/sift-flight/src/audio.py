"""The soundtrack of the flight, all synthesised here from the cue sheet (cues.py) - no samples.

THE MUSIC is eight bars at 128 bpm in E major, one bar to each part of the flight, and it loops as the video does:
  bar 1  on the spire: a held breath - a soft pad, the wind, a riser, the melody's pick-up; the step off, the elytra
         snapping open
  bar 2  the drop, as the dive peaks: the kick, the bass pumping on the off-beats, supersaw chords, the lead's hook
         (E - B - C#m - A under it, bars 2 to 5)
  bar 4  the first firework: an impact, the boost, the ribs of the fossil whipping past
  bar 5  the lake of ichor and its falls, wide open
  bar 6  the climb between the coral towers: F#m - B, a snare roll and a riser building
  bar 7  the second firework, the peak: the hook an octave up, the firework show over the valley
  bar 8  the swing round the spire and the landing: A - B resolving to E as the feet touch down, the drums stopping,
         the pad left holding - which is where bar 1 picks up.

THE EFFECTS follow the flight: the wind with the speed (a breeze on the spire, a roar in the dive), the step off the
edge and the elytra opening, the rockets' launch, hiss and crackle, whooshes as each tree, rib and tower goes past
(panned to its side, louder the closer it is), the ichor falls' roar and the lake's bubbling as they pass, the show's
rockets whistling up and bursting, the landing. The mix wraps round - what rings on past the last frame is heard under
the first - so the loop has no seam. Loudness normalised to -14 LUFS with a peak limiter.
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


def sine_sweep(dur, f0, f1, tau_f):
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


def varispeed(x, rate):
    """Play x at a time-varying rate (per output sample): Doppler shifts, slowing down."""
    ph = np.cumsum(rate)
    ph = ph[ph < len(x) - 1]
    return np.interp(ph, np.arange(len(x)), x)


def modal(dur, f0, ratios, amps, taus, rng, detune=0.004):
    """Sum of decaying inharmonic partials (struck metal)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a, tau in zip(ratios, amps, taus):
        f = f0 * r * (1 + rng.uniform(-detune, detune))
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    return x


def saw(freq, rng=None):
    """A sawtooth from a per-sample frequency array (band-limit it with a low-pass after)."""
    ph = np.cumsum(freq) / SR + (rng.uniform() if rng is not None else 0.0)
    return 2.0 * np.mod(ph, 1.0) - 1.0


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
        s = int(round(t * SR))
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

    def add2(self, L, R, t, gain=1.0):
        """A stereo signal."""
        s = int(round(t * SR))
        if s >= self.n:
            return
        if s < 0:
            L, R = L[-s:], R[-s:]
            s = 0
        e = min(self.n, s + len(L))
        self.L[s:e] += L[:e - s] * gain
        self.R[s:e] += R[:e - s] * gain

    def add_dyn(self, x, t, gain, pan):
        """A mono signal with per-sample gain and pan arrays (the same length as x)."""
        s = int(round(t * SR))
        if s >= self.n:
            return
        if s < 0:
            x, gain, pan = x[-s:], gain[-s:], pan[-s:]
            s = 0
        e = min(self.n, s + len(x))
        m = e - s
        p = np.clip(pan[:m], -1, 1)
        self.L[s:e] += x[:m] * gain[:m] * np.cos((p + 1) * np.pi / 4) * np.sqrt(2)
        self.R[s:e] += x[:m] * gain[:m] * np.sin((p + 1) * np.pi / 4) * np.sqrt(2)


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


def reverb(L, R, seed=3, decay=2.0, wet=0.3, pre=0.018, dark=5200.0):
    """Convolution with a generated stereo hall: exponentially decaying, darkening noise."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    n = int(decay * 1.6 * SR)
    t = np.arange(n) / SR
    out = []
    for ch, x in enumerate((L, R)):
        ir = rng.standard_normal(n) * np.exp(-t / (decay / 6.9))
        ir = lp(ir, dark, 2) * 0.7 + lp(ir, 1400, 2) * 0.3
        ir[:int(pre * SR)] = 0.0
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        y = fftconvolve(x, ir)[:len(x)]
        out.append(x * (1 - wet) + y * wet * 1.4)
    return out[0], out[1]


def delay(L, R, t_d, fb=0.35, wet=0.3, damp=3500.0):
    """A ping-pong delay (t_d seconds), darkening on each repeat."""
    d = int(t_d * SR)
    oL, oR = np.zeros_like(L), np.zeros_like(R)
    xL, xR = L.copy(), R.copy()
    g = wet
    for k in range(6):
        xL, xR = lp(np.concatenate([np.zeros(d), xR[:-d]]), damp, 1), lp(np.concatenate([np.zeros(d), xL[:-d]]), damp, 1)
        oL += g * xL
        oR += g * xR
        g *= fb
    return L + oL, R + oR


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


# ---------------------------------------------------------------------------------------------
# drums
# ---------------------------------------------------------------------------------------------
_DRUM = {}


def _cached(key, fn):
    if key not in _DRUM:
        _DRUM[key] = fn()
    return _DRUM[key]


def kick(v=0):
    def make():
        rng = np.random.default_rng(100 + v)
        dur = 0.5
        n = int(dur * SR)
        body = sine_sweep(dur, 165.0, 47.0, 0.032) * expenv(n, 0.30, 0.001)
        click = hp(rng.standard_normal(n), 2500, 2) * expenv(n, 0.003)
        x = body + 0.35 * click
        return norm(np.tanh(1.6 * x), 0.95)
    return _cached(('kick', v), make)


def snare(v=0, tight=1.0):
    def make():
        rng = np.random.default_rng(200 + v)
        dur = 0.4
        n = int(dur * SR)
        tone = (np.sin(2 * np.pi * 186 * np.arange(n) / SR) + 0.5 * np.sin(2 * np.pi * 330 * np.arange(n) / SR)) \
            * expenv(n, 0.05 * tight, 0.001)
        rattle = bp(rng.standard_normal(n), 1400, 9500, 2) * expenv(n, 0.13 * tight, 0.001)
        crack = hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.008)
        return norm(0.7 * tone + 1.0 * rattle + 0.5 * crack, 0.9)
    return _cached(('snare', v, tight), make)


def clap(v=0):
    def make():
        rng = np.random.default_rng(300 + v)
        dur = 0.35
        n = int(dur * SR)
        x = np.zeros(n)
        for k, dt in enumerate((0.0, 0.011, 0.023, 0.034)):
            s = int(dt * SR)
            m = n - s
            x[s:] += bp(rng.standard_normal(m), 900, 5200, 2) * expenv(m, 0.006 if k < 3 else 0.11)
        return norm(x, 0.85)
    return _cached(('clap', v), make)


def hat(v=0, open_=False):
    def make():
        rng = np.random.default_rng(400 + v)
        dur = 0.35 if open_ else 0.06
        n = int(dur * SR)
        x = hp(rng.standard_normal(n), 7000, 2) + 0.3 * modal(dur, 5300, (1.0, 1.47, 1.93), (1, 0.7, 0.5),
                                                             (0.05, 0.04, 0.03), rng)
        return norm(x * expenv(n, 0.11 if open_ else 0.018, 0.0005), 0.6)
    return _cached(('hat', v, open_), make)


def taiko(v=0):
    def make():
        rng = np.random.default_rng(500 + v)
        dur = 0.9
        n = int(dur * SR)
        body = sine_sweep(dur, 128.0, 58.0, 0.05) * expenv(n, 0.34, 0.002)
        skin = lp(rng.standard_normal(n), 600, 2) * expenv(n, 0.05)
        return norm(np.tanh(1.3 * (body + 0.5 * skin)), 0.95)
    return _cached(('taiko', v), make)


def tom(f0, v=0):
    def make():
        rng = np.random.default_rng(600 + v)
        dur = 0.6
        n = int(dur * SR)
        body = sine_sweep(dur, f0 * 1.5, f0, 0.04) * expenv(n, 0.22, 0.001)
        skin = bp(rng.standard_normal(n), 300, 3000, 2) * expenv(n, 0.02)
        return norm(body + 0.3 * skin, 0.9)
    return _cached(('tom', f0, v), make)


def crash(v=0, dur=2.6):
    def make():
        rng = np.random.default_rng(700 + v)
        n = int(dur * SR)
        x = hp(rng.standard_normal(n), 3200, 2)
        x += 0.5 * modal(dur, 3150, (1.0, 1.41, 1.93, 2.66, 3.3), (1.0, 0.8, 0.7, 0.5, 0.4),
                         (0.9, 0.7, 0.6, 0.5, 0.4), rng, 0.02)
        return norm(x * expenv(n, 0.75, 0.001), 0.6)
    return _cached(('crash', v, dur), make)


def impact(rng, size=1.0):
    """A cinematic hit: a sub drop, a noise punch, a crash."""
    dur = 3.2
    n = int(dur * SR)
    sub = sine_sweep(dur, 90.0, 29.0, 0.35) * expenv(n, 0.9 * size, 0.002)
    punch = lp(rng.standard_normal(n), 1800, 2) * expenv(n, 0.05, 0.001)
    x = 1.0 * sub + 0.6 * punch
    x[:len(crash(1, 3.0))] += 0.5 * crash(1, 3.0)[:n]
    return norm(np.tanh(1.4 * x), 0.95)


def riser(rng, dur, f0=300.0, f1=6000.0):
    """White noise sweeping up, and a saw gliding up an octave, swelling."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    nz = shaped_noise(dur, lambda tt: f0 * (f1 / f0) ** (tt / dur), lambda tt: 0.9 + 0 * tt,
                      lambda tt: (tt / dur) ** 2, rng)
    glide = saw(110.0 * 2 ** (u * 1.0) * np.ones(n), rng) + saw(110.0 * 1.006 * 2 ** (u * 1.0) * np.ones(n), rng)
    glide = lp(glide, 500 + 4000 * u.mean(), 2) * u ** 2.5
    return norm(norm(nz) + 0.35 * norm(glide), 0.8)


def reverse_crash(rng, dur):
    """A crash played backwards: it swells into the moment."""
    c = crash(2, max(dur, 1.0) + 0.3)
    x = c[:int(dur * SR)][::-1].copy()
    x += 0.4 * hp(rng.standard_normal(len(x)), 1500, 2) * (np.arange(len(x)) / len(x)) ** 4
    return norm(x, 0.7)


def snare_roll(rng, dur, start_div=4, end_div=16, bpm=140.0):
    """A snare roll accelerating from start_div to end_div notes per bar, crescendo."""
    out = np.zeros(int((dur + 0.4) * SR))
    t = 0.0
    while t < dur:
        u = t / dur
        div = start_div * (end_div / start_div) ** u
        step = 240.0 / bpm / div
        s = snare(int(t * 97) % 3, tight=0.7)
        i = int(t * SR)
        m = min(len(s), len(out) - i)
        out[i:i + m] += s[:m] * (0.25 + 0.75 * u ** 1.5)
        t += step
    return out


# ---------------------------------------------------------------------------------------------
# synths
# ---------------------------------------------------------------------------------------------
def supersaw(freqs, dur, rng, attack=0.01, release=0.25, cutoff=4200.0, voices=7, spread=0.012):
    """Detuned saw stack (stereo): odd voices left, even right. Returns (L, R)."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    L, R = np.zeros(n), np.zeros(n)
    dets = np.linspace(-spread, spread, voices)
    for f in freqs:
        for k, d in enumerate(dets):
            x = saw(np.full(n, f * (1 + d)), rng)
            if k % 2:
                L += x
            else:
                R += x
            if k == voices // 2:
                L += 0.5 * x
                R += 0.5 * x
    env = np.clip(t / max(attack, 1e-3), 0, 1) * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / release, 0, 1))
    L = lp(lp(L, cutoff, 2), cutoff * 1.5, 2) * env
    R = lp(lp(R, cutoff, 2), cutoff * 1.5, 2) * env
    s = 1.0 / (voices * len(freqs))
    return L * s, R * s


def pluck(m, dur, rng, bright=1.0, decay=0.35):
    """A plucked synth: harmonics whose upper partials die first, a little chorus."""
    f0 = midi_hz(m)
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 14):
        fk = k * f0
        if fk > 12000:
            break
        amp = (1.0 / k) ** (1.2 - 0.3 * bright)
        tau = decay / (1 + 0.55 * (k - 1))
        for d in (-0.0035, 0.0035):
            x += 0.5 * amp * np.sin(2 * np.pi * fk * (1 + d) * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    x *= np.clip(t / 0.002, 0, 1) * np.where(t < dur, 1.0, np.exp(-np.clip(t - dur, 0, None) / 0.05))
    return norm(x, 0.8)


def lead(m, dur, rng, vib=0.004, bright=5000.0):
    """A sustained supersaw lead with vibrato that comes in after the attack."""
    f0 = midi_hz(m)
    n = int((dur + 0.18) * SR)
    t = np.arange(n) / SR
    vibe = 1 + vib * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.18) / 0.2, 0, 1)
    x = np.zeros(n)
    for d in (-0.006, 0.0, 0.006):
        x += saw(f0 * (1 + d) * vibe, rng)
    x = lp(x, bright, 2)
    env = np.clip(t / 0.006, 0, 1) * (0.7 + 0.3 * np.exp(-t / 0.12)) * \
        np.where(t < dur, 1.0, np.clip(1 - (t - dur) / 0.18, 0, 1))
    return norm(x * env, 0.8)


def reese(m, dur, rng, cutoff=650.0, drive=2.2):
    """The drum-and-bass bass: two detuned saws beating, low-passed and driven."""
    f0 = midi_hz(m)
    n = int((dur + 0.08) * SR)
    t = np.arange(n) / SR
    x = saw(np.full(n, f0 * 0.994), rng) + saw(np.full(n, f0 * 1.006), rng)
    x = lp(x, cutoff, 2)
    x = np.tanh(drive * x)
    x = lp(x, cutoff * 1.6, 2)
    sub = np.sin(2 * np.pi * f0 * t)
    env = np.clip(t / 0.004, 0, 1) * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / 0.08, 0, 1))
    return norm((0.8 * norm(x) + 0.7 * sub) * env, 0.9)


def subbass(m, dur, attack=0.004):
    f0 = midi_hz(m)
    n = int((dur + 0.06) * SR)
    t = np.arange(n) / SR
    env = np.clip(t / attack, 0, 1) * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / 0.06, 0, 1))
    return np.sin(2 * np.pi * f0 * t) * env


def choir(ms, dur, rng, attack=0.35, release=0.8, vowel='a'):
    """A choir chord: detuned saw voices with vibrato through the formants of a sung vowel. Returns (L, R)."""
    F = {'a': ((800, 1.0), (1150, 0.55), (2900, 0.22)), 'o': ((500, 1.0), (850, 0.5), (2800, 0.12))}[vowel]
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    L, R = np.zeros(n), np.zeros(n)
    for m in ms:
        f0 = midi_hz(m)
        for k in range(5):
            vibe = 1 + 0.0045 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6.28))
            x = saw(f0 * (1 + rng.uniform(-0.004, 0.004)) * vibe, rng)
            if k % 2:
                L += x
            else:
                R += x
    out = []
    for x in (L, R):
        y = sum(a * bp(x, fc * 0.85, fc * 1.15, 2) for fc, a in F)
        out.append(y)
    env = np.clip(t / attack, 0, 1) ** 1.5 * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / release, 0, 1))
    s = 1.0 / (5 * len(ms))
    return out[0] * env * s * 6, out[1] * env * s * 6


def bell(m, rng, dur=1.6, bright=1.0):
    """A glassy bell for the End's arpeggios."""
    f0 = midi_hz(m)
    x = modal(dur, f0, (1.0, 2.0, 3.0, 4.2, 5.4), (1.0, 0.45 * bright, 0.25 * bright, 0.12, 0.06),
              (0.9, 0.5, 0.3, 0.18, 0.1), rng, 0.001)
    x *= np.clip(np.arange(len(x)) / SR / 0.002, 0, 1)
    return norm(x, 0.8)


def braam(ms, dur, rng):
    """The brass hit: low saws, the filter snapping open and closing, driven hard."""
    n = int((dur + 0.6) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in ms:
        for d in (-0.008, 0.0, 0.008):
            x += saw(np.full(n, midi_hz(m) * (1 + d)), rng)
    dark = lp(x, 500, 2)
    bright = lp(x, 2600, 2) * np.exp(-t / 0.35)
    y = np.tanh(1.8 * (dark + bright) / len(ms))
    env = np.clip(t / 0.03, 0, 1) * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / 0.6, 0, 1))
    return norm(y * env, 0.9)


def note_block(m, rng):
    """A note block (the game's harp): a bright pluck that dies away quickly."""
    dur = 0.9
    x = modal(dur, midi_hz(m), (1.0, 2.0, 3.0, 4.1), (1.0, 0.35, 0.15, 0.05), (0.45, 0.22, 0.12, 0.07), rng, 0.0015)
    n = len(x)
    x += 0.15 * hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.003)
    return norm(x * np.clip(np.arange(n) / SR / 0.0015, 0, 1), 0.8)


def drone(ms, dur, rng, cutoff=900.0):
    n = int(dur * SR)
    x = np.zeros(n)
    for m in ms:
        for d in (-0.004, 0.004):
            x += saw(np.full(n, midi_hz(m) * (1 + d)), rng)
    x = lp(x, cutoff, 2)
    return norm(x, 0.8)


# ---------------------------------------------------------------------------------------------
# sound effects
# ---------------------------------------------------------------------------------------------
def whoosh(rng, dur, f_hi, f_lo, width=1.0):
    """Air torn past something: band-passed noise swelling in, its pitch falling as it goes by."""
    n = int(dur * SR)
    x = shaped_noise(dur, lambda t: f_lo + (f_hi - f_lo) * np.exp(-3.0 * t / dur), lambda t: width + 0 * t,
                     lambda t: np.clip(t / (0.35 * dur), 0, 1) ** 2 * np.exp(-np.clip(t - 0.35 * dur, 0, None) /
                                                                            (0.22 * dur)), rng)
    return norm(x[:n], 0.9)


def grass_step(rng):
    """A step on grass: a few crunchy bursts."""
    n = int(0.22 * SR)
    x = np.zeros(n)
    for k in range(5):
        s = int(rng.uniform(0.0, 0.05) * SR)
        m = n - s
        x[s:] += bp(rng.standard_normal(m), 600, 5200, 2) * expenv(m, rng.uniform(0.008, 0.02)) * rng.uniform(0.4, 1)
    x += 0.5 * lp(rng.standard_normal(n), 220, 2) * expenv(n, 0.03)
    return norm(x, 0.8)


def elytra_open(rng):
    """The wings snapping open: a low whump of air caught, a flap of fabric, the air starting to rush."""
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    whump = lp(rng.standard_normal(n), 240, 2) * expenv(n, 0.09, 0.006)
    snap = hp(rng.standard_normal(n), 2400, 2) * expenv(n, 0.018, 0.001)
    flap = bp(rng.standard_normal(n), 300, 2500, 2) * expenv(n, 0.05, 0.002) * (1 + 0.6 * np.sin(2 * np.pi * 26 * t))
    air = shaped_noise(0.9, lambda tt: 500 + 1800 * tt, lambda tt: 1.1 + 0 * tt,
                       lambda tt: np.clip(tt / 0.3, 0, 1) * np.exp(-tt / 0.5), rng)[:n]
    return norm(1.0 * norm(whump) + 0.45 * norm(snap) + 0.55 * norm(flap) + 0.5 * norm(air), 0.9)


def rocket_launch(rng):
    """A firework rocket lit in the hand: a pop, then a fierce hiss that climbs and fades as it drives you on."""
    dur = 1.3
    n = int(dur * SR)
    pop = lp(rng.standard_normal(n), 900, 2) * expenv(n, 0.02, 0.001)
    hiss = shaped_noise(dur, lambda t: 1800 + 3400 * np.clip(t / 0.5, 0, 1), lambda t: 0.9 + 0 * t,
                        lambda t: np.clip(t / 0.03, 0, 1) * np.exp(-t / 0.55), rng)[:n]
    roar = lp(rng.standard_normal(n), 400, 2) * np.clip(np.arange(n) / SR / 0.05, 0, 1) * expenv(n, 0.5)
    return norm(0.8 * norm(pop) + 1.0 * norm(hiss) + 0.45 * norm(roar), 0.9)


def crackle(rng, dur, rate=60.0, spread=1.0):
    """The twinkle of a firework's stars: a crackle of tiny pops scattered in time and space. Returns (L, R)."""
    n = int((dur + 0.1) * SR)
    L, R = np.zeros(n), np.zeros(n)
    k = int(rate * dur)
    ts = np.sort(rng.uniform(0, dur, k)) + rng.uniform(0, 0.02, k)
    for i, t in enumerate(ts):
        m = int(rng.uniform(0.004, 0.012) * SR)
        s = int(t * SR)
        if s + m >= n:
            continue
        pop = hp(rng.standard_normal(m), rng.uniform(2500, 6000), 2) * expenv(m, m / SR / 4)
        a = rng.uniform(0.3, 1.0) * (1.0 - 0.6 * t / dur)
        p = rng.uniform(-spread, spread)
        L[s:s + m] += pop * a * np.cos((p + 1) * np.pi / 4)
        R[s:s + m] += pop * a * np.sin((p + 1) * np.pi / 4)
    return norm(L, 0.8), norm(R, 0.8)


def boom(rng, size=1.0):
    """A firework bursting far off: a deep thump, a crack, a rolling tail."""
    dur = 2.4
    n = int(dur * SR)
    thump = sine_sweep(dur, 110.0, 38.0, 0.08) * expenv(n, 0.35 * size, 0.003)
    crack = bp(rng.standard_normal(n), 400, 5000, 2) * expenv(n, 0.03, 0.001)
    roll = lp(rng.standard_normal(n), 700, 2) * expenv(n, 0.7, 0.02)
    return norm(np.tanh(1.3 * (1.0 * thump + 0.5 * norm(crack) + 0.35 * norm(roll))), 0.95)


def whistle(rng, dur):
    """A rocket climbing: a whistle rising in pitch, a sparkling hiss under it."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 900 + 1900 * (t / dur) ** 1.4
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    env = np.clip(t / 0.08, 0, 1) * np.clip((dur - t) / 0.05, 0, 1)
    hiss = hp(rng.standard_normal(n), 3000, 2) * 0.3
    return norm((tone * 0.6 + hiss) * env * (0.5 + 0.5 * t / dur), 0.8)


def falls_roar(rng, dur):
    """The ichor falls: a thick, slow roar (lower than water's), churning."""
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 90, 1800, 2) + 0.4 * bp(rng.standard_normal(n), 1800, 5000, 2)
    churn = 0.75 + 0.25 * np.sin(2 * np.pi * 3.1 * np.arange(n) / SR) * np.sin(2 * np.pi * 0.7 * np.arange(n) / SR)
    return norm(x * churn, 0.9)


def gloop(rng):
    """A bubble of ichor bursting: a thick low pop."""
    dur = 0.18
    n = int(dur * SR)
    x = sine_sweep(dur, rng.uniform(260, 420), rng.uniform(70, 110), 0.03) * expenv(n, 0.05, 0.002)
    return norm(x + 0.2 * lp(rng.standard_normal(n), 800, 2) * expenv(n, 0.01), 0.8)


def squeak(rng):
    """A blub's chirp."""
    dur = 0.16
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = rng.uniform(1300, 1700) * (1 + 0.35 * np.sin(np.pi * t / dur)) * (1 + 0.03 * np.sin(2 * np.pi * 40 * t))
    return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 2, 0.7)


def touchdown(rng):
    """Landing on the spire's top: a soft thud on grass, the wings folding away."""
    n = int(0.8 * SR)
    thud = lp(rng.standard_normal(n), 160, 2) * expenv(n, 0.07, 0.003)
    body = sine_sweep(0.8, 90.0, 45.0, 0.05) * expenv(n, 0.09, 0.002)
    fold = shaped_noise(0.8, lambda t: 1400 - 800 * t, lambda t: 1.2 + 0 * t,
                        lambda t: np.clip((t - 0.08) / 0.1, 0, 1) * np.exp(-np.clip(t - 0.18, 0, None) / 0.18), rng)[:n]
    x = 1.0 * norm(thud) + 0.8 * norm(body) + 0.4 * norm(fold)
    x[:int(0.22 * SR)] += 0.5 * grass_step(rng)[:int(0.22 * SR)]
    return norm(x, 0.9)


# ---------------------------------------------------------------------------------------------
# the score: 8 bars at 128 bpm in E major
# ---------------------------------------------------------------------------------------------
BPM = 128.0
BAR = 240.0 / BPM
BEAT = BAR / 4
TAIL = 4.0                  # what rings on past the end is wrapped round under the start


def bt(bar, beat=0.0):
    """The time of a beat (bars from 1, beats from 0)."""
    return (bar - 1) * BAR + beat * BEAT


# the chords, a bar each from bar 2 (bar 6 and bar 8 change mid-bar): (beat, voicing, bass)
E_, B_, CSM, A_ = (64, 68, 71, 76), (63, 66, 71, 75), (64, 68, 73, 76), (64, 69, 73, 76)
FSM7 = (64, 66, 69, 73)
CHORDS = {2: [(0, E_, 40)], 3: [(0, B_, 35)], 4: [(0, CSM, 37)], 5: [(0, A_, 33)],
          6: [(0, FSM7, 42), (2, B_, 35)], 7: [(0, E_, 40)], 8: [(0, A_, 33), (1, B_, 35), (2, E_, 40)]}
PAD_END = (52, 59, 64, 66, 68, 71)       # E add9: from the touchdown round to the drop
# the hook, per bar: (eighth, midi, eighths)
HOOK = {1: [(6, 71, 1), (7, 73, 1)],
        2: [(0, 76, 2), (2, 76, 1), (3, 78, 1), (4, 80, 2), (6, 78, 1), (7, 76, 1)],
        3: [(0, 75, 2), (2, 75, 1), (3, 76, 1), (4, 78, 2), (6, 76, 1), (7, 75, 1)],
        4: [(0, 73, 2), (2, 73, 1), (3, 75, 1), (4, 76, 2), (6, 75, 1), (7, 73, 1)],
        5: [(0, 71, 2), (2, 73, 2), (4, 76, 2), (6, 78, 2)],
        6: [(0, 81, 3), (3, 80, 1), (4, 78, 2), (6, 75, 1), (7, 78, 1)],
        7: [(0, 80, 2), (2, 80, 1), (3, 81, 1), (4, 83, 2), (6, 81, 1), (7, 80, 1)],
        8: [(0, 81, 2), (2, 80, 1), (3, 78, 1), (4, 80, 6)]}


def _chord_spans():
    """(start, length, voicing, bass) of every chord from the drop to the touchdown."""
    out = []
    for bar in range(2, 9):
        ch = CHORDS[bar]
        for k, (b0, v, bass) in enumerate(ch):
            b1 = ch[k + 1][0] if k + 1 < len(ch) else 4
            if bar == 8 and k == len(ch) - 1:
                continue                                  # the last E is the pad's (it holds round the loop)
            out.append((bt(bar, b0), (b1 - b0) * BEAT, v, bass))
    return out


def score(rng):
    """The music as three mixes over the video and its tail: the main bus, the pumping bus (ducked by the kick) and
    the echo bus (a dotted-eighth ping-pong)."""
    dur = BAR * 8 + TAIL
    mus, pump, dly = Mix(dur), Mix(dur), Mix(dur)
    kicks = []
    t_touch = bt(8, 2)

    # ---- the pad: E add9 from the touchdown round the loop to the drop; the chords' pad under bars 2-8
    L, R = supersaw([midi_hz(m) for m in PAD_END], BAR * 2 - BEAT * 2 + BAR, rng, attack=0.35, release=1.2,
                    cutoff=1500, voices=5, spread=0.009)
    mus.add2(L, R, t_touch, 0.55)
    L, R = choir(PAD_END[1:5], BAR * 2 - BEAT * 2 + BAR, rng, attack=0.5, release=1.2, vowel='o')
    mus.add2(L, R, t_touch, 0.22)
    for (t0, ln, v, bass) in _chord_spans():
        L, R = supersaw([midi_hz(m) for m in v] + [midi_hz(v[0] - 12)], ln, rng, attack=0.012, release=0.28,
                        cutoff=3400 if t0 < bt(6) else 3000 + 1800 * np.clip((t0 - bt(6)) / BAR, 0, 1))
        pump.add2(L, R, t0, 0.50)
        L, R = supersaw([midi_hz(m - 12) for m in v[:3]], ln, rng, attack=0.2, release=0.6, cutoff=1200, voices=5)
        pump.add2(L, R, t0, 0.30)
        # the bass: sub on the beat, the octave on the off-beats
        steps = int(round(ln / (BEAT / 2)))
        for e8 in range(steps):
            m = bass + (12 if e8 % 2 else 0)
            x = reese(m, BEAT * 0.42, rng, cutoff=820, drive=1.3)
            pump.add(x, t0 + e8 * BEAT / 2, 0.40 if e8 % 2 else 0.52, 0.0)
        # the arpeggio: sixteenths up and down the chord
        seq = list(v) + [v[-1] + 12] + list(v[::-1][1:-1])
        for s16 in range(int(round(ln / (BEAT / 4)))):
            m = seq[s16 % len(seq)] + 12
            dly.add(pluck(m, BEAT * 0.2, rng, bright=0.8, decay=0.12), t0 + s16 * BEAT / 4, 0.16,
                    0.55 * np.sin(s16 * 1.1))
    # the final chord at the touchdown: a hit on top of the pad
    L, R = supersaw([midi_hz(m) for m in E_] + [midi_hz(52)], BEAT * 1.2, rng, attack=0.005, release=0.9,
                    cutoff=4200)
    mus.add2(L, R, t_touch, 0.42)
    mus.add(subbass(28, BEAT * 3), t_touch, 0.55, 0.0)

    # ---- the hook: a supersaw lead doubled by a pluck through the echo
    for bar, notes in HOOK.items():
        for (e8, m, ln) in notes:
            t0 = bt(bar, e8 / 2)
            d = ln * BEAT / 2
            up = 12 if bar == 7 else 0
            dly.add(pluck(m, d * 0.9, rng, bright=1.0, decay=0.45), t0, 0.34, -0.12)
            mus.add(lead(m, d * 0.94, rng, bright=5200 if bar < 6 else 6500), t0, 0.19 if bar > 1 else 0.12, 0.1)
            if up:
                mus.add(lead(m + up, d * 0.94, rng, bright=7000), t0, 0.09, -0.2)
    # the choir comes in for the peak and the landing
    L, R = choir((64, 68, 71, 76), BAR * 1.0, rng, attack=0.25, release=0.5, vowel='a')
    mus.add2(L, R, bt(7), 0.30)
    L, R = choir((64, 69, 73, 76), BEAT * 1.0, rng, attack=0.08, release=0.3, vowel='a')
    mus.add2(L, R, bt(8), 0.26)
    L, R = choir((63, 66, 71, 75), BEAT * 1.0, rng, attack=0.08, release=0.3, vowel='a')
    mus.add2(L, R, bt(8, 1), 0.26)

    # ---- drums: four on the floor from the drop to the touchdown; claps on 2 and 4; hats
    for bar in range(2, 9):
        for q in range(4):
            t = bt(bar, q)
            if t > t_touch + 1e-6:
                break
            kicks.append(t)
            mus.add(kick(q % 2), t, 0.92 if t < t_touch - 1e-6 else 1.0)
            if q in (1, 3) and bar < 8:
                mus.add(clap(q), t, 0.46, 0.05)
                mus.add(snare(q, 0.8), t, 0.26, -0.05)
        if bar < 8:
            for s16 in range(16):
                op = s16 % 4 == 2
                if bar < 5 and not op and s16 % 2:
                    continue
                g = (0.17 if op else 0.10) * (0.75 + 0.25 * (s16 % 2 == 0))
                mus.add(hat(s16 % 3, op), bt(bar, s16 / 4), g, 0.35 * np.sin(s16 * 1.3))
    for bar in (2, 4, 5, 7):
        mus.add(crash(bar), bt(bar), 0.46, 0.25 if bar % 2 else -0.25)
    mus.add(crash(8, 3.2), t_touch, 0.5, 0.0)

    # ---- the builds and the hits
    # bar 1: the held breath - a riser from the step off, a roll, the reversed crash into the drop
    mus.add(riser(rng, bt(2) - 0.47, 250, 7000), 0.47, 0.30, 0.0)
    mus.add(snare_roll(rng, BEAT * 2, 8, 32, BPM), bt(1, 2), 0.24, 0.0)
    mus.add(reverse_crash(rng, BEAT * 2), bt(2) - BEAT * 2, 0.40, 0.0)
    mus.add(impact(rng, 1.1), bt(2), 0.78, 0.0)
    # bar 4: the first firework
    mus.add(impact(rng, 0.8), FIREWORK_T[0], 0.52, 0.0)
    # bar 6: the build to the peak
    mus.add(snare_roll(rng, BAR - BEAT * 0.25, 4, 32, BPM), bt(6), 0.30, 0.0)
    mus.add(riser(rng, BAR, 200, 9000), bt(6), 0.30, 0.0)
    mus.add(reverse_crash(rng, BEAT * 2), bt(7) - BEAT * 2, 0.42, 0.0)
    # bar 7: the second firework, the peak
    mus.add(impact(rng, 1.3), bt(7), 0.82, 0.0)
    # the touchdown
    mus.add(impact(rng, 0.6), t_touch, 0.42, 0.0)
    return mus, pump, dly, kicks


FIREWORK_T = (5.625, 11.25)


# ---------------------------------------------------------------------------------------------
# the mix
# ---------------------------------------------------------------------------------------------
def sidechain(t_kicks, n, depth=0.6, rel=0.14):
    """Gain curve that dips on every kick (the pumping of the pads and bass)."""
    g = np.ones(n)
    for tk in t_kicks:
        s = int(tk * SR)
        m = min(n - s, int(rel * 4 * SR))
        if m <= 0:
            continue
        u = np.arange(m) / SR
        g[s:s + m] = np.minimum(g[s:s + m], 1 - depth * np.exp(-u / rel) * np.clip(u / 0.004, 0, 1))
    return g


def _per_sample(values, fps, n):
    return np.interp(np.arange(n) / SR * fps, np.arange(len(values)), values)


def _wrap(x, n):
    """Fold what rings on past n back onto the start (the video loops)."""
    y = x[:n].copy()
    k = min(len(x) - n, n)
    y[:k] += x[n:n + k]
    return y


def build(cues, out_path, seed=5, stems=False):
    """Write the soundtrack for the cue sheet to out_path (48 kHz 16-bit stereo WAV)."""
    rng = np.random.default_rng(seed)
    frames, fps = cues['frames'], cues['fps']
    ev = cues['events']
    dur = cues['duration']
    n = int(round(dur * SR))
    N = n + int(TAIL * SR)
    tt = np.arange(N) / SR
    sfx = Mix(dur + TAIL)
    amb = Mix(dur + TAIL)
    v = _per_sample([c['v'] for c in frames] + [0.0] * int(TAIL * fps + 2), fps, N)
    agl = _per_sample([c['agl'] for c in frames] + [frames[-1]['agl']] * int(TAIL * fps + 2), fps, N)

    # ---- the wind: a breeze on the spire, a roar with the speed; the wings' flutter when fast
    speed = np.clip(v / 50.0, 0, 1.4)
    env = 0.10 + 0.9 * speed ** 1.6
    env = env * (0.85 + 0.15 * np.abs(lp(rng.standard_normal(N), 1.5, 1)) * 40)
    w1 = shaped_noise(dur + TAIL + 0.5, lambda t: 300 + 1300 * np.interp(t, tt, speed), lambda t: 1.4 + 0 * t,
                      lambda t: np.interp(t, tt, env), rng)[:N]
    w2 = shaped_noise(dur + TAIL + 0.5, lambda t: 2200 + 2600 * np.interp(t, tt, speed), lambda t: 0.7 + 0 * t,
                      lambda t: np.interp(t, tt, env * speed * 0.6), rng)[:N]
    flutter = 1 + 0.18 * speed * np.sin(2 * np.pi * 17.0 * tt) * np.sin(2 * np.pi * 0.9 * tt)
    wL = norm((w1 + w2) * flutter)
    wR = norm((np.roll(w1, 1100) + np.roll(w2, 700)) * flutter)
    sfx.add2(0.34 * wL, 0.34 * wR, 0.0)
    # the ground rushing under when low and fast
    low = np.clip((10.0 - agl) / 8.0, 0, 1) * speed
    rush = lp(rng.standard_normal(N), 350, 2) * lp(np.concatenate([low, np.zeros(SR)])[:N], 3, 1)
    sfx.add(norm(rush), 0.0, 0.22, 0.0)

    # ---- the step off, the wings opening
    sfx.add(grass_step(rng), ev['step'] - 0.03, 0.55, 0.1)
    sfx.add(elytra_open(rng), ev['elytra'] - 0.02, 0.72, 0.0)

    # ---- the rockets in the hand
    for tf in ev['fireworks']:
        sfx.add(rocket_launch(rng), tf - 0.01, 0.80, 0.15)
        L, R = crackle(rng, 1.0, 45.0, 0.9)
        sfx.add2(L * 0.30, R * 0.30, tf + 0.08)

    # ---- things going past: trees, the fossil's ribs, the coral towers, the arch
    for kind, items, base in (('tree', ev['trees'], 0.34), ('tower', ev['towers'], 0.36)):
        for e in items:
            g = base * float(np.clip(5.0 / max(e['d'], 1.5), 0.15, 1.6))
            d = float(np.clip(0.34 - 0.002 * e['v'], 0.18, 0.34))
            f_hi = 900 + 30 * e['v']
            x = whoosh(rng, d, f_hi, 260, 1.0)
            sfx.add(x, e['t'] - d * 0.4, g, 0.8 * e['side'])
    for k, e in enumerate(ev['ribs']):
        x = whoosh(rng, 0.20, 2400, 420, 0.9)
        sfx.add(x, e['t'] - 0.08, 0.30, 0.55 if k % 2 else -0.55)
    x = whoosh(rng, 0.9, 1800, 160, 1.3)
    sfx.add(x, ev['arch'] - 0.35, 0.75, 0.0)

    # ---- the lake: the falls roaring past, the ichor's gloops under the flight
    for e in ev['falls']:
        span = 2.2
        x = falls_roar(rng, span)
        env_ = np.exp(-0.5 * ((np.arange(len(x)) / SR - span * 0.55) / (span * 0.2)) ** 2)
        sfx.add(x * env_, e['t'] - span * 0.55, 0.55 * float(np.clip(18.0 / max(e['d'], 6.0), 0.3, 1.2)),
                0.7 * e['side'])
    a, b = ev['lake']
    t = a
    while t < b:
        amb.add(gloop(rng), t, rng.uniform(0.10, 0.22), rng.uniform(-0.8, 0.8))
        t += rng.exponential(0.07)
    for e in ev['blubs']:
        amb.add(squeak(rng), e['t'] - 0.05, 0.25, 0.6 * e['side'])

    # ---- the show over the valley: rockets whistling up, bursting, crackling
    for s in ev['show']:
        rise = s['t'] - s['launch']
        amb.add(whistle(rng, rise), s['launch'], 0.16, rng.uniform(-0.4, 0.4))
        sfx.add(boom(rng, 1.0), s['t'], 0.62, rng.uniform(-0.3, 0.3))
        L, R = crackle(rng, 1.5, 70.0, 1.0)
        amb.add2(L * 0.30, R * 0.30, s['t'] + 0.35)

    # ---- the touchdown
    sfx.add(touchdown(rng), ev['touchdown'] - 0.02, 0.75, 0.0)

    # ---- the music
    mus, pump, dly, kicks = score(rng)
    sc = sidechain(kicks, N, depth=0.55, rel=0.13)
    dL, dR = delay(dly.L[:N], dly.R[:N], BEAT * 0.75, fb=0.32, wet=0.30)
    M_L = mus.L[:N] + pump.L[:N] * sc + dL * (0.6 + 0.4 * sc)
    M_R = mus.R[:N] + pump.R[:N] * sc + dR * (0.6 + 0.4 * sc)
    M_L, M_R = reverb(M_L, M_R, seed=4, decay=1.7, wet=0.18)

    # ---- the buses
    sL, sR = reverb(sfx.L[:N], sfx.R[:N], seed=6, decay=1.2, wet=0.12)
    aL, aR = reverb(amb.L[:N], amb.R[:N], seed=7, decay=2.4, wet=0.35)
    MUSIC = 0.66
    L = sL + aL + MUSIC * M_L
    R = sR + aR + MUSIC * M_R
    if stems:
        _write(out_path[:-4] + '_sfx.wav', 0.25 * _wrap(sL + aL, n), 0.25 * _wrap(sR + aR, n))
        _write(out_path[:-4] + '_music.wav', 0.25 * MUSIC * _wrap(M_L, n), 0.25 * MUSIC * _wrap(M_R, n))

    # ---- master: fold the tail round, a little phone EQ, -14 LUFS, peaks under -1.2 dBFS; processed three loops
    # long and the middle one kept, so the compressor and limiter have no seam either
    L, R = _wrap(L, n), _wrap(R, n)
    L3, R3 = np.tile(L, 3), np.tile(R, 3)
    L3, R3 = hp(L3, 30.0, 2), hp(R3, 30.0, 2)
    L3 = L3 - 0.35 * lp(L3, 70.0, 2) + 0.18 * bp(L3, 1800.0, 4500.0, 1)
    R3 = R3 - 0.35 * lp(R3, 70.0, 2) + 0.18 * bp(R3, 1800.0, 4500.0, 1)
    g = 10 ** ((-18.0 - integrated_lufs(L3[n:2 * n], R3[n:2 * n])) / 20.0)
    L3, R3 = compress(L3 * g, R3 * g, thresh_db=-23.0, ratio=2.5)
    g = 10 ** ((-14.0 - integrated_lufs(L3[n:2 * n], R3[n:2 * n])) / 20.0)
    L3, R3 = limiter(L3 * g, R3 * g, ceiling=10 ** (-1.2 / 20))
    L, R = L3[n:2 * n].copy(), R3[n:2 * n].copy()
    lufs = integrated_lufs(L, R)
    pcm = _write(out_path, L, R)
    print(f'[audio] {dur:.2f}s, {lufs:.1f} LUFS integrated, peak {20 * np.log10(np.abs(pcm).max() / 32768):.1f} '
          f'dBFS', flush=True)
    return out_path


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fh:
        build(json.load(fh), sys.argv[2], stems='--stems' in sys.argv)
