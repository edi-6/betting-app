"""The soundtrack of the ride, all synthesised here from the cue sheet (cues.py) - no samples.

THE MUSIC is one piece in D that changes with the dimension, each section fitted bar for bar to the ride:
  OVERWORLD  (the drop to the portal, 6 bars at 135 bpm) a bright, heroic theme in D major: four-on-the-floor,
             supersaw chords pumping under the kick, an octave bass, a plucked lead with the hook. It opens on an
             impact the instant the cart tips over the crest; the water of the fall muffles it for a moment; two bars
             of build (Em, A) lead into the portal.
  NETHER     (158 bpm) the same key turned dark (D Phrygian): half-time drum and bass, a distorted reese bass, a
             menacing plucked ostinato, taiko. The ghast's shot stops the drums dead; a riser pulls into the blast.
  THE JUMP   (slow motion) no beat at all: the blast rings in the ears, a heartbeat, a low drone and a choir swelling
             under the fireball that comes past - and the whole mix sucked into a reversed crash that lands with the
             cart. The groove slams back in on the landing, full drum and bass, until the End portal.
  THE END    (5 bars at 133 bpm) epic D minor: choir, bell arpeggios, taiko and toms, a brass hit as the dragon swoops,
             the lead climbing into the exit portal; D major in the white flash.
  THE LIFT   (2 bars) a held dominant, a snare roll and a riser over the clicking chain: the build that the loop
             resolves - the last frame's riser lands on the first frame's impact.

THE EFFECTS follow the physics frame by frame: the wheels' roar and the clack of the rails with the speed (silent
in the air), the wind with its square, the lift chain's ratchet, the water of the fall, lava bubbling and the roar of
the lava falls by distance, the fire on the bridge; the portals (the hum as the cart nears one, the travel whoosh, the
End portal's plunge), the ghasts' moan and shriek, the fireballs (moving sources, panned and Doppler-shifted), the
blast, the span's pieces hissing into the lava, the landing's clang and screech, the dragon's wings and roars, the
crystals' hum, the chirp of the toasts. Loudness normalised to -14 LUFS with a peak limiter.
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
def make_explosion(rng, size=1.0):
    """The game's blast, built to read on phone speakers: crack, punchy body, a roaring burst sweeping down, a
    crackling tail, a short sub thump."""
    dur = 2.0 + 0.6 * size
    n = int(dur * SR)
    t = np.arange(n) / SR
    crk = bp(rng.standard_normal(n), 1500, 12000) * expenv(n, 0.005)
    thump = sine_sweep(dur, rng.uniform(110, 135), rng.uniform(44, 52), 0.05) * expenv(n, 0.16 + 0.04 * size, 0.002)
    body = bp(rng.standard_normal(n), 160, 900, 2) * expenv(n, 0.18 + 0.08 * size, 0.003)
    roar = shaped_noise(dur, lambda tt: 450 + 3800 * np.exp(-tt / (0.09 + 0.05 * size)), lambda tt: 1.2 + 0 * tt,
                        lambda tt: np.exp(-tt / (0.30 + 0.16 * size)) * np.clip(tt / 0.004, 0, 1), rng)
    crackle = np.zeros(n)
    for _ in range(int(170 * size)):
        p = int(min(n - 400, rng.exponential(0.3 + 0.18 * size) * SR))
        m = int(rng.integers(40, 260))
        crackle[p:p + m] += rng.standard_normal(m) * np.hanning(m) * rng.uniform(0.3, 1.0) * np.exp(-p / SR / 0.7)
    crackle = bp(crackle, 900, 7000)
    rumble = bp(rng.standard_normal(n), 45, 220, 2) * expenv(n, 0.36 + 0.12 * size, 0.02)
    x = 0.55 * norm(crk) + 0.7 * norm(thump) + 0.75 * norm(body) + 1.0 * norm(roar) + 0.4 * norm(crackle) \
        + 0.35 * norm(rumble)
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    x = hp(x, 36.0, 2)
    return norm(x * np.clip((dur - t) / 0.3, 0, 1), 0.95)


def make_clack(rng):
    """A wheel over a rail joint: a hard metallic tick and a low knock."""
    dur = 0.08
    n = int(dur * SR)
    x = modal(dur, rng.uniform(1100, 1500), (1.0, 2.31, 3.9), (1.0, 0.5, 0.25), (0.012, 0.008, 0.005), rng)
    x += 0.8 * np.sin(2 * np.pi * rng.uniform(85, 110) * np.arange(n) / SR) * expenv(n, 0.018)
    x += 0.3 * hp(rng.standard_normal(n), 2000, 2) * expenv(n, 0.002)
    return norm(x * np.clip(np.arange(n) / SR / 0.0005, 0, 1), 0.8)


def make_chain_click(rng):
    """The lift chain's anti-rollback ratchet."""
    dur = 0.07
    n = int(dur * SR)
    x = modal(dur, rng.uniform(1900, 2300), (1.0, 1.53, 2.7), (1.0, 0.6, 0.3), (0.01, 0.008, 0.005), rng)
    x += 0.6 * modal(dur, rng.uniform(520, 640), (1.0, 2.2), (1.0, 0.4), (0.02, 0.012), rng)
    x += 0.4 * hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.0015)
    return norm(x, 0.8)


def make_portal_whoosh(rng, dur=2.2):
    """Travelling through a nether portal: a long whoosh that rises and wobbles."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = lambda tt: np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 1.5
    x = shaped_noise(dur, lambda tt: 350 * (1 + 5.0 * (tt / dur)) * (1 + 0.25 * np.sin(2 * np.pi * 7.0 * tt)),
                     lambda tt: 0.8 + 0 * tt, env, rng)
    f = 150 * 2 ** (1.6 * t / dur) * (1 + 0.03 * np.sin(2 * np.pi * 6.5 * t))
    tone = lp(saw(f, rng) + saw(f * 1.5, rng) * 0.5, 1800, 2) * env(t)
    return norm(norm(x) + 0.35 * norm(tone), 0.85)


def make_portal_hum(rng, dur):
    """A nether portal's warbling hum (it swells as the cart nears it)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, a in ((92.0, 1.0), (138.5, 0.7), (184.0, 0.5), (277.0, 0.25)):
        x += a * np.sin(2 * np.pi * f * t + 3.0 * np.sin(2 * np.pi * 5.3 * t + f))
    x += 0.4 * bp(rng.standard_normal(n), 400, 1600, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t))
    return norm(x * (t / dur) ** 2, 0.8)


def make_end_plunge(rng):
    """Diving into the End portal: a deep 'vwoom' falling away, air rushing, a shimmer."""
    dur = 2.4
    n = int(dur * SR)
    low = sine_sweep(dur, 110.0, 32.0, 0.5) * expenv(n, 0.9, 0.01)
    air = shaped_noise(dur, lambda tt: 2400 * np.exp(-tt / 0.5) + 200, lambda tt: 1.0 + 0 * tt,
                       lambda tt: np.exp(-tt / 0.6) * np.clip(tt / 0.03, 0, 1), rng)
    shim = modal(dur, 1318.5, (1.0, 1.5, 2.0, 3.0), (1, 0.6, 0.4, 0.2), (0.9, 0.7, 0.5, 0.3), rng, 0.003)
    return norm(norm(low) + 0.7 * norm(air) + 0.18 * norm(shim), 0.9)


def make_flash(rng):
    """The exit portal's white flash: a bright cluster of bells over a swelling hiss and a soft boom."""
    dur = 2.4
    n = int(dur * SR)
    x = np.zeros(n)
    for m in (86, 90, 93, 98, 102):
        b = bell(m, rng, dur)
        x[:len(b)] += b[:n] * 0.5
    x += 0.5 * hp(rng.standard_normal(n), 5000, 2) * expenv(n, 0.25, 0.01)
    x += 0.8 * sine_sweep(dur, 80.0, 40.0, 0.3) * expenv(n, 0.5, 0.005)
    return norm(x, 0.85)


def make_water_rush(rng, dur):
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 250, 5000, 2) * (0.75 + 0.25 * np.abs(lp(rng.standard_normal(n), 12, 1)) * 10)
    x += 0.6 * lp(rng.standard_normal(n), 300, 2)
    return norm(x, 0.8)


def make_splash(rng):
    dur = 0.9
    n = int(dur * SR)
    x = shaped_noise(dur, lambda tt: 3000 * np.exp(-tt / 0.2) + 500, lambda tt: 1.1 + 0 * tt,
                     lambda tt: np.exp(-tt / 0.22) * np.clip(tt / 0.005, 0, 1), rng)
    x += 0.5 * lp(rng.standard_normal(n), 250, 2) * expenv(n, 0.12)
    return norm(x, 0.8)


def make_lava_pop(rng):
    """A bubble of lava bursting: a low blip and a spit."""
    dur = 0.18
    n = int(dur * SR)
    f = rng.uniform(70, 160)
    x = sine_sweep(dur, f * 1.8, f, 0.02) * expenv(n, 0.05, 0.002)
    x += 0.3 * bp(rng.standard_normal(n), 800, 4000, 2) * expenv(n, 0.02)
    return norm(x, 0.7)


def make_sizzle(rng, dur=0.5):
    """A piece of the span going into the lava: a plop and a hiss."""
    n = int(dur * SR)
    x = 0.8 * sine_sweep(dur, 220, 90, 0.03) * expenv(n, 0.05, 0.001)
    x += bp(rng.standard_normal(n), 2500, 11000, 2) * expenv(n, dur * 0.35, 0.01)
    return norm(x, 0.7)


def make_ghast_moan(rng, dur=1.9):
    """The ghast's cry: a high, mournful wail, gliding, with a breathy edge."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    f = 620 + 380 * np.sin(np.pi * np.clip(u * 1.3, 0, 1)) - 260 * np.clip(u - 0.6, 0, 1)
    f = f * (1 + 0.03 * np.sin(2 * np.pi * 6.2 * t))
    x = saw(f, rng) * 0.6 + np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = bp(x, 450, 3500, 2)
    x += 0.35 * bp(rng.standard_normal(n), 1500, 4500, 2)
    env = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.8
    return norm(x * env, 0.8)


def make_ghast_shriek(rng, dur=0.55):
    """The shriek as it fires, and the fireball's launch: a sharp rising cry and a 'fwoosh'."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 1050 * (1 + 0.6 * np.clip(t / 0.15, 0, 1)) * (1 + 0.04 * np.sin(2 * np.pi * 11 * t))
    cry = bp(saw(f, rng) + 0.5 * saw(f * 1.5, rng), 700, 6000, 2) * np.sin(np.pi * np.clip(t / (dur * 0.7), 0, 1))
    whoosh = shaped_noise(dur, lambda tt: 600 + 2500 * np.clip(tt / 0.2, 0, 1), lambda tt: 1.0 + 0 * tt,
                          lambda tt: np.clip((tt - 0.12) / 0.05, 0, 1) * np.exp(-np.clip(tt - 0.17, 0, None) / 0.18),
                          rng)
    thump = np.zeros(n)
    s = int(0.14 * SR)
    thump[s:] = sine_sweep(dur - 0.14, 140, 60, 0.03)[:n - s] * expenv(n - s, 0.08)
    return norm(0.8 * norm(cry) + 0.9 * norm(whoosh) + 0.6 * thump, 0.85)


def make_fire_loop(rng, dur):
    """A fireball / a fire: a roaring flutter with crackles."""
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 150, 2500, 2) * (0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 18, 1)) * 12)
    cr = np.zeros(n)
    for _ in range(int(dur * 38)):
        p = int(rng.uniform(0, n - 300))
        m = int(rng.integers(30, 200))
        cr[p:p + m] += rng.standard_normal(m) * np.hanning(m) * rng.uniform(0.2, 1.0)
    return norm(norm(x) + 0.5 * norm(hp(cr, 1500, 2)), 0.8)


def make_landing(rng):
    """The cart slamming onto the rails: an iron clang, a thump, and the wheels screeching."""
    dur = 1.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    clang = modal(dur, 205.0, (1.0, 2.43, 3.87, 5.31, 7.1), (1.0, 0.7, 0.5, 0.35, 0.2), (0.9, 0.6, 0.4, 0.3, 0.2),
                  rng, 0.01)
    thump = sine_sweep(dur, 120, 45, 0.04) * expenv(n, 0.14, 0.001)
    f = 2700 * np.exp(-t / 0.9) + 1700
    screech = bp(saw(f * (1 + 0.01 * np.sin(2 * np.pi * 37 * t)), rng), 1200, 7000, 2) * \
        np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.35)
    grit = hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.25, 0.002)
    return norm(0.8 * norm(clang) + 1.0 * norm(thump) + 0.35 * norm(screech) + 0.35 * norm(grit), 0.95)


def make_dragon_roar(rng, dur=2.3):
    """The ender dragon's growl: a low rattling roar, rising and falling, with breath."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    f = 72 + 55 * np.sin(np.pi * np.clip(u * 1.2, 0, 1)) ** 0.7
    rattle = 0.55 + 0.45 * np.sin(2 * np.pi * 27 * t + 2.0 * np.sin(2 * np.pi * 3.1 * t))
    x = (saw(f, rng) + 0.7 * saw(f * 1.01, rng) + 0.4 * saw(f * 2.02, rng)) * rattle
    x = 0.8 * bp(x, 180, 900, 2) + 0.5 * bp(x, 1000, 2600, 2) + 0.35 * lp(x, 180, 2)
    breath = bp(rng.standard_normal(n), 400, 3500, 2) * rattle
    env = np.clip(u / 0.08, 0, 1) * np.clip((1 - u) / 0.35, 0, 1)
    return norm(np.tanh(2.0 * norm(x + 0.4 * norm(breath))) * env, 0.95)


def make_wing_flap(rng):
    dur = 0.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = lp(rng.standard_normal(n), 520, 2) + 0.3 * bp(rng.standard_normal(n), 500, 1500, 2)
    env = np.sin(np.pi * np.clip(t / 0.45, 0, 1)) ** 2
    return norm(x * env, 0.8)


def make_toast(rng):
    """The advancement toast sliding in: a soft whoosh and a two-note chime."""
    dur = 1.4
    n = int(dur * SR)
    x = np.zeros(n)
    sw = shaped_noise(0.3, lambda tt: 800 + 3000 * tt / 0.3, lambda tt: 0.8 + 0 * tt,
                      lambda tt: np.sin(np.pi * np.clip(tt / 0.3, 0, 1)) ** 2, rng)
    x[:len(sw)] += 0.4 * norm(sw)
    for k, m in enumerate((88, 95)):
        b = bell(m, rng, 1.0, 0.7)
        s = int((0.16 + 0.09 * k) * SR)
        x[s:s + len(b)] += 0.5 * b[:n - s]
    return norm(x, 0.7)


def make_heartbeat(rng):
    dur = 0.5
    n = int(dur * SR)
    x = np.zeros(n)
    for dt, a in ((0.0, 1.0), (0.16, 0.7)):
        s = int(dt * SR)
        x[s:] += a * sine_sweep((n - s) / SR, 70, 42, 0.03)[:n - s] * expenv(n - s, 0.06, 0.004)
    return norm(lp(x, 200, 2), 0.9)


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


# ---------------------------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------------------------
class Grid:
    """Bars of 4 beats between t0 and t1."""

    def __init__(self, t0, t1, bars):
        self.t0, self.bars = t0, bars
        self.bar = (t1 - t0) / bars
        self.beat = self.bar / 4.0
        self.bpm = 60.0 / self.beat

    def t(self, bar, beat=0.0):
        return self.t0 + bar * self.bar + beat * self.beat


# the Overworld theme: D A Bm G | Em A (the build)
A_CHORDS = [(62, 66, 69), (61, 64, 69), (62, 66, 71), (62, 67, 71), (64, 67, 71), (61, 64, 69)]
A_BASS = [38, 33, 35, 31, 28, 33]
A_HOOK = [
    [(0, 74, 0.5), (0.5, 78, 0.5), (1, 81, 1.0), (2, 78, 0.5), (2.5, 81, 0.5), (3, 86, 1.0)],
    [(0, 85, 1.5), (1.5, 81, 0.5), (2, 76, 1.0), (3, 81, 1.0)],
    [(0, 83, 0.5), (0.5, 81, 0.5), (1, 78, 1.0), (2, 74, 0.5), (2.5, 78, 0.5), (3, 83, 1.0)],
    [(0, 81, 1.5), (1.5, 79, 0.5), (2, 78, 1.0), (3, 76, 1.0)],
    [(0, 79, 1.0), (1, 83, 1.0), (2, 88, 2.0)],
    [(0, 88, 0.5), (0.5, 85, 0.5), (1, 81, 0.5), (1.5, 76, 0.5)],
]
# the Nether: Dm Eb Dm C | (after the landing) Dm Bb
N_ROOTS = [38, 39, 38, 36]
N_ARP = [[62, 69, 74, 75, 74, 69, 65, 69], [63, 70, 75, 74, 75, 70, 67, 70], [62, 69, 74, 77, 74, 69, 65, 69],
         [60, 67, 72, 74, 72, 67, 64, 67]]
N2_ROOTS = [38, 34]
N2_ARP = [[62, 69, 74, 77, 74, 69, 65, 69], [58, 65, 70, 74, 70, 65, 62, 65]]
N2_LEAD = [[(0, 86, 1.0), (1, 84, 0.5), (1.5, 81, 0.5), (2, 82, 1.0), (3, 81, 1.0)],
           [(0, 77, 1.0), (1, 81, 1.0), (2, 82, 1.5)]]
# the End: Dm Bb Gm A Bb-A
E_CHORDS = [(62, 65, 69), (58, 62, 65), (55, 58, 62), (57, 61, 64), (58, 62, 65)]
E_BASS = [38, 34, 31, 33, 34]
E_ARP = [[62, 65, 69, 74], [58, 62, 65, 70], [55, 58, 62, 67], [57, 61, 64, 69], [58, 62, 65, 70]]
E_LEAD = [[], [], [(0, 79, 1.5), (1.5, 77, 0.5), (2, 74, 2.0)], [(0, 76, 1.0), (1, 79, 1.0), (2, 81, 2.0)],
          [(0, 82, 2.0), (2, 81, 2.0)]]


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


def score(ev, dur, rng):
    """The music (stereo) as three mixes: the pumping bus (pads, bass: ducked by the kick), the echo bus (the
    hook's plucks, the End's bells: a dotted-eighth ping-pong) and the rest."""
    mus = Mix(dur)
    pump = Mix(dur)
    dly = Mix(dur)
    kicks = []

    # ---- OVERWORLD: 6 bars from the crest to the portal
    g = Grid(0.0, ev['portal_a'], 6)
    mus.add(impact(rng, 1.0), 0.0, 0.9, 0.0)
    for b in range(6):
        build = b >= 4
        # drums
        for q in range(4):
            if b == 5 and q >= 2:
                break
            kicks.append(g.t(b, q))
            mus.add(kick(q % 2), g.t(b, q), 0.95)
            if q in (1, 3) and not (b == 5):
                mus.add(clap(q), g.t(b, q), 0.55, 0.05)
                mus.add(snare(q, 0.8), g.t(b, q), 0.35, -0.05)
        for s16 in range(16):
            if b == 5 and s16 >= 8:
                break
            op = s16 % 4 == 2
            mus.add(hat(s16 % 3, op), g.t(b, s16 / 4), (0.20 if op else 0.13) * (0.75 + 0.25 * (s16 % 2 == 0)),
                    0.35 * np.sin(s16 * 1.3))
        if b in (0, 4):
            mus.add(crash(b), g.t(b), 0.5, 0.25)
        # chords (pumping), the octave bass on the off-beats
        L, R = supersaw([midi_hz(m) for m in A_CHORDS[b]] + [midi_hz(A_CHORDS[b][0] - 12)], g.bar, rng,
                        attack=0.01, release=0.3, cutoff=2600 + 700 * b if not build else 4200)
        pump.add2(L, R, g.t(b), 0.55)
        for e8 in range(8):
            m = A_BASS[b] + (12 if e8 % 2 else 0)
            x = reese(m, g.beat * 0.45, rng, cutoff=900, drive=1.4)
            pump.add(x, g.t(b, e8 / 2 + (0.0 if e8 % 2 else 0.0)), 0.5 if e8 % 2 else 0.62, 0.0)
        # the hook
        for (bt, m, ln) in A_HOOK[b]:
            p = pluck(m, ln * g.beat * 0.9, rng, bright=1.0, decay=0.4)
            dly.add(p, g.t(b, bt), 0.42, -0.15)
            ld = lead(m, ln * g.beat * 0.92, rng)
            mus.add(ld, g.t(b, bt), 0.20, 0.15)
    # the build into the portal: snare roll and riser over bars 5-6, a reversed crash into the cut
    mus.add(snare_roll(rng, g.bar * 1.5, 4, 32, g.bpm), g.t(4, 2), 0.42, 0.0)
    mus.add(riser(rng, g.bar * 1.5), g.t(4, 2), 0.35, 0.0)
    mus.add(reverse_crash(rng, g.bar * 0.5), g.t(5, 2), 0.45, 0.0)

    # ---- NETHER: 4 bars from the portal to the blast
    g = Grid(ev['portal_a'], ev['explosion'], 4)
    for b in range(4):
        last = b == 3
        mus.add(taiko(b % 2), g.t(b), 0.7, -0.1)
        kicks.append(g.t(b))
        mus.add(kick(0), g.t(b), 1.0)
        if not last:
            kicks.append(g.t(b, 2.5))
            mus.add(kick(1), g.t(b, 2.5), 0.85)
            mus.add(snare(1), g.t(b, 2), 0.75, 0.0)
            mus.add(clap(1), g.t(b, 2), 0.3, 0.0)
            for s16 in range(16):
                mus.add(hat(s16 % 2), g.t(b, s16 / 4), 0.11 * (1.0 if s16 % 2 == 0 else 0.55), 0.3)
            if b == 2:
                # a tom fill into the ghast's bar
                for k, (bt, f0) in enumerate(((3.0, 150), (3.25, 130), (3.5, 110), (3.75, 90))):
                    mus.add(tom(f0, k), g.t(b, bt), 0.55, 0.4 - 0.25 * k)
        else:
            # the ghast fires on beat 2: the drums stop dead; a riser and a roll pull into the blast
            mus.add(snare(2), g.t(b, 1), 0.8, 0.0)
            mus.add(crash(3, 1.2), g.t(b, 1), 0.35, 0.0)
            mus.add(snare_roll(rng, g.beat * 2.0, 8, 32, g.bpm), g.t(b, 2), 0.45, 0.0)
            mus.add(riser(rng, g.beat * 3.0, 200, 8000), g.t(b, 1), 0.45, 0.0)
        # the reese, the ostinato
        ln = g.bar if not last else g.beat * 1.0
        pump.add(reese(N_ROOTS[b], ln * 0.98, rng, cutoff=560 + 90 * b, drive=2.6), g.t(b), 0.75, 0.0)
        for s16 in range(16 if not last else 4):
            m = N_ARP[b][s16 % 8]
            mus.add(pluck(m, g.beat * 0.22, rng, bright=0.7, decay=0.12), g.t(b, s16 / 4), 0.30,
                    0.45 * np.sin(s16 * 0.9))
        L, R = supersaw([midi_hz(N_ROOTS[b] + 24), midi_hz(N_ROOTS[b] + 31)], ln, rng, attack=0.05, release=0.3,
                        cutoff=1400)
        pump.add2(L, R, g.t(b), 0.35)

    # ---- THE JUMP (slow motion): no beat; the blast, the ring, a heartbeat, a drone and a choir swelling
    t_ex, t_land = ev['explosion'], ev['land']
    mus.add(impact(rng, 1.4), t_ex, 1.0, 0.0)
    span = t_land - t_ex
    dr = drone((26, 38, 45), span + 0.4, rng, cutoff=500)
    dr *= np.clip(np.arange(len(dr)) / SR / 1.0, 0, 1) * np.clip((span + 0.4 - np.arange(len(dr)) / SR) / 0.3, 0, 1)
    mus.add(dr, t_ex + 0.2, 0.32, 0.0)
    L, R = choir((50, 57, 62, 65, 69), span - 0.9, rng, attack=span * 0.6, release=0.3)
    mus.add2(L, R, t_ex + 0.9, 0.55)
    tb = t_ex + 0.9
    while tb < t_land - 0.5:
        mus.add(make_heartbeat(rng), tb, 0.8, 0.0)
        tb += 0.95
    mus.add(reverse_crash(rng, 1.6), t_land - 1.6, 0.7, 0.0)
    mus.add(riser(rng, 2.2, 150, 9000), t_land - 2.2, 0.35, 0.0)

    # ---- THE LANDING: the groove slams back in, full drum and bass, two bars to the End portal
    g = Grid(t_land, t_land + 2 * 1.5215, 2)
    mus.add(impact(rng, 0.8), t_land, 0.8, 0.0)
    mus.add(crash(4), t_land, 0.55, -0.2)
    for b in range(2):
        for (bt, kind) in ((0, 'k'), (1, 's'), (1.75, 'k'), (2.5, 'k'), (3, 's')):
            tt = g.t(b, bt)
            if tt >= ev['portal_b'] - 0.02:
                continue
            if kind == 'k':
                kicks.append(tt)
                mus.add(kick(0), tt, 1.0)
            else:
                mus.add(snare(0), tt, 0.8, 0.0)
                mus.add(clap(0), tt, 0.3, 0.0)
        for s16 in range(16):
            tt = g.t(b, s16 / 4)
            if tt < ev['portal_b'] - 0.02:
                mus.add(hat(s16 % 3, s16 % 4 == 2), tt, 0.13 if s16 % 2 == 0 else 0.08, -0.3)
        ln = g.bar if b == 0 else ev['portal_b'] - g.t(b)
        pump.add(reese(N2_ROOTS[b], ln * 0.99, rng, cutoff=700, drive=3.0), g.t(b), 0.8, 0.0)
        for s16 in range(16):
            tt = g.t(b, s16 / 4)
            if tt < ev['portal_b'] - 0.05:
                mus.add(pluck(N2_ARP[b][s16 % 8], g.beat * 0.22, rng, bright=0.8, decay=0.12), tt, 0.28,
                        0.45 * np.sin(s16 * 0.9))
        for (bt, m, lnb) in N2_LEAD[b]:
            tt = g.t(b, bt)
            if tt < ev['portal_b'] - 0.1:
                mus.add(lead(m, min(lnb * g.beat, ev['portal_b'] - tt) * 0.95, rng, bright=3800), tt, 0.3, 0.1)
    mus.add(riser(rng, g.bar * 0.9, 300, 9000), ev['portal_b'] - g.bar * 0.9, 0.35, 0.0)

    # ---- THE END: 5 bars from the portal to the exit portal
    g = Grid(ev['portal_b'], ev['portal_c'], 5)
    mus.add(impact(rng, 1.1), ev['portal_b'], 0.8, 0.0)
    for b in range(5):
        last = b == 4
        for q in range(4):
            kicks.append(g.t(b, q))
            mus.add(kick(1), g.t(b, q), 0.8 if q % 2 == 0 else 0.6)
        mus.add(taiko(0), g.t(b, 0), 0.75, -0.2)
        mus.add(taiko(1), g.t(b, 2), 0.6, 0.2)
        if not last:
            mus.add(snare(1), g.t(b, 2), 0.55, 0.0)
        for s8 in range(8):
            mus.add(hat(1), g.t(b, s8 / 2 + 0.5 * (s8 % 2 == 0) * 0), 0.07, 0.25)
        if b in (0, 2):
            mus.add(crash(b), g.t(b), 0.45, -0.2)
        if b == 1:
            for k, (bt, f0) in enumerate(((3.0, 140), (3.25, 120), (3.5, 100), (3.75, 85))):
                mus.add(tom(f0, k), g.t(b, bt), 0.6, 0.3 - 0.2 * k)
        L, R = choir(E_CHORDS[b], g.bar, rng, attack=0.25 if b else 0.5, release=0.5)
        pump.add2(L, R, g.t(b), 0.65)
        pump.add(subbass(E_BASS[b], g.bar * 0.98, 0.02), g.t(b), 0.55, 0.0)
        pump.add(reese(E_BASS[b] + 12, g.bar * 0.98, rng, cutoff=500, drive=1.5), g.t(b), 0.3, 0.0)
        for s16 in range(16):
            m = E_ARP[b][s16 % 4] + (12 if (s16 // 4) % 2 else 0)
            dly.add(bell(m, rng, 0.9, 0.8), g.t(b, s16 / 4), 0.16, 0.5 * np.sin(s16 * 1.1))
        for (bt, m, ln) in E_LEAD[b]:
            mus.add(lead(m, ln * g.beat * 0.95, rng, bright=4200), g.t(b, bt), 0.30, -0.1)
            mus.add(lead(m + 12, ln * g.beat * 0.95, rng, bright=3000), g.t(b, bt), 0.1, 0.2)
    # the brass hit with the dragon's swoop
    mus.add(braam((26, 38, 45, 50), 1.3, rng), ev['swoop'] - 0.05, 0.6, 0.0)
    mus.add(snare_roll(rng, g.bar, 4, 32, g.bpm), g.t(4), 0.4, 0.0)
    mus.add(riser(rng, g.bar, 250, 9000), g.t(4), 0.35, 0.0)

    # ---- THE WHITE FLASH and THE LIFT: D major for an instant, then the dominant held under the chain, building
    t_c, t_e = ev['portal_c'], ev['end']
    for k, m in enumerate((50, 57, 62, 66, 69, 74)):
        mus.add(bell(m + 12, rng, 2.0, 0.6), t_c + 0.01 * k, 0.12, (k - 2.5) * 0.15)
    L, R = choir((50, 57, 62, 66, 69), 0.9, rng, attack=0.02, release=0.9)
    mus.add2(L, R, t_c, 0.45)
    span = t_e - t_c
    g = Grid(t_c, t_e, 2)
    L, R = supersaw([midi_hz(m) for m in (57, 62, 64, 69)], span, rng, attack=span * 0.7, release=0.01,
                    cutoff=1800)
    mus.add2(L, R, t_c + 0.4, 0.5)
    mus.add(subbass(33, span - 0.6, 0.4), t_c + 0.6, 0.35, 0.0)
    mus.add(snare_roll(rng, span - 0.7, 4, 32, 140.0), t_c + 0.7, 0.42, 0.0)
    mus.add(riser(rng, span - 0.3, 200, 10000), t_c + 0.3, 0.4, 0.0)
    for q in range(8):
        tt = g.t(q // 4, q % 4)
        if tt > t_c + 0.6:
            mus.add(kick(0), tt, 0.35 + 0.06 * q)
    # a reversed crash lands on the loop point (the first frame's impact)
    mus.add(reverse_crash(rng, 1.0), t_e - 1.0, 0.5, 0.0)

    # the pumping: pads and bass duck under every kick
    sc = sidechain(kicks, pump.n, depth=0.55, rel=0.12)
    pump.L *= sc
    pump.R *= sc
    return mus, pump, dly


# ---------------------------------------------------------------------------------------------
# the mix
# ---------------------------------------------------------------------------------------------
def _per_sample(values, fps, n):
    return np.interp(np.arange(n) / SR * fps, np.arange(len(values)), values)


def _source_track(frames, key, fps, n, ref=6.0, maxg=1.4):
    """Per-sample gain (distance attenuation), pan (left/right in the rider's view) and distance for a named
    sound source; gain 0 where the source doesn't exist."""
    gs, ps, ds = [], [], []
    for c in frames:
        p = c['src'].get(key)
        if p is None:
            gs.append(0.0)
            ps.append(0.0)
            ds.append(1e3)
            continue
        d = np.asarray(p) - np.asarray(c['eye'])
        dist = float(np.linalg.norm(d))
        u = d / max(dist, 1e-3)
        gs.append(min(maxg, ref / max(dist, ref * 0.5)))
        ps.append(float(np.clip(np.dot(u, c['R']) * 1.2, -1, 1)))
        ds.append(dist)
    g = np.clip(lp(np.concatenate([_per_sample(gs, fps, n), np.zeros(SR)])[:n], 12.0, 1), 0, None)
    p = _per_sample(ps, fps, n)
    return g, p, _per_sample(ds, fps, n)


def build(cues, out_path, seed=5, stems=False):
    """Write the soundtrack for the cue sheet to out_path (48 kHz 16-bit stereo WAV)."""
    rng = np.random.default_rng(seed)
    frames, fps = cues['frames'], cues['fps']
    ev = cues['events']
    dur = cues['nframes'] / fps
    n = int(dur * SR) + SR
    tt = np.arange(n) / SR
    sfx = Mix(dur)
    slo = Mix(dur)                 # the slow-motion bus: low-passed and drowned in reverb
    amb = Mix(dur)
    t_ex, t_land = ev['explosion'], ev['land']

    world = [c['w'] for c in frames]
    v = _per_sample([c['v'] for c in frames], fps, n)
    k = _per_sample([c['k'] for c in frames], fps, n)
    air = _per_sample([1.0 if c['air'] else 0.0 for c in frames], fps, n)
    lift = _per_sample([1.0 if c['lift'] else 0.0 for c in frames], fps, n)
    gf = _per_sample([c['g'] for c in frames], fps, n)
    in_w = {w: _per_sample([1.0 if x == w else 0.0 for x in world], fps, n) for w in ('over', 'nether', 'end')}
    seg_d = _per_sample([1.0 if c['seg'] == 'D' else 0.0 for c in frames], fps, n)
    veff = v * k                                  # speed as heard (the slow motion slows everything)

    # ---- the wheels: a roar with the speed, and the clack of the joints (every 2 blocks), none in the air
    roll_env = np.clip(veff / 45.0, 0, 1.3) ** 1.1 * (1 - air) * (1 - 0.7 * lift)
    base = shaped_noise(dur + 1.0, lambda t: 170 + 14 * np.interp(t, tt, veff), lambda t: 1.2 + 0 * t,
                        lambda t: np.interp(t, tt, roll_env), rng)[:n]
    rumble = lp(rng.standard_normal(n), 120, 2) * roll_env
    sfx.add(0.55 * norm(base) + 0.35 * norm(rumble), 0.0, 0.55, 0.0)
    clacks = [make_clack(rng) for _ in range(12)]
    dist = np.cumsum(veff / SR)                   # blocks travelled, as heard
    joint = 2.0
    marks = np.floor(dist / joint)
    idx = np.nonzero(np.diff(marks) > 0)[0]
    for i in idx:
        t = i / SR
        if air[i] > 0.5 or lift[i] > 0.5:
            continue
        gain = 0.30 * min(1.0, (veff[i] / 30.0)) ** 0.8
        if gain < 0.02:
            continue
        x = clacks[int(rng.integers(len(clacks)))]
        if k[i] < 0.95:
            x = resample(x, max(0.45, k[i] ** 0.5))
        sfx.add(x, t, gain, rng.uniform(-0.3, 0.3))
        sfx.add(x, t + 0.8 / max(v[i], 1.0) / max(k[i], 0.2), gain * 0.7, rng.uniform(-0.3, 0.3))

    # ---- wind: with the square of the speed, gusting; a rush in the air
    wind_env = np.clip(veff / 45.0, 0, 1.5) ** 2 * (0.75 + 0.25 * np.abs(lp(rng.standard_normal(n), 0.8, 1)) * 30)
    wind_env *= 1 + 0.6 * air
    w1 = shaped_noise(dur + 1.0, lambda t: 380 + 22 * np.interp(t, tt, veff), lambda t: 1.5 + 0 * t,
                      lambda t: np.interp(t, tt, np.clip(wind_env, 0, 3)), rng)[:n]
    w2 = shaped_noise(dur + 1.0, lambda t: 1600 + 40 * np.interp(t, tt, veff), lambda t: 0.8 + 0 * t,
                      lambda t: np.interp(t, tt, np.clip(wind_env, 0, 3) * 0.5), rng)[:n]
    wL = norm(w1 + w2)
    wR = norm(np.roll(w1, 900) + np.roll(w2, 1300))
    sfx.add2(0.30 * wL, 0.30 * wR, 0.0)
    # the pull-outs at the bottoms of drops: a low whoosh of g
    genv = np.clip(gf - 1.6, 0, 2.5) / 2.5
    gw = lp(rng.standard_normal(n), 260, 2) * lp(np.concatenate([genv, np.zeros(SR)])[:n], 4, 1)
    sfx.add(norm(gw), 0.0, 0.35, 0.0)

    # ---- the lift chain (the last seconds): the ratchet clicking, birds, a breeze
    cl = [make_chain_click(rng) for _ in range(6)]
    t = ev['portal_c'] + 0.1
    while t < dur - 0.02:
        i = int(t * SR)
        if lift[min(i, n - 1)] > 0.5:
            sfx.add(cl[int(rng.integers(len(cl)))], t, 0.42, rng.uniform(-0.15, 0.15))
        t += 1.0 / 7.5
    tb = ev['portal_c'] + 0.6
    while tb < dur - 0.4:
        amb.add(make_chirp(rng), tb, 0.06 * rng.uniform(0.6, 1.0), rng.uniform(-0.9, 0.9))
        tb += rng.uniform(0.7, 1.4)
    tb = 0.8
    while tb < ev['portal_a'] - 1.0:
        amb.add(make_chirp(rng), tb, 0.04 * rng.uniform(0.5, 1.0), rng.uniform(-0.9, 0.9))
        tb += rng.uniform(1.5, 3.0)

    # ---- the overworld: the river, the waterfall, the portal
    water = make_water_rush(rng, dur + 1.2)[:n]
    fall_g, fall_p, fall_d = _source_track(frames, 'fall', fps, n, ref=10.0, maxg=1.6)
    river = in_w['over'] * (1 - seg_d) * np.clip((ev['portal_a'] - 1.5 - tt) / 1.0, 0, 1) * \
        np.clip((tt - ev['bottom'] + 1.0) / 1.0, 0, 1)
    amb.add(0.35 * water * river, 0.0, 1.0, -0.2)
    sfx.add_dyn(water, 0.0, 0.45 * fall_g * in_w['over'], fall_p)
    if 'waterfall' in ev:
        sfx.add(make_splash(rng), ev['waterfall'] - 0.05, 0.9, 0.0)
        sfx.add(make_splash(rng), ev['waterfall'] + 0.12, 0.6, 0.3)
    sfx.add(make_portal_hum(rng, 1.8), ev['portal_a'] - 1.8, 0.45, 0.0)
    sfx.add(make_portal_whoosh(rng, 2.2), ev['portal_a'] - 0.5, 0.75, 0.0)
    sfx.add(make_toast(rng), ev['toast_b'], 0.35, 0.4)

    # ---- the nether: its hum, lava bubbling, the lava falls, the ghasts, the blast, the jump, the bridge fires
    nz = in_w['nether']
    neth = lp(rng.standard_normal(n), 90, 2) + 0.3 * bp(rng.standard_normal(n), 150, 400, 2)
    amb.add(norm(neth) * nz, 0.0, 0.22, 0.0)
    t = ev['portal_a'] + 0.3
    while t < ev['portal_b']:
        amb.add(make_lava_pop(rng), t, rng.uniform(0.05, 0.14), rng.uniform(-0.8, 0.8))
        t += rng.exponential(0.16)
    lf_g, lf_p, _ = _source_track(frames, 'lavafall', fps, n, ref=14.0)
    roar = bp(rng.standard_normal(n), 120, 2200, 2) * (0.7 + 0.3 * np.abs(lp(rng.standard_normal(n), 6, 1)) * 10)
    sfx.add_dyn(norm(roar), 0.0, 0.40 * lf_g * nz, lf_p)
    # the ghasts
    g1, p1, _ = _source_track(frames, 'ghast1', fps, n, ref=24.0)
    i = int(ev['ghast_moan'] * SR)
    x = make_ghast_moan(rng, 1.9)
    sfx.add_dyn(x, ev['ghast_moan'], 0.75 * g1[i:i + len(x)], p1[i:i + len(x)])
    i = int((ev['ghast_shoot'] - 0.35) * SR)
    x = make_ghast_shriek(rng)
    sfx.add_dyn(x, ev['ghast_shoot'] - 0.35, 0.95 * np.maximum(g1[i:i + len(x)], 0.5), p1[i:i + len(x)])
    g2, p2, _ = _source_track(frames, 'ghast2', fps, n, ref=24.0)
    x = resample(make_ghast_shriek(rng), 0.85)
    i = int((ev['fb2_shoot'] - 0.3) * SR)
    slo.add_dyn(x, ev['fb2_shoot'] - 0.3, 0.6 * np.maximum(g2[i:i + len(x)], 0.25), p2[i:i + len(x)])
    # the fireballs: moving sources, panned, the second Doppler-shifted as it comes past
    for key, gain in (('fb1', 0.8), ('fb2', 1.1)):
        fg, fp, fd = _source_track(frames, key, fps, n, ref=5.0, maxg=2.5)
        on = np.nonzero(fg > 1e-4)[0]
        if not len(on):
            continue
        a, b = on[0], on[-1]
        m = b - a
        loop = make_fire_loop(rng, m / SR * 1.6 + 1.0)
        dd = np.gradient(fd[a:b]) * SR                    # blocks per second, video time
        rate = np.clip(343.0 / (343.0 + 3.0 * dd), 0.5, 1.8)
        if key == 'fb2':
            rate = rate * np.clip(0.55 + 0.45 * k[a:b], 0.5, 1.0)
        x = varispeed(loop, rate)
        m = min(m, len(x))
        (slo if key == 'fb2' else sfx).add_dyn(x[:m], a / SR, gain * fg[a:a + m], fp[a:a + m])
    # the fireball coming past: a deep whoosh with it
    x = shaped_noise(1.6, lambda t: 300 + 1500 * np.exp(-((t - 0.9) / 0.35) ** 2), lambda t: 1.0 + 0 * t,
                     lambda t: np.exp(-((t - 0.9) / 0.4) ** 2), rng)
    slo.add(norm(x), ev['fb2_pass'] - 0.9, 0.9, 0.5)
    # the blast (at full speed: the slow motion comes after it) and the ring in the ears
    sfx.add(make_explosion(rng, 1.8), t_ex, 1.25, 0.0)
    sfx.add(resample(make_explosion(rng, 1.2), 0.6), t_ex + 0.05, 0.6, 0.0)
    ring = np.sin(2 * np.pi * 3950 * np.arange(int(3.0 * SR)) / SR) * expenv(int(3.0 * SR), 1.0, 0.05)
    slo.add(ring, t_ex + 0.1, 0.06, 0.0)
    # the span's pieces going into the lava: plops and hisses, slowed
    for ts in ev['sinks'][::3]:
        if ts < t_land + 0.8:
            kk = float(np.interp(ts, tt, k))
            x = resample(make_sizzle(rng, 0.5), max(0.5, kk ** 0.5))
            slo.add(x, ts, 0.18 * rng.uniform(0.6, 1.0), rng.uniform(-0.5, 0.5))
    # take-off: the wheels leave the rails (a clunk, then only the wind)
    x = resample(make_clack(rng), 0.6)
    slo.add(x, ev['takeoff'], 0.6, 0.0)
    # the landing: at full speed again
    sfx.add(make_landing(rng), t_land, 1.2, 0.0)
    sp = np.zeros(int(0.8 * SR))
    for _ in range(90):
        p = int(rng.uniform(0, 0.5) * SR)
        m = int(rng.integers(30, 150))
        sp[p:p + m] += rng.standard_normal(m) * np.hanning(m) * rng.uniform(0.2, 1.0)
    sfx.add(norm(hp(sp, 3000, 2)) * expenv(len(sp), 0.3), t_land + 0.02, 0.35, 0.0)
    # the bridge's fires
    fg, fpn, _ = _source_track(frames, 'fire', fps, n, ref=5.0)
    fl = make_fire_loop(rng, dur + 1.2)[:n]
    sfx.add_dyn(fl, 0.0, 0.35 * fg * nz * (tt > t_land), fpn)
    # the End portal
    sfx.add(make_end_plunge(rng), ev['portal_b'] - 0.15, 0.95, 0.0)
    sfx.add(make_toast(rng), ev['toast_c'], 0.35, 0.4)

    # ---- the End: the void's air, the crystals, the dragon's wings and roars
    ez = in_w['end']
    void = bp(rng.standard_normal(n), 250, 1100, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.23 * tt))
    amb.add(norm(void) * ez, 0.0, 0.14, 0.0)
    cg, cp, _ = _source_track(frames, 'crystal', fps, n, ref=10.0)
    hum = np.sin(2 * np.pi * 164.8 * tt) + 0.7 * np.sin(2 * np.pi * 247.9 * tt) + 0.4 * np.sin(2 * np.pi * 330.6 * tt)
    hum *= 0.7 + 0.3 * np.sin(2 * np.pi * 1.7 * tt)
    sfx.add_dyn(hum, 0.0, 0.10 * cg * ez, cp)
    dg, dp, dd = _source_track(frames, 'dragon', fps, n, ref=18.0, maxg=1.6)
    flap = make_wing_flap(rng)
    t = ev['portal_b'] + 0.4
    while t < ev['perch']:
        i = int(t * SR)
        sfx.add(flap, t, 0.9 * dg[i], dp[i])
        t += 0.95
    for (te, gain) in ((ev['swoop'] - 0.25, 1.1), (ev['roar'] - 0.5, 1.2)):
        x = make_dragon_roar(rng, 2.3)
        i = int(te * SR)
        sfx.add_dyn(x, te, gain * np.clip(dg[i:i + len(x)] * 1.4, 0.45, 1.3), dp[i:i + len(x)])
    x = shaped_noise(1.4, lambda t: 200 + 900 * np.exp(-((t - 0.6) / 0.25) ** 2), lambda t: 1.1 + 0 * t,
                     lambda t: np.exp(-((t - 0.6) / 0.3) ** 2), rng)
    sfx.add(norm(x), ev['swoop'] - 0.6, 0.8, 0.0)

    # ---- the exit portal and the flash
    sfx.add(make_end_plunge(rng), ev['portal_c'] - 0.2, 0.7, 0.0)
    sfx.add(make_flash(rng), ev['portal_c'], 0.5, 0.0)

    # ---- the music
    mus, pump, dly = score(ev, dur, rng)
    dL, dR = delay(dly.L[:n], dly.R[:n], 0.335, fb=0.35, wet=0.32)
    M_L = mus.L[:n] + pump.L[:n] + dL
    M_R = mus.R[:n] + pump.R[:n] + dR
    # the water muffles it as the cart goes through the fall
    if 'waterfall' in ev:
        wet = np.exp(-0.5 * ((tt - ev['waterfall']) / 0.12) ** 2)
        M_L = M_L * (1 - 0.75 * wet) + lp(M_L, 500, 2) * 0.75 * wet
        M_R = M_R * (1 - 0.75 * wet) + lp(M_R, 500, 2) * 0.75 * wet
    M_L, M_R = reverb(M_L, M_R, seed=4, decay=1.6, wet=0.18)
    # duck the music under the big moments
    duck = np.ones(n)
    for te, depth, ln in ((t_ex, 0.35, 0.8), (t_land, 0.3, 0.5), (ev['swoop'], 0.3, 1.2), (ev['roar'], 0.35, 1.4),
                          (ev['ghast_shoot'] - 0.3, 0.2, 0.6)):
        duck *= 1 - depth * np.exp(-0.5 * ((tt - te - ln * 0.4) / (ln * 0.5)) ** 2)
    M_L, M_R = M_L * duck, M_R * duck

    # ---- the buses
    sL, sR = reverb(sfx.L[:n], sfx.R[:n], seed=6, decay=1.1, wet=0.12)
    oL, oR = lp(slo.L[:n], 3200, 2), lp(slo.R[:n], 3200, 2)
    oL, oR = reverb(oL, oR, seed=8, decay=3.2, wet=0.45, dark=2600.0)
    # in the slow motion the world goes muffled: the SFX bus low-passed while k < 1
    slow = np.clip((1 - k) / 0.7, 0, 1)
    sL = sL * (1 - slow) + lp(sL, 700, 2) * slow
    sR = sR * (1 - slow) + lp(sR, 700, 2) * slow
    MUSIC = 0.62
    L = sL + oL + amb.L[:n] + MUSIC * M_L
    R = sR + oR + amb.R[:n] + MUSIC * M_R
    if stems:
        _write(out_path[:-4] + '_sfx.wav', 0.25 * (sL + oL + amb.L[:n]), 0.25 * (sR + oR + amb.R[:n]))
        _write(out_path[:-4] + '_music.wav', 0.25 * MUSIC * M_L, 0.25 * MUSIC * M_R)

    # ---- master: a little phone EQ, loudness -14 LUFS, peaks under -1.2 dBFS; loop-friendly edges
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    L = L - 0.35 * lp(L, 70.0, 2) + 0.18 * bp(L, 1800.0, 4500.0, 1)
    R = R - 0.35 * lp(R, 70.0, 2) + 0.18 * bp(R, 1800.0, 4500.0, 1)
    L, R = L[:int(dur * SR)], R[:int(dur * SR)]
    g = 10 ** ((-18.0 - integrated_lufs(L, R)) / 20.0)
    L, R = compress(L * g, R * g, thresh_db=-23.0, ratio=2.5)
    g = 10 ** ((-14.0 - integrated_lufs(L, R)) / 20.0)
    L, R = limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    lufs = integrated_lufs(L, R)
    f0, f1 = int(0.002 * SR), int(0.012 * SR)
    for ch in (L, R):
        ch[:f0] *= np.linspace(0, 1, f0)
        ch[-f1:] *= np.linspace(1, 0, f1)
    pcm = _write(out_path, L, R)
    print(f'[audio] {dur:.2f}s, {lufs:.1f} LUFS integrated, peak {20 * np.log10(np.abs(pcm).max() / 32768):.1f} '
          f'dBFS', flush=True)
    return out_path


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fh:
        build(json.load(fh), sys.argv[2], stems='--stems' in sys.argv)
