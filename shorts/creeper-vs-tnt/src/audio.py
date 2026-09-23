"""Procedural sound design driven by the simulation cue sheet.

All sounds are synthesised here (no samples): TNT fuse hiss, explosions (crack, thump, roar and crackle) with a
rolling-thunder bed for mass detonations, flying-block whooshes, flesh/gunpowder spray, dirt clods, debris
patter, collapse booms, UI pops/stamps and a light wind bed. Mixed in stereo with
panning from the 3D event positions, then loudness-normalised (BS.1770 style) to -14 LUFS with a
peak limiter.
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


def make_boom(rng, dur=2.2):
    n = int(dur * SR)
    sub = sine_sweep(dur, 92, 44, 0.2) * expenv(n, 0.38, 0.004)
    rumble = lp(rng.standard_normal(n), 200, 4) * expenv(n, 0.42, 0.01)
    crack = bp(rng.standard_normal(n), 900, 6000) * expenv(n, 0.02)
    debris = grains(dur, 260, 400, 3500, 0.35, rng, glen=(0.002, 0.012))
    x = 1.0 * norm(sub) + 0.7 * norm(rumble) + 0.5 * norm(crack) + 0.5 * norm(debris)
    x = np.tanh(x * 1.3) / np.tanh(1.3)
    return norm(x, 0.95)


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


# ---------------------------------------------------------------------------------------------
# mixing
# ---------------------------------------------------------------------------------------------
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


def make_spray(rng, dur=0.35, lo=300, hi=2500):
    """Wet/gritty spray of blasted flesh or gunpowder."""
    x = shaped_noise(dur, lambda tt: 0.5 * (lo + hi) * np.exp(-tt / 0.25), lambda tt: 1.1 + 0 * tt,
                     lambda tt: np.exp(-tt / 0.09) * np.clip(tt / 0.003, 0, 1), rng)
    x += 0.5 * grains(dur, 40, lo, hi, 0.06, rng)
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


# ---------------------------------------------------------------------------------------------
# build from cues
# ---------------------------------------------------------------------------------------------
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


def build(cues, fps, out_path, seed=7):
    rng = np.random.default_rng(seed)
    nfr = len(cues)
    dur = nfr / fps
    mix = Mix(dur + 4.0)
    booms = [make_explosion(rng, 1.0) for _ in range(8)]
    big_booms = [make_explosion(rng, 1.8) for _ in range(4)]
    far_booms = [make_explosion(rng, 1.2, distant=True) for _ in range(6)]
    sprays = [make_spray(rng) for _ in range(8)]
    powders = [make_spray(rng, 0.45, 1500, 7000) for _ in range(6)]
    gravels = [make_crunch(rng, 250, 1400, 0.35) for _ in range(12)]
    ticks = [make_tick(rng) for _ in range(24)]
    collapse = [make_boom(rng) for _ in range(2)]
    pop = make_pop(rng)
    xs = make_x_sound(rng)
    ck = make_check_sound(rng)
    hurt = make_hurt(rng)
    mix.add(make_wind(rng, dur + 2.0), 0.0, 0.11, pan=-0.6)
    mix.add(make_wind(rng, dur + 2.0), 0.0, 0.11, pan=0.6)
    hit = make_hit(rng)

    # segments (hook shots / rounds)
    segs = []
    for i, c in enumerate(cues):
        key = (c['seg'], c.get('shot', 0))
        if not segs or segs[-1][0] != key:
            segs.append([key, i, i])
        segs[-1][2] = i

    # fuse hiss while TNT is primed, and the rush of the formation before the first blast
    for (key, a0, b0) in segs:
        seg = key[0]
        c0 = cues[a0]
        primed = np.array([cues[i]['events'].get('primed', [0])[0] for i in range(a0, b0 + 1)], float)
        if primed.max() > 0:
            L = (b0 - a0 + 1) / fps
            hiss = make_fuse(rng, L + 0.2)
            env = np.interp(np.arange(len(hiss)) / SR * fps, np.arange(len(primed)), primed)
            env = np.clip(0.35 + 0.65 * np.log1p(env) / np.log1p(10000), 0, 1) * (env > 0)
            env = lp(env, 12, 1)
            mix.add(hiss * env, a0 / fps, 0.42)
        if seg == 'hook':
            heavy = 1.0 if c0['round'] in ('r4', 'r5') else 0.5
            mix.add(hp(make_whoosh(rng, (b0 - a0 + 1) / fps + 0.05, heavy), 160.0, 2), a0 / fps, 0.8)
            mix.add(hit, a0 / fps, 0.55 if a0 == 0 else 0.42)
            continue
        if key[1] not in (0, None):
            continue
        first = next((i for i in range(a0, b0 + 1) if cues[i]['events'].get('explode')), None)
        if first is not None:
            t_launch = a0 / fps + c0['launch']
            heavy = float(np.clip(np.log10(max(1, c0['n_tnt'])) / 4.0, 0, 1))
            w = make_whoosh(rng, max(0.2, first / fps - t_launch + 0.05), heavy)
            mix.add(w, t_launch, 0.35 + 0.3 * heavy)

    hook_end = next((i for i, c in enumerate(cues) if c['seg'] != 'hook'), 0)
    if hook_end > 0:
        mix.add(hp(make_riser(rng, hook_end / fps), 180.0, 2), 0.0, 0.5)
        mix.add(hit, hook_end / fps, 0.5)                     # lands on the cut into round 1

    # rolling thunder under mass detonations
    rate = np.array([c['events'].get('explode', [0])[0] for c in cues], float)
    if rate.max() > 6:
        bed = make_rumble_bed(rng, dur + 1.0)
        env = np.clip(np.log1p(rate) / np.log1p(400), 0, 1) ** 1.3
        env = np.interp(np.arange(len(bed)) / SR * fps, np.arange(len(env)), env)
        env = lp(np.concatenate([env, np.zeros(SR)])[:len(bed)], 3.0, 1)       # slow attack, long tail
        mix.add(bed * env * 1.8, 0.0, 0.22)

    last_hurt = -99
    last_big = -99
    booms_in_seg = {}
    for i, c in enumerate(cues):
        t = i / fps
        ev = c['events']
        cam = c.get('cam')
        tgt = [0.0, 0.0, 14.0]
        if c.get('label'):
            mix.add(pop, t, 0.5)
        if c.get('stamp') == 'x':
            mix.add(xs, t + 0.02, 0.55)
        if c.get('stamp') == 'check':
            mix.add(ck, t + 0.02, 0.6)
        if 'hp' in c and c['hp'] < c['hp_prev'] and (i - last_hurt) > fps * 0.45:
            mix.add(hurt, t, 0.5)
            last_hurt = i

        def put(bank, count, cap, gain, pos, spread=1.0 / fps, pitch=(0.85, 1.2), pan_spread=0.5):
            if count <= 0:
                return
            pan, g = _pan_gain(pos, cam, tgt)
            cc = min(float(count), float(cap))
            k = int(np.floor(cc)) + int(rng.random() < (cc - np.floor(cc)))
            if k <= 0:
                return
            total = gain * g * (0.45 + 0.55 * min(1.0, np.log1p(count) / np.log1p(cap * 4)))
            gg = total / np.sqrt(k)
            for _ in range(k):
                s = bank[rng.integers(len(bank))]
                s = resample(s, rng.uniform(*pitch))
                mix.add(s, t + rng.uniform(0, spread), gg * rng.uniform(0.7, 1.0),
                        pan + rng.uniform(-pan_spread, pan_spread))

        if 'explode' in ev:
            n, pos = ev['explode']
            pan, g = _pan_gain(pos, cam, tgt)
            if n <= 3:
                for _ in range(n):
                    mix.add(resample(booms[rng.integers(len(booms))], rng.uniform(0.9, 1.1)),
                            t + rng.uniform(0, 1.0 / fps), 0.85 * g, pan + rng.uniform(-0.2, 0.2))
            else:
                if i - last_big >= 3:                    # a big full-bodied blast every 0.1 s at most
                    mix.add(resample(big_booms[rng.integers(len(big_booms))], rng.uniform(0.85, 1.05)),
                            t, 0.75 * g, pan + rng.uniform(-0.3, 0.3))
                    last_big = i
                put(far_booms, n / 6.0, 3, 0.45, pos, pitch=(0.8, 1.15), pan_spread=0.7)
        if 'blast' in ev:
            n, pos = ev['blast']
            put(sprays, n / 400.0, 3, 0.35, pos, pan_spread=0.6)
        if 'powder' in ev:
            n, _ = ev['powder']
            put(powders, n / 300.0, 2, 0.22, None, pitch=(0.9, 1.3))
        if 'ground' in ev:
            n, pos = ev['ground']
            put(gravels, n / 10.0, 4, 0.4, pos, pitch=(0.7, 1.0), pan_spread=0.6)
        if 'crumble' in ev:
            n, _ = ev['crumble']
            put(gravels, n / 150.0, 2, 0.2, None, pitch=(0.7, 1.0))
        if 'debris_ground' in ev:
            n, pos = ev['debris_ground']
            put(ticks, n / 6.0, 8, 0.25, pos, pitch=(0.8, 1.4), pan_spread=0.6)
        if 'shatter' in ev or 'detach' in ev:
            big = ev.get('shatter', [0])[0] + ev.get('detach', [0])[0]
            nb = booms_in_seg.get(c['seg'], 0)
            if big > 400 and nb < 1:
                pos = (ev.get('shatter') or ev.get('detach'))[1]
                pan, g = _pan_gain(pos, cam, tgt)
                mix.add(collapse[rng.integers(len(collapse))], t, 0.5 * g, pan)
                booms_in_seg[c['seg']] = nb + 1

    L, R = mix.L[:int(dur * SR)], mix.R[:int(dur * SR)]
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
