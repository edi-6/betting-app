"""The zoom's soundtrack, synthesised here (no samples), and a seamless loop: everything is mixed round a circle of
exactly the video's length (notes and reverb tails that run past the end come back in at the start, and the
compressor and limiter run over three laps so the seam has no jump).

* Music: a soft felt piano and strings in D, the chords changing on the story's beats (Steve, the chunk, the hand-over
  to the map, the far side of the world, the Earth, the pixel) and the arpeggio's notes on a musical clock that's
  stretched a little so each section holds exactly sixteen of them. It builds as the zoom speeds up, holds its breath
  at the world border, lands on the pixel, and comes back to where it began.
* An endless rising (Shepard-Risset) tone under it all, rising three octaves per loop in step with the zoom.
* A chime as each milestone appears, a deep hit and a swell for "1 PIXEL", soft whooshes at the hand-overs.
* Ambience that follows the camera: the campfire's crackle and birds close to Steve (at both ends of the loop),
  high wind over the map.
Loudness -14 LUFS, peaks under -1.2 dBFS.
"""
import numpy as np

import audio as A
import zhud as ZH
import ztimeline as TL

SR = A.SR
# section starts (as view widths): the eye, Steve, the chunk, the hand-over, kilometres, the far side, the Earth, the pixel
SECTIONS_L = [TL.L0, 3.3, 24.0, TL.HANDOVER, 1600.0, 1.0e5, 1.0e7, TL.WORLD]
# chords (MIDI notes): bass, then the voicing
CHORDS = [
    (38, [62, 66, 69, 73, 76]),     # Dmaj9
    (35, [59, 62, 66, 69, 76]),     # Bm11
    (31, [59, 62, 66, 69, 73]),     # Gmaj7#11-ish
    (33, [57, 61, 64, 69, 71]),     # Aadd9
    (42, [57, 62, 66, 69, 74]),     # D/F#
    (43, [59, 62, 66, 71, 74]),     # Gmaj7
    (40, [59, 62, 67, 69, 74]),     # Em9 / A7sus4 colour
    (38, [62, 66, 69, 74, 78]),     # D (the pixel)
]


class LoopMix:
    def __init__(self, T):
        self.n = int(round(T * SR))
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, x, t, gain=1.0, pan=0.0):
        x = np.asarray(x, float) * gain
        pan = float(np.clip(pan, -1, 1))
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        s = int(round(t * SR)) % self.n
        k = 0
        while k < len(x):
            m = min(len(x) - k, self.n - s)
            self.L[s:s + m] += x[k:k + m] * gl
            self.R[s:s + m] += x[k:k + m] * gr
            k += m
            s = 0

    def add_stereo(self, L, R):
        self.L += L
        self.R += R


def _tile3(x):
    return np.concatenate([x, x, x])


def _mid(x, n):
    return x[n:2 * n]


def loop_reverb(L, R, **kw):
    n = len(L)
    a, b = A.reverb(_tile3(L), _tile3(R), **kw)
    return _mid(a, n), _mid(b, n)


def env_over_time(tl, f, n):
    """A control signal f(L, t) sampled per frame and smoothed, as a per-sample envelope."""
    fps = tl.fps
    vals = np.array([f(tl.frame(i)[1], i / fps) for i in range(tl.n)])
    tt = np.arange(n) / SR * fps
    v = np.interp(tt, np.arange(tl.n + 1), np.concatenate([vals, vals[:1]]))
    return A.lp(np.concatenate([v, v, v]), 4.0, 1)[n:2 * n]


def shepard(tl, n, rng, octaves=3, f_lo=55.0, n_part=8):
    """An endless rising tone: octave-spaced sines under a fixed bell-shaped spectrum (zero at its ends, so a partial
    wraps from the top to the bottom unheard), sliding up with the zoom. It runs half a second past the loop's end
    and that overlap is cross-faded into its start (it has the same spectrum there), so the seam is smooth too."""
    m = int(0.5 * SR)
    t = np.arange(n + m) / SR
    u = np.interp(t % tl.T, tl._t, tl._u) + np.where(t >= tl.T, TL.U, 0.0)
    pos = octaves * u / TL.U
    out = np.zeros(n + m)
    for k in range(n_part):
        q = np.mod(k + pos, n_part)
        f = f_lo * 2.0 ** q
        amp = np.sin(np.pi * q / n_part) ** 2
        ph = np.cumsum(2 * np.pi * f / SR)
        out += amp * np.sin(ph + rng.uniform(0, 6.28))
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    head = out[:m] * w + out[n:n + m] * (1 - w)
    out = out[:n].copy()
    out[:m] = head
    return out / n_part


def crackle(rng, n):
    """A campfire: pops and a soft hiss."""
    x = A.lp(rng.standard_normal(n), 2500, 2) * 0.12
    pops = (rng.random(n) < 14.0 / SR).astype(float)
    pops = np.convolve(pops * rng.uniform(0.3, 1.0, n), np.exp(-np.arange(int(0.006 * SR)) / (0.0012 * SR)))[:n]
    x += A.hp(pops * rng.standard_normal(n), 1200, 2) * 2.2
    return x


def birds(rng, T):
    out = []
    t = 0.4
    while t < T:
        out.append(t)
        t += rng.uniform(1.4, 3.4)
    return out


def chirp(rng):
    d = rng.uniform(0.08, 0.16)
    m = int(d * SR)
    tt = np.arange(m) / SR
    f0 = rng.uniform(2600, 4200)
    f = f0 * (1 + 0.35 * np.sin(np.pi * tt / d)) * (1 + 0.2 * np.sin(2 * np.pi * rng.uniform(25, 45) * tt))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * tt / d) ** 2
    reps = int(rng.integers(1, 4))
    gap = int(rng.uniform(0.05, 0.12) * SR)
    y = np.zeros(reps * (m + gap))
    for r in range(reps):
        y[r * (m + gap):r * (m + gap) + m] += x * (0.8 ** r)
    return y


def boom(rng, dur=3.2):
    """The deep hit for the pixel: a sub drop and a soft thud with a long tail."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 30 + 70 * np.exp(-t / 0.12)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 1.1)
    x += 0.35 * A.lp(rng.standard_normal(n), 300, 2) * np.exp(-t / 0.07)
    return A.norm(x, 0.95)


def build(tl, out_path, seed=7):
    rng = np.random.default_rng(seed)
    T = tl.T
    mus, sfx, amb = LoopMix(T), LoopMix(T), LoopMix(T)
    n = mus.n
    secs = [tl.t_at_L(L) for L in SECTIONS_L] + [T]

    def m2t(m):
        """Musical time (sixteenths; 16 a section) to seconds."""
        k = int(np.clip(m // 16, 0, 7))
        a, b = secs[k], secs[k + 1]
        return a + (m - 16 * k) / 16.0 * (b - a)

    # energy of the music through the loop (0..1): quiet on the eye and Steve, building over the map, full at the
    # pixel, settling back down for the loop
    def energy(t):
        return float(np.interp(t, [0.0, secs[1], secs[3], secs[4], secs[5], secs[6], secs[7] - 0.3, secs[7] + 1.2,
                                   T - 1.2, T],
                               [0.30, 0.38, 0.55, 0.70, 0.82, 0.86, 0.72, 1.0, 0.36, 0.30]))

    # ---- piano: bass on each section, an arpeggio in sixteenths
    pat = [0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 1]
    for k in range(8):
        bass, voic = CHORDS[k]
        t0 = m2t(16 * k)
        e = energy(t0)
        mus.add(A.piano(bass, secs[k + 1] - t0, 0.55 + 0.3 * e, seed=k), t0, 0.42, -0.15)
        mus.add(A.piano(bass + 12, secs[k + 1] - t0, 0.4 + 0.3 * e, seed=k + 3), t0 + 0.01, 0.3, -0.1)
        for j in range(16):
            if k == 7 and j < 2:
                continue                                        # a breath before the pixel's chord rings
            m = 16 * k + j
            t = m2t(m)
            note = voic[pat[j]] + (12 if (k >= 4 and j % 4 == 3) else 0)
            vel = (0.32 + 0.38 * energy(t)) * (1.0 if j % 4 == 0 else 0.78)
            mus.add(A.piano(note, 0.42, vel, seed=m), t, 0.38, 0.25 if j % 2 else -0.05)
    # the pixel's chord: wide, struck together
    t7 = secs[7]
    for v, note in enumerate([38, 50, 57, 62, 66, 69, 74, 78, 81]):
        mus.add(A.piano(note, 3.0, 0.85, seed=40 + v), t7 + 0.004 * v, 0.42, -0.4 + 0.1 * v)

    # ---- strings: a pad on every chord, louder as it builds, swelling into the pixel
    for k in range(8):
        bass, voic = CHORDS[k]
        t0, t1 = secs[k], secs[k + 1]
        e = energy(t0)
        fr = [A.midi_hz(m) for m in [bass + 12] + voic[:4]]
        s = A.strings(fr, t1 - t0 + 0.3, rng, attack=0.8 if k else 0.4, release=1.2, bright=1500 + 1800 * e)
        mus.add(s, t0 - 0.15, 0.10 + 0.55 * e ** 1.6, 0.0)

    # ---- the endless rising tone
    sh = shepard(tl, n, rng)
    lvl = env_over_time(tl, lambda L, t: 0.25 + 0.75 * np.clip(tl.speed_at(t) / 1.4, 0, 1), n)
    sh = A.lp(sh, 2400, 2) * lvl * 0.16
    mus.add_stereo(sh, np.roll(sh, int(0.011 * SR)))

    # ---- milestones: a chime as each one appears
    for (rng_l, lines) in ZH.LABELS:
        t = tl.t_at_L(rng_l[1]) - 0.25
        top = 86 if 'PIXEL' not in ' '.join(lines) else 93
        x = np.zeros(int(2.4 * SR))
        for j, m in enumerate((top - 12, top - 5, top)):
            b = A.bell(m, rng, 2.0)
            s0 = int(j * 0.06 * SR)
            x[s0:s0 + len(b)] += b[:len(x) - s0]
        sfx.add(A.norm(x, 0.7), t, 0.32, 0.15)
    t_pix_label = tl.t_at_L(1.0) - 0.1
    sfx.add(A.bell(93, rng, 1.6), t_pix_label, 0.18, 0.2)

    # ---- hand-overs and the pixel
    sfx.add(A.make_whoosh(rng, 0.9), tl.t_at_L(TL.HANDOVER) - 0.5, 0.22, -0.2)
    sfx.add(A.make_whoosh(rng, 1.1), tl.t_at_L(TL.EYE_L) - 0.6, 0.25, 0.2)
    sw = A.make_swell(rng, 1.6)
    sw[-int(0.012 * SR):] *= np.linspace(1, 0, int(0.012 * SR))
    sfx.add(sw, t7 - 1.6, 0.5, 0.0)
    sfx.add(boom(rng), t7, 0.95, 0.0)
    sfx.add(A.make_levelup(rng), t7 + 0.05, 0.28, 0.0)

    # ---- ambience: close to Steve (either end of the loop) the campfire and birds, over the map high wind
    def L3(L):
        return L / TL.RATIO if L >= TL.EYE_L else L
    near = env_over_time(tl, lambda L, t: float(np.clip(1.0 - np.log(max(L3(L), 0.4) / 0.4) / np.log(60.0), 0, 1))
                         if (L < TL.HANDOVER or L >= TL.EYE_L) else 0.0, n)
    cr = crackle(rng, n) * near ** 1.5
    amb.add_stereo(cr * 0.5, np.roll(cr, 300) * 0.42)
    for tb in birds(rng, T):
        g = float(np.interp(tb * SR, np.arange(n), near))
        if g > 0.05:
            amb.add(chirp(rng), tb, 0.05 * g, rng.uniform(-0.8, 0.8))
    high = env_over_time(tl, lambda L, t: TL.ramp(L, 60.0, 500.0, 2.0e7, 4.0e7), n)
    w = A.bp(rng.standard_normal(n), 200, 1600, 2)
    w = w / (np.abs(w).max() + 1e-9) * (0.55 + 0.45 * np.sin(2 * np.pi * np.arange(n) / n * 3))
    amb.add_stereo(w * high * 0.07, np.roll(w, 4000) * high * 0.07)

    # ---- mix, reverb and master, all round the loop
    mL, mR = loop_reverb(mus.L, mus.R, decay=2.6, wet=0.34)
    sL, sR = loop_reverb(sfx.L, sfx.R, seed=5, decay=2.0, wet=0.25)
    L = mL * 0.9 + sL + amb.L
    R = mR * 0.9 + sR + amb.R
    L3_, R3_ = _tile3(L), _tile3(R)
    L3_, R3_ = A.hp(L3_, 30.0, 2), A.hp(R3_, 30.0, 2)
    # for phone speakers: less mud, more presence
    L3_ = L3_ - 0.35 * A.lp(L3_, 90.0, 2) + 0.28 * A.bp(L3_, 1800.0, 5000.0, 1)
    R3_ = R3_ - 0.35 * A.lp(R3_, 90.0, 2) + 0.28 * A.bp(R3_, 1800.0, 5000.0, 1)
    g = 10 ** ((-18.0 - A.integrated_lufs(L3_, R3_)) / 20.0)
    L3_, R3_ = A.compress(L3_ * g, R3_ * g, thresh_db=-24.0, ratio=2.2)
    g = 10 ** ((-14.0 - A.integrated_lufs(L3_, R3_)) / 20.0)
    L3_, R3_ = A.limiter(L3_ * g, R3_ * g, ceiling=10 ** (-1.2 / 20))
    L, R = _mid(L3_, n), _mid(R3_, n)
    lufs = A.integrated_lufs(L, R)
    A._write(out_path, L, R)
    peak = 20 * np.log10(max(np.abs(L).max(), np.abs(R).max()) + 1e-12)
    print(f'[audio] {T:.2f}s loop, {lufs:.1f} LUFS, peak {peak:.1f} dBFS, seam jump '
          f'{abs(L[0] - L[-1]):.4f}', flush=True)
    return L, R


if __name__ == '__main__':
    import sys
    build(TL.Timeline(), sys.argv[1] if len(sys.argv) > 1 else 'zoom_audio.wav')
