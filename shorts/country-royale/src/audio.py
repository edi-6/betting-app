"""Music and sound for the battle royale, all synthesised (no samples), driven by the cue sheet of the edit.

The music is an original track at 128 BPM in A minor that follows the match: it starts on the first frame with a hit
as the balls burst out, grooves (four-on-the-floor kick, claps, a pumping bass, a plucked arpeggio and a hook) while
the field thins out, drops into a bigger second half at TOP 10 (wide saw chords, faster hats, the hook an octave up),
strips back to a heartbeat and a ticking clock for the final two, while every bounce off the ring plays the next
note of a climbing motif, and lands on an A major fanfare when the winner is decided.

The effects come from the simulation: every hard bounce off the ring rings a note that depends on where it hits the
ring (low at the bottom, high at the top), balls knock together, each one that goes out whistles down and pops
(a little higher each time), lands on the board with a thunk and a ding, and the milestones get a riser and a hit.
Loudness normalised to -14 LUFS, true peak under -1 dBFS.
"""
import math

import numpy as np

from sound import (SR, Mix, bell, bp, compress, expenv, hp, integrated_lufs, limiter, lp, make_crash, make_heartbeat,
                   make_whoosh, midi_hz, modal, norm, reverb, sine_sweep, _write)

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
S16 = BEAT / 4
# A minor: Am - F - C - G
PROG = [(45, (57, 60, 64)), (41, (57, 60, 65)), (48, (55, 60, 64)), (43, (55, 59, 62))]
# the hook, over two bars: (16th step, midi note, length in 16ths)
HOOK = [(0, 76, 2), (2, 76, 1), (3, 79, 2), (6, 76, 2), (8, 74, 2), (10, 72, 2), (12, 74, 3), (16, 72, 2),
        (18, 72, 1), (19, 76, 2), (22, 72, 2), (24, 71, 2), (26, 69, 4)]
PENTA = [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84, 86, 88]
DUEL = [69, 72, 76, 81, 79, 76, 74, 76, 72, 74, 76, 79, 81, 84, 83, 81, 88, 86, 84, 81, 76, 79, 81, 84, 88]


def _put(x, c, p, g=1.0):
    if p >= len(x) or p + len(c) <= 0:
        return
    if p < 0:
        c, p = c[-p:], 0
    m = min(len(c), len(x) - p)
    x[p:p + m] += c[:m] * g


# ----------------------------------------------------------------------------------------------------------------
# oscillators and instruments

def saw(f, n, ph0=0.0):
    """Band-limited sawtooth (polyBLEP), f constant or per sample."""
    f = np.broadcast_to(np.asarray(f, float), (n,))
    dt = f / SR
    ph = (ph0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    t = ph
    m1 = t < dt
    x = t[m1] / dt[m1]
    y[m1] -= x + x - x * x - 1
    m2 = t > 1 - dt
    x = (t[m2] - 1) / dt[m2]
    y[m2] -= x * x + x + x + 1
    return y


def square(f, n, ph0=0.0, pw=0.5):
    return 0.5 * (saw(f, n, ph0) - saw(f, n, ph0 + pw))


def adsr(n, a, d, s, r, hold):
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    e = np.where(t > hold, e * np.exp(-(t - hold) / max(r, 1e-4)), e)
    return e


def kick(rng):
    dur = 0.45
    n = int(dur * SR)
    x = sine_sweep(dur, 190, 46, 0.03) * expenv(n, 0.2, 0.0005)
    x += 0.35 * hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.003)
    return norm(np.tanh(1.6 * x), 0.95)


def clap(rng):
    dur = 0.3
    n = int(dur * SR)
    x = np.zeros(n)
    for k, off in enumerate((0.0, 0.011, 0.023)):
        p = int(off * SR)
        b = bp(rng.standard_normal(n - p), 900, 2600, 2) * expenv(n - p, 0.012 if k < 2 else 0.11)
        x[p:] += b * (0.7 if k < 2 else 1.0)
    return norm(x, 0.8)


def hat(rng, open_=False):
    dur = 0.35 if open_ else 0.06
    n = int(dur * SR)
    x = hp(rng.standard_normal(n), 7500, 2) * expenv(n, 0.11 if open_ else 0.018)
    return norm(x, 0.5)


def snare(rng):
    dur = 0.3
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 1500, 9000, 2) * expenv(n, 0.08, 0.0008)
    x += 0.7 * sine_sweep(dur, 260, 180, 0.03) * expenv(n, 0.05, 0.0008)
    return norm(x, 0.85)


def bass(m, dur):
    """Sub sine plus a filtered saw so it carries on a phone speaker."""
    n = int((dur + 0.05) * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    env = adsr(n, 0.004, 0.12, 0.75, 0.04, dur)
    sub = np.sin(2 * np.pi * f * t)
    grit = lp(saw(f * 2, n), 900, 2) * 0.35
    return np.tanh(1.3 * (sub + grit)) * env * 0.8


def pluck(m, dur, bright=1.0):
    n = int((dur + 0.25) * SR)
    f = midi_hz(m)
    raw = saw(f, n) * 0.6 + square(f * 1.003, n) * 0.4
    hi = lp(raw, 5200 * bright, 2)
    lo = lp(raw, 900, 2)
    t = np.arange(n) / SR
    fe = np.exp(-t / 0.05)
    x = hi * fe + lo * (1 - fe)
    return x * np.exp(-t / (0.18 + 0.4 * dur)) * np.clip(t / 0.002, 0, 1) * 0.6


def lead(m, dur, rng):
    n = int((dur + 0.18) * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.12) / 0.15, 0, 1)
    x = square(f * vib, n, pw=0.35) * 0.5 + saw(f * vib * 1.004, n) * 0.35
    x = lp(x, 4200, 2)
    return x * adsr(n, 0.006, 0.18, 0.62, 0.09, dur) * 0.55


def supersaw(notes, dur, rng, cutoff=3800, attack=0.02):
    n = int((dur + 0.2) * SR)
    x = np.zeros(n)
    for m in notes:
        f = midi_hz(m)
        for k in range(7):
            det = 1 + (k - 3) * 0.0042
            x += saw(f * det, n, rng.uniform()) * (0.5 if k != 3 else 0.8)
    x = lp(x, cutoff, 2) / (len(notes) * 5)
    return x * adsr(n, attack, 0.3, 0.8, 0.12, dur)


def brass(notes, dur, rng):
    """A brassy swell for the fanfare: detuned saws whose brightness opens with the attack."""
    n = int((dur + 0.3) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        f = midi_hz(m)
        vib = 1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)) * np.clip((t - 0.2) / 0.3, 0, 1)
        for det in (0.997, 1.0, 1.003):
            x += saw(f * det * vib, n, rng.uniform())
    x /= 3 * len(notes)
    dark, bright = lp(x, 900, 2), lp(x, 5000, 2)
    o = np.clip(t / 0.09, 0, 1)
    y = dark * (1 - o) + bright * o
    return y * adsr(n, 0.035, 0.4, 0.85, 0.2, dur) * 0.9


def riser(rng, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    noise = rng.standard_normal(n)
    a = bp(noise, 400, 1200, 2)
    b = bp(noise, 2500, 7000, 2)
    x = a * (1 - u) + b * u
    x += 0.4 * np.sin(2 * np.pi * np.cumsum(200 + 1600 * u ** 2) / SR)
    return x * u ** 2.2 * 0.7


def impact(rng):
    dur = 2.2
    n = int(dur * SR)
    x = sine_sweep(dur, 110, 32, 0.25) * expenv(n, 0.6, 0.0005)
    x += 0.5 * lp(rng.standard_normal(n), 1800, 2) * expenv(n, 0.12, 0.0005)
    return norm(np.tanh(1.4 * x), 0.95)


def tick(rng):
    dur = 0.12
    x = modal(dur, 1850, (1.0, 2.7, 4.1), (1.0, 0.4, 0.2), (0.03, 0.015, 0.01), rng)
    return norm(x, 0.5)


def marimba(m, rng):
    dur = 0.7
    f = midi_hz(m)
    x = modal(dur, f, (1.0, 3.93, 9.2), (1.0, 0.28, 0.08), (0.32, 0.06, 0.02), rng, 0.002)
    n = len(x)
    x += 0.15 * bp(rng.standard_normal(n), f * 2, f * 6, 2) * expenv(n, 0.004)
    return norm(x, 0.7)


def knock(rng):
    dur = 0.09
    x = modal(dur, rng.uniform(700, 1100), (1.0, 2.3, 3.7), (1.0, 0.5, 0.25), (0.018, 0.01, 0.006), rng)
    return norm(x, 0.5)


def whistle_out(rng, f0):
    """The ball going out: a quick downward whistle."""
    dur = 0.32
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1.0 - 0.55 * (t / dur) ** 0.8)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    x *= np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.6 * np.exp(-t / 0.2)
    x += 0.4 * bp(rng.standard_normal(n), 1500, 6000, 2) * expenv(n, 0.01)
    return norm(x, 0.7)


def thunk(rng):
    dur = 0.25
    n = int(dur * SR)
    x = sine_sweep(dur, 220, 90, 0.03) * expenv(n, 0.06, 0.0008)
    x += 0.3 * bp(rng.standard_normal(n), 300, 2500, 2) * expenv(n, 0.008)
    return norm(x, 0.8)


def popper(rng):
    """A party popper: a crack and a fizz of paper."""
    dur = 0.6
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 800, 8000, 2) * expenv(n, 0.006)
    fz = hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.15) * (rng.uniform(size=n) > 0.7)
    return norm(x + 0.5 * fz, 0.8)


def applause(rng, dur):
    """Applause: thousands of single claps at random."""
    n = int(dur * SR)
    x = np.zeros(n)
    cl = [bp(rng.standard_normal(int(0.03 * SR)), 900 + 600 * k, 3500 + 500 * k, 2) *
          expenv(int(0.03 * SR), 0.006) for k in range(6)]
    t = np.arange(n) / SR
    env = np.clip(t / 0.5, 0, 1) * np.clip((dur - t) / 1.2, 0, 1)
    count = int(dur * 900)
    for p in rng.uniform(0, dur - 0.04, count):
        c = cl[rng.integers(0, 6)]
        k = int(p * SR)
        x[k:k + len(c)] += c * rng.uniform(0.2, 1.0) * env[k]
    return norm(x, 0.6)


# ----------------------------------------------------------------------------------------------------------------
# the score

class Score:
    def __init__(self, dur, rng):
        self.mus = Mix(dur)
        self.duck = np.ones(self.mus.n)
        self.rng = rng
        self.K, self.C = kick(rng), clap(rng)
        self.HC = [hat(rng) for _ in range(4)]
        self.HO = hat(rng, True)
        self.SN = snare(rng)

    def _sidechain(self, t):
        a = int(t * SR)
        n = int(0.32 * SR)
        e = 1 - 0.62 * np.exp(-np.arange(n) / SR / 0.075) * np.clip(np.arange(n) / SR / 0.004, 0, 1)
        e = np.minimum(e, 1)
        m = min(n, len(self.duck) - a)
        if m > 0:
            self.duck[a:a + m] = np.minimum(self.duck[a:a + m], e[:m])

    def groove(self, t0, t1, big=False, start_bar=0):
        """The track from t0 to t1. Notes are cut at t1."""
        mus, rng = self.mus, self.rng
        pads = Mix(t1 + 1.0)
        bar = start_bar
        t = t0

        def add(x, at, g, pan=0.0, dst=None):
            if at >= t1:
                return
            x = x[:int((t1 - at + 0.02) * SR)]
            if len(x) > 64:
                x = x.copy()
                k = min(len(x), int(0.02 * SR))
                x[-k:] *= np.linspace(1, 0, k)
            (dst or mus).add(x, at, g, pan)

        while t < t1 - 1e-6:
            root, chord = PROG[bar % 4]
            # drums
            for b in range(4):
                add(self.K, t + b * BEAT, 0.82)
                self._sidechain(t + b * BEAT)
                if b in (1, 3) and (bar > start_bar or big or bar > 0):
                    add(self.C, t + b * BEAT, 0.42, 0.05)
                add(self.HO if big and b % 2 == 1 else self.HC[b % 4], t + b * BEAT + BEAT / 2, 0.16, 0.3)
                if big:
                    for s in (1, 3):
                        add(self.HC[(b + s) % 4], t + b * BEAT + s * S16, 0.08, -0.3)
            # bass in eighths
            for e in range(8):
                m = root - 12 if e % 2 == 0 else root
                add(bass(m, BEAT / 2 * 0.9), t + e * BEAT / 2, 0.42, dst=pads)
            # plucked arpeggio in sixteenths
            arp = [chord[0], chord[1], chord[2], chord[1] + 12, chord[2], chord[1], chord[0] + 12, chord[2]]
            for s in range(16):
                m = arp[s % 8] + 12
                add(pluck(m, S16 * 0.9, 1.0 if big else 0.8), t + s * S16, 0.16 if big else 0.13,
                    -0.35 if s % 2 else 0.35, dst=pads)
            # the hook, every other two bars in the first part, always in the second (an octave up)
            if (bar - start_bar) % 4 in (2, 3) or big:
                half = (bar - start_bar) % 2
                for (step, m, ln) in HOOK:
                    if (step >= 16) == bool(half):
                        st = t + (step - 16 * half) * S16
                        add(lead(m + (12 if big else 0), ln * S16 * 0.95, rng), st, 0.2 if big else 0.17, 0.0)
            if big:
                add(supersaw([root + 12] + list(chord), BAR * 0.98, rng), t, 0.4, 0.0, dst=pads)
            bar += 1
            t += BAR
        n = min(len(pads.L), self.mus.n)
        d = self.duck[:n]
        self.mus.L[:n] += pads.L[:n] * d
        self.mus.R[:n] += pads.R[:n] * d
        return bar


def build(cues, out_path, seed=21):
    rng = np.random.default_rng(seed)
    dur = float(cues['duration'])
    win = float(cues['win'])
    ms = {label: t for t, label in cues['milestones']}
    m10 = ms.get('TOP 10', win * 0.6)
    m2 = ms.get('FINAL 2', win - 6)
    sc = Score(dur + 2, rng)
    mus = sc.mus
    fx = Mix(dur + 2)

    # -- music ---------------------------------------------------------------------------------------------------
    sc.groove(0.0, m10)
    sc.groove(m10, m2, big=True)
    mus.add(impact(rng), 0.0, 0.55)
    mus.add(make_crash(rng, 2.2), 0.0, 0.35, 0.2)
    for t_m, label in cues['milestones']:
        mus.add(riser(rng, 1.4), t_m - 1.4, 0.32)
        mus.add(impact(rng), t_m, 0.5)
        mus.add(make_crash(rng, 2.0), t_m, 0.32, -0.2)
    # the final two: a heartbeat speeding up, a clock, a low drone; every bounce a note of the duel motif
    hb = make_heartbeat(rng)
    t = m2 + 0.2
    while t < win - 0.3:
        u = (t - m2) / max(win - m2, 1e-3)
        mus.add(hb, t, 0.5)
        t += 0.75 - 0.3 * u
    t = m2
    k = 0
    while t < win - 0.2:
        mus.add(tick(rng), t, 0.16 if k % 2 == 0 else 0.1, 0.4 if k % 2 else -0.4)
        t += BEAT
        k += 1
    n_dr = int((win - m2 + 0.5) * SR)
    tt = np.arange(n_dr) / SR
    drone = (np.sin(2 * np.pi * midi_hz(33) * tt) + 0.5 * lp(saw(midi_hz(45), n_dr), 500, 2)) * \
        np.clip(tt / 1.0, 0, 1) * np.clip((win + 0.3 - m2 - tt) / 0.3, 0, 1)
    mus.add(drone * 0.5, m2, 0.32)
    pad = supersaw([57, 60, 64], win - m2 + 0.4, rng, cutoff=1400, attack=0.8)
    mus.add(pad, m2, 0.16)
    # into the last one going out: a long riser and the suck before the hit
    mus.add(riser(rng, 2.6), win - 2.6, 0.38)
    mus.add(make_whoosh(rng, 0.9), win - 0.5, 0.5)
    # the winner: the hit, the fanfare, a happy groove in A major to the end
    mus.add(impact(rng), win, 0.75)
    mus.add(make_crash(rng, 3.0), win, 0.45, 0.25)
    mus.add(make_crash(rng, 3.0), win + 0.02, 0.35, -0.25)
    mus.add(brass([45, 57, 61, 64, 69], 1.6, rng), win + 0.05, 0.55)
    fan = [(0.0, 69, 0.22), (0.25, 73, 0.22), (0.5, 76, 0.22), (0.75, 81, 1.1), (1.9, 78, 0.22), (2.15, 81, 0.22),
           (2.4, 85, 1.4)]
    for (dt, m, ln) in fan:
        mus.add(brass([m, m - 12], ln, rng), win + 0.9 + dt, 0.42, 0.1)
    happy = [(45, (61, 64, 69)), (52, (59, 64, 68)), (54, (61, 66, 69)), (50, (62, 66, 69))]
    t = win + 0.9
    hb_k = kick(rng)
    while t < dur:
        root, chord = happy[int((t - win - 0.9) / BAR) % 4]
        for b in range(4):
            mus.add(hb_k, t + b * BEAT, 0.6)
            if b in (1, 3):
                mus.add(sc.C, t + b * BEAT, 0.35)
            mus.add(sc.HC[b], t + b * BEAT + BEAT / 2, 0.12, 0.3)
        mus.add(supersaw([root + 12] + list(chord), BAR, rng, cutoff=4500), t, 0.22)
        for e in range(8):
            mus.add(bass(root - 12 if e % 2 == 0 else root, BEAT / 2 * 0.9), t + e * BEAT / 2, 0.32)
        t += BAR
    end_chord = brass([45, 57, 64, 69, 73], 1.4, rng)
    mus.add(end_chord, dur - 1.3, 0.35)

    # -- effects -------------------------------------------------------------------------------------------------
    whs = [whistle_out(rng, f) for f in np.geomspace(700, 1500, 12)]
    th = thunk(rng)
    for t, i, place, x in cues['elims']:
        k = int(np.clip((50 - place) / 49 * 11, 0, 11))
        fx.add(whs[k], t, 0.32, float(np.clip(x, -1, 1)) * 0.7)
    for t, place in cues['lands']:
        fx.add(th, t, 0.28, 0.0)
        fx.add(bell(84 + (place % 5) * 2, rng, 0.6, 0.7), t, 0.06, 0.3)
    # bounces: the strongest hit off the ring in each 60 ms rings a note by where it hit; knocks between balls
    imps = sorted(cues['impacts'], key=lambda r: r[0])
    notes = {m: marimba(m, rng) for m in PENTA + DUEL}
    knocks = [knock(rng) for _ in range(6)]
    last_wall, last_knock = -1.0, -1.0
    duel_k = 0
    for (t, a, b, imp, x, y, n_alive) in imps:
        if b < 0:
            if t - last_wall < (0.06 if n_alive > 6 else 0.03) or imp < (0.6 if n_alive > 10 else 0.25):
                continue
            last_wall = t
            if t >= m2 and n_alive <= 2:
                m = DUEL[duel_k % len(DUEL)]
                duel_k += 1
                g = 0.34
            else:
                h = (1 - y) / 2                 # 0 at the bottom of the ring, 1 at the top
                m = PENTA[int(np.clip(h * (len(PENTA) - 1) + 0.5, 0, len(PENTA) - 1))]
                g = (0.07 if n_alive > 20 else 0.11 if n_alive > 6 else 0.18) * min(1.0, imp / 2.5 + 0.3)
            fx.add(notes[m], t, g, float(np.clip(x, -1, 1)) * 0.6)
        else:
            if t - last_knock < 0.045 or imp < 0.4:
                continue
            last_knock = t
            fx.add(knocks[int(rng.integers(0, 6))], t, 0.05 * min(1.0, imp / 2.0 + 0.3),
                   float(np.clip(x, -1, 1)) * 0.6)
    # the celebration
    fx.add(applause(rng, dur - win + 0.2), win + 0.1, 0.22)
    for k in range(4):
        fx.add(popper(rng), win + 0.35 + k * 0.13, 0.3, (-0.7, 0.7, -0.4, 0.4)[k])
    fx.add(bell(93, rng, 1.6, 1.0), win + 1.42, 0.2)               # the crown lands
    fx.add(bell(88, rng, 1.6, 1.0), win + 1.44, 0.14)

    # -- mix -----------------------------------------------------------------------------------------------------
    mL, mR = reverb(mus.L, mus.R, seed=4, decay=1.6, wet=0.16)
    fL, fR = reverb(fx.L, fx.R, seed=5, decay=1.1, wet=0.14)
    L = 0.8 * mL + fL
    R = 0.8 * mR + fR
    n = int(dur * SR)
    L, R = L[:n], R[:n]
    L, R = hp(L, 28, 2), hp(R, 28, 2)
    L, R = compress(L, R, thresh_db=-15.0, ratio=2.0)
    lufs = integrated_lufs(L, R)
    g = 10 ** ((-14.0 - lufs) / 20.0)
    L, R = L * g, R * g
    L, R = limiter(L, R, ceiling=0.84)
    L, R = true_peak(L, R, -1.5)
    fade = np.clip((dur - np.arange(n) / SR) / 0.3, 0, 1)
    L, R = L * fade, R * fade
    _write(out_path, L, R)
    print(f'[audio] {dur:.2f}s, {integrated_lufs(L, R):.1f} LUFS', flush=True)


def true_peak(L, R, ceiling_db=-1.5):
    """Keep the inter-sample peaks (4x oversampled) under the ceiling."""
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import resample_poly
    c = 10 ** (ceiling_db / 20)
    peak = np.maximum(np.abs(resample_poly(L, 4, 1)), np.abs(resample_poly(R, 4, 1)))
    peak = peak[:len(L) * 4].reshape(-1, 4).max(1)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-9))
    w = int(0.004 * SR)
    g = minimum_filter1d(need, size=2 * w + 1)
    g = np.convolve(g, np.ones(w) / w, 'same')
    return L * g, R * g
