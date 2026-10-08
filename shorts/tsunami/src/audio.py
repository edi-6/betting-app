"""The soundtrack, synthesised here from nothing (no samples), following the edit (edit.py) and the simulations.

Music at 120 bpm that climbs with the rounds: a big low hit and the wall's roar for the cold open; a light plucked
tune for one block; drums and a bass line for ten; for a hundred, the church bell, strings and toms, a drop into
slow motion with booms; for a thousand, the music falls away to a drone and a heartbeat under the roar, stops dead
for the sponge, and comes back as a little victory tune.
The effects: the sea, each wave's roar and crash, water rushing through the streets, every block that breaks (a
crack for wood, a crunch for stone, thousands of them become a roar of their own; slowed down with the picture), the
bell, the villager's 'hmm', the sponge going down and drinking the sea, the stamps. Loudness to -14 LUFS, limited.
"""
import os

import numpy as np

import edit as E
import sim as SM
import village as VL
import blocks as BL
from sound import (SR, Mix, bp, compress, expenv, hp, integrated_lufs, limiter, lp, midi_hz, modal, norm,
                   reverb, shaped_noise, sine_sweep, strings, _write)

BPM = 120.0
BEAT = 60.0 / BPM


# -- instruments -------------------------------------------------------------------------------------------------------
def pluck(m, dur=0.35, vel=0.8, rng=None):
    """A marimba-ish pluck: a sine with a short bright attack."""
    f = midi_hz(m)
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / (0.22 + 0.4 * dur))
    x += 0.35 * np.sin(2 * np.pi * 4.0 * f * t) * np.exp(-t / 0.03)
    x += 0.15 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.08)
    x *= np.clip(t / 0.002, 0, 1)
    return x * vel


def bass(m, dur, vel=0.8):
    f = midi_hz(m)
    n = int((dur + 0.15) * SR)
    t = np.arange(n) / SR
    saw = 2 * np.mod(f * t, 1.0) - 1
    x = lp(saw, 220 + 900 * np.exp(-t[0] * 0), 2) * 0.7 + 0.6 * np.sin(2 * np.pi * f * t)
    env = np.clip(t / 0.005, 0, 1) * np.where(t < dur, np.exp(-t / (dur * 2.5)), np.exp(-(t - dur) / 0.04) *
                                              np.exp(-dur / (dur * 2.5)))
    return lp(x, 900, 2) * env * vel


def kick(vel=1.0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    x = sine_sweep(0.45, 150.0, 45.0, 0.04) * np.exp(-t / 0.16)
    x += 0.25 * lp(np.random.default_rng(1).standard_normal(n), 3000, 2) * np.exp(-t / 0.004)
    return norm(x, 0.9) * vel


def snare(vel=1.0, rng=None):
    rng = rng or np.random.default_rng(2)
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 1500, 9000, 2) * np.exp(-t / 0.07)
    x += 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return norm(x, 0.8) * vel


def hat(vel=1.0, rng=None, open_=False):
    rng = rng or np.random.default_rng(3)
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 7000, 2) * np.exp(-t / (0.08 if open_ else 0.015))
    return norm(x, 0.5) * vel


def tom(f0=110.0, vel=1.0):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    x = sine_sweep(0.6, f0 * 1.6, f0, 0.05) * np.exp(-t / 0.22)
    x += 0.2 * lp(np.random.default_rng(4).standard_normal(n), 1500, 2) * np.exp(-t / 0.01)
    return norm(x, 0.9) * vel


def braam(notes, dur, rng, bright=1.0):
    """A huge low brass hit: detuned saws through a filter that opens and closes, driven hard."""
    n = int((dur + 1.0) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        f = midi_hz(m)
        for d in (-0.008, -0.003, 0.0, 0.004, 0.009):
            x += 2 * np.mod(f * (1 + d) * t + rng.random(), 1.0) - 1
    env = np.clip(t / 0.02, 0, 1) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    cut = 180 + 1500 * bright * np.exp(-t / 0.35)
    # a moving low-pass: blocks of fixed cutoff, cross-faded
    out = np.zeros(n)
    blk = int(0.05 * SR)
    for s in range(0, n, blk):
        seg = x[max(0, s - blk):s + blk]
        y = lp(seg, float(cut[min(s, n - 1)]), 2)
        out[s:s + blk] = y[-min(blk, n - s):] if s > 0 else y[:min(blk, n - s)]
    out = np.tanh(2.2 * norm(out)) * env
    sub = np.sin(2 * np.pi * midi_hz(min(notes) - 12) * t) * env
    return norm(out + 0.7 * sub, 0.95)


def church_bell(rng, f0=196.0, dur=4.5):
    """A big bronze bell: hum, prime, minor third, fifth, nominal, with long decays."""
    x = modal(dur, f0, (0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011),
              (0.6, 1.0, 0.8, 0.45, 0.6, 0.3, 0.25, 0.15), (4.0, 3.0, 2.2, 1.6, 1.4, 0.9, 0.8, 0.6), rng, 0.002)
    n = len(x)
    x += 0.3 * hp(rng.standard_normal(n), 2000, 2) * expenv(n, 0.01)
    return norm(x * np.clip(np.arange(n) / SR / 0.003, 0, 1), 0.9)


# -- effects ------------------------------------------------------------------------------------------------------------
def villager_hmm(rng, f_base=175.0, dur=0.6, up=False):
    """The villager's 'hmm': a nasal hum with a little pitch hook."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    f = f_base * (1.0 + 0.16 * np.sin(np.pi * np.clip(u * 1.3, 0, 1)) - (0.0 if up else 0.18) * u ** 2 +
                  (0.22 * u ** 2 if up else 0.0))
    f *= 1.0 + 0.012 * np.sin(2 * np.pi * 5.5 * t)
    ph = np.cumsum(f) / SR
    src = np.zeros(n)
    for k in range(1, 18):
        src += np.sin(2 * np.pi * k * ph) / k ** 1.1
    # nasal formants: a strong low one, a honk in the middle, a little air on top
    y = 1.0 * bp(src, 180, 420, 2) + 0.55 * bp(src, 900, 1500, 2) + 0.18 * bp(src, 2200, 3200, 2)
    y = y - 0.3 * bp(y, 650, 800, 2)
    env = np.clip(t / 0.05, 0, 1) * np.clip((dur - t) / 0.14, 0, 1) ** 1.3
    br = bp(rng.standard_normal(n), 400, 2500, 2) * np.exp(-t / 0.04) * 0.15
    return norm((y + br) * env, 0.8)


def sea(rng, dur, level=1.0):
    """The sea on a calm evening: slow swells of soft surf."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * t / 5.3) ** 2 * (0.7 + 0.3 * np.sin(2 * np.pi * t / 13.0))
    x = lp(rng.standard_normal(n), 900, 2) * 0.6 + 0.35 * bp(rng.standard_normal(n), 1500, 6000, 2) * swell ** 2
    return norm(x * swell, 0.5) * level


def roar(rng, dur, size=1.0, rise=True):
    """A wave's roar: deep, rumbling, churning; it swells as it comes."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    churn = 0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 6.0, 1)) * 12
    lo = lp(rng.standard_normal(n), 160 + 60 / size, 2)
    mid = bp(rng.standard_normal(n), 250, 1800 / size ** 0.3, 2)
    hi = bp(rng.standard_normal(n), 2500, 7000, 2)
    x = 1.0 * norm(lo) + 0.7 * norm(mid) * churn + 0.18 * norm(hi) * churn
    env = (t / dur) ** 1.6 if rise else np.ones(n)
    return norm(np.tanh(1.2 * x) * env * np.clip(t / 0.3, 0, 1), 0.9)


def crash(rng, size=1.0, dur=None):
    """A wave breaking: a thump, a huge rush of white noise, hiss of the spray."""
    dur = dur or (1.4 + 1.6 * size)
    n = int(dur * SR)
    t = np.arange(n) / SR
    thump = sine_sweep(dur, 90.0, 35.0, 0.15) * np.exp(-t / (0.25 * size + 0.1))
    body = shaped_noise(dur, lambda tt: 500 + 1800 * np.exp(-tt / 0.4), lambda tt: 1.6 + 0 * tt,
                        lambda tt: np.clip(tt / 0.02, 0, 1) * np.exp(-tt / (0.5 + 0.6 * size)), rng)
    spray = hp(rng.standard_normal(n), 3000, 2) * np.exp(-t / (0.6 + 0.4 * size)) * np.clip(t / 0.05, 0, 1)
    x = 0.8 * norm(thump) * min(1.0, size) + 1.0 * norm(body) + 0.35 * norm(spray)
    return norm(np.tanh(1.4 * x) * np.clip((dur - t) / 0.3, 0, 1), 0.95)


def small_wave(rng):
    """The one-block wave: a soft break on the sand, then the fizz of it running up the beach."""
    dur = 2.4
    n = int(dur * SR)
    t = np.arange(n) / SR
    br = shaped_noise(dur, lambda tt: 700 + 900 * np.exp(-tt / 0.25), lambda tt: 1.4 + 0 * tt,
                      lambda tt: np.clip(tt / 0.04, 0, 1) * np.exp(-tt / 0.45), rng)
    fizz = hp(rng.standard_normal(n), 2500, 2) * np.clip((t - 0.3) / 0.3, 0, 1) * np.exp(-np.clip(t - 0.6, 0, None)
                                                                                         / 0.7)
    fizz *= 0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 30, 1)) * 15
    return norm(1.0 * norm(br) + 0.3 * norm(fizz), 0.8)


def rush(rng, dur):
    """Water rushing through the streets."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    tur = 0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 9.0, 1)) * 14
    x = bp(rng.standard_normal(n), 200, 2200, 2) * tur + 0.3 * hp(rng.standard_normal(n), 3500, 2) * tur
    x += 0.6 * lp(rng.standard_normal(n), 120, 2)
    return norm(x * np.clip(t / 0.4, 0, 1) * np.clip((dur - t) / 0.5, 0, 1), 0.8)


def crack(rng, low=1.0, stone=False):
    """A block breaking loose: wood cracks, stone crunches."""
    if stone:
        dur = 0.25
        n = int(dur * SR)
        t = np.arange(n) / SR
        x = lp(rng.standard_normal(n), 1800 * low, 2) * np.exp(-t / 0.05)
        x += 0.6 * sine_sweep(dur, 140 * low, 60 * low, 0.04) * np.exp(-t / 0.06)
    else:
        dur = 0.18
        n = int(dur * SR)
        t = np.arange(n) / SR
        x = modal(dur, rng.uniform(500, 1100) * low, (1.0, 1.7, 2.9), (1.0, 0.6, 0.3), (0.03, 0.02, 0.012), rng)
        x += 0.8 * bp(rng.standard_normal(n), 1200 * low, 6000 * low, 2) * np.exp(-t / 0.012)
    return norm(x * np.clip(t / 0.0005, 0, 1), 0.8)


def place(rng):
    """A sponge put down: a soft, squishy thump."""
    dur = 0.35
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sine_sweep(dur, 260.0, 120.0, 0.03) * np.exp(-t / 0.05)
    x += 0.8 * bp(rng.standard_normal(n), 300, 2500, 2) * np.exp(-t / 0.035)
    sq = np.sin(2 * np.pi * np.cumsum(500 + 900 * np.exp(-t / 0.04)) / SR) * np.exp(-t / 0.03) * 0.3
    return norm(x + sq, 0.9)


def slurp(rng, dur=2.2):
    """The sponge drinking the sea: a vast sucking roar sweeping up, gurgling, ending in a gulp."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    suck = shaped_noise(dur, lambda tt: 160 + 3400 * (tt / dur) ** 2.2, lambda tt: 0.45 + 0 * tt,
                        lambda tt: np.clip(tt / 0.08, 0, 1) * (0.5 + 0.5 * (tt / dur)), rng)
    rumble = lp(rng.standard_normal(n), 140, 2) * (1 - u) ** 0.7
    gurgle = np.zeros(n)
    k = 0
    while k < n:
        d = int(rng.uniform(0.02, 0.07) * SR)
        f0 = rng.uniform(250, 700) * (1 + 1.5 * u[min(k, n - 1)])
        tt = np.arange(d) / SR
        g = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 2.5 * tt / (d / SR))) / SR) * np.sin(np.pi * tt / (d / SR))
        gurgle[k:k + d] += g[:n - k] * rng.uniform(0.3, 1.0)
        k += int(rng.uniform(0.015, 0.06) * SR)
    x = 1.0 * norm(suck) + 0.8 * norm(rumble) + 0.35 * norm(gurgle)
    x *= np.clip((dur - t) / 0.06, 0, 1)
    gulp_n = int(0.4 * SR)
    tg = np.arange(gulp_n) / SR
    gulp = sine_sweep(0.4, 320.0, 70.0, 0.06) * np.exp(-tg / 0.09) + 0.4 * lp(rng.standard_normal(gulp_n), 900, 2) \
        * np.exp(-tg / 0.03)
    out = np.concatenate([norm(np.tanh(1.3 * x), 0.9), norm(gulp, 1.0)])
    return out


def whoosh(rng, dur=0.45, lo=400, hi=4000):
    x = shaped_noise(dur, lambda tt: lo + hi * np.sin(np.pi * np.clip(tt / dur, 0, 1)), lambda tt: 0.8 + 0 * tt,
                     lambda tt: np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2, rng)
    return norm(x, 0.8)


def boom(rng, size=1.0):
    dur = 1.6 + size
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sine_sweep(dur, 85.0, 30.0, 0.25) * np.exp(-t / (0.5 + 0.4 * size))
    x += 0.5 * lp(rng.standard_normal(n), 300, 2) * np.exp(-t / 0.25)
    x += 0.3 * bp(rng.standard_normal(n), 800, 5000, 2) * np.exp(-t / 0.03)
    return norm(np.tanh(1.5 * x), 0.95)


def ding(rng, ok=True):
    """The verdict: a bright two-note chime, or a low buzz."""
    if ok:
        out = np.zeros(int(1.6 * SR))
        for k, m in enumerate((84, 91)):
            f = midi_hz(m)
            n = len(out) - int(k * 0.11 * SR)
            t = np.arange(n) / SR
            x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.2)) * \
                np.exp(-t / 0.5)
            out[int(k * 0.11 * SR):] += x * np.clip(t / 0.003, 0, 1)
        return norm(out, 0.8)
    dur = 0.9
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * 98 * t)) + np.sign(np.sin(2 * np.pi * 103.5 * t))
    x = lp(x, 1400, 2) * np.clip(t / 0.01, 0, 1) * np.clip((0.6 - t) / 0.05, 0, 1)
    x += 0.9 * sine_sweep(dur, 110.0, 45.0, 0.2) * np.exp(-t / 0.3)
    return norm(x, 0.85)


def heartbeat(rng):
    n = int(0.6 * SR)
    x = np.zeros(n)
    for (s, a) in ((0.0, 1.0), (0.2, 0.7)):
        m = n - int(s * SR)
        x[int(s * SR):] += a * sine_sweep(m / SR, 70.0, 40.0, 0.05) * expenv(m, 0.1, 0.004)
    return norm(lp(x, 180, 2), 0.9)


# -- when things happen ---------------------------------------------------------------------------------------------
def shot(name):
    return next(s for s in E.SHOTS if s.name == name)


def screen_time(rnd, t_round, names=None):
    """The first screen time at which a shot of this round shows round time t_round (None if none does)."""
    for s in E.SHOTS:
        if s.rnd != rnd or (names and s.name not in names):
            continue
        us = np.linspace(0, 1, 400)
        ts = np.array([s.clock(u) for u in us])
        if ts[0] <= t_round <= ts[-1]:
            k = int(np.searchsorted(ts, t_round))
            return s.start + us[min(k, len(us) - 1)] * s.dur
    return None


def break_events():
    """Every block breaking on screen: (screen time, stone?, speed of the picture)."""
    out = []
    ev = {}
    for n in (2, 3):
        m = np.load(os.path.join(SM.round_dir(n), 'meta.npz'))
        e = m['events'].reshape(-1, 2)
        kinds = m['kinds']
        stone = np.isin(kinds, [BL.KIND_ID[k] for k in ('cobble', 'mossy_cobble', 'bricks', 'mossy_bricks',
                                                        'stone', 'terracotta')])
        ev[n] = (e[:, 0], stone)
    for s in E.SHOTS:
        if s.rnd not in ev:
            continue
        times, stone = ev[s.rnd]
        nf = int(round(s.dur * E.FPS))
        prev = s.clock(0.0)
        for i in range(1, nf + 1):
            u = i / nf
            T = s.clock(u)
            sel = np.nonzero((times > prev) & (times <= T))[0]
            speed = (T - prev) * E.FPS
            ts = s.start + u * s.dur
            for j in sel:
                out.append((ts + np.random.default_rng(int(j)).uniform(-0.5, 0.5) / E.FPS, bool(stone[j]), speed))
            prev = T
    return out


# -- the mix ------------------------------------------------------------------------------------------------------------
def build(out_path, seed=7):
    rng = np.random.default_rng(seed)
    D = E.DURATION + 1.5
    fx = Mix(D)
    mus = Mix(D)
    S = {s.name: s for s in E.SHOTS}

    def st(name):
        return S[name].start

    def en(name):
        return S[name].start + S[name].dur

    # ---------------- cold open: the wall --------------------------------------------------------------------------
    fx.add(boom(rng, 1.4), 0.0, 1.0)
    mus.add(braam([33, 40, 45], 1.2, rng), 0.0, 0.9)
    fx.add(roar(rng, st('r1') + 0.3, 2.0, rise=False) * np.clip(1 - np.arange(int((st('r1') + 0.3) * SR)) /
                                                                  SR / (st('r1') + 0.3), 0, 1) ** 0.5,
           0.0, 0.55)
    fx.add(whoosh(rng, 0.4), st('r1') - 0.3, 0.5)

    # ---------------- round 1: one block --------------------------------------------------------------------------
    r1 = S['r1']
    fx.add(sea(rng, r1.dur + 0.5, 1.0), r1.start, 0.5)
    t_break = screen_time(1, 14.45)
    if t_break is not None:
        fx.add(small_wave(rng), t_break, 0.75)
    t_feet = screen_time(1, 15.95)
    if t_feet is not None:
        fx.add(hp(rng.standard_normal(int(0.8 * SR)), 2000, 2) * expenv(int(0.8 * SR), 0.3, 0.05) * 0.4, t_feet,
               0.4, -0.1)
    t_hmm = screen_time(1, 16.25)
    if t_hmm is not None:
        fx.add(villager_hmm(rng, 170.0, 0.62), t_hmm, 0.95)
    fx.add(boom(rng, 0.2) * 0.5, r1.start, 0.45)
    # the tune: a light pluck figure in C
    melody = [72, 76, 79, 76, 81, 79, 76, 74, 72, 74, 76, 79, 77, 76, 74, 72]
    tb = r1.start
    k = 0
    while tb < en('r1') - 0.1:
        mus.add(pluck(melody[k % len(melody)], 0.22, 0.55), tb, 0.55, 0.15 * np.sin(k))
        if k % 4 == 0:
            mus.add(pluck(48 + [0, 5, 7, 5][(k // 4) % 4], 0.6, 0.6), tb, 0.6, -0.2)
            mus.add(kick(0.5), tb, 0.45)
        tb += BEAT / 2
        k += 1

    # ---------------- round 2: ten blocks -------------------------------------------------------------------------
    t2 = st('r2_coming')
    fx.add(boom(rng, 0.6), t2, 0.75)
    fx.add(whoosh(rng, 0.5), t2 - 0.35, 0.45)
    fx.add(roar(rng, st('r2_hit') - t2 + 0.3, 1.0), t2, 0.8)
    t_hit = screen_time(2, 13.45)
    if t_hit is not None:
        fx.add(crash(rng, 0.8), t_hit, 0.7)
    t_houses = screen_time(2, 14.2)
    if t_houses is not None:
        fx.add(crash(rng, 1.4), t_houses, 1.0)
        fx.add(boom(rng, 0.6), t_houses, 0.6)
    fx.add(rush(rng, en('r2_flood') - st('r2_hit') + 0.6), st('r2_hit') + 0.2, 0.55)
    # drums and bass, A minor
    bass_line = [45, 45, 48, 43]
    tb = t2
    k = 0
    while tb < en('r2_flood') - 0.05:
        mus.add(kick(0.8), tb, 0.6)
        if k % 2 == 1:
            mus.add(snare(0.7, rng), tb, 0.45, 0.1)
        for h in range(2):
            mus.add(hat(0.4, rng), tb + h * BEAT / 2, 0.25, 0.3)
        mus.add(bass(bass_line[(k // 4) % 4], BEAT * 0.9, 0.8), tb, 0.5)
        mus.add(pluck([69, 72, 76, 72][k % 4] + (3 if (k // 4) % 4 == 2 else 0), 0.2, 0.4), tb + BEAT / 2, 0.35,
                -0.2)
        tb += BEAT
        k += 1

    # ---------------- round 3: a hundred --------------------------------------------------------------------------
    t3 = st('r3_horizon')
    fx.add(boom(rng, 1.0), t3, 0.85)
    fx.add(whoosh(rng, 0.5), t3 - 0.35, 0.5)
    for k, tb in enumerate((t3 + 0.15, t3 + 1.05, st('r3_look_up') + 0.3)):
        fx.add(church_bell(rng, 196.0), tb, 0.55 - 0.08 * k, -0.25)
    fx.add(roar(rng, st('r3_impact') - t3 + 0.2, 2.2), t3, 1.0)
    mus.add(strings([midi_hz(m) for m in (45, 52, 57, 60)], st('r3_impact') - t3, rng, attack=1.2, release=0.8,
                    bright=1600), t3, 0.7)
    for k in range(int((st('r3_impact') - t3) / (BEAT / 2))):
        tb = t3 + k * BEAT / 2
        if k % 2 == 0 or k > 8:
            mus.add(tom(90.0 + 20 * (k % 3), 0.5 + 0.04 * k), tb, 0.5, 0.2 * np.sin(k))
    # slow motion: the music drops out to a drone, every hit lands low and heavy
    imp = S['r3_impact']
    mus.add(braam([33, 40], imp.dur * 0.7, rng, 0.6), imp.start, 0.6)
    t_wall = screen_time(3, 11.0, ('r3_impact',))
    if t_wall is not None:
        fx.add(crash(rng, 3.0, 3.6), t_wall, 1.0)
        fx.add(boom(rng, 2.0), t_wall, 0.9)
    fx.add(rush(rng, en('r3_gone') - st('r3_gone') + 0.4), st('r3_gone'), 0.5)
    fx.add(roar(rng, en('r3_gone') - st('r3_gone'), 1.5, rise=False), st('r3_gone'), 0.45)

    # breaking blocks (rounds 2 and 3), thinned out where there are too many to hear one by one
    evs = break_events()
    last = -1.0
    dens = np.zeros(int(D * 100) + 1)
    for (ts, stone, speed) in evs:
        dens[int(ts * 100)] += 1
    for (ts, stone, speed) in evs:
        local = dens[int(ts * 100)]
        if local > 3 and rng.random() > 3.0 / local:
            continue
        low = float(np.clip(speed, 0.35, 1.0)) ** 0.5
        c = crack(rng, low * rng.uniform(0.8, 1.2), stone)
        if speed < 0.8:
            c = np.interp(np.arange(int(len(c) / low)) * low, np.arange(len(c)), c)
        fx.add(c, ts, 0.18 + 0.1 * rng.random(), rng.uniform(-0.6, 0.6))
        last = ts
    # where blocks break in their thousands, a continuous crunching roar
    env = np.convolve(dens, np.ones(15) / 15, mode='same')
    if env.max() > 0:
        n = int(D * SR)
        e = np.interp(np.arange(n) / SR * 100, np.arange(len(env)), env)
        e = np.clip(e / 6.0, 0, 1.2)
        bed = bp(rng.standard_normal(n), 300, 4000, 2) * (0.6 + 0.4 * np.abs(lp(rng.standard_normal(n), 25, 1))
                                                          * 15)
        fx.add(norm(bed) * e, 0.0, 0.35)

    # ---------------- round 4: a thousand -------------------------------------------------------------------------
    t4 = st('r4_wide')
    fx.add(boom(rng, 2.0), t4, 1.0)
    mus.add(braam([28, 35, 40], 1.6, rng, 1.2), t4, 0.9)
    fx.add(whoosh(rng, 0.6), t4 - 0.45, 0.5)
    sp = S['r4_sponge']
    t_place = sp.start + sp.extra['place'] * sp.dur
    t_drink = sp.start + sp.extra['drink'][0] * sp.dur
    t_dry = sp.start + sp.extra['drink'][1] * sp.dur
    # the wall's roar: deep and everywhere, growing, until the sponge goes down
    n_roar = int((t_place - t4) * SR)
    rr = roar(rng, t_place - t4 + 0.2, 3.0, rise=False)
    rr *= np.clip(0.45 + 0.55 * np.arange(len(rr)) / SR / (t_place - t4), 0, 1)
    rr[n_roar:] *= np.exp(-np.arange(len(rr) - n_roar) / SR / 0.05)
    fx.add(rr, t4, 0.9)
    # a drone and a heartbeat under it
    dn = int((t_place - t4) * SR)
    td = np.arange(dn) / SR
    drone = sum(a * np.tanh(1.4 * np.sin(2 * np.pi * f * td)) for f, a in ((41.2, 1.0), (43.65, 0.7), (82.4, 0.4)))
    mus.add(lp(drone, 300, 2) * np.clip(td / 1.5, 0, 1) * np.clip((dn / SR - td) / 0.05, 0, 1), t4, 0.45)
    tb = st('r4_street')
    k = 0
    while tb < t_place - 0.3:
        mus.add(heartbeat(rng), tb, 0.7)
        tb += 0.75 - 0.03 * k
        k += 1
    mus.add(strings([midi_hz(m) for m in (52, 53, 59, 64, 65)], t_place - st('r4_street'), rng, attack=2.5,
                    release=0.05, bright=2600), st('r4_street'), 0.55)
    # the swing and the sponge going down... and silence
    fx.add(whoosh(rng, 0.22, 600, 2500), t_place - 0.12, 0.35)
    fx.add(place(rng), t_place, 1.0)
    # the drink
    sl = slurp(rng, t_dry - t_drink)
    fx.add(sl, t_drink, 1.0)
    fx.add(boom(rng, 0.8) * 0.6, t_drink, 0.6)
    # calm: a breeze, the wet sponge, the villager
    calm = int((D - t_dry) * SR)
    tc = np.arange(calm) / SR
    breeze = bp(rng.standard_normal(calm), 300, 2500, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * tc / 6.0) ** 2)
    fx.add(norm(breeze) * np.clip(tc / 1.0, 0, 1), t_dry, 0.12)
    hm = S['r4_hmm']
    for (uu, tt) in ((0.05, 0.0), (0.16, 0.0), (0.27, 0.0), (0.38, 0.0)):
        stp = bp(rng.standard_normal(int(0.1 * SR)), 300, 3000, 2) * expenv(int(0.1 * SR), 0.03, 0.003)
        fx.add(norm(stp, 0.4), hm.start + uu * hm.dur, 0.25, -0.3)
    fx.add(villager_hmm(rng, 165.0, 0.7), hm.start + 0.50 * hm.dur, 1.0)
    fx.add(villager_hmm(rng, 185.0, 0.5, up=True), hm.start + 0.84 * hm.dur, 0.9)

    # the sponge turning wet: a squelch
    t_wet = sp.start + (sp.extra['drink'][0] + 0.45 * (sp.extra['drink'][1] - sp.extra['drink'][0])) * sp.dur
    sq_n = int(0.5 * SR)
    tq = np.arange(sq_n) / SR
    squelch = bp(rng.standard_normal(sq_n), 250, 1800, 2) * np.exp(-tq / 0.08) * (1 + np.sin(2 * np.pi * 23 * tq))
    fx.add(norm(squelch, 0.6), t_wet, 0.5, -0.2)

    # the village losing hearts: a dull hit for every half heart (not too many at once)
    import playback as PB
    import village as VLG
    import scene as SCN
    vil = VLG.build(SCN.world_data()['H'])
    pbs = {n: PB.Playback(n, vil) for n in (2, 3)}
    last_hit = -1.0
    prev_half = 20
    for s_ in E.SHOTS:
        if s_.rnd not in (2, 3):
            prev_half = 20
            continue
        nf = int(round(s_.dur * E.FPS))
        for i in range(nf):
            u = i / nf
            frac = 1.0 - pbs[s_.rnd].broken(s_.clock(u)) / 8022.0
            half = int(np.ceil(20 * frac - 0.25)) if frac > 0 else 0
            ts = s_.start + u * s_.dur
            if half < prev_half and ts - last_hit > 0.09:
                hn = int(0.18 * SR)
                th = np.arange(hn) / SR
                hit = sine_sweep(0.18, 210.0, 90.0, 0.02) * np.exp(-th / 0.05) + 0.5 * bp(
                    rng.standard_normal(hn), 400, 2500, 2) * np.exp(-th / 0.015)
                fx.add(norm(hit, 0.7), ts, 0.35)
                last_hit = ts
            prev_half = half

    # ---------------- the round titles and the verdicts -------------------------------------------------------------
    import main as MN
    for n in (1, 2, 3, 4):
        tv = MN.verdict_time(n)
        fx.add(ding(rng, MN.OK[n]), tv, 0.7 if MN.OK[n] else 0.8)
        if n == 4:
            # the little victory tune
            for k, m in enumerate((72, 76, 79, 84, 79, 84)):
                mus.add(pluck(m, 0.3 if k < 5 else 0.9, 0.8), tv + 0.15 + k * BEAT / 3, 0.75, 0.1 * np.sin(k))
            mus.add(pluck(48, 1.2, 0.7), tv + 0.15, 0.6)
            mus.add(pluck(55, 1.2, 0.6), tv + 0.15 + 5 * BEAT / 3, 0.5)

    # ---------------- mix down -----------------------------------------------------------------------------------------
    mL, mR = reverb(mus.L, mus.R, decay=1.8, wet=0.22)
    fL, fR = reverb(fx.L, fx.R, seed=5, decay=1.2, wet=0.12)
    # take the edge off the hiss of all that water (a gentle high shelf from 6 kHz)
    lo_l, lo_r = lp(fL, 6000.0, 2), lp(fR, 6000.0, 2)
    fL, fR = lo_l + 0.6 * (fL - lo_l), lo_r + 0.6 * (fR - lo_r)
    L = fL + 0.55 * mL
    R = fR + 0.55 * mR
    L, R = hp(L, 30.0, 2), hp(R, 30.0, 2)
    L, R = compress(L, R, -20.0, 2.5)
    n = int(round(E.DURATION * SR))
    L, R = L[:n], R[:n]
    fade = np.clip((n - np.arange(n)) / (0.3 * SR), 0, 1)
    L, R = L * fade, R * fade
    lufs = integrated_lufs(L, R)
    g = 10 ** ((-14.0 - lufs) / 20)
    L, R = limiter(L * g, R * g, 0.89)
    _write(out_path, L, R)
    print(f'[audio] {out_path}: {n / SR:.1f}s, {integrated_lufs(L, R):.1f} LUFS', flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                                'cache', 'audio.wav'))
